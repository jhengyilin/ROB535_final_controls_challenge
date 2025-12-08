import numpy as np
import json
import os
import sys
import time
import importlib
from typing import List, Tuple, Optional, Callable, Dict
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# Ensure we can import from the current directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tinyphysics import TinyPhysicsSimulator, TinyPhysicsModel


@dataclass
class CMAESConfig:
    """Configuration for CMA-ES optimization."""
    
    # Population and generations
    population_size: int = 40  
    max_generations: int = 100
    
    # Convergence criteria
    target_cost: float = 30.0
    stagnation_generations: int = 20  
    
    # Initial distribution
    initial_sigma: float = 0.2 
    
    # Evaluation settings
    num_segments_per_eval: int = 20
    num_workers: int = 8 
    use_adaptive_segments: bool = True
    parallelize_population: bool = True  # Parallelize candidate evaluation
    population_workers: Optional[int] = None  # Workers for population-level parallelism  
    
    # Data paths
    data_dir: str = "./data"
    model_path: str = "./models/tinyphysics.onnx"
    
    # Controller to optimize
    controller_name: str = "custom_velocity"
    
    # Normalization
    use_normalization: bool = True
    
    # Warm start
    warm_start_file: Optional[str] = None


# Parameter definitions for multi-step lookahead controller
PARAM_NAMES = [
    'kp', 'ki', 'kd', 
    'k_roll', 
    'k_ff_immediate', 'k_ff_lookahead',
    'lookahead_start_idx', 'lookahead_end_idx', 'lookahead_decay',
    'error_filter_alpha',
    'action_smoothing',
    'velocity_scaling',
    # New velocity-dependent parameters
    'velocity_kp_scale',
    'velocity_ki_scale',
    'velocity_kd_scale',
    'velocity_ff_scale',
    'future_velocity_weight',
    'velocity_lookahead_scale',
]

DEFAULT_PARAMS = {
    'kp': 0.28,
    'ki': 0.025,
    'kd': 0.09,
    'k_roll': 0.92,
    'k_ff_immediate': 0.18,
    'k_ff_lookahead': 0.14,
    'lookahead_start_idx': 2,
    'lookahead_end_idx': 6,
    'lookahead_decay': 0.8,
    'error_filter_alpha': 0.35,
    'action_smoothing': 0.28,
    'velocity_scaling': 0.018,
    # New velocity-dependent parameters (initialized to 0 for backward compatibility)
    'velocity_kp_scale': 0.0,
    'velocity_ki_scale': 0.0,
    'velocity_kd_scale': 0.0,
    'velocity_ff_scale': 0.0,
    'future_velocity_weight': 0.0,
    'velocity_lookahead_scale': 0.0,
}

# Parameter bounds for multi-step lookahead
PARAM_BOUNDS = {
    'kp': (0.0, 1.5),              
    'ki': (0.0, 1.5),              
    'kd': (0.0, 1.5),              
    'k_roll': (0.0, 3.0),          
    'k_ff_immediate': (0.0, 1.5),  
    'k_ff_lookahead': (0.0, 1.5),  
    'lookahead_start_idx': (0, 10),      # Start of window
    'lookahead_end_idx': (1, 20),        # End of window
    'lookahead_decay': (0.5, 1.0),       # Decay factor
    'error_filter_alpha': (0.0, 1.0),
    'action_smoothing': (0.0, 1.0), 
    'velocity_scaling': (-0.2, 0.2),
    # New velocity-dependent parameter bounds
    'velocity_kp_scale': (-1.0, 1.0),      # Velocity scaling for P term
    'velocity_ki_scale': (-1.0, 1.0),      # Velocity scaling for I term
    'velocity_kd_scale': (-1.0, 1.0),      # Velocity scaling for D term
    'velocity_ff_scale': (-1.0, 1.0),     # Velocity scaling for feedforward
    'future_velocity_weight': (-0.5, 0.5), # Weight for future velocity in feedforward
    'velocity_lookahead_scale': (-0.3, 0.3), # Velocity-based lookahead weighting
}


def params_to_vector(params: Dict[str, float]) -> np.ndarray:
    """Convert parameter dict to vector."""
    return np.array([params[name] for name in PARAM_NAMES])


def vector_to_params(vector: np.ndarray) -> Dict[str, float]:
    """Convert parameter vector to dict."""
    return {name: float(vector[i]) for i, name in enumerate(PARAM_NAMES)}


def normalize_param_vector(params_vec: np.ndarray) -> np.ndarray:
    """Normalize parameter vector to [0, 1] range."""
    normalized = np.zeros_like(params_vec)
    for i, name in enumerate(PARAM_NAMES):
        low, high = PARAM_BOUNDS[name]
        normalized[i] = (params_vec[i] - low) / (high - low)
    return normalized


def denormalize_param_vector(normalized_vec: np.ndarray) -> np.ndarray:
    """Denormalize from [0, 1] back to physical parameter range."""
    physical = np.zeros_like(normalized_vec)
    for i, name in enumerate(PARAM_NAMES):
        low, high = PARAM_BOUNDS[name]
        physical[i] = low + normalized_vec[i] * (high - low)
    
    # Ensure lookahead indices are valid integers and ordered
    start_idx = PARAM_NAMES.index('lookahead_start_idx')
    end_idx = PARAM_NAMES.index('lookahead_end_idx')
    physical[start_idx] = max(0, int(physical[start_idx]))
    physical[end_idx] = max(physical[start_idx] + 1, int(physical[end_idx]))
    
    return physical


def get_search_bounds(normalized: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """Get lower and upper bounds for all parameters."""
    if normalized:
        return np.zeros(len(PARAM_NAMES)), np.ones(len(PARAM_NAMES))
    else:
        lower = np.array([PARAM_BOUNDS[name][0] for name in PARAM_NAMES])
        upper = np.array([PARAM_BOUNDS[name][1] for name in PARAM_NAMES])
        return lower, upper


class CMAESOptimizer:
    """Covariance Matrix Adaptation Evolution Strategy."""
    
    def __init__(
        self,
        objective_fn: Callable[[np.ndarray], float],
        initial_point: np.ndarray,
        sigma: float = 0.2,
        population_size: Optional[int] = None,
        bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        generation_callback: Optional[Callable] = None
    ):
        self.objective_fn = objective_fn
        self.dim = len(initial_point)
        self.bounds = bounds
        self.generation_callback = generation_callback
        
        # Population size (lambda)
        if population_size is None:
            self.lam = max(4 + int(3 * np.log(self.dim)), 12)
        else:
            self.lam = max(population_size, 12)
            
        # Number of parents (mu)
        self.mu = self.lam // 2
        
        # Recombination weights
        weights = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.weights = weights / weights.sum()
        
        # Effective selection mass
        self.mu_eff = 1.0 / (self.weights ** 2).sum()
        
        # Learning rates
        self.c_sigma = (self.mu_eff + 2) / (self.dim + self.mu_eff + 5)
        self.d_sigma = 1 + 2 * max(0, np.sqrt((self.mu_eff - 1) / (self.dim + 1)) - 1) + self.c_sigma
        self.cc = (4 + self.mu_eff / self.dim) / (self.dim + 4 + 2 * self.mu_eff / self.dim)
        self.c1 = 2 / ((self.dim + 1.3) ** 2 + self.mu_eff)
        self.cmu = min(1 - self.c1, 2 * (self.mu_eff - 2 + 1 / self.mu_eff) / 
                      ((self.dim + 2) ** 2 + self.mu_eff))
        
        # Initialize distribution
        self.mean = initial_point.copy()
        self.sigma = sigma
        self.C = np.eye(self.dim)
        self.p_sigma = np.zeros(self.dim)
        self.p_c = np.zeros(self.dim)
        
        # Expected norm of N(0,I)
        self.chi_n = np.sqrt(self.dim) * (1 - 1/(4*self.dim) + 1/(21*self.dim**2))
        
        # History tracking
        self.generation = 0
        self.best_cost = float('inf')
        self.best_params = initial_point.copy()
        self.cost_history = []
        
        print(f"\n=== CMA-ES Configuration (Multi-Step Lookahead) ===")
        print(f"Dimension: {self.dim}")
        print(f"Population size (λ): {self.lam}")
        print(f"Parents (μ): {self.mu}")
        print(f"Initial sigma: {self.sigma:.4f}")
        
    def _sample_population(self) -> List[np.ndarray]:
        """Sample candidate solutions from current distribution."""
        eigenvalues, eigenvectors = np.linalg.eigh(self.C)
        eigenvalues = np.maximum(eigenvalues, 1e-10)
        
        B = eigenvectors
        D = np.diag(np.sqrt(eigenvalues))
        
        population = []
        clipped_count = np.zeros(self.dim)
        
        for _ in range(self.lam):
            z = np.random.randn(self.dim)
            x = self.mean + self.sigma * (B @ D @ z)
            
            if self.bounds is not None:
                x_unclipped = x.copy()
                x = np.clip(x, self.bounds[0], self.bounds[1])
                clipped_count += (x != x_unclipped).astype(int)
                
            population.append(x)
        
        if self.generation == 0 or self.generation % 10 == 0:
            pop_array = np.array(population)
            print(f"\n=== Population Diversity (Generation {self.generation}) ===")
            for i, name in enumerate(PARAM_NAMES):
                clip_pct = 100 * clipped_count[i] / self.lam
                warning = " ⚠️ HIGH CLIPPING!" if clip_pct > 20 else ""
                print(f"  {name:25s}: std={pop_array[:, i].std():.4f}, "
                      f"range=[{pop_array[:, i].min():.4f}, {pop_array[:, i].max():.4f}], "
                      f"clipped={clip_pct:.1f}%{warning}")
            print()
            
        return population
    
    def _update(self, population: List[np.ndarray], costs: List[float]):
        """Update distribution parameters based on evaluations."""
        indices = np.argsort(costs)
        
        if costs[indices[0]] < self.best_cost:
            improvement = self.best_cost - costs[indices[0]]
            self.best_cost = costs[indices[0]]
            self.best_params = population[indices[0]].copy()
            print(f"  ✓ New best! Cost: {self.best_cost:.4f} (improved by {improvement:.4f})")
            
        self.cost_history.append(self.best_cost)
        
        selected = [population[i] for i in indices[:self.mu]]
        old_mean = self.mean.copy()
        
        self.mean = np.zeros(self.dim)
        for i, x in enumerate(selected):
            self.mean += self.weights[i] * x
            
        eigenvalues, eigenvectors = np.linalg.eigh(self.C)
        eigenvalues = np.maximum(eigenvalues, 1e-10)
        B = eigenvectors
        D_inv = np.diag(1.0 / np.sqrt(eigenvalues))
        
        self.p_sigma = ((1 - self.c_sigma) * self.p_sigma + 
                        np.sqrt(self.c_sigma * (2 - self.c_sigma) * self.mu_eff) *
                        B @ D_inv @ B.T @ (self.mean - old_mean) / self.sigma)
        
        self.sigma *= np.exp((self.c_sigma / self.d_sigma) * 
                             (np.linalg.norm(self.p_sigma) / self.chi_n - 1))
        
        h_sigma = 1 if (np.linalg.norm(self.p_sigma) / 
                       np.sqrt(1 - (1 - self.c_sigma) ** (2 * (self.generation + 1)))) < \
                       (1.4 + 2 / (self.dim + 1)) * self.chi_n else 0
        
        self.p_c = ((1 - self.cc) * self.p_c + 
                   h_sigma * np.sqrt(self.cc * (2 - self.cc) * self.mu_eff) *
                   (self.mean - old_mean) / self.sigma)
        
        c1_term = self.c1 * np.outer(self.p_c, self.p_c)
        
        cmu_term = np.zeros((self.dim, self.dim))
        for i, x in enumerate(selected):
            y = (x - old_mean) / self.sigma
            cmu_term += self.weights[i] * np.outer(y, y)
        cmu_term *= self.cmu
        
        self.C = ((1 - self.c1 - self.cmu + 
                  (1 - h_sigma) * self.c1 * self.cc * (2 - self.cc)) * self.C +
                 c1_term + cmu_term)
        
        self.C = (self.C + self.C.T) / 2
        eigenvalues, eigenvectors = np.linalg.eigh(self.C)
        eigenvalues = np.maximum(eigenvalues, 1e-10)
        eigenvalues = np.minimum(eigenvalues, 1e6)
        self.C = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
        
        self.generation += 1
        
    def optimize(
        self, 
        max_generations: int = 100,
        target_cost: float = 0.0,
        stagnation_limit: int = 20,
        callback: Optional[Callable] = None,
        parallelize_population: bool = True,
        population_workers: Optional[int] = None
    ) -> Tuple[np.ndarray, float]:
        """Run full optimization loop."""
        stagnation_counter = 0
        improvement_window = []
        window_size = 10
        
        # Determine population-level parallelism
        if population_workers is None:
            # Use half of available CPU cores for population-level parallelism
            try:
                cpu_count = os.cpu_count() or 10
                population_workers = max(4, min(cpu_count // 2, 8))
            except:
                population_workers = 4
        
        for gen in range(max_generations):
            print(f"\n{'='*60}")
            print(f"Generation {gen}")
            print(f"{'='*60}")
            print(f"Sigma: {self.sigma:.4f}")
            
            population = self._sample_population()
            
            print(f"Evaluating {len(population)} candidates...")
            
            if parallelize_population and len(population) > 1:
                # Parallelize population evaluation
                # Get config info from stored attributes
                config_dict = getattr(self, '_eval_config_dict', None)
                all_segments = getattr(self, '_eval_all_segments', None)
                
                if config_dict is None or all_segments is None:
                    # Fallback to sequential if config not set up
                    print("  ⚠️  Config not set up for parallel evaluation, using sequential")
                    costs = []
                    for i, params in enumerate(population):
                        cost = self.objective_fn(params)
                        costs.append(cost)
                        if (i + 1) % 10 == 0 or (i + 1) == len(population):
                            print(f"  Progress: {i+1}/{len(population)}, "
                                  f"best so far: {min(costs):.4f}")
                else:
                    costs = []
                    completed = 0
                    with ProcessPoolExecutor(max_workers=population_workers) as executor:
                        futures = {
                            executor.submit(
                                evaluate_candidate_for_population, 
                                params, 
                                config_dict,
                                all_segments,
                                self.generation
                            ): i 
                            for i, params in enumerate(population)
                        }
                        
                        for future in as_completed(futures):
                            try:
                                cost = future.result(timeout=300)  # 5 min timeout per candidate
                                idx = futures[future]
                                costs.append((idx, cost))
                                completed += 1
                                if completed % 10 == 0 or completed == len(population):
                                    best_so_far = min([c[1] for c in costs]) if costs else float('inf')
                                    print(f"  Progress: {completed}/{len(population)}, "
                                          f"best so far: {best_so_far:.4f}")
                            except Exception as e:
                                print(f"  ⚠️  Candidate evaluation failed: {e}")
                                idx = futures[future]
                                costs.append((idx, 1000.0))  # Penalty for failed evaluation
                                completed += 1
                    
                    # Sort by original index and extract costs
                    costs = [cost for _, cost in sorted(costs, key=lambda x: x[0])]
            else:
                # Sequential evaluation (original behavior)
                costs = []
                for i, params in enumerate(population):
                    cost = self.objective_fn(params)
                    costs.append(cost)
                    if (i + 1) % 10 == 0 or (i + 1) == len(population):
                        print(f"  Progress: {i+1}/{len(population)}, "
                              f"best so far: {min(costs):.4f}")
            
            print(f"\nGeneration statistics:")
            print(f"  Min cost:  {min(costs):.4f}")
            print(f"  Max cost:  {max(costs):.4f}")
            print(f"  Mean cost: {np.mean(costs):.4f}")
            print(f"  Std cost:  {np.std(costs):.4f}")
            
            self._update(population, costs)
            
            if self.best_cost <= target_cost:
                print(f"\nTarget cost {target_cost} reached at generation {gen}!")
                break
            
            improvement_window.append(self.best_cost)
            if len(improvement_window) > window_size:
                improvement_window.pop(0)
            
            if len(improvement_window) >= window_size:
                window_improvement = improvement_window[0] - improvement_window[-1]
                if window_improvement < 0.02:
                    stagnation_counter += 1
                    print(f"  ⚠️  Stagnation: {stagnation_counter}/{stagnation_limit} "
                          f"(improvement: {window_improvement:.4f})")
                else:
                    stagnation_counter = 0
            
            if self.sigma < 1e-8:
                print(f"\n⚠️  Sigma collapsed to {self.sigma:.2e}, stopping early")
                break
            
            if stagnation_counter >= stagnation_limit:
                print(f"\n⚠️  Stagnation detected at generation {gen}")
                break
                
            if callback:
                callback(gen, self.best_cost, self.best_params)
            
            if self.generation_callback:
                self.generation_callback(gen)
                
        return self.best_params, self.best_cost


def evaluate_controller_on_segment(
    params: np.ndarray,
    segment_path: str,
    model_path: str,
    controller_name: str
) -> float:
    """Evaluate a controller with given parameters on a single segment."""
    try:
        param_dict = vector_to_params(params)
        controller_module = importlib.import_module(f'controllers.{controller_name}')
        controller = controller_module.Controller(params=param_dict)
        
        model = TinyPhysicsModel(model_path, debug=False)
        sim = TinyPhysicsSimulator(model, segment_path, controller=controller, debug=False)
        
        cost = sim.rollout()
        return cost['total_cost']
        
    except Exception as e:
        print(f"Error evaluating segment {segment_path}: {e}")
        return 1000.0


def evaluate_controller_batch(
    params: np.ndarray,
    segment_paths: List[str],
    model_path: str,
    controller_name: str,
    num_workers: int = 4
) -> float:
    """Evaluate a controller on multiple segments in parallel."""
    
    if num_workers <= 1:
        costs = [evaluate_controller_on_segment(params, seg, model_path, controller_name) 
                for seg in segment_paths]
    else:
        costs = []
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(
                    evaluate_controller_on_segment, 
                    params, seg, model_path, controller_name
                ): seg for seg in segment_paths
            }
            
            for future in as_completed(futures):
                try:
                    cost = future.result(timeout=60)
                    costs.append(cost)
                except Exception as e:
                    print(f"Evaluation failed: {e}")
                    costs.append(1000.0)
    
    return float(np.mean(costs)) if costs else 1000.0


def evaluate_candidate_for_population(
    params: np.ndarray,
    config_dict: dict,
    all_segments: List[str],
    generation: int
) -> float:
    """
    Top-level function to evaluate a candidate - can be pickled for multiprocessing.
    
    Args:
        params: Parameter vector to evaluate
        config_dict: Dictionary containing configuration (must be picklable)
        all_segments: List of all available segment paths
        generation: Current generation number (for adaptive segment selection)
    
    Returns:
        Cost value for the candidate
    """
    # Extract config values
    use_normalization = config_dict['use_normalization']
    use_adaptive_segments = config_dict['use_adaptive_segments']
    num_segments_per_eval = config_dict['num_segments_per_eval']
    model_path = config_dict['model_path']
    controller_name = config_dict['controller_name']
    num_workers = config_dict['num_workers']
    
    # Get segments (adaptive logic)
    if not use_adaptive_segments:
        segments = all_segments[:num_segments_per_eval]
    else:
        if generation < 15:
            n = max(15, num_segments_per_eval // 2)
        elif generation < 30:
            n = num_segments_per_eval
        else:
            n = min(len(all_segments), int(num_segments_per_eval * 1.5))
        segments = list(np.random.choice(all_segments, n, replace=False))
    
    # Denormalize if needed
    if use_normalization:
        params_phys = denormalize_param_vector(params)
    else:
        params_phys = params
    
    return evaluate_controller_batch(
        params_phys, segments, model_path, 
        controller_name, num_workers
    )


def load_params_from_file(filepath: str) -> Optional[Dict[str, float]]:
    """Load parameters from JSON file with backward compatibility for missing velocity params."""
    try:
        with open(filepath, 'r') as f:
            loaded = json.load(f)
        
        # Remove metadata fields
        params = {k: v for k, v in loaded.items() if not k.startswith('_')}
        
        # Ensure all required parameters exist (backward compatibility)
        for key in PARAM_NAMES:
            if key not in params:
                if key in DEFAULT_PARAMS:
                    params[key] = DEFAULT_PARAMS[key]
                    print(f"  ⚠️  Missing parameter '{key}', using default: {DEFAULT_PARAMS[key]}")
                else:
                    params[key] = 0.0
                    print(f"  ⚠️  Missing parameter '{key}', using 0.0")
        
        return params
    except FileNotFoundError:
        print(f"  ⚠️  Warm start file not found: {filepath}")
        return None
    except Exception as e:
        print(f"  ⚠️  Error loading warm start file {filepath}: {e}")
        return None


def get_segment_paths(data_dir: str, num_segments: Optional[int] = None) -> List[str]:
    """Get paths to route segments for evaluation."""
    data_path = Path(data_dir)
    segments = [str(f) for f in data_path.glob('*.csv')]
    
    if len(segments) == 0:
        print(f"WARNING: No CSV files found in {data_dir}")
        return []
    
    print(f"Found {len(segments)} total segments in {data_dir}")
    
    if num_segments and len(segments) > num_segments:
        np.random.seed(42)
        segments = list(np.random.choice(segments, num_segments, replace=False))
        print(f"Selected {num_segments} segments for evaluation")
        
    return sorted(segments)


def train_cmaes(config: CMAESConfig):
    """Main training loop for CMA-ES optimization."""
    print("=" * 80)
    print("CMA-ES Multi-Step Lookahead Controller Optimization")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Controller: {config.controller_name}")
    print(f"  Population size: {config.population_size}")
    print(f"  Max generations: {config.max_generations}")
    print(f"  Initial sigma: {config.initial_sigma}")
    print(f"  Segments per eval: {config.num_segments_per_eval}")
    print(f"  Workers (segment-level): {config.num_workers}")
    print(f"  Parallelize population: {config.parallelize_population}")
    if config.parallelize_population:
        print(f"  Population workers: {config.population_workers}")
    print(f"  Use normalization: {config.use_normalization}")
    print(f"  Adaptive segments: {config.use_adaptive_segments}")
    if config.warm_start_file:
        print(f"  Warm start: {config.warm_start_file}")
    else:
        print(f"  Warm start: None (using default parameters)")

    hyperparameters = {
        'population_size': config.population_size,
        'max_generations': config.max_generations,
        'initial_sigma': config.initial_sigma,
        'num_segments_per_eval': config.num_segments_per_eval,
        'num_workers': config.num_workers,
        'use_normalization': config.use_normalization,
        'use_adaptive_segments': config.use_adaptive_segments,
        'target_cost': config.target_cost,
        'stagnation_generations': config.stagnation_generations,
        'controller_name': config.controller_name,
        'data_dir': config.data_dir,
        'model_path': config.model_path,
    }
    
    all_segments = get_segment_paths(config.data_dir)
    
    if len(all_segments) == 0:
        print("ERROR: No segments found!")
        return None, None
    
    # Load initial parameters (warm start or defaults)
    if config.warm_start_file:
        print(f"\n🔄 Attempting warm start from: {config.warm_start_file}")
        warm_start_params = load_params_from_file(config.warm_start_file)
        if warm_start_params:
            print(f"  ✓ Loaded {len(warm_start_params)} parameters from warm start file")
            initial_params = params_to_vector(warm_start_params)
        else:
            print(f"  ✗ Failed to load warm start, using default parameters")
            initial_params = params_to_vector(DEFAULT_PARAMS)
    else:
        initial_params = params_to_vector(DEFAULT_PARAMS)
    
    if config.use_normalization:
        print("\n✓ Using parameter normalization to [0, 1]")
        initial_params = normalize_param_vector(initial_params)
        bounds = get_search_bounds(normalized=True)
    else:
        print("\n✗ Using physical parameter space")
        bounds = get_search_bounds(normalized=False)
    
    print(f"\nOptimizing {len(initial_params)} parameters (including 5-step lookahead):")
    for name, value in DEFAULT_PARAMS.items():
        low, high = PARAM_BOUNDS[name]
        print(f"  {name:25s}: {value:8.4f} (bounds: [{low:6.2f}, {high:6.2f}])")
    
    # Adaptive segment selection
    class AdaptiveSegments:
        def __init__(self, all_segs, initial_n):
            self.all_segments = all_segs
            self.current_n = initial_n
            self.generation = 0
            
        def get_segments(self):
            if not config.use_adaptive_segments:
                return self.all_segments[:self.current_n]
            
            if self.generation < 15:
                n = max(15, self.current_n // 2)
            elif self.generation < 30:
                n = self.current_n
            else:
                n = min(len(self.all_segments), int(self.current_n * 1.5))
            
            return list(np.random.choice(self.all_segments, n, replace=False))
        
        def update_generation(self, gen):
            self.generation = gen
    
    adaptive_segments = AdaptiveSegments(all_segments, config.num_segments_per_eval)
    
    print("\nEvaluating initial parameters...")
    segments_for_init = adaptive_segments.get_segments()
    
    if config.use_normalization:
        initial_params_phys = denormalize_param_vector(initial_params)
    else:
        initial_params_phys = initial_params
        
    initial_cost = evaluate_controller_batch(
        initial_params_phys, segments_for_init, config.model_path, 
        config.controller_name, config.num_workers
    )
    print(f"Initial cost: {initial_cost:.4f}")
    
    def objective(params: np.ndarray) -> float:
        segments = adaptive_segments.get_segments()
        
        if config.use_normalization:
            params_phys = denormalize_param_vector(params)
        else:
            params_phys = params
        
        return evaluate_controller_batch(
            params_phys, segments, config.model_path, 
            config.controller_name, config.num_workers
        )
    
    optimizer = CMAESOptimizer(
        objective_fn=objective,
        initial_point=initial_params,
        sigma=config.initial_sigma,
        population_size=config.population_size,
        bounds=bounds,
        generation_callback=adaptive_segments.update_generation
    )
    
    # Store config info for parallel population evaluation
    # This allows the picklable function to access config without closure
    config_dict = {
        'use_normalization': config.use_normalization,
        'use_adaptive_segments': config.use_adaptive_segments,
        'num_segments_per_eval': config.num_segments_per_eval,
        'model_path': config.model_path,
        'controller_name': config.controller_name,
        'num_workers': config.num_workers,
    }
    optimizer._eval_config_dict = config_dict
    optimizer._eval_all_segments = all_segments
    
    # Determine population workers if not set
    if config.population_workers is None:
        try:
            cpu_count = os.cpu_count() or 10
            config.population_workers = max(4, min(cpu_count // 2, 8))
        except:
            config.population_workers = 4
    
    def callback(gen, best_cost, best_params):
        if config.use_normalization:
            best_params_phys = denormalize_param_vector(best_params)
        else:
            best_params_phys = best_params
            
        param_dict = vector_to_params(best_params_phys)
        
        print(f"\n📊 Generation {gen:3d} Summary:")
        print(f"  Best Cost: {best_cost:.4f}")
        print(f"  Key params: kp={param_dict['kp']:.4f}, ki={param_dict['ki']:.4f}, "
              f"lookahead=[{int(param_dict['lookahead_start_idx'])}-{int(param_dict['lookahead_end_idx'])}], "
              f"decay={param_dict['lookahead_decay']:.3f}")
        
        save_params(best_params_phys, "optimized_params_velocity.json", gen, 
                   best_cost, config.population_size, hyperparameters=hyperparameters)
        
        if gen % 5 == 0 and gen > 0:
            save_params(best_params_phys, f"checkpoint_velocity_gen{gen}.json", gen, 
                       best_cost, config.population_size, hyperparameters=hyperparameters)
    
    print("\n" + "=" * 80)
    print("Starting optimization...")
    print("=" * 80)
    start_time = time.time()
    
    best_params, best_cost = optimizer.optimize(
        max_generations=config.max_generations,
        target_cost=config.target_cost,
        stagnation_limit=config.stagnation_generations,
        callback=callback,
        parallelize_population=config.parallelize_population,
        population_workers=config.population_workers
    )
    
    elapsed = time.time() - start_time
    
    if config.use_normalization:
        best_params = denormalize_param_vector(best_params)
    
    print(f"\n" + "=" * 80)
    print(f"Optimization complete in {elapsed/60:.1f} minutes ({elapsed:.1f} seconds)")
    print(f"=" * 80)
    print(f"Initial cost:     {initial_cost:.4f}")
    print(f"Final best cost:  {best_cost:.4f}")
    print(f"Improvement:      {initial_cost - best_cost:.4f} ({100*(initial_cost - best_cost)/initial_cost:.1f}%)")
    print(f"Total generations: {optimizer.generation}")
    
    save_params(best_params, "optimized_params_velocity_final.json", optimizer.generation, 
               best_cost, config.population_size, completed=True, hyperparameters=hyperparameters)
    
    return best_params, best_cost


def save_params(params: np.ndarray, filename: str, generation: int = None, 
               cost: float = None, population_size: int = None, completed: bool = False, 
               hyperparameters: dict = None):
    """Save optimized parameters to JSON file with metadata."""
    param_dict = vector_to_params(params)
    
    meta = {
        'description': 'Multi-step lookahead optimized with CMA-ES',
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'parameters': len(PARAM_NAMES),
        'lookahead_steps': 5,
    }
    
    if generation is not None:
        meta['generation'] = generation
    
    if cost is not None:
        meta['cost'] = float(cost)
    
    if population_size is not None:
        meta['population_size'] = population_size
    
    meta['status'] = 'completed' if completed else 'in_progress'
    
    if hyperparameters is not None:
        meta['hyperparameters'] = hyperparameters
    
    param_dict['_meta'] = meta
    
    temp_filename = filename + '.tmp'
    try:
        with open(temp_filename, 'w') as f:
            json.dump(param_dict, f, indent=2)
        
        os.replace(temp_filename, filename)
        
        if generation is not None and generation % 10 == 0:
            print(f"Saved checkpoint: {filename}")
    except Exception as e:
        print(f"Warning: Failed to save {filename}: {e}")
        if os.path.exists(temp_filename):
            os.remove(temp_filename)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Train multi-step lookahead controller with CMA-ES",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--data_dir", type=str, default="./data",
                       help="Directory containing segment CSV files")
    parser.add_argument("--model_path", type=str, default="./models/tinyphysics.onnx",
                       help="Path to ONNX model")
    parser.add_argument("--population_size", type=int, default=40,
                       help="CMA-ES population size")
    parser.add_argument("--max_generations", type=int, default=100,
                       help="Maximum number of generations")
    parser.add_argument("--num_segments", type=int, default=25,
                       help="Number of segments per evaluation")
    parser.add_argument("--num_workers", type=int, default=8,
                       help="Number of parallel workers for segment evaluation (default: 8 for M2 Pro)")
    parser.add_argument("--population_workers", type=int, default=None,
                       help="Number of workers for population-level parallelism (default: auto-detect)")
    parser.add_argument("--no-parallel-population", action="store_true",
                       help="Disable population-level parallelism (use sequential evaluation)")
    parser.add_argument("--sigma", type=float, default=0.2,
                       help="Initial step size (sigma)")
    parser.add_argument("--no-normalize", action="store_true",
                       help="Disable parameter normalization")
    parser.add_argument("--no-adaptive", action="store_true",
                       help="Disable adaptive segment evaluation")
    parser.add_argument("--target_cost", type=float, default=30.0,
                       help="Target cost to reach")
    parser.add_argument("--warm_start", type=str, default="optimized_params_velocity.json",
                       help="Path to JSON file with optimized parameters for warm start (default: optimized_params_velocity.json). Use 'none' to disable.")
    
    args = parser.parse_args()
    
    config = CMAESConfig(
        controller_name="custom_velocity",
        data_dir=args.data_dir,
        model_path=args.model_path,
        population_size=args.population_size,
        max_generations=args.max_generations,
        num_segments_per_eval=args.num_segments,
        num_workers=args.num_workers,
        initial_sigma=args.sigma,
        use_normalization=not args.no_normalize,
        use_adaptive_segments=not args.no_adaptive,
        target_cost=args.target_cost,
        warm_start_file=None if (args.warm_start.lower() == 'none' or not os.path.exists(args.warm_start)) else args.warm_start,
        parallelize_population=not args.no_parallel_population,
        population_workers=args.population_workers,
    )
    
    print("\n🚀 Starting CMA-ES optimization for multi-step lookahead controller!\n")
    
    best_params, best_cost = train_cmaes(config)
        
    if best_params is not None:
        print("\n" + "=" * 80)
        print("FINAL OPTIMIZED PARAMETERS (Multi-Step Lookahead)")
        print("=" * 80)
        print(f"Best Cost: {best_cost:.4f}\n")
        print("Parameters:")
        param_dict = vector_to_params(best_params)
        for k, v in param_dict.items():
            print(f"  '{k}': {v:.6f},")
        print(f"\n✅ Training complete! Parameters saved to optimized_params_velocity_final.json")