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
    target_cost: float = 48.0
    stagnation_generations: int = 20  
    
    # Initial distribution
    initial_sigma: float = 0.2 
    
    # Evaluation settings
    num_segments_per_eval: int = 20
    num_workers: int = 4
    use_adaptive_segments: bool = True  
    
    # Data paths
    data_dir: str = "./data"
    model_path: str = "./models/tinyphysics.onnx"
    
    # Controller to optimize
    controller_name: str = "custom_single_step"
    
    # Normalization
    use_normalization: bool = True 


# Parameter definitions for the custom controller
PARAM_NAMES = [
    'kp', 'ki', 'kd', 
    'k_roll', 
    'k_ff_immediate', 'k_ff_lookahead',
    'lookahead_idx',
    'error_filter_alpha',
    'action_smoothing',
    'velocity_scaling'
]

DEFAULT_PARAMS = {
    'kp': 0.28,
    'ki': 0.025,
    'kd': 0.09,
    'k_roll': 0.92,
    'k_ff_immediate': 0.18,
    'k_ff_lookahead': 0.14,
    'lookahead_idx': 7,
    'error_filter_alpha': 0.35,
    'action_smoothing': 0.28,
    'velocity_scaling': 0.018,
}

PARAM_BOUNDS = {
    'kp': (0.0, 1.5),              
    'ki': (0.0, 1.5),              
    'kd': (0.0, 1.5),              
    'k_roll': (0.0, 3.0),          
    'k_ff_immediate': (0.0, 1.5),  
    'k_ff_lookahead': (0.0, 1.5),  
    'lookahead_idx': (1, 30),     
    'error_filter_alpha': (0.0, 1.0),
    'action_smoothing': (0.0, 1.0), 
    'velocity_scaling': (-0.2, 0.2), 
}


def params_to_vector(params: Dict[str, float]) -> np.ndarray:
    """Convert parameter dict to vector."""
    return np.array([params[name] for name in PARAM_NAMES])


def vector_to_params(vector: np.ndarray) -> Dict[str, float]:
    """Convert parameter vector to dict."""
    return {name: float(vector[i]) for i, name in enumerate(PARAM_NAMES)}


def normalize_param_vector(params_vec: np.ndarray) -> np.ndarray:
    """
    Normalize parameter vector to [0, 1] range.
    
    This is CRITICAL for CMA-ES to work properly with parameters
    that have different scales (e.g., velocity_scaling vs k_roll).
    """
    normalized = np.zeros_like(params_vec)
    for i, name in enumerate(PARAM_NAMES):
        low, high = PARAM_BOUNDS[name]
        # Map [low, high] → [0, 1]
        normalized[i] = (params_vec[i] - low) / (high - low)
    return normalized


def denormalize_param_vector(normalized_vec: np.ndarray) -> np.ndarray:
    """
    Denormalize from [0, 1] back to physical parameter range.
    """
    physical = np.zeros_like(normalized_vec)
    for i, name in enumerate(PARAM_NAMES):
        low, high = PARAM_BOUNDS[name]
        # Map [0, 1] → [low, high]
        physical[i] = low + normalized_vec[i] * (high - low)
    return physical


def get_search_bounds(normalized: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """Get lower and upper bounds for all parameters."""
    if normalized:
        # All parameters in [0, 1]
        return np.zeros(len(PARAM_NAMES)), np.ones(len(PARAM_NAMES))
    else:
        # Physical bounds
        lower = np.array([PARAM_BOUNDS[name][0] for name in PARAM_NAMES])
        upper = np.array([PARAM_BOUNDS[name][1] for name in PARAM_NAMES])
        return lower, upper


class CMAESOptimizer:
    """
    Covariance Matrix Adaptation Evolution Strategy.
    
    Implements proper CMA-ES with:
    - Adaptive covariance matrix
    - Step-size control
    - Rank-μ and rank-1 updates
    - Clipping diagnostics
    """
    
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
            # Default: 4 + 3*ln(n)
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
        
        # Learning rates for step-size control
        self.c_sigma = (self.mu_eff + 2) / (self.dim + self.mu_eff + 5)
        self.d_sigma = 1 + 2 * max(0, np.sqrt((self.mu_eff - 1) / (self.dim + 1)) - 1) + self.c_sigma
        
        # Learning rates for covariance matrix
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
        
        print(f"\n=== CMA-ES Configuration ===")
        print(f"Dimension: {self.dim}")
        print(f"Population size (λ): {self.lam}")
        print(f"Parents (μ): {self.mu}")
        print(f"Initial sigma: {self.sigma:.4f}")
        print(f"μ_eff: {self.mu_eff:.2f}")
        
    def _sample_population(self) -> List[np.ndarray]:
        """Sample candidate solutions from current distribution."""
        # Eigendecomposition of covariance matrix
        eigenvalues, eigenvectors = np.linalg.eigh(self.C)
        eigenvalues = np.maximum(eigenvalues, 1e-10)
        
        B = eigenvectors
        D = np.diag(np.sqrt(eigenvalues))
        
        population = []
        clipped_count = np.zeros(self.dim)  # Track clipping
        
        for _ in range(self.lam):
            z = np.random.randn(self.dim)
            # Sample: x = m + σ * B * D * z
            x = self.mean + self.sigma * (B @ D @ z)
            
            # Clip to bounds and track violations
            if self.bounds is not None:
                x_unclipped = x.copy()
                x = np.clip(x, self.bounds[0], self.bounds[1])
                clipped_count += (x != x_unclipped).astype(int)
                
            population.append(x)
        
        # Diagnostic output for first generation and every 10 generations
        if self.generation == 0 or self.generation % 10 == 0:
            pop_array = np.array(population)
            print(f"\n=== Population Diversity (Generation {self.generation}) ===")
            for i, name in enumerate(PARAM_NAMES):
                clip_pct = 100 * clipped_count[i] / self.lam
                warning = " ⚠️ HIGH CLIPPING!" if clip_pct > 20 else ""
                print(f"  {name:20s}: std={pop_array[:, i].std():.4f}, "
                      f"range=[{pop_array[:, i].min():.4f}, {pop_array[:, i].max():.4f}], "
                      f"clipped={clip_pct:.1f}%{warning}")
            print()
            
        return population
    
    def _update(self, population: List[np.ndarray], costs: List[float]):
        """Update distribution parameters based on evaluations."""
        # Sort by cost (lower is better)
        indices = np.argsort(costs)
        
        # Update best
        if costs[indices[0]] < self.best_cost:
            improvement = self.best_cost - costs[indices[0]]
            self.best_cost = costs[indices[0]]
            self.best_params = population[indices[0]].copy()
            print(f"  ✓ New best! Cost: {self.best_cost:.4f} (improved by {improvement:.4f})")
            
        self.cost_history.append(self.best_cost)
        
        # Select top mu solutions
        selected = [population[i] for i in indices[:self.mu]]
        old_mean = self.mean.copy()
        
        # Update mean (weighted recombination)
        self.mean = np.zeros(self.dim)
        for i, x in enumerate(selected):
            self.mean += self.weights[i] * x
            
        # Eigendecomposition for path updates
        eigenvalues, eigenvectors = np.linalg.eigh(self.C)
        eigenvalues = np.maximum(eigenvalues, 1e-10)
        B = eigenvectors
        D_inv = np.diag(1.0 / np.sqrt(eigenvalues))
        
        # Update evolution path for sigma (step-size control)
        self.p_sigma = ((1 - self.c_sigma) * self.p_sigma + 
                        np.sqrt(self.c_sigma * (2 - self.c_sigma) * self.mu_eff) *
                        B @ D_inv @ B.T @ (self.mean - old_mean) / self.sigma)
        
        # Update sigma (step size adaptation)
        self.sigma *= np.exp((self.c_sigma / self.d_sigma) * 
                             (np.linalg.norm(self.p_sigma) / self.chi_n - 1))
        
        # Heaviside function for rank-one update
        h_sigma = 1 if (np.linalg.norm(self.p_sigma) / 
                       np.sqrt(1 - (1 - self.c_sigma) ** (2 * (self.generation + 1)))) < \
                       (1.4 + 2 / (self.dim + 1)) * self.chi_n else 0
        
        # Update evolution path for covariance
        self.p_c = ((1 - self.cc) * self.p_c + 
                   h_sigma * np.sqrt(self.cc * (2 - self.cc) * self.mu_eff) *
                   (self.mean - old_mean) / self.sigma)
        
        # Rank-one update
        c1_term = self.c1 * np.outer(self.p_c, self.p_c)
        
        # Rank-mu update
        cmu_term = np.zeros((self.dim, self.dim))
        for i, x in enumerate(selected):
            y = (x - old_mean) / self.sigma
            cmu_term += self.weights[i] * np.outer(y, y)
        cmu_term *= self.cmu
        
        # Update covariance matrix
        self.C = ((1 - self.c1 - self.cmu + 
                  (1 - h_sigma) * self.c1 * self.cc * (2 - self.cc)) * self.C +
                 c1_term + cmu_term)
        
        # Ensure C remains symmetric and positive definite
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
        callback: Optional[Callable] = None
    ) -> Tuple[np.ndarray, float]:
        """Run full optimization loop."""
        stagnation_counter = 0
        
        # Track improvement over longer window
        improvement_window = []
        window_size = 10
        
        for gen in range(max_generations):
            print(f"\n{'='*60}")
            print(f"Generation {gen}")
            print(f"{'='*60}")
            print(f"Sigma: {self.sigma:.4f}")
            
            # Sample population
            population = self._sample_population()
            
            # Evaluate all candidates
            print(f"Evaluating {len(population)} candidates...")
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
            
            # Update distribution
            self._update(population, costs)
            
            # Check for target reached
            if self.best_cost <= target_cost:
                print(f"\nTarget cost {target_cost} reached at generation {gen}!")
                break
            
            # Track improvement over window
            improvement_window.append(self.best_cost)
            if len(improvement_window) > window_size:
                improvement_window.pop(0)
            
            # Check for stagnation
            if len(improvement_window) >= window_size:
                window_improvement = improvement_window[0] - improvement_window[-1]
                if window_improvement < 0.02:  # Increased threshold from 0.01
                    stagnation_counter += 1
                    print(f"  ⚠️  Stagnation: {stagnation_counter}/{stagnation_limit} "
                          f"(improvement: {window_improvement:.4f})")
                else:
                    stagnation_counter = 0
            
            # Early stop on sigma collapse
            if self.sigma < 1e-8:
                print(f"\n⚠️  Sigma collapsed to {self.sigma:.2e}, stopping early")
                break
            
            if stagnation_counter >= stagnation_limit:
                print(f"\n⚠️  Stagnation detected at generation {gen}")
                break
                
            # Callback for logging
            if callback:
                callback(gen, self.best_cost, self.best_params)
            
            # Generation callback (for adaptive segments)
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
        import traceback
        traceback.print_exc()
        return 1000.0


def evaluate_controller_batch(
    params: np.ndarray,
    segment_paths: List[str],
    model_path: str,
    controller_name: str,
    num_workers: int = 4
) -> float:
    """Evaluate a controller on multiple segments, optionally in parallel."""
    
    if num_workers <= 1:
        costs = []
        for seg in segment_paths:
            cost = evaluate_controller_on_segment(params, seg, model_path, controller_name)
            costs.append(cost)
    else:
        costs = []
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(
                    evaluate_controller_on_segment, 
                    params, 
                    seg, 
                    model_path, 
                    controller_name
                ): seg for seg in segment_paths
            }
            
            for future in as_completed(futures):
                try:
                    cost = future.result(timeout=60)
                    costs.append(cost)
                except Exception as e:
                    print(f"Evaluation failed: {e}")
                    costs.append(1000.0)
    
    if len(costs) == 0:
        return 1000.0
        
    return float(np.mean(costs))


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
    print("CMA-ES Controller Optimization with Advanced Features")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Controller: {config.controller_name}")
    print(f"  Population size: {config.population_size}")
    print(f"  Max generations: {config.max_generations}")
    print(f"  Initial sigma: {config.initial_sigma}")
    print(f"  Segments per eval: {config.num_segments_per_eval}")
    print(f"  Workers: {config.num_workers}")
    print(f"  Use normalization: {config.use_normalization}")
    print(f"  Adaptive segments: {config.use_adaptive_segments}")

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
    
    # Load all available segments
    all_segments = get_segment_paths(config.data_dir)
    
    if len(all_segments) == 0:
        print("ERROR: No segments found! Please ensure data_dir is correct.")
        return None, None
    
    # Prepare initial parameters
    initial_params = params_to_vector(DEFAULT_PARAMS)
    
    if config.use_normalization:
        print("\n✓ Using parameter normalization to [0, 1]")
        initial_params = normalize_param_vector(initial_params)
        bounds = get_search_bounds(normalized=True)
    else:
        print("\n✗ Using physical parameter space (not normalized)")
        bounds = get_search_bounds(normalized=False)
    
    print(f"\nOptimizing {len(initial_params)} parameters:")
    for name, value in DEFAULT_PARAMS.items():
        low, high = PARAM_BOUNDS[name]
        print(f"  {name:20s}: {value:8.4f} (bounds: [{low:6.2f}, {high:6.2f}])")
    
    # Adaptive segment selection
    class AdaptiveSegments:
        def __init__(self, all_segs, initial_n):
            self.all_segments = all_segs
            self.current_n = initial_n
            self.generation = 0
            
        def get_segments(self):
            """Get current segment set based on generation."""
            if not config.use_adaptive_segments:
                return self.all_segments[:self.current_n]
            
            # Adaptive strategy: start small, increase as we converge
            if self.generation < 15:
                n = max(15, self.current_n // 2)  # Fast exploration
            elif self.generation < 30:
                n = self.current_n  # Normal
            else:
                n = min(len(self.all_segments), int(self.current_n * 1.5))  # High precision
            
            return list(np.random.choice(self.all_segments, n, replace=False))
        
        def update_generation(self, gen):
            self.generation = gen
    
    adaptive_segments = AdaptiveSegments(all_segments, config.num_segments_per_eval)
    
    # Evaluate initial parameters
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
    
    # Define objective function
    def objective(params: np.ndarray) -> float:
        """Objective function that handles normalization."""
        # Get current segment set
        segments = adaptive_segments.get_segments()
        
        # Denormalize if needed
        if config.use_normalization:
            params_phys = denormalize_param_vector(params)
        else:
            params_phys = params
        
        return evaluate_controller_batch(
            params_phys, segments, config.model_path, 
            config.controller_name, config.num_workers
        )
    
    # Create optimizer
    optimizer = CMAESOptimizer(
        objective_fn=objective,
        initial_point=initial_params,
        sigma=config.initial_sigma,
        population_size=config.population_size,
        bounds=bounds,
        generation_callback=adaptive_segments.update_generation
    )
    
    # Callback for saving checkpoints
    def callback(gen, best_cost, best_params):
        # Denormalize for saving
        if config.use_normalization:
            best_params_phys = denormalize_param_vector(best_params)
        else:
            best_params_phys = best_params
            
        param_dict = vector_to_params(best_params_phys)
        
        print(f"\n📊 Generation {gen:3d} Summary:")
        print(f"  Best Cost: {best_cost:.4f}")
        print(f"  Key params: kp={param_dict['kp']:.4f}, ki={param_dict['ki']:.4f}, "
              f"kd={param_dict['kd']:.4f}, k_roll={param_dict['k_roll']:.4f}")
        
        # Save current best parameters
        save_params(best_params_phys, "optimized_params.json", gen, best_cost, 
                   config.population_size, hyperparameters=hyperparameters)
        
        # Save detailed checkpoint every 5 generations
        if gen % 5 == 0 and gen > 0:
            save_params(best_params_phys, f"checkpoint_gen{gen}.json", gen, 
                       best_cost, config.population_size, hyperparameters=hyperparameters)  # ← ADD hyperparameters=hyperparameters

    
    print("\n" + "=" * 80)
    print("Starting optimization...")
    print("=" * 80)
    start_time = time.time()
    
    best_params, best_cost = optimizer.optimize(
        max_generations=config.max_generations,
        target_cost=config.target_cost,
        stagnation_limit=config.stagnation_generations,
        callback=callback
    )
    
    elapsed = time.time() - start_time
    
    # Denormalize final result
    if config.use_normalization:
        best_params = denormalize_param_vector(best_params)
    
    print(f"\n" + "=" * 80)
    print(f"Optimization complete in {elapsed/60:.1f} minutes ({elapsed:.1f} seconds)")
    print(f"=" * 80)
    print(f"Initial cost:     {initial_cost:.4f}")
    print(f"Final best cost:  {best_cost:.4f}")
    print(f"Improvement:      {initial_cost - best_cost:.4f} ({100*(initial_cost - best_cost)/initial_cost:.1f}%)")
    print(f"Total generations: {optimizer.generation}")
    
    # Final save with completion marker
    save_params(best_params, "optimized_params_final.json", optimizer.generation, 
               best_cost, config.population_size, completed=True, hyperparameters=hyperparameters)
    
    return best_params, best_cost


def save_params(params: np.ndarray, filename: str, generation: int = None, 
               cost: float = None, population_size: int = None, completed: bool = False, hyperparameters: dict = None):
    """Save optimized parameters to JSON file with metadata."""
    param_dict = vector_to_params(params)
    
    # Build metadata
    meta = {
        'description': 'Optimized with CMA-ES',
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'parameters': len(PARAM_NAMES),
    }
    
    if generation is not None:
        meta['generation'] = generation
    
    if cost is not None:
        meta['cost'] = float(cost)
    
    if population_size is not None:
        meta['population_size'] = population_size
    
    if completed:
        meta['status'] = 'completed'
    else:
        meta['status'] = 'in_progress'
    
    if hyperparameters is not None:
        meta['hyperparameters'] = hyperparameters
    param_dict['_meta'] = meta
    
    # Write to file atomically (write to temp, then rename)
    temp_filename = filename + '.tmp'
    try:
        with open(temp_filename, 'w') as f:
            json.dump(param_dict, f, indent=2)
        
        # Atomic rename (prevents corruption if interrupted)
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
        description="Train controller with CMA-ES",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--controller", type=str, default="custom",
                       help="Controller module name")
    parser.add_argument("--data_dir", type=str, default="./data",
                       help="Directory containing segment CSV files")
    parser.add_argument("--model_path", type=str, default="./models/tinyphysics.onnx",
                       help="Path to ONNX model")
    parser.add_argument("--population_size", type=int, default=40,
                       help="CMA-ES population size (λ)")
    parser.add_argument("--max_generations", type=int, default=100,
                       help="Maximum number of generations")
    parser.add_argument("--num_segments", type=int, default=25,
                       help="Number of segments per evaluation")
    parser.add_argument("--num_workers", type=int, default=4,
                       help="Number of parallel workers")
    parser.add_argument("--sigma", type=float, default=0.2,
                       help="Initial step size (sigma)")
    parser.add_argument("--no-normalize", action="store_true",
                       help="Disable parameter normalization")
    parser.add_argument("--no-adaptive", action="store_true",
                       help="Disable adaptive segment evaluation")
    parser.add_argument("--target_cost", type=float, default=45.0,
                       help="Target cost to reach")
    
    args = parser.parse_args()
    
    config = CMAESConfig(
        controller_name=args.controller,
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
    )
    
    print("\nStarting CMA-ES optimization with enhanced features!\n")
    
    best_params, best_cost = train_cmaes(config)
        
    if best_params is not None:
        print("\n" + "=" * 80)
        print("FINAL OPTIMIZED PARAMETERS")
        print("=" * 80)
        print(f"Best Cost: {best_cost:.4f}\n")
        print("Parameters:")
        param_dict = vector_to_params(best_params)
        for k, v in param_dict.items():
            print(f"  '{k}': {v:.6f},")
        print("\n✅ Training complete! Parameters saved to optimized_params_final.json")