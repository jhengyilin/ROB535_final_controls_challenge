# ROB 535 Final Project

## Overview

This final project implements a lateral acceleration controller for autonomous vehicle steering using **CMA-ES (Covariance Matrix Adaptation Evolution Strategy)**. The controller is optimized to minimize both lateral acceleration tracking error and jerk (rate of change of acceleration) across diverse driving scenarios.

## Controller Architecture

The controller uses a multi-step lookahead approach that:

1. Computes weighted average of future target lateral accelerations over a configurable window (with typical step size 2-8)
2. Scales PID gains based on current and future vehicle acceleration and/or velocity
3. Combines feedback (PID) and feedforward terms for robust control
4. Filters errors and smooths actions to reduce oscillations


## File Structure

```
ROB535_final_controls_challenge/
├── train_cmaes_velocity.py          # Main CMA-ES training script
├── eval.py                          # Evaluation script for comparing controllers
├── tinyphysics.py                   # Physics simulator and model interface
├── controllers/
│   ├── custom_single_step.py        
│   ├── custom_multi_step.py         
│   ├── custom_velocity.py           # velocity as a factor
│   ├── ffpid.py                     # lookahead pid
│   ├── pid.py                       # Baseline PID controller
│   └── ...
├── data/                            # Training/evaluation data (CSV files)
│   ├── 00000.csv
│   ├── 00001.csv
│   └── ...
├── models/
│   └── tinyphysics.onnx            # ONNX neural network physics model
└── optimized_params_<controller>.json   # Optimized parameters (output)

```

## Data Format

The training data consists of CSV files, each representing a driving segment. Each CSV file contains the following columns:

| Column | Description | Units |
|--------|-------------|-------|
| `t` | Time | seconds |
| `vEgo` | Ego vehicle velocity | m/s |
| `aEgo` | Ego vehicle forward acceleration | m/s² |
| `roll` | Road roll angle | radians |
| `targetLateralAcceleration` | Desired lateral acceleration | m/s² |
| `steerCommand` | Steering command (ground truth) | normalized [-2, 2] |

### Example Data Snippet

```csv
t,vEgo,aEgo,roll,targetLateralAcceleration,steerCommand
0.0,33.770,0.017,0.037,1.004,-0.330
0.1,33.764,-0.039,0.037,1.050,-0.335
0.2,33.756,-0.068,0.037,1.056,-0.333
...
```

## Controller Training

### Training Process

The CMA-ES optimization process:

1. **Initialization**: Starts with default PID parameters or loads from a warm-start file
2. **Population Sampling**: Generates a population of candidate parameter sets from a multivariate Gaussian distribution
3. **Evaluation**: Each candidate is evaluated on a subset of driving segments by:
   - Running the controller in the physics simulator
   - Computing total cost: `(lataccel_cost × 50) + jerk_cost`
4. **Selection**: Best candidates (lowest cost) are selected as parents
5. **Update**: Distribution parameters (mean, covariance, step size) are updated based on successful candidates
6. **Iteration**: Process repeats for multiple generations until convergence


### Running Training

```bash
# Basic training with default settings
# replace <controller> with "single_step, multi_step, or velocity"
python train_cmaes_<controller>.py

# Custom configuration
python train_cmaes_<controller>.py \
    --data_dir ./data \
    --model_path ./models/tinyphysics.onnx \
    --population_size 40 \
    --max_generations 100 \
    --num_segments 50 \
    --num_workers 4 \
    --sigma 0.2 \
    --target_cost 30.0 \
    --warm_start optimized_params_*.json

# Disable normalization (use physical parameter space)
python train_cmaes_<controller>.py --no-normalize

# Disable adaptive segment selection
python train_cmaes_<controller>.py --no-adaptive
```

### Training Arguments

- `--data_dir`: Directory containing CSV segment files (default: `./data`)
- `--model_path`: Path to ONNX physics model (default: `./models/tinyphysics.onnx`)
- `--population_size`: CMA-ES population size λ (default: 40)
- `--max_generations`: Maximum optimization generations (default: 100)
- `--num_segments`: Number of segments per evaluation (default: 25)
- `--num_workers`: Parallel workers for evaluation (default: 4)
- `--sigma`: Initial step size (default: 0.2)
- `--target_cost`: Target cost to reach (default: 30.0)
- `--warm_start`: Path to JSON file with previous parameters (default: `optimized_params_velocity.json`)
- `--no-normalize`: Disable parameter normalization to [0,1]
- `--no-adaptive`: Disable adaptive segment selection

### Output Files

- `optimized_params_<controller>.json`: Best parameters found (updated each generation)
- `optimized_params_<controller>_final.json`: Final optimized parameters
- `checkpoint_<controller>_gen*.json`: Checkpoints every 5 generations

## Evaluation

### Running Evaluation

Evaluate the optimized controller against a baseline:

```bash
python eval.py \
    --model_path ./models/tinyphysics.onnx \
    --data_path ./data \
    --num_segs 100 \
    --test_controller custom_<controller> \
    --baseline_controller pid
```

This generates:
- Cost statistics: Mean lataccel_cost, jerk_cost, and total_cost for both controllers
- Visual comparison: Histograms of cost distributions
- Sample rollouts: Plots showing lateral acceleration tracking for sample segments
- HTML report: `report.html` with all results

### Cost Metrics

The evaluation computes two cost components:

1. **Lateral Acceleration Cost**:
   ```
   lataccel_cost = Σ(actual_lataccel - target_lataccel)² / steps × 100
   ```

2. **Jerk Cost**:
   ```
   jerk_cost = Σ((lataccel_t - lataccel_{t-1}) / Δt)² / (steps - 1) × 100
   ```

3. **Total Cost**:
   ```
   total_cost = (lataccel_cost × 50) + jerk_cost
   ```

The controller aims to minimize total cost, balancing tracking accuracy with smoothness.

## Controller Usage

The optimized controller can be used in the simulator:

```bash
# Single segment test
python tinyphysics.py \
    --model_path ./models/tinyphysics.onnx \
    --data_path ./data/00000.csv \
    --controller <controller_name> \
    --debug

# Batch evaluation
python tinyphysics.py \
    --model_path ./models/tinyphysics.onnx \
    --data_path ./data \
    --num_segs 100 \
    --controller <controller_name>
```
