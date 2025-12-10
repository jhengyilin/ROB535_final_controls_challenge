import json
import glob
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

# --- Configuration for IEEE/LaTeX Style ---
# This block sets the global font and style parameters for LaTeX rendering
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "mathtext.fontset": "custom",
    "mathtext.rm": "Times New Roman",
    "mathtext.it": "Times New Roman:italic",
    "mathtext.bf": "Times New Roman:bold",
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.figsize": (6, 4) # Standard 6x4 single-column figure size
})

def load_cma_es_logs(base_dir="train_checkpoint"):
    """
    Recursively loads all JSON checkpoint files from specified subdirectories.
    Also loads optimized_params_*_final.json files from the root directory.
    """
    all_data = []
    
    # Define how to map directory/description to plot labels
    controller_map = {
        'multi_step': 'Multi-Step Lookahead',
        'single_step': 'Single-Step Lookahead',
        'velocity': 'Multi-Step Lookahead with Velocity-Dependent Gain'
    }

    # Search for JSON files in the specified subdirectories
    search_patterns = [
        os.path.join(base_dir, 'multi_step', 'checkpoint_*.json'),
        os.path.join(base_dir, 'single_step', 'checkpoint_*.json'),
        os.path.join(base_dir, 'velocity', 'checkpoint_*.json')
    ]
    
    for pattern in search_patterns:
        # Determine the controller key from the search path
        controller_key = pattern.split(os.path.sep)[-2]
        controller_name = controller_map.get(controller_key, 'Unknown')
        
        for file_path in glob.glob(pattern):
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    
                    meta = data.get('_meta', {})
                    generation = meta.get('generation')
                    cost = meta.get('cost')
                    
                    if generation is not None and cost is not None:
                        # Extract parameter values for later plotting
                        params = {k: v for k, v in data.items() if k != '_meta'}
                        
                        entry = {
                            'generation': int(generation),
                            'cost': float(cost),
                            'controller': controller_name,
                            **params
                        }
                        all_data.append(entry)
            except Exception as e:
                print(f"Error processing file {file_path}: {e}")

    # Load optimized_params_*_final.json files from root directory and subdirectories
    final_params_map = {
        'optimized_params_multistep_final.json': ('Multi-Step Lookahead', 'multi_step'),
        'optimized_params_single_final.json': ('Single-Step Lookahead', 'single_step'),
        'optimized_params_velocity_final.json': ('Multi-Step Lookahead with Velocity-Dependent Gain', 'velocity')
    }
    
    # Get the directory containing the script (or use current directory)
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        script_dir = os.getcwd()
    
    for filename, (controller_name, subdir) in final_params_map.items():
        # Try root directory first
        root_path = os.path.join(script_dir, filename)
        # Then try subdirectory
        subdir_path = os.path.join(script_dir, base_dir, subdir, filename)
        
        for path in [root_path, subdir_path]:
            if os.path.exists(path):
                try:
                    with open(path, 'r') as f:
                        data = json.load(f)
                        
                        meta = data.get('_meta', {})
                        generation = meta.get('generation')
                        cost = meta.get('cost')
                        
                        if generation is not None and cost is not None:
                            # Extract parameter values for later plotting
                            params = {k: v for k, v in data.items() if k != '_meta'}
                            
                            entry = {
                                'generation': int(generation),
                                'cost': float(cost),
                                'controller': controller_name,
                                **params
                            }
                            all_data.append(entry)
                            break  # Found the file, no need to check other paths
                except Exception as e:
                    print(f"Error processing file {path}: {e}")

    if not all_data:
        print("No valid data found to plot. Check your 'training_checkpoint' directory structure and file names.")
        return pd.DataFrame()

    df = pd.DataFrame(all_data)
    # Ensure all parameter names are normalized for plotting
    df = df.rename(columns={
        'kp': '$K_p$', 'ki': '$K_i$', 'kd': '$K_d$',
        'k_ff_lookahead': '$K_{ff,k}$',
        'lookahead_decay': '$w$'
    })
    return df

def plot_convergence(df, filename='convergence_plot_ieee.pdf'):
    """Plots the Total Cost vs. Generation for all controllers (IEEE style)."""
    if df.empty: return
    
    plt.figure()
    
    # Define distinct markers and linestyles for IEEE clarity
    styles = {
        'Single-Step Lookahead': ('#1f77b4', 'o', '-'),
        'Multi-Step Lookahead': ('#ff7f0e', 's', '--'),
        'Multi-Step Lookahead with Velocity-Dependent Gain': ('#2ca02c', '^', ':')
    }

    for name, group in df.groupby('controller'):
        if name in styles:
            group = group.sort_values('generation')
            color, marker, linestyle = styles[name]
            plt.plot(
                group['generation'], 
                group['cost'], 
                label=name, 
                color=color,
                marker=marker,
                linestyle=linestyle,
                markersize=4,
                linewidth=1.5
            )

    plt.title('Evolutionary Optimization Convergence', fontsize=14)
    plt.xlabel('Generation', fontsize=12)
    # Use LaTeX for the y-axis label to match the cost function J
    plt.ylabel('Total Cost, $J$', fontsize=12) 
    plt.legend(loc='best', frameon=False) # Remove frame for cleaner look
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.yscale('log') # Use log scale for clarity in showing large initial drop
    plt.tight_layout()
    plt.savefig(filename, format='pdf', bbox_inches='tight')
    print(f"Convergence Plot saved to {filename}")

def plot_parameter_evolution(df, controller_name, params_to_plot, filename_suffix='param_evolution_ieee.pdf'):
    """Plots the evolution of key parameters for a specific controller (IEEE style)."""
    if df.empty: return

    subset = df[df['controller'] == controller_name].sort_values('generation')
    if subset.empty: return
    
    # Create a figure with a subplot for each parameter
    fig, axes = plt.subplots(len(params_to_plot), 1, sharex=True, figsize=(6, 6))
    if len(params_to_plot) == 1: axes = [axes] # Ensure axes is iterable if only 1 parameter
    
    colors = plt.cm.get_cmap('viridis', len(params_to_plot))
    
    for i, param in enumerate(params_to_plot):
        if param in subset.columns:
            axes[i].plot(
                subset['generation'], 
                subset[param], 
                color=colors(i),
                linestyle='-',
                linewidth=1.5,
                marker='.',
                markersize=3
            )
            # Use the LaTeX column name as the y-label
            axes[i].set_ylabel(param, fontsize=10) 
            axes[i].grid(True, linestyle=':', alpha=0.5)
    
    # Set the x-axis label only on the bottom plot
    axes[-1].set_xlabel('Generation', fontsize=12)
    fig.suptitle(f'{controller_name} Parameter Evolution', fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust for suptitle
    
    filename = f"{controller_name.lower().replace('-', '_').replace(' ', '_')}_{filename_suffix}"
    plt.savefig(filename, format='pdf', bbox_inches='tight')
    print(f"Parameter Evolution Plot saved to {filename}")

# --- Execution ---
df_logs = load_cma_es_logs()

if not df_logs.empty:
    # 1. Plot the main convergence curve for all three controllers
    plot_convergence(df_logs)

    # 2. Plot the evolution of key parameters for the best performing controller
    # The Multi-Step Lookahead controller achieved the lowest total cost 
    # best_controller = 'Multi-Step Lookahead'
    # key_params_multi_step = ['$K_p$', '$K_{ff,k}$', '$w$'] # Kp, feedforward gain, and decay weight
    # plot_parameter_evolution(df_logs, best_controller, key_params_multi_step)

    # 3. Plot the evolution of key parameters for the Multi-Step Lookahead with Velocity-Dependent Gain controller
    # This shows how velocity-dependent gains (betas) evolved
    # velocity_controller = 'Multi-Step Lookahead with Velocity-Dependent Gain'
    # key_params_velocity = ['$K_p$', '$K_i$', '$\beta_p$', '$\beta_i$'] 
    # plot_parameter_evolution(df_logs, velocity_controller, key_params_velocity, filename_suffix='velocity_param_evolution_ieee.pdf')