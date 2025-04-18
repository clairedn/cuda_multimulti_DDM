import numpy as np
import os, math, re, argparse, glob
import time
from collections import defaultdict
import scipy.optimize as opt
import multiprocessing

# Helper to check data dimensions
def validate_data_dimensions(data, lambdas, taus, desc=""):
    expected = (len(lambdas), len(taus))
    if data.shape != expected:
        raise ValueError(f"Dimension mismatch in {desc}: expected {expected}, got {data.shape}")
    return True

def read_data_file(fname):
    try:
        with open(fname, "r") as f:
            lines = f.readlines()
            if len(lines) < 3: 
                raise ValueError("File must contain lambda, tau, and ISF data lines.")
            
            # Get basic data from first two lines
            lambdas = np.array([float(n) for n in lines[0].split()])
            taus = np.array([float(n) for n in lines[1].split()])
            angle_info_list = []
            
            # Check if file has angle sections
            has_angles = any(l.strip().startswith('# Angle section') for l in lines[2:])
            
            if not has_angles:
                # Simple case - just get all data lines (no angles)
                data_lines = [l for l in lines[2:] if l.strip() and not l.strip().startswith('#')]
                if not data_lines: 
                    raise ValueError("No ISF data lines found.")
                data = np.array([[float(n) for n in line.split()] for line in data_lines])
                validate_data_dimensions(data, lambdas, taus, "radial average data")
                angle_info_list.append(("Radial Average", data))
                return lambdas, taus, angle_info_list
            
            # Handle angle sections case
            current_data = []
            current_desc = ""
            
            for line in lines[2:]:
                line = line.strip()
                if not line: continue
                
                if line.startswith('# Angle section'):
                    # Save previous angle data if we have any
                    if current_data:
                        data = np.array(current_data)
                        validate_data_dimensions(data, lambdas, taus, current_desc)
                        angle_info_list.append((current_desc, data))
                        current_data = []
                    
                    # Process angle metadata
                    match = re.search(r'# Angle section \(radial direction\) (\d+)\s*\(center angle:\s*(-?\d+\.?\d*)\s*degrees,\s*range:\s*(-?\d+\.?\d*)\s*to\s*(-?\d+\.?\d*)\s*degrees\)', line)
                    current_desc = f"Angle {int(match.group(1))}: Center {float(match.group(2)):.1f}°, Range: {float(match.group(3)):.1f}° to {float(match.group(4)):.1f}°" if match else line.strip('# ')
                
                elif not line.startswith('#'):
                    # Add data line
                    current_data.append([float(n) for n in line.split()])
            
            # Don't forget the last angle section
            if current_data:
                data = np.array(current_data)
                validate_data_dimensions(data, lambdas, taus, current_desc)
                angle_info_list.append((current_desc, data))
            
            if not angle_info_list: 
                raise ValueError("No valid data blocks parsed")
            
            return lambdas, taus, angle_info_list
            
    except FileNotFoundError:
        print(f"Error: File not found at {fname}")
        return None, None, None
    except Exception as e:
        print(f"Error reading file {fname}: {e}")
        return None, None, None

# Extract metadata from filename
def parse_filename(filename):
    info = {'episode': -1, 'window': -1, 'scale': -1, 'tile': -1}
    
    # Get episode and window
    episode_match = re.search(r'episode(\d+)-(\d+)', filename)
    if episode_match:
        info['episode'] = int(episode_match.group(1))
        info['window'] = int(episode_match.group(2))
    
    # Get scale and tile
    scale_match = re.search(r'scale(\d+)-(\d+)', filename)
    if scale_match:
        info['scale'] = int(scale_match.group(1))
        info['tile'] = int(scale_match.group(2))
        
    return info

# Avoid overflow in exp calculation
def safe_exp_calc(exponent):
    return np.where(exponent < 700, np.exp(-exponent), 0)

# Fitting model function
def generic_exp_model(tau, A, Gamma, beta, B):
    exponent = (Gamma * tau) ** beta
    return A * (1 - safe_exp_calc(exponent)) + B

# Supported fitting models
fitting_models = {
    'generic_exp': {
        'func': generic_exp_model,
        'latex': r"$I(q,\tau) = A(1-e^{-(\Gamma\tau)^{\beta}}) + B$",
        'params': ['A', 'Gamma', 'beta', 'B']
    }
}

# Generate initial parameters and bounds for fitting
def get_model_params(A_init, Gamma_init, B_init):
    return {
        'p0': [A_init, Gamma_init, 1.0, B_init],
        'bounds': ([A_init*0.5, Gamma_init*0.1, 0.5, B_init-0.1], 
                   [A_init*1.5, Gamma_init*10, 2.0, B_init+0.1])
    }

def batch_fit_curves(taus, q_values, data_array, q_indices):
    model_func = fitting_models['generic_exp']['func']
    param_names = fitting_models['generic_exp']['params']
    fit_results = []
    
    for i in q_indices:
        try:
            # Get data for this q-value
            y_data = data_array.T[:, i]
            
            # Estimate initial parameters
            B_init = np.min(y_data)
            A_init = np.max(y_data) - B_init
            
            mid_idx = len(taus) // 2
            Gamma_init = 1.0 / taus[mid_idx] if taus[mid_idx] > 0 and A_init > 1e-6 else 1.0
            
            # Run fitting
            params = get_model_params(A_init, Gamma_init, B_init)
            popt, _ = opt.curve_fit(
                model_func, taus, y_data, 
                p0=params['p0'], 
                bounds=params['bounds'],
                method='trf',
            )
            
            # Store results
            params_dict = {'q': q_values[i]}
            for j, name in enumerate(param_names):
                params_dict[name] = float(popt[j])
            fit_results.append((i, params_dict, popt))
            
        except Exception as e:
            print(f"Warning: q={q_values[i]:.4f} fit failed: {e}")
            pass
            
    return fit_results

def get_output_filename(data_entry, model_name, fit_data, angle_info_list, output_dir=None):
    # Figure out base filename
    base_filename = data_entry.get('file_path', '')
    if os.path.exists(base_filename):
        base_filename = os.path.splitext(base_filename)[0]
    elif 'info' in data_entry:
        info = data_entry['info']
        base_filename = f"episode{info['episode']}-{info['window']}_scale{info['scale']}-{info['tile']}"
    
    # Add appropriate suffix
    if len(angle_info_list) == 1:
        info = data_entry.get('info', {})
        angle_idx = info.get('selected_angle', 0)
        if not info.get('selected_angle'):
            match = re.search(r'Angle\s+(\d+)', angle_info_list[0][0])
            if match: angle_idx = int(match.group(1))
        suffix = f"_fit_{model_name}" if fit_data else f"_angle{angle_idx}"
    else:
        suffix = f"_fit_{model_name}" if fit_data else "_all_angles"
    
    # Build full path
    output_file = f"{base_filename}{suffix}"
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, os.path.basename(output_file))
    
    return output_file

# Configure plot axis appearance
def configure_axis(ax, desc, is_multiple, idx, num_angles):
    # Set up ticks and grid
    ax.tick_params(axis='x', which='both', direction='in', top=True)
    ax.tick_params(axis='y', which='both', direction='in', right=True)
    ax.grid(True, which='both', linestyle='--', alpha=0.5)
    ax.set_title(desc, fontsize=10)
    
    # Add axis labels only where needed
    if is_multiple:
        # Only bottom row gets x label
        if idx // 2 == math.ceil(num_angles / 2) - 1:
            ax.set_xlabel(r'Lag time, $\tau$ [s]', fontsize=15)
        # Only left column gets y label
        if idx % 2 == 0:
            ax.set_ylabel(r'I(q, $\tau$) [a. u.]', fontsize=15)
    else:
        # Single plot gets both labels
        ax.set_xlabel(r'Lag time, $\tau$ [s]', fontsize=15)
        ax.set_ylabel(r'I(q, $\tau$) [a. u.]', fontsize=15)

# Save fit parameters to file
def save_fit_parameters(all_angle_fit_params, param_file, model_name):
    try:
        with open(param_file, 'w') as f:
            f.write(f"Fitting model: {fitting_models[model_name]['latex']}\n\n")
            for angle_desc, fit_params in all_angle_fit_params:
                f.write(f"{angle_desc}\n")
                f.write(f"{'q (2π/λ)':<10} {' '.join(f'{name:<10}' for name in fitting_models[model_name]['params'])}\n")
                f.write("-" * (11 + 11 * len(fitting_models[model_name]['params'])) + "\n")
                
                for params in fit_params:
                    values = [f"{params['q']:<10.2f}"]
                    values.extend([f"{params.get(name, float('nan')):<10.3f}" for name in fitting_models[model_name]['params']])
                    f.write(' '.join(values) + '\n')
                f.write("\n\n")
    except Exception as e:
        print(f"Error saving parameters: {param_file}: {e}")

# Main data processing function
def plot_data(data_entry, max_q=20, output_dir=None, fit_data=False, model_name='generic_exp', connect_points=False, generate_plots=True):
    # Extract data
    lambdas = data_entry['lambdas']
    taus = data_entry['taus']
    angle_info_list = data_entry['angle_info_list']
    
    # Validation
    if lambdas is None or not angle_info_list: 
        return None
    
    # Calculate q-values (q = 2π/λ)
    q_values = 2 * np.pi / lambdas
    plot_count = min(len(q_values), max_q)
    q_indices = list(range(plot_count))
    
    # Initialize matplotlib stuff if needed
    plt = None
    can_plot = False
    
    if generate_plots:
        try:
            # For plotting - only import if needed
            import matplotlib.pyplot as plt
            import matplotlib.cm as cm
            can_plot = True
            
            # Get colormap
            colors = cm.get_cmap("jet", plot_count)
            
            # Create figure(s)
            if len(angle_info_list) == 1:
                # Single plot
                fig, ax = plt.subplots(figsize=(10, 7))
                axes = [ax]
            else:
                # Multiple subplots (one per angle)
                ncols = 2
                nrows = math.ceil(len(angle_info_list) / ncols)
                fig, axes_grid = plt.subplots(nrows=nrows, ncols=ncols, 
                                             figsize=(ncols*6, nrows*5),
                                             sharex=True, sharey=True)
                axes = axes_grid.flatten()
                
                # Add a title if we're showing fit results
                if fit_data and model_name:
                    plt.figtext(0.5, 0.96, f"Fitted Model: {model_name}\n{fitting_models[model_name]['latex']}", ha='center', fontsize=15)
                    
        except ImportError:
            print("Warning: matplotlib import failed, will generate fit parameters only")
    
    # Where we'll store fitting results
    all_angle_fit_params = []
    
    # Process each angle
    for idx, (desc, data) in enumerate(angle_info_list):
        # Plot raw data if we have matplotlib
        if can_plot and idx < len(axes):
            ax = axes[idx]
            configure_axis(ax, desc, len(angle_info_list) > 1, idx, len(angle_info_list))
            
            # Plot each q-value
            for i in q_indices:
                # Add connecting lines if requested
                linestyle = '-' if connect_points and not fit_data else ''
                # Only add label for first row or single plot
                label = f'q={q_values[i]:.2f}' if idx == 0 or len(angle_info_list) == 1 else ""
                ax.semilogx(taus, data.T[:, i], f'o{linestyle}', markersize=4, color=colors(i), label=label)
        
        # Perform fitting if requested
        fit_params = []
        if fit_data:
            fit_results = batch_fit_curves(taus, q_values, data, q_indices)
            
            # Process each fit result
            for i, params_dict, popt in fit_results:
                # Store parameters
                fit_params.append(params_dict)
                
                # Plot fitted curve
                if can_plot and idx < len(axes):
                    # Generate smooth curve for plotting
                    tau_fine = np.logspace(np.log10(taus.min()), np.log10(taus.max()), 100)
                    fit_curve = generic_exp_model(tau_fine, *popt)
                    axes[idx].semilogx(tau_fine, fit_curve, '-', color=colors(i), alpha=0.8)
            
            # Store all fit parameters for this angle
            if fit_params:
                all_angle_fit_params.append((desc, fit_params))
    
    # Finalize and save plots
    if can_plot:
        # Hide unused subplots
        if len(angle_info_list) > 1:
            for i in range(len(angle_info_list), len(axes)):
                if i < len(axes): 
                    axes[i].set_visible(False)
            
            # Add legend showing q-values
            try:
                handles, labels = axes[0].get_legend_handles_labels()
                if handles:
                    fig.legend(handles, labels, title="q values", loc='center left', bbox_to_anchor=(0.88, 0.5), fontsize=11)
            except IndexError:
                pass
        
        # Save figure to file
        output_file = get_output_filename(data_entry, model_name, fit_data, angle_info_list, output_dir)
        plt.tight_layout(rect=[0, 0, 0.88 if len(angle_info_list) > 1 else 1, 0.96])
        plt.savefig(f"{output_file}.png")
        plt.close(fig)

    # Save fit parameters
    if fit_data and all_angle_fit_params:
        param_file = get_output_filename(data_entry, model_name, fit_data, angle_info_list, output_dir) + '.txt'
        save_fit_parameters(all_angle_fit_params, param_file, model_name)
        
    return all_angle_fit_params

# Create key for grouping similar files for averaging
def create_group_key(data, average_episodes, average_tiles):
    info = data['info']
    key_parts = [f"ep{info['episode']}"]
    if not average_episodes:
        key_parts.append(f"win{info['window']}")
    key_parts.append(f"sc{info['scale']}")
    if not average_tiles:
        key_parts.append(f"tile{info['tile']}")
    return "_".join(key_parts)

# Build output identifier for averaged data
def build_output_id(episode, window, scale, tile, average_episodes, average_tiles, average):
    parts = [f"episode{episode}"]
    if not average_episodes:
        parts.append(f"win{window}")
    parts.append(f"scale{scale}")
    if not average_tiles:
        parts.append(f"tile{tile}")
    return "_".join(parts) + f"_avg_{average}"

# Average data from multiple ISF files
def average_angle_data(angle_data_list, angle_desc_list, angle_idx, average):
    if not angle_data_list:
        return None, None
        
    # Make sure all data has same shape
    ref_shape = angle_data_list[0].shape
    valid_data = [d for d in angle_data_list if d.shape == ref_shape]
    
    if not valid_data:
        return None, None
    
    # Take mean of all data arrays    
    averaged_data = np.mean(valid_data, axis=0)
    
    # Create descriptive name
    common_desc = angle_desc_list[0] if angle_desc_list else f"Angle {angle_idx}"
    common_desc += f" (averaged {len(valid_data)} {average})"
    
    return common_desc, averaged_data

# Check if file is a valid ISF data file
def is_valid_data_file(filepath):
    excluded_extensions = ('.png', '.txt')
    return not filepath.lower().endswith(excluded_extensions)

# Main function to load and process multiple data files
def load_and_process_files(file_paths, average='individual', specific_angle=None):
    if not file_paths:
        return []
        
    # Set averaging modes
    average_tiles = average == 'tiles'
    average_episodes = average == 'episodes'
    
    # Load all files
    all_data = []
    print(f"Reading {len(file_paths)} files...")
    
    for file_path in file_paths:
        try:
            # Read file data
            lambdas, taus, angle_info_list = read_data_file(file_path)
            if lambdas is None or not angle_info_list:
                continue
            
            # Parse filename for metadata    
            file_info = parse_filename(os.path.basename(file_path))
            
            # Handle specific angle selection
            if specific_angle is not None and specific_angle < len(angle_info_list):
                angle_info_list = [angle_info_list[specific_angle]]
                file_info['selected_angle'] = specific_angle
            elif specific_angle is not None:
                continue
            
            # Store all data
            all_data.append({'file_path': file_path,'info': file_info,'lambdas': lambdas,'taus': taus,'angle_info_list': angle_info_list})
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
    
    if not all_data:
        return []
    
    # If not averaging, just return all files
    if average == 'individual':
        return all_data
    
    # Group data for averaging
    grouped_data = defaultdict(list)
    for data in all_data:
        key = create_group_key(data, average_episodes, average_tiles)
        grouped_data[key].append(data)
    
    # Do the averaging
    print(f"Averaging {len(grouped_data)} groups...")
    averaged_results = []
    
    for key, group in grouped_data.items():
        # Get basic info from first file in group
        first_info = group[0]['info']
        episode = first_info['episode']
        scale = first_info['scale']
        window = -1 if average_episodes else first_info['window'] 
        tile = -1 if average_tiles else first_info['tile']
        
        # Create output identifier
        file_id = build_output_id(episode, window, scale, tile, average_episodes, average_tiles, average)
        
        # Get data dimensions from first file
        lambdas = group[0]['lambdas']
        taus = group[0]['taus']
        max_angles = max((len(data['angle_info_list']) for data in group), default=0)
            
        if max_angles == 0:
            continue
             
        # Process each angle across all files
        averaged_angle_info = []
        for angle_idx in range(max_angles):
            # Collect data for this angle
            angle_desc_list = []
            angle_data_list = []
            
            for data in group:
                if angle_idx < len(data['angle_info_list']):
                    desc, angle_data = data['angle_info_list'][angle_idx]
                    angle_desc_list.append(desc)
                    angle_data_list.append(angle_data)
            
            # Average the data
            try:
                common_desc, averaged_data = average_angle_data(angle_data_list, angle_desc_list, angle_idx, average)
                
                if common_desc and averaged_data is not None:
                    averaged_angle_info.append((common_desc, averaged_data))
            except Exception:
                pass
        
        # Store averaged results
        if averaged_angle_info:
            averaged_results.append({
                'file_path': file_id,
                'info': {'episode': episode, 'window': window, 'scale': scale, 'tile': tile, 'averaged': average},
                'lambdas': lambdas,
                'taus': taus,
                'angle_info_list': averaged_angle_info
            })
    
    return averaged_results

# Wrapper for parallel processing
def process_entry_wrapper(args_dict):
    data_entry = args_dict.pop('data_entry') 
    
    try:
        return plot_data(data_entry, **args_dict)
    except Exception as e:
        print(f"Error processing {data_entry.get('file_path', 'unknown')}: {e}")
        return None 

# Find ISF data files matching pattern
def find_data_files(input_pattern):
    files = glob.glob(input_pattern)
    return [p for p in files if is_valid_data_file(p)]

# Main entry point
def main():
    # Parse command line args
    parser = argparse.ArgumentParser(description="Process ISF data files")
    
    # Main options
    parser.add_argument("--input", default="episode300-0_scale1024-0", help="Input file pattern (wildcards supported)")
    parser.add_argument("--output-dir", default=None, help="Output directory")
    parser.add_argument("--processes", type=int, default=None, help="Number of CPU cores for parallel processing")
    
    # Processing options
    proc_group = parser.add_argument_group('Processing Options')
    proc_group.add_argument("--mode", choices=['individual', 'tiles', 'episodes'], default='individual', help="Processing mode (individual/tiles/episodes)")
    proc_group.add_argument("--angle", type=int, default=None, help="Process specific angle")
    proc_group.add_argument("--max-q", type=int, default=20, help="Maximum q values to process")
    
    # Fitting options
    fit_group = parser.add_argument_group('Fitting Options')
    fit_group.add_argument("--fit", action="store_true", help="Perform curve fitting")
    fit_group.add_argument("--model", choices=['generic_exp'], default='generic_exp', help="Fitting model type")
    
    # Plot options
    plot_group = parser.add_argument_group('Plot Options')
    plot_group.add_argument("--plots", action="store_true", help="Generate plots")
    plot_group.add_argument("--connect-points", action="store_true", help="Connect data points")
    
    args = parser.parse_args()
    
    # Find data files
    file_paths = find_data_files(args.input)
    
    if not file_paths:
        print(f"No data files found: {args.input}")
        return
        
    # Load and preprocess files
    processed_data = load_and_process_files(file_paths, average=args.mode,specific_angle=args.angle)
    
    if not processed_data:
        print("No valid data entries after loading")
        return
    
    # Set up parallel processing
    num_processes = args.processes if args.processes else multiprocessing.cpu_count()
    print(f"Processing {len(processed_data)} entries using {num_processes} CPU cores...")
    overall_start = time.time()
    
    # Common arguments for all tasks
    common_args = {'max_q': args.max_q, 'output_dir': args.output_dir, 'fit_data': args.fit, 'model_name': args.model, 'connect_points': args.connect_points, 'generate_plots': args.plots}
    
    # Create task list
    tasks = [{'data_entry': entry, **common_args} for entry in processed_data]

    # Process files in parallel
    with multiprocessing.Pool(processes=num_processes) as pool:
        chunksize = max(1, len(tasks) // (num_processes * 2))
        results = list(pool.map(process_entry_wrapper, tasks, chunksize=chunksize))
         
    # Generate report
    total_elapsed = time.time() - overall_start
    successful_count = sum(1 for r in results if r is not None)
    
    # Calculate stats
    files_per_second = successful_count / total_elapsed if total_elapsed > 0 else 0
    time_per_file = total_elapsed / successful_count if successful_count > 0 else 0
    
    # Print summary
    print(f"\n=== Processing Summary ===")
    print(f"Completed in {total_elapsed:.2f}s - Processed {successful_count}/{len(processed_data)} files")
    print(f"Performance: {files_per_second:.2f} files/second ({time_per_file:.2f}s per file)")
    if args.fit:
        print(f"Fitting: Enabled (model: {args.model})")
    if args.plots:
        print(f"Plots: Generated (max q-values: {args.max_q})")
    if args.output_dir:
        print(f"Output saved to: {args.output_dir}")

if __name__ == "__main__":
    multiprocessing.freeze_support()  # for Windows
    main()