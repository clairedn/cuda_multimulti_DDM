import subprocess
import os

def check_file(prompt, file_type):
    while True:
        path = input(prompt)
        if not path:
            print(f"Error: {file_type} file path cannot be empty. Try again.")
            continue
        if not os.path.exists(path):
            print(f"Error: {file_type} file '{path}' does not exist. Try again.")
            continue
        if not os.access(path, os.R_OK):
            print(f"Error: Cannot read {file_type} file '{path}'. Check permissions.")
            continue
        return path

def get_yn(prompt):
    while True:
        ans = input(prompt).lower()
        if ans in ['y', 'n']: 
            return ans
        print("Please enter 'y' or 'n'.")

def get_num(prompt, default=None, min_val=None, as_int=True):
    default_str = f" [{default}]" if default is not None else ""
    while True:
        val = input(f"{prompt}{default_str}: ") or (str(default) if default is not None else "")
        if not val:
            print("Error: Value cannot be empty. Try again.")
            continue
        try:
            if as_int:
                num = int(val)
            else:
                num = float(val)
            if min_val is not None and num < min_val:
                print(f"Error: Value must be at least {min_val}.")
                continue
            return val
        except ValueError:
            print(f"Error: Please enter a valid {'integer' if as_int else 'number'}.")

def main():
    print("=== multimultiDDM Analysis Tool ===")
    
    video = check_file("Video file path: ", "Video")
    lambda_file = check_file("Lambda file path: ", "Lambda")
    tau_file = check_file("Tau file path: ", "Tau")
    scale_file = check_file("Scale file path: ", "Scale")
    episode_file = check_file("Episode file path: ", "Episode")
    frames = get_num("Number of frames (required)", None, 1)

    cmd = ["python", "pipeline.py", "--input", video, "--lambda-file", lambda_file, "--tau-file", tau_file, "--scale-file", scale_file, "--episode-file", episode_file, "--frames", frames]
    
    if get_yn("Show more options? (y/n): ") == 'y':
        output_prefix = input("Output prefix [output_]: ") or "output_"
        cmd.extend(["--output", output_prefix])
        
        offset = get_num("Frame offset", 0, 0)
        if offset != "0": 
            cmd.extend(["--offset", offset])
        
        x_off = get_num("X offset", 0)
        if x_off != "0": 
            cmd.extend(["--x-off", x_off])
        
        y_off = get_num("Y offset", 0)
        if y_off != "0": 
            cmd.extend(["--y-off", y_off])
        
        if get_yn("Enable angle analysis? (y/n): ") == 'y':
            cmd.append("--enable-angle")
            angle_count = get_num("Number of angle sections", 8, 1)
            cmd.extend(["--angle-count", angle_count])
        
        if get_yn("Perform curve fitting? (y/n): ") == 'y':
            cmd.append("--fit")
            
            max_q = get_num("Maximum q values to process", 20, 1)
            cmd.extend(["--max-q", max_q])
            
            while True:
                mode = input("Processing mode (individual/tiles/episodes) [individual]: ") or "individual"
                if mode in ['individual', 'tiles', 'episodes']:
                    break
                print("Error: Processing mode must be 'individual', 'tiles', or 'episodes'.")
            cmd.extend(["--mode", mode])
         
            fit_angle = input("Process specific angle (leave empty for all): ")
            if fit_angle:
                try:
                    int(fit_angle)
                    cmd.extend(["--fit-angle", fit_angle])
                except ValueError:
                    print("Warning: Invalid angle format. Using all angles.")
        
        if get_yn("Generate plots? (y/n): ") == 'y':
            cmd.append("--plots")
            if get_yn("Connect data points? (y/n): ") == 'y':
                cmd.append("--connect-points")
       
        output_dir = input("Output directory (leave empty for current directory): ")
        if output_dir:
            try:
                os.makedirs(output_dir, exist_ok=True)
                cmd.extend(["--output-dir", output_dir])
            except OSError as e:
                print(f"Warning: Cannot create output directory: {e}. Using current directory.")
        
        processes = input("Number of CPU cores for processing (leave empty for all available): ")
        if processes:
            try:
                proc_int = int(processes)
                if proc_int <= 0:
                    print("Warning: Invalid number of cores. Using all available cores.")
                else:
                    cmd.extend(["--processes", processes])
            except ValueError:
                print("Warning: Invalid format for number of cores. Using all available cores.")
        
        if get_yn("Verbose output? (y/n): ") == 'y':
            cmd.append("--verbose")
    
    print("\nExecuting command:", " ".join(cmd))
    subprocess.run(cmd)

if __name__ == "__main__":
    main()