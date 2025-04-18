"""
multimultiDDM Analysis Pipeline

Examples:
  # Basic usage
  python pipeline.py --input video.mp4 --lambda-file lambda.txt --tau-file tau.txt
                    --scale-file scale.txt --episode-file episode.txt --frames 1000
                    
  # Enable fitting with plots
  python pipeline.py --input video.mp4 --lambda-file lambda.txt --tau-file tau.txt
                    --scale-file scale.txt --episode-file episode.txt --frames 1000 
                    --fit --plots
"""

import os
import sys
import glob
import argparse
import subprocess
import time
import shutil 

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Run multimultiDDM pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    req = parser.add_argument_group('Required')
    req.add_argument("--input", required=True, help="Input video file")
    req.add_argument("--lambda-file", required=True, help="Lambda values file")
    req.add_argument("--tau-file", required=True, help="Tau values file")
    req.add_argument("--scale-file", required=True, help="Scale values file")
    req.add_argument("--episode-file", required=True, help="Episode values file")
    req.add_argument("--frames", type=int, required=True, help="Number of frames to analyze")

    analysis = parser.add_argument_group('Analysis')
    analysis.add_argument("--output", default="output_", help="Output prefix")
    analysis.add_argument("--offset", type=int, default=0, help="Frame offset")
    analysis.add_argument("--x-off", type=int, default=0, help="X offset")
    analysis.add_argument("--y-off", type=int, default=0, help="Y offset")
    analysis.add_argument("--enable-angle", action="store_true", help="Enable angle analysis")
    analysis.add_argument("--angle-count", type=int, default=8, help="Number of angle sections")
    
    fit = parser.add_argument_group('Fitting')
    fit.add_argument("--fit", action="store_true", help="Perform curve fitting")
    fit.add_argument("--max-q", type=int, default=20, help="Maximum q values to process")
    fit.add_argument("--mode", choices=["individual", "tiles", "episodes"], 
                     default="individual", help="Processing mode")
    fit.add_argument("--fit-angle", type=int, default=None, help="Process specific angle")
    fit.add_argument("--connect-points", action="store_true", help="Connect data points")
    fit.add_argument("--plots", action="store_true", help="Generate plots")
    fit.add_argument("--output-dir", default=None, help="Output directory")
    fit.add_argument("--processes", type=int, default=None, help="CPU cores for fitting")
    
    other = parser.add_argument_group('Other')
    other.add_argument("--verbose", action="store_true", help="Verbose output")
    
    return parser.parse_args()

def run_cmd(cmd, description, verbose=False):
    print(f"\n=== {description} ===")
    start_time = time.time()
    
    try:
        result = subprocess.run(cmd, check=True, text=True, capture_output=True)
        if verbose and result.stdout.strip():
            print(result.stdout.strip())
        if result.stderr.strip():
            print(f"Note: {result.stderr.strip()}")
    except subprocess.CalledProcessError as e:
        print(f"Error: Command failed (code: {e.returncode})")
        if e.stderr: 
            print(f"{e.stderr.strip()}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    elapsed = time.time() - start_time
    print(f"Completed in {elapsed:.2f} seconds")
    return elapsed

def verify_prerequisites(args, fit_script="fitting.py"):
    ddm_exe = shutil.which("multimultiDDM") or shutil.which("./multimultiDDM")
    if not ddm_exe or not os.path.exists(fit_script):
        print("Error: Required executables not found")
        sys.exit(1)
    
    required_files = [args.input, args.lambda_file, args.tau_file, args.scale_file, args.episode_file]
    for path in required_files:
        if not os.path.exists(path) or not os.access(path, os.R_OK):
            print(f"Error: Cannot access file {path}")
            sys.exit(1)
         
    return ddm_exe

def create_output_dirs(args):
    output_dir = os.path.dirname(args.output)
    if output_dir:
            os.makedirs(output_dir, exist_ok=True)
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

def build_ddm_command(args, ddm_exe):
    cmd = [ddm_exe, "-f", args.input, "-N", str(args.frames), "-T", args.tau_file, "-Q", args.lambda_file, "-E", args.episode_file, "-S", args.scale_file, "-o", args.output]

    optional_flags = {
        "-v": args.verbose,
        "-s": args.offset if args.offset > 0 else None,
        "-x": args.x_off if args.x_off > 0 else None,
        "-y": args.y_off if args.y_off > 0 else None,
        "-A": args.enable_angle,
        "-n": str(args.angle_count) if args.enable_angle else None
    }
    
    for flag, value in optional_flags.items():
        if value is True:
            cmd.append(flag)
        elif value is not None and value is not False:
            cmd.extend([flag, str(value)])
            
    return cmd

def find_isf_files(output_prefix):
    isf_pattern = f"{output_prefix}episode*_scale*"
    return [f for f in glob.glob(isf_pattern) 
            if not f.lower().endswith(('.txt', '.png'))]

def build_fitting_command(args, fit_script="fitting.py"):
    cmd = [sys.executable, fit_script, "--input", f"{args.output}episode*_scale*", "--max-q", str(args.max_q), "--mode", args.mode]
    
    if args.output_dir:
        cmd.extend(["--output-dir", args.output_dir])
    if args.fit:
        cmd.append("--fit")
    if args.processes:
        cmd.extend(["--processes", str(args.processes)])
    if args.fit_angle is not None:
        cmd.extend(["--angle", str(args.fit_angle)])
    if args.connect_points: 
        cmd.append("--connect-points")
    if args.plots: 
        cmd.append("--plots")
        
    return cmd

def print_summary(total_time, analysis_time, fitting_time, frames, args):
    fps = frames / total_time
    
    print(f"\n=== Pipeline Summary ===")
    print(f"Total time: {total_time:.2f}s (Analysis: {analysis_time:.2f}s, Fitting: {fitting_time:.2f}s)")
    print(f"Processing speed: {fps:.2f} frames/second")
    
    if args.fit or args.plots:
        output_loc = args.output_dir if args.output_dir else "."
        print(f"\n=== Output Files ===")
        if args.fit:
            print(f"Fit parameters: {output_loc}/output_episode*_scale*_fit_generic_exp.txt")
        if args.plots: 
            print(f"Plot files: {output_loc}/output_episode*_scale*_fit_generic_exp.png")

def main():
    pipeline_start = time.time()
    args = parse_arguments()
    
    ddm_exe = verify_prerequisites(args)
    create_output_dirs(args)
    
    ddm_cmd = build_ddm_command(args, ddm_exe)    
    ddm_elapsed = run_cmd(ddm_cmd, "Running multimultiDDM analysis", args.verbose)
    
    matching_files = find_isf_files(args.output)
    fit_elapsed = 0
    
    if not matching_files:
        print("Warning: No ISF data files found. Skipping fitting phase")
    else:
        print(f"Found {len(matching_files)} ISF files")
        fit_cmd = build_fitting_command(args)
        fit_elapsed = run_cmd(fit_cmd, "Running fitting", args.verbose)

    total_elapsed = time.time() - pipeline_start
    print_summary(total_elapsed, ddm_elapsed, fit_elapsed, args.frames, args)
        
if __name__ == "__main__":
    main()