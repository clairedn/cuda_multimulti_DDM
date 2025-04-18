## Guideline 

### Prerequisites
- NVIDIA GPU with CUDA support
- CUDA Toolkit
- OpenCV 4
- Python Libraries: `numpy`, `scipy`, `matplotlib` (matplotlib is optional and only needed for plotting)

### Compilation on w1

Clone the repository and compile the code (compiler warnings about C++ inheritance and function overriding in OpenCV could be ignored):

```bash
git clone https://github.com/clairedn/cuda_multimulti_DDM.git
cd cuda_multimulti_DDM

# Compile CUDA components 
nvcc -c azimuthal_average.cu -o azimuthal_average.o -O3 -std=c++17 --use_fast_math -I/usr/local/include/opencv4
nvcc -c DDM.cu -o DDM.o -O3 -std=c++17 --use_fast_math -I/usr/local/include/opencv4

# Compile C++ components
g++ -c main.cpp -o main.o -O3 -std=c++17 -I/usr/local/include/opencv4
g++ -c video_reader.cpp -o video_reader.o -O3 -std=c++17 -I/usr/local/include/opencv4
g++ -c debug.cpp -o debug.o -O3 -std=c++17 -I/usr/local/include/opencv4

# Link everything
nvcc azimuthal_average.o DDM.o main.o video_reader.o debug.o -o multimultiDDM -L/usr/local/lib -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_videoio -lcufft -lnvToolsExt

```

It uses CUDA for GPU acceleration and supports:
- Multiple spatial scales (tile)
- Multiple temporal scales (time window/episode length)
- Angular analysis (optional)

## Usage (Recommended Method: Interactive GUI)

A simple way to run the pipeline is using the interactive command-line tool `gui.py`.

1.  **Ensure Prerequisites and Compilation are complete.**
2.  **Run the script:**
    ```bash
    python gui.py
    ```
3.  **Follow the prompts:** It will guide you through providing paths to the video and parameter files, setting the number of frames, and configuring all optional analysis, fitting, and plotting parameters. Pressing Enter at a prompt often accepts the default value shown in brackets `[]`. If you answer 'n' to the initial 'Show more options?' prompt, the script will skip all optional settings and execute the pipeline using the default values for those parameters.

### Example GUI Session

Below is an example of a complete interactive session with `gui.py`, showing the prompts and user inputs:

```
=== multimultiDDM Analysis Tool ===
Video file path: video.mp4
Lambda file path: lambda.txt
Tau file path: tau.txt
Scale file path: scale.txt
Episode file path: episode.txt
Number of frames (required): 900
Show more options? (y/n): y
Output prefix [output_]: 
Frame offset [0]: 
X offset [0]: 
Y offset [0]: 
Enable angle analysis? (y/n): y
Number of angle sections [8]: 4
Perform curve fitting? (y/n): y
Maximum q values to process [20]: 15
Processing mode (individual/tiles/episodes) [individual]: episodes
Process specific angle (leave empty for all): 
Generate plots? (y/n): n
Output directory (leave empty for current directory): 
Number of CPU cores for processing (leave empty for all available): 
Verbose output? (y/n): n

Executing command: python pipeline.py --input video.mp4 --lambda-file lambda.txt --tau-file tau.txt --scale-file scale.txt --episode-file episode.txt --frames 900 --output output_ --enable-angle --angle-count 4 --fit --max-q 15 --mode episodes

=== Running multimultiDDM analysis ===
Completed in 3.97 seconds
Found 45 ISF files

=== Running fitting ===
Completed in 0.63 seconds

=== Pipeline Summary ===
Total time: 4.60s (Analysis: 3.97s, Fitting: 0.63s)
Processing speed: 195.60 frames/second

=== Output Files ===
Fit parameters: ./output_episode*_scale*_fit_generic_exp.txt
```

This example shows how to:
- Process a video file with 900 frames
- Enable angle analysis with 4 angle sections
- Perform curve fitting on the first 15 q values (the largest 15 q values, corresponding to the smallest 15 lambda values)
- Use "episodes" processing mode
- Skip plot generation for faster processing

The GUI automatically constructs and executes the appropriate command-line parameters for `pipeline.py`.

## Pipeline Scripts Overview

*   **`gui.py`:** An interactive command-line tool to easily configure and launch the pipeline. It gathers all necessary parameters and calls `pipeline.py`.
*   **`pipeline.py`:** It takes arguments (provided by `gui.py`), runs the compiled `multimultiDDM` executable, and then optionally calls `fitting.py` if needed.
*   **`fitting.py`:** Reads the output files from `multimultiDDM`, performs curve fitting on the ISF data, and generate plots if needed.

## Output File Structure

### 1. Raw ISF Data Files

For each analysis run, the program generates multiple ISF data files - one for each unique combination of:
- Episode length
- Window index
- Scale value
- Tile index

The number of files depends on these parameters. For example, with:
- 1 episode value (e.g., 100 frames)
- 8 windows (total frames ÷ episode length)
- 2 scale values (e.g., 512 and 1024)
- 1 tile for scale 1024 (main scale)
- 4 tiles for scale 512 (when using main scale of 1024)

This would generate: 1 episode × 8 windows × (1 tile for scale 1024 + 4 tiles for scale 512) = 40 files.

**Example ISF file list:**
```
output_episode100-0_scale512-0
output_episode100-0_scale512-1
output_episode100-0_scale512-2
output_episode100-0_scale512-3
output_episode100-0_scale1024-0
output_episode100-1_scale512-0
...
output_episode100-7_scale1024-0
```

Each file contains the raw ISF data which shows how the image structure changes over different lag times (tau) and spatial wavelengths (lambda). This forms a 2D data matrix where:
- Each row represents a different lambda value (or corresponding q value)
- Each column represents a different tau value
- Cell values are the ISF intensity at that specific combination of lambda and tau

The file format is:
- Line 1: Lambda values (spatial wavelengths in pixels)
- Line 2: Tau values (lag times, converted to seconds using the video frame rate)
- Remaining lines: ISF values organized as a matrix where:
  - Each row corresponds to one lambda value
  - Each column corresponds to one tau value
- When angle analysis is enabled, it repeats for each angle section, with angle information headers

These raw ISF data files serve as input for the fitting process, which extracts dynamic parameters from the tau-dependent behavior at each q value.

### 2. Fitting Output Files

When fitting is enabled (`--fit` option), additional files are generated with the suffix `_fit_generic_exp.txt`:

```
episode100-1_scale512-0_fit_generic_exp.txt
episode100-1_scale512-1_fit_generic_exp.txt
...
```

These files contain:
- A header describing the fitting model: `I(q,τ) = A(1-e^{-(Γτ)^β}) + B`
- For each angle section (if enabled):
  - The angle description (center angle and range)
  - A table of parameters for each q value (see "Fitting Parameter Files" section below for detailed parameter descriptions)

## Fitting Parameter Files

The fitting process in `fitting.py` uses a curve-fitting approach to extract dynamic parameters from the Image Structure Function data. For each spatial frequency q, the ISF is analyzed as a function of lag time (τ) using the model:

```
I(q,τ) = A(1-e^{-(Γτ)^β}) + B
```

Where τ is the independent variable (lag time), and the fitted parameters are:
- A: Amplitude of the decay function, related to the contrast of the dynamic signal. Higher A values indicate stronger structural changes.
- Γ (Gamma): Decay rate parameter, inversely proportional to the characteristic relaxation time (τₒ = 1/Γ). Higher Γ values indicate faster dynamics (shorter characteristic times).
- β (beta): Stretching/compressing exponent, indicates whether relaxation is simple exponential (β=1) or stretched/compressed (β≠1).
- B: Baseline offset, represents the non-decaying component at infinite time delays.

Note that the fitting is performed separately for each q value. The program fits how the ISF changes with τ at fixed q values, resulting in one set of parameters (A, Γ, β, B) per q value. Therefore, if you analyze 15 q values, you will get 15 sets of fitting parameters, each describing the dynamics at a different spatial scale.

The physical interpretation of these parameters is important for understanding the dynamics:
- Higher Γ values indicate faster dynamics at that particular q value
- β < 1 suggests stretched exponential relaxation, often seen in complex systems with multiple relaxation pathways or heterogeneous dynamics
- β > 1 indicates compressed exponential behavior, sometimes observed in actively driven systems or systems under stress
- A/B ratio can indicate the relative contribution of dynamic vs. static components in the signal

When using different processing modes:
- `individual`: Each ISF file gets its own fitting file
- `tiles`: Files for the same scale but different tiles are averaged before fitting
- `episodes`: Files for the same window size but different indices are averaged before fitting

These fitting parameters provide quantitative information about the dynamics at different spatial scales, which can be related to physical properties of the sample.

The `--max-q` parameter (default: 20) controls how many q values are included in the fitting process. Note that lambda values are sorted in ascending order, while the corresponding q values are effectively sorted in descending order. So setting `--max-q 15` selects the 15 largest q values (corresponding to the 15 smallest lambda values) for fitting. This is useful when you want to focus on smaller spatial scales (higher q values) or reduce computational load while retaining the most relevant dynamics information.

## Processing Modes

When using `fitting.py` (either directly or through the pipeline), there are three different processing modes that determine how data files are combined before fitting:

### 1. `individual` Mode (Default Mode)
- **Working Principle**: Each ISF file is processed individually without any averaging
- **File Processing**: Generates an independent fitting file for each ISF file
- **Use Case**: When you need to preserve all details and analyze the unique dynamic properties of each time window and spatial region
- **Output File Count**: Same as the number of input ISF files
- **Output Naming**: Original filename with `_fit_generic_exp` suffix

### 2. `tiles` Mode
- **Working Principle**: Averages ISF files with the same scale but different tile positions
- **File Processing**: Combines and averages all tile data from the same scale before fitting
- **Averaging Strategy**: Groups by `episode<value>-<index>_scale<value>`, ignoring tile index differences
- **Use Case**: When your sample is spatially homogeneous and you want to reduce spatial sampling noise
- **Output File Count**: `Number of time windows × Number of scale values`
- **Output Naming**: `episode<value>-<index>_scale<value>_avg_tiles_fit_generic_exp`

### 3. `episodes` Mode
- **Working Principle**: Averages ISF files with the same window size (episode value) but different window indices
- **File Processing**: Combines and averages all window index data with the same episode value before fitting
- **Averaging Strategy**: Groups by `episode<value>_scale<value>-<index>`, ignoring window index differences
- **Use Case**: When your sample is temporally stationary and you want to increase statistical reliability
- **Output File Count**: `Number of window size values × Number of scale values × Number of tiles`
- **Output Naming**: `episode<value>_scale<value>-<index>_avg_episodes_fit_generic_exp`

### Choosing the Right Processing Mode

- If your sample exhibits consistent dynamic properties throughout the observation time, `episodes` mode can provide better statistical averaging and reduce noise in fitting parameters.
- If your sample is spatially homogeneous, `tiles` mode can provide cleaner averaged data.
- If you want to preserve all temporal and spatial details, or if your sample is non-uniform in time or space, `individual` mode is most appropriate.

You can select the processing mode through the GUI or using the command-line parameter `--mode`:
```bash
python pipeline.py --mode episodes ... other parameters ...
```

## Performance Optimization

### Multi-CPU Processing

The fitting process supports parallel computation across multiple CPU cores, which can significantly speed up the analysis, especially for large datasets or when processing many files:

```bash
# Example: Use 4 CPU cores for fitting
python pipeline.py --processes 4 ... other parameters ...
```

If `--processes` is not specified, it will automatically use all available CPU cores. For optimal performance:
- On systems with limited memory, consider using fewer cores to avoid memory pressure
- For systems with many CPU cores, using about 50-75% of available cores may provide better balance of performance and resource usage

During processing, it displays real-time performance metrics, including:
- Processing time for each phase (DDM analysis, fitting)
- Overall frames per second processing rate
- Count of processed files and remaining tasks

When the verbose mode is enabled (using the `-v` flag with `multimultiDDM` or selecting 'y' for verbose output in the GUI), additional memory usage statistics will be displayed:
```
Memory Allocations Done.
Total memory allocated
Device:
    Explicitly allocated:    0.123456 GB 
    Total allocated:         0.234567 GB
    Free memory remaining:   7.654321 GB
Host:   0.012345 GB
```

These statistics help monitor resource usage and optimize performance for specific hardware configuration.

### Memory Management

The CUDA multi-multi DDM pipeline is designed to handle memory during large-scale video processing:

1. **Triple-Buffer System**: It implements a triple-buffer system to overlap computation and data transfer, maximizing GPU utilization
2. **Chunk-Based Processing**: Videos are processed in smaller chunks (default: 30 frames) to limit memory usage
3. **Scale-Based Memory Allocation**: Memory is allocated according to the maximum scale and then reused for smaller scales
4. **Stream Management**: Optional dual-stream processing for systems with sufficient GPU memory

To optimize memory usage for specific hardware:

- **For Systems with Limited GPU Memory**:
  - Reduce the chunk size with `-C` parameter (e.g., `-C 15` for smaller chunks)
  - Disable multi-stream processing with `-Z` flag
  - Process videos with lower resolution or smaller frames if possible
  - Consider using the rolling purge option for very long videos

- **For High-Performance Systems**:
  - Increase chunk size for better efficiency (e.g., `-C 60`)
  - Keep multi-stream enabled for maximum throughput
  - Utilize more CPU cores for fitting with `--processes`

### Handling Large Video Files

For very long video files, it includes a rolling purge feature that processes accumulators periodically:

```bash
# Process and save results every 10 chunks
./multimultiDDM -f video.mp4 ... other parameters ... -G 10
```

This feature:
- Reduces peak memory usage by periodically clearing accumulators
- Allows for checkpointing during long-running analyses
- Generates intermediate results that can be analyzed before the entire video is processed

Combined with appropriate chunk sizing, the rolling purge option makes it possible to process video files of arbitrary length, limited only by storage capacity rather than memory.

When working with multi-GB video files, consider:
1. Splitting analysis into episodes with manageable window sizes
2. Using the benchmark mode for initial testing to verify memory usage before committing to full analysis
3. Monitoring system memory usage during processing
> **Note:** The advanced memory management options (`-C` for chunk size, `-Z` for disabling multi-stream processing, and `-G` for rolling purge) are currently available only when using the direct command-line interface with `multimultiDDM`. These options are not included in the interactive `gui.py` tool (can be added later if needed)

## Input Files

It requires several input files to define parameters:

### Creating Input Files

Create the necessary input files directly using terminal commands. The following is just a random example. 

```bash
# Create tau values (lag times)
echo -e "1\n2\n3\n4\n5" > tau.txt

# Create lambda values (length scales)
echo -e "2\n10\n50\n70\n100" > lambda.txt

# Create episode values (time windows)
echo -e "100\n300" > episode.txt

# Create scale values (tile sizes)
echo -e "512\n1024" > scale.txt
```

### Parameter Description

- **tau.txt**: Contains lag times that represent the frame separations to analyze (e.g., 1 means compare consecutive frames, 5 means compare frames that are 5 frames apart). Values are sorted in ascending order and must be positive integers.

- **lambda.txt**: Contains length scales in pixels that define the spatial scales at which dynamics are probed. Values are sorted in ascending order and must be positive. The largest lambda should be smaller than the smallest scale value.

- **episode.txt**: Contains time window sizes for temporal analysis. Each value defines a window size in frames. For each window size, it:
  - Divides the total video into multiple windows of that size
  - Processes each window independently to capture dynamics at different time/temporal positions
  - Values are sorted in ascending order during processing

- **scale.txt**: Contains tile sizes for spatial subdivision of each frame. Values:
  - Must be powers of 2 (512, 1024, etc.)
  - Are sorted in descending order during processing
  - Define the size of tiles (in pixels) used for spatial segmentation
  - The largest scale is used as the main scale for organizing tiles

## Processing Pipeline

1. Checks all input parameters 
   - Validates tau, lambda, scale, and episode values
   - Sorts parameters in appropriate order (tau and lambda ascending, scale descending, episode ascending)

2. For each episode size in `episode.txt`:
   - Divides total video into windows of specified size
   - Creates multiple time windows for analysis
   - Each window is processed independently

3. For each time window:
   - Positions video to start frame of window
   - Processes frames in memory-efficient chunks (default chunk size is 30 but could be revised using -C)

4. For each scale value in `scale.txt`:
   - Divides each frame into tiles based on specified scale

5. Loads video frames in chunks for efficient memory management
   - Implements circular/triple-buffer 
   - Uses CUDA streams for overlapping memory transfers and computation

6. First applies Fast Fourier Transform to each frame
   - Computes FFT for each tile at each scale
   - Uses CUFFT library for GPU-accelerated transform

7. For each tau value in `tau.txt`:
   - Computes differences between FFTs of frames separated by tau

8. Computes |FFT(I(t+τ) - I(t))|² for each frame pair
   - Accumulates results for all available frame pairs

9. For each lambda value in `lambda.txt`:
   - Calculates the corresponding q value for spatial frequency analysis
   - Creates azimuthal average masks in Fourier space for each q-value

10. **Angular Analysis** (if enabled):
    - Divides each annular mask into angular segments
    - Creates separate masks for different angle ranges
    - Default 8 segments covering 180° (only right half-circle due to FFT symmetry)

11. **Azimuthal Averaging**: Uses masks to extract dynamics at specific spatial frequencies
    - Computes radial averages for each q-value
    - Computes sector averages for each angle segment when angular analysis is enabled

12. Writes ISF results to files 
    - Separate file for each combination of episode, window index, scale, and tile index
    - Includes metadata about parameters used in analysis in the first two rows (lambda values and tau values (converted in seconds)) 

## Usage

### Command Line Arguments

```
 ~~ multiscale DDM - CUDA ~~ 

  Usage ./multimultiDDM [OPTION]..
  -h           Print out this help.
   REQUIRED ARGS
  -o PATH      Output file-path.
  -N INT       Number of frames to analyse.
  -Q PATH      Specify path to lambda-value file (line separated).
  -T PATH      Specify path to tau-value file (line separated).
  -S PATH      Specify path to scale-value file (line separated).
  -E PATH      Specify path to episode-value file (line separated).
   INPUT ARGS
  -f PATH      Specify path to input video (either -f or -W option must be given).
  -W INT       Use web-camera as input video, (web-camera number can be supplied, defaults to first connected camera).
  -B           Benchmark mode, will perform analysis on random data.
   OPTIONAL ARGS
  -s OFFSET    Set first frame offset (default 0).
  -x OFFSET    Set x-offset (default 0).
  -y OFFSET    Set y-offset (default 0).
  -I           Use frame indices for tau-labels not real time.
  -v           Verbose mode on.
  -Z           Turn off multi-stream (smaller memory footprint - slower execution time).
  -t INT       Set the q-vector mask tolerance - percent (integer only) (default 20 i.e. radial mask (1 - 1.2) * q).
  -C INT       Set main chunk frame count, a buffer 3x chunk frame count will be allocated in memory (default 30 frames).
  -G SIZE      Sub-divide analysis, buffer will be output and purged every SIZE chunks
  -M           Set if using movie-file format.
  -F FPS       Force the analysis to assume a specific frame-rate, over-rides other options.
  -A           Enable angle analysis
  -n INT       Set angle count (default is 8)
```

### Example Command

```bash
./multimultiDDM -f video.mp4 -N 900 -T tau.txt -Q lambda.txt -E episode.txt -S scale.txt -A -n 16
```

This command:
- Chooses the video file `video.mp4` 
- Analyzes a total of 900 frames (`-N 900`) 
- Uses lag times from `tau.txt`
- Uses length scales from `lambda.txt`
- Uses time windows from `episode.txt` for temporal subdivision of the full video
- Uses tile sizes from `scale.txt` for spatial subdivision of each frame
- Enables angle analysis (`-A`) with 16 angle sections (`-n 16`)  (this feature is not enabled by default and the default angle sections without `-n` is 8)
  
Note: 
- You can use either relative/absolute paths for all input files
- The above example doesn't specify an output file path with `-o`. Without this parameter, output files will be generated in the current directory with names following the pattern: `episode<window_size>-<window_index>_scale<tile_size>-<tile_index>`
- To specify an output path, add `-o /output_path/prefix` to the command. It will use this prefix for all output files.
  
## Temporal Windows Analysis

The episode values in `episode.txt` define different time window sizes. It processes each episode value separately, following these steps:

1. For each episode value (let's call it E), the program divides the total video length (N frames) into multiple windows:
   - Number of windows = ceiling(N ÷ E)
   - Each window contains E frames, except possibly the last window which may contain fewer frames if N is not a multiple of E

2.  For each window:
   - Video is positioned to the starting frame of the window
   - Frames within the window are loaded in chunks for efficiency
   - DDM is performed on frames within the window
   - Results are accumulated 

3. For example, with total frames N = 900 and episode value E = 100:
     - Creates 9 windows (900 ÷ 100 = 9)
     - Windows span frames 0-99, 100-199, 200-299, ..., 800-899
     - Each window is processed independently, capturing dynamics at different times in the video

4. For each episode value in the file, the above process is repeated
   - Allows comparison across different temporal scales
   - For example, with episode values of 100 and 300:
     - Episode 100: Creates 9 windows of 100 frames each (fine temporal resolution)
     - Episode 300: Creates 3 windows of 300 frames each (coarser temporal resolution)

## Angular Analysis

When enabled with the `-A` flag, it performs angular analysis (might be useful for anisotropy):

1. The 2D Fourier space is divided into angle segments:
   - Default is 8 segments covering half-circle (180°) from -90° to +90°
   - Each segment represents a different orientation in real space
   - For example, with 8 segments, each covers 22.5° of the semicircle
   - Segments are centered around angles from -90° to +90° (using half range due to FFT symmetry)

2. Results for each angle segment are written to the output file:
   - Each angle section includes metadata with its center angle and range
   - For example: "Angle section (radial direction) 0 (center angle: -78.75 degrees, range: -90.0 to -67.5 degrees)"

## Memory Management

1. **Chunk Processing**: Frames are loaded and processed in chunks:
   - Default chunk size is 30 frames (`-C` option)
   - Memory allocation is 3× chunk size for triple-buffer processing

2. **Rolling Purge**: For very long videos, accumulated results can be output periodically:
   - Controlled by the `-G` option
   - After processing G chunks, results are written to disk and buffers cleared
   - Allows processing of arbitrarily long videos with limited memory

3. **Multi-Stream Processing**: By default, uses CUDA streams for parallel processing:
   - Can be disabled with `-Z` option for lower memory usage
   - Improves performance by overlapping computation and data transfer

## Additional Features

- **Webcam Input**: Instead of a video file, you can use a webcam as input with the `-W` option
- **Benchmark Mode**: Test performance using random data with the `-B` option
- **Custom Frame Rate**: Force a specific frame rate with `-F` when video metadata is incorrect
- **Q-vector Tolerance**: Adjust tolerance factor for q-vector mask with `-t` (affects the width of azimuthal average masks)
- **Offsets**: Set frame, x, and y offsets with `-s`, `-x`, and `-y` options for specific analysis regions 

## Cleaning and Recompiling

If you need to clean all compiled files (.o files and executable) and recompile, use the following commands:

```bash
# Clean all .o files and executable
rm -f *.o multimultiDDM

# Recompile
# Compile CUDA components 
nvcc -c azimuthal_average.cu -o azimuthal_average.o -O3 -std=c++17 --use_fast_math -I/usr/local/include/opencv4
nvcc -c DDM.cu -o DDM.o -O3 -std=c++17 --use_fast_math -I/usr/local/include/opencv4

# Compile C++ components
g++ -c main.cpp -o main.o -O3 -std=c++17 -I/usr/local/include/opencv4
g++ -c video_reader.cpp -o video_reader.o -O3 -std=c++17 -I/usr/local/include/opencv4
g++ -c debug.cpp -o debug.o -O3 -std=c++17 -I/usr/local/include/opencv4

# Link everything
nvcc azimuthal_average.o DDM.o main.o video_reader.o debug.o -o multimultiDDM -L/usr/local/lib -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_videoio -lcufft -lnvToolsExt
```

If you only want to recompile a specific file (for example, if you modified DDM.cu), you can use:

```bash
# Only recompile the modified file
rm -f DDM.o multimultiDDM
nvcc -c DDM.cu -o DDM.o -O3 -std=c++17 --use_fast_math -I/usr/local/include/opencv4

# Relink
nvcc azimuthal_average.o DDM.o main.o video_reader.o debug.o -o multimultiDDM -L/usr/local/lib -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_videoio -lcufft -lnvToolsExt
```

Then run the program again after compilation:

```bash
# Example command to run the program
./multimultiDDM -f video.mp4 -N 900 -T tau.txt -Q lambda.txt -E episode.txt -S scale.txt 
```
