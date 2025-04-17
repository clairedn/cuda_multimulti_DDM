## Guideline 

### Prerequisites
- NVIDIA GPU with CUDA support
- CUDA Toolkit
- OpenCV 4
- C++17 compatible compiler
- Python 3.6+ (for helper scripts)
- Python Libraries: `numpy`, `scipy`, `matplotlib` (matplotlib is optional for plotting)
  ```bash
  pip install numpy scipy matplotlib
  ```

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

The simplest way to run the pipeline is using the interactive command-line tool `gui.py`.

1.  **Ensure Prerequisites and Compilation are complete.**
2.  **Run the script:**
    ```bash
    python gui.py
    ```
3.  **Follow the prompts:** The script will guide you through providing paths to the video and parameter files, setting the number of frames, and configuring all optional analysis, fitting, and plotting parameters. Pressing Enter at a prompt often accepts the default value shown in brackets `[]`.

## Pipeline Scripts Overview

*   **`gui.py`:** An interactive command-line tool to easily configure and launch the analysis pipeline. It gathers all necessary parameters and calls `pipeline.py`.
*   **`pipeline.py`:** The main pipeline script. It takes arguments (usually provided by `gui.py`), runs the compiled `multimultiDDM` executable, and then optionally calls `fitting.py` for post-processing.
*   **`fitting.py`:** Reads the output files from `multimultiDDM`, performs curve fitting on the Image Structure Function data, and can generate plots if requested.

## Input Files

It requires several input files to define parameters:

### Creating Input Files

Create the necessary input files directly using terminal commands. The following is just a random example. 

```bash
# Create tau values (lag times)
echo -e "1\n2\n3\n4\n5" > tau.txt

# Create lambda values (length scales, where q = 2π/lambda)
echo -e "2\n10\n50\n70\n100" > lambda.txt

# Create episode values (time windows)
echo -e "100\n300" > episode.txt

# Create scale values (tile sizes)
echo -e "512\n1024" > scale.txt
```

### Parameter Description

- **tau.txt**: Contains lag times that represent the frame separations to analyze (e.g., 1 means compare consecutive frames, 5 means compare frames that are 5 frames apart). Values are sorted in ascending order and must be positive integers.

- **lambda.txt**: Contains length scales in pixels related to q-vectors by the formula q = 2π/lambda, where q is the wave vector in Fourier space. These define the spatial scales at which dynamics are probed. Values are sorted in ascending order and must be positive. The largest lambda should be smaller than the smallest scale value.

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
   - Calculates q = 2π/lambda for spatial frequency analysis
   - Creates azimuthal average masks in Fourier space for each q-value

10. **Angular Analysis** (if enabled):
    - Divides each annular mask into angular segments
    - Creates separate masks for different angle ranges
    - Default 8 segments covering 180° (only right half-circle due to FFT symmetry)

11. **Azimuthal Averaging**: Uses masks to extract dynamics at specific spatial frequencies
    - Computes radial averages for each q-value
    - Computes sector averages for each angle segment when angular analysis is enabled

12. Writes ISF results to files (fitting is not included now)
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
- Uses length scales from `lambda.txt` (where q = 2π/lambda) 
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
   - Default is 8 segments covering half-circle (180°)
   - Each segment represents a different range in real space
   - For example, with 8 segments, each covers 22.5° of the semicircle

2. Results for each angle segment are written to the output file:
   - Includes angle information (center angle and range)

## Output Files

The analysis generates output files with the below naming convention:

```
episode<window_size>-<window_index>_scale<tile_size>-<tile_index>
```

Example: `episode100-0_scale512-0`

This represents:
- ISF values calculated for:
  - Episode length of 100 frames
  - Window index 0 (first time window, frames 0-99)
  - Scale (tile size) of 512 pixels
  - Tile index 0 (first spatial tile)

The output file contains:
- Line 1: Lambda values (length scales)
- Line 2: Tau values (time delays) converted in seconds (uses video frame rate)
- Remaining lines: ISF values for each combination of lambda and tau
- When angle analysis is enabled, ISF values are organized by angle section

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
