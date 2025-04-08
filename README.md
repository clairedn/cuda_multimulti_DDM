# CUDA multi multi DDM

## Installation

### Prerequisites
- NVIDIA GPU with CUDA support
- CUDA Toolkit
- OpenCV 4
- C++17 compatible compiler

### Compilation

Clone the repository and compile the code:

```bash
git clone https://github.com/clairedn/cuda_multimulti_DDM.git
cd cuda_multimulti_DDM

# Compile CUDA components
nvcc -c azimuthal_average.cu -o azimuthal_average.o -O3 -std=c++17 --use_fast_math -I/usr/local/include/opencv4 -I/usr/local/include/opencv4/opencv2
nvcc -c DDM.cu -o DDM.o -O3 -std=c++17 --use_fast_math -I/usr/local/include/opencv4 -I/usr/local/include/opencv4/opencv2

# Compile C++ components
g++ -c main.cpp -o main.o -O3 -std=c++17 -I/usr/local/include/opencv4 -I/usr/local/include/opencv4/opencv2
g++ -c video_reader.cpp -o video_reader.o -O3 -std=c++17 -I/usr/local/include/opencv4 -I/usr/local/include/opencv4/opencv2
g++ -c debug.cpp -o debug.o -O3 -std=c++17 -I/usr/local/include/opencv4 -I/usr/local/include/opencv4/opencv2

# Link everything
nvcc azimuthal_average.o DDM.o main.o video_reader.o debug.o -o multiDDM \
-L/usr/local/lib -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_videoio -lcufft -lnvToolsExt
```

## Input Files

The DDM analysis requires several input files to define parameters:

### Creating Input Files

You can create the necessary input files directly using terminal commands:

```bash
# Create tau values (time intervals)
echo -e "1\n2\n3\n4\n5" > tau.txt

# Create lambda values (length scales)
echo -e "2\n10\n50\n70\n100" > lambda.txt

# Create episode values (time windows)
echo -e "100\n300" > episode.txt

# Create scale values (spatial scales)
echo -e "512\n1024" > scale.txt
```

## Usage

### Command Line Arguments

```
 ~~ multiscale DDM - CUDA ~~ 

  Usage ./multiDDM [OPTION]..
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
  -Z           Turn off multi-steam (smaller memory footprint - slower execution time).
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
./multiDDM -f diff_1.0.mp4 -T tau.txt -Q lambda.txt -E episode.txt -S scale.txt -A -N 900
```

This command:
- Analyzes the video file `diff_1.0.mp4`
- Uses the time intervals from `tau.txt`
- Uses the length scales from `lambda.txt`
- Uses the time windows from `episode.txt`
- Uses the spatial scales from `scale.txt`
- Enables angle analysis (`-A`)
- Analyzes 900 frames (`-N 900`)

## Output

The analysis generates output files with naming convention based on the episode and scale values:

Example: `episode100-0_scale512-0`

This represents:
- ISF (Intermediate Scattering Function) values for episode length (100 frames)
- Scale 512
- For the first time window and first tile

## Additional Features

- **Webcam Input**: Instead of a video file, you can use a webcam as input
- **Benchmark Mode**: Test performance using random data
- **Angle Analysis**: Enable angular sector analysis for directional information
- **Multi-stream Processing**: Use GPU streams for faster processing (can be disabled with `-Z`)

## Advanced Options

- **Custom Frame Rate**: Force a specific frame rate with `-F`
- **Q-vector Tolerance**: Adjust tolerance factor for q-vector mask with `-t`
- **Memory Management**: Control buffer size with `-C` and `-G` options
- **Offsets**: Set frame, x, and y offsets with `-s`, `-x`, and `-y` options 