
# CUDA and C++ compilers
NVCC = nvcc
CC = g++-9 

# OpenCV header file path 
INCLUDES = -I/usr/include/opencv4 -I/usr/include/opencv4/opencv2 -I/usr/local/cuda/include

# OpenCV and CUDA libraries 
LDFLAGS = -L/usr/lib/aarch64-linux-gnu -L/usr/local/cuda/lib64 -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_videoio -lcufft -lnvToolsExt

# Compilation options (C++ and CUDA)
CPPFLAGS = -O3 -std=c++11 -g $(INCLUDES)
NVCCFLAGS = -O3 -std=c++11 -gencode arch=compute_53,code=sm_53 $(INCLUDES)

# Source files
CU_SRC = azimuthal_average.cu DDM.cu
CPP_SRC = main.cpp video_reader.cpp debug.cpp
OBJ = $(CU_SRC:.cu=.o) $(CPP_SRC:.cpp=.o)
EXEC = multiDDM

# Compile CUDA files
%.o: %.cu
	$(NVCC) -c $< -o $@ $(NVCCFLAGS)

# Compile C++ files
%.o: %.cpp
	$(CC) -c $< -o $@ $(CPPFLAGS)

# Link the executable
$(EXEC): $(OBJ)
	$(NVCC) $(OBJ) -o $(EXEC) $(LDFLAGS)

# Clean up
clean:
	rm -f *.o $(EXEC)
