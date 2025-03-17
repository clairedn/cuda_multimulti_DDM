#include <cuda_runtime.h>

#ifndef CONSTANTS_HPP_
#define CONSTANTS_HPP_

// Block-size constants for use in main-code
// Note the maximum number of threads-per-block is 1025, however
// better performance is achieved through smaller block sizes
// This however is some-what empirical so more analysis is needed

int const BLOCKSIZE_X = 16;
int const BLOCKSIZE_Y = 16;
int const BLOCKSIZE = 256;

// If we want to scale the pixel-values from input video then we create a look-up table
//__constant__ float dk_uchar_float_lookup[256];

#endif