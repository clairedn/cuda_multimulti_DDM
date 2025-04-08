#include <cuda_runtime.h>
#include "constants.hpp"
#include "video_reader.hpp"

#ifndef KERNEL_DDM
#define KERNEL_DDM

///////////////////////////////////////////////////////
// GPU function to parse the input video to get ready for FFT
///////////////////////////////////////////////////////
__global__ void parseBufferScale(const unsigned char* __restrict__ d_buffer,
                                 float* __restrict__ d_parsed,
                                 video_info_struct info,
                                 int scale,
                                 int main_scale,
                                 int frame_count) {

    int x = blockIdx.x * BLOCKSIZE_X + threadIdx.x;
    int y = blockIdx.y * BLOCKSIZE_Y + threadIdx.y;

    if (x < main_scale && y < main_scale) {
        int local_x = x % scale;	// this operation is somewhat slow
        int local_y = y % scale;

        int tile_x = x / scale;
        int tile_y = y / scale;

        int tile_width = main_scale / scale;

        int new_idx = ((tile_y * tile_width) + tile_x) * (scale * scale) + (local_y * scale) + local_x;

        for (int f = 0; f < frame_count; f++) {
            d_parsed[f * main_scale * main_scale + new_idx] =
                    static_cast<float>(d_buffer[info.bpp * (f * info.h * info.w + (y + info.y_off) * info.w+ (x + info.x_off))]);
        }
    }
}

///////////////////////////////////////////////////////
// More optimised GPU function to parse the input video to get ready for FFT,
// Only works if frame size is a power of 2 - we take shortcut to avoid modulo operation
///////////////////////////////////////////////////////
__global__ void parseBufferScalePow2(const unsigned char* __restrict__ d_buffer,
                                    float* __restrict__ d_parsed,
                                    const unsigned int channel_pp,
                                    const unsigned int channel_idx,
                                    const unsigned int img_width,
                                    const unsigned int img_height,
                                    const unsigned int x_offset,
                                    const unsigned int y_offset,
                                    const unsigned int scale,
                                    const unsigned int main_scale,
                                    const unsigned int frame_count) {

    const unsigned int x = blockIdx.x * BLOCKSIZE_X + threadIdx.x;
    const unsigned int y = blockIdx.y * BLOCKSIZE_Y + threadIdx.y;

    if (x < main_scale && y < main_scale) {
        const unsigned int local_x = x & (scale - 1); // x % scale - bit twiddling hack
        const unsigned int local_y = y & (scale - 1); // y % scale

        const unsigned int tile_x = x / scale;
        const unsigned int tile_y = y / scale;

        const unsigned int tile_width = main_scale / scale;

        const unsigned int new_idx = ((tile_y * tile_width) + tile_x) * (scale * scale) + (local_y * scale) + local_x;

        for (unsigned int f = 0; f < frame_count; f++) {

//            d_parsed[f * main_scale * main_scale + new_idx] = dk_uchar_float_lookup
//            		[d_buffer[channel_pp * (f * img_width * img_height + (y + y_offset) * img_width + (x + x_offset)) + channel_idx]];

            d_parsed[f * main_scale * main_scale + new_idx] = static_cast<float>(
            		d_buffer[channel_pp * (f * img_width * img_height + (y + y_offset) * img_width + (x + x_offset)) + channel_idx]);
        }
    }
}


///////////////////////////////////////////////////////
// GPU function to add the absolute magnitude of FFT to accumulator array
///////////////////////////////////////////////////////
__global__ void processFFT(const cufftComplex* __restrict__ d_dataA,
                           const cufftComplex* __restrict__ d_dataB,
                           float * __restrict__ d_odata,
                           float fft_norm,
                           int frame_size) {

    unsigned int i = blockIdx.x * BLOCKSIZE + threadIdx.x;

    if (i < frame_size) {
        cufftComplex val;
        val.x = fft_norm * (d_dataA[i].x - d_dataB[i].x);
        val.y = fft_norm * (d_dataA[i].y - d_dataB[i].y);

        d_odata[i] += val.x * val.x + val.y * val.y;

    }
}

///////////////////////////////////////////////////////
// Simple GPU function to combine two accumulator arrays - for use if using CUDA streams
///////////////////////////////////////////////////////
__global__ void combineAccum(float* __restrict__ d_dataA,
                             const float* __restrict__ d_dataB,
                             int tau_count,
                             unsigned int frame_size) {

    unsigned int i = blockIdx.x * BLOCKSIZE + threadIdx.x;

    if (i < frame_size) {
        for (int t = 0; t < tau_count; t++) {
            d_dataA[frame_size * t + i] += d_dataB[frame_size * t + i];
        }
    }
}


#endif