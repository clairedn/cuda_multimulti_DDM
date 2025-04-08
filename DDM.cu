#include <stdio.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <nvToolsExt.h>
#include <stdbool.h>
#include <unistd.h>
#include <stdlib.h>

#include <opencv2/opencv.hpp>
#include <string>
#include <chrono>
#include <algorithm>
#include <iostream>

#include "azimuthal_average.cuh"
#include "debug.hpp"
#include "constants.hpp"
#include "video_reader.hpp"

#include "DDM_kernel.cuh"


// Function to swap two pointers
template <class T> inline void swap(T*& A, T*& B) {
    T* tmp = A;
    A = B;
    B = tmp;
}


// Function to swap three pointers, positional order important
template <class T> inline void rotateThreePtr(T*& A, T*& B, T*& C) {
    T* tmp = A;
    A = B;
    B = C;
    C = tmp;
}


///////////////////////////////////////////////////////
// If we choose to dual-stream the code then we must
// combine the FFT intensity accumulator associated with
// each stream.
///////////////////////////////////////////////////////
inline void combineAccumulators(float **d_accum_list_A,
                                float **d_accum_list_B,
                                int *scale_arr,
								int scale_count,
                                int tau_count) {

    dim3 blockDim(BLOCKSIZE);
    int main_scale = scale_arr[0];

    for (int s = 0; s < scale_count; s++) {
        int scale = scale_arr[s];
        int tile_count = (main_scale / scale) * (main_scale / scale);
        int frame_size = (scale / 2 + 1) * scale * tile_count;

        int gridDim = ceil(frame_size / static_cast<float>(BLOCKSIZE));

        combineAccum<<<gridDim, blockDim>>>(d_accum_list_A[s], d_accum_list_B[s], tau_count, frame_size);
    }
}


///////////////////////////////////////////////////////
//	This function handles analysis of the I(q, tau)
//  accumulator. Given the inputed values of q it
//  handles calculation of the azimuthal averages.
///////////////////////////////////////////////////////
void analyse_accums(int *scale_arr,	int scale_count,
					float *lambda_arr,	int lambda_count,
					int *tau_arr,	int tau_count,
					int frames_analysed,
					float mask_tolerance,
		            std::string file_out,
		            float **accum_list,
		            int framerate,
		            int window_size,
		            int window_index,
		            bool enable_angle_analysis,
		            int angle_count) {

	int main_scale = scale_arr[0]; // the largest length-scale

	bool *d_masks; // device pointer to reference the boolean azimuthal masks

	int element_count = (main_scale / 2 + 1) * main_scale;
	int total_masks = lambda_count;

    // Calculate total number of masks needed: 
    // When angle analysis is enabled: one mask per (q-value × angle segment) combination; Otherwise: one mask per q-value only
	total_masks = enable_angle_analysis ? lambda_count * angle_count : lambda_count;

	// Allocate GPU memory for all azimuthal masks
	// element_count: number of elements in FFT result ((main_scale/2 + 1) * main_scale)
	// total_masks: total number of masks (lambda_count * angle_count in angle analysis mode or just lambda_count in regular mode)
	// This large allocation creates all q-value and angle segment masks at once for GPU computation
	gpuErrorCheck(cudaMalloc((void** ) &d_masks, sizeof(bool) * element_count * total_masks));

	int *h_pixel_counts = new int[total_masks * scale_count](); // host array to hold the number of pixels in each mask

	float normalisation = 1.0 / static_cast<float>(frames_analysed);

	float *q_pixel_radius = new float[lambda_count]; // host array to hold temporary q values for each length-scale

	for (int s = 0; s < scale_count; s++) {
    int scale = scale_arr[s];
    int tile_count = (main_scale / scale) * (main_scale / scale);
    int tile_size = (scale / 2 + 1) * scale;

    for (int i = 0; i < lambda_count; i++) {
        q_pixel_radius[i] = static_cast<float>(scale) / (lambda_arr[i]);
    }
        
        buildAzimuthMask(d_masks, h_pixel_counts, q_pixel_radius, lambda_count, mask_tolerance, scale, scale, enable_angle_analysis, angle_count);

        for (int tile_idx = 0; tile_idx < tile_count; tile_idx++) { 
           
            std::string tmp_filename = file_out + "episode" + std::to_string(window_size) + "-" + std::to_string(window_index) + "_scale" + std::to_string(scale) + "-" + std::to_string(tile_idx);

            float *d_accum_tmp = accum_list[s] + tile_size * tile_idx;

            float *ISF = analyseFFTDevice(d_accum_tmp, d_masks, h_pixel_counts, normalisation, tau_count, lambda_count, tile_count, scale, scale, enable_angle_analysis, angle_count);
            
            writeIqtToFile(tmp_filename, ISF, lambda_arr, lambda_count, tau_arr, tau_count, framerate, enable_angle_analysis, angle_count);
            
            delete[] ISF;
        }
    }
    gpuErrorCheck(cudaFree(d_masks));
    delete[] h_pixel_counts;
    delete[] q_pixel_radius;

    verbose("\n[Results for analysis window size = %d frames]\n", frames_analysed);
}


////////////////////////////////////////////////////////////////////////////////
//  This function handles the parsing of on-device raw (uchar) data into a float
//  array, and the multi-scale FFT of this data to a list of cufftComplex arrays.
////////////////////////////////////////////////////////////////////////////////
void parseChunk(unsigned char *d_raw_in,
                cufftComplex **d_fft_list_out,
                float *d_workspace,
                int *scale_arr,
				int scale_count,
                int frame_count,
                video_info_struct info,
                cufftHandle *fft_plan_list,
                cudaStream_t stream) {

    int main_scale = scale_arr[0];

    int x_dim = static_cast<int>(ceil(main_scale / static_cast<float>(BLOCKSIZE_X)));
    int y_dim = static_cast<int>(ceil(main_scale / static_cast<float>(BLOCKSIZE_Y)));

    dim3 gridDim(x_dim, y_dim);
    dim3 blockDim(BLOCKSIZE_X, BLOCKSIZE_Y);

    for (int s = 0; s < scale_count; s++) {
        int scale = scale_arr[s];
        
        parseBufferScalePow2<<<gridDim, blockDim, 0, stream>>>(d_raw_in, d_workspace, info.bpp, 0, info.w, info.h, info.x_off, info.y_off, scale, main_scale, frame_count);
        cufftSetStream(fft_plan_list[s], stream);

        int exe_code = cufftExecR2C(fft_plan_list[s], d_workspace, d_fft_list_out[s]);
        conditionAssert(exe_code == CUFFT_SUCCESS, "cuFFT execution failed", true);
    }
}


////////////////////////////////////////////////////////////////////////////////
//  This function handles the analysis of the FFT, i.e. handles the calculation
//  of the difference functions. Makes use of 3-section circular buffer 
////////////////////////////////////////////////////////////////////////////////
void analyseChunk(cufftComplex **d_fft_buffer1,
                  cufftComplex **d_fft_buffer2,
                  float **d_fft_accum_list,
                  int scale_count,
                  int *scale_vector,
                  int frame_count,
                  int chunk_frame_count,
                  int frame_offset,
                  int tau_count,
                  int *tau_vector,
                  cudaStream_t stream) {

    dim3 blockDim(BLOCKSIZE);

    int main_scale = scale_vector[0];

    for (int s = 0; s < scale_count; s++) {
        int scale = scale_vector[s];
        int tile_count = (main_scale / scale) * (main_scale / scale);
        int frame_size = (scale / 2 + 1) * scale * tile_count;

        int px_count = scale * scale;
        float fft_norm = 1.0f / px_count; // factor to normalise the FFT

        dim3 gridDim(static_cast<int>(ceil(frame_size / static_cast<float>(BLOCKSIZE))));

        int frames_left = chunk_frame_count - frame_offset;

        cufftComplex *tmp; // index pointer
        for (int t = 0; t < tau_count; t++) {
            if (tau_vector[t] < frames_left) { // check to see if second index is in next chunk
                tmp = d_fft_buffer1[s] + (frame_offset + tau_vector[t]) * frame_size;
            } else {
                tmp = d_fft_buffer2[s] + (tau_vector[t] - frames_left) * frame_size;
            }

            float *accum_out = d_fft_accum_list[s] + frame_size * t; // tmp pointer for position in accumulator array

            processFFT<<<gridDim, blockDim, 0, stream>>>(d_fft_buffer1[s] + frame_size * frame_offset, tmp, accum_out, fft_norm, frame_size);
        }
    }
}


////////////////////////////////////////////////////////////////////////////////
//  Main multi-DDM function
////////////////////////////////////////////////////////////////////////////////
void runDDM(std::string file_in,
            std::string file_out,
            int *tau_vector,
			int tau_count,
            float *lambda_arr,
			int lambda_count,
            int *scale_vector,
			int scale_count,
            int x_offset,
			int y_offset,
            int *episode_vector, 
            int episode_count,   
            int total_frames,
			int frame_offset,
            int chunk_frame_count,
            bool multistream,
            bool use_webcam,
            int webcam_idx,
            float mask_tolerance,
			bool use_moviefile,
			bool use_index_fps,
			bool use_explicit_fps,
			float explicit_fps,
            int dump_accum_after,
			bool benchmark_mode,
            bool enable_angle_analysis,
            int angle_count) {

    auto start_time = std::chrono::high_resolution_clock::now();
    verbose("[multiDDM Begin]\n");

    //////////
    ///  CUDA Check
    //////////

    cudaDeviceProp deviceProp;
    deviceProp.major = 0;
    deviceProp.minor = 0;
    int dev = 0;

    // This will pick the best possible CUDA capable device
    gpuErrorCheck(cudaGetDeviceProperties(&deviceProp, dev));

    // Get information on CUDA device
    verbose("[Device Info] Device found, %d Multi-Processors, SM %d.%d compute capabilities.\n",
            deviceProp.multiProcessorCount, deviceProp.major, deviceProp.minor);

    //////////
    ///  Sort Parameter Arrays
    //////////

    // Can make assumptions later if tau / q / scale arrays are in known order

    std::sort(tau_vector, tau_vector + tau_count);
    std::sort(lambda_arr, lambda_arr + lambda_count);
    std::sort(scale_vector, scale_vector + scale_count, std::greater<int>());
    // Sort window sizes in ascending order for efficient processing (starting with smaller windows) but could also be descending
    std::sort(episode_vector, episode_vector + episode_count);

    //////////
    ///  Parameter Check
    //////////

    verbose("Scale list:\n");
    for (int s = 0; s < scale_count; s++) {
        unsigned int scale = scale_vector[s];
        //printf("\t%d\n", scale);
        conditionAssert(!(scale == 0) && !(scale & (scale - 1)), "scales must be powers of two (> 0)", true);

        if (s < scale_count - 1)
            conditionAssert((scale_vector[s] > scale_vector[s + 1]), "scales should be descending order", true);
    }

    verbose("Episode list (time window sizes):\n");
    for (int e = 0; e < episode_count; e++) {
        int window_size = episode_vector[e];
        //printf("\t%d\n", window_size);
        conditionAssert(window_size >= 0, "Window size should be non-negative", true);
        
        if (e < episode_count - 1) {
            conditionAssert(window_size < episode_vector[e + 1], "Window sizes should be in ascending order", true);
        }
        
        if (window_size == 0) {
            verbose("Warning: Window with size 0 at position %d will be skipped\n", e+1);
        } else if (window_size > total_frames) {
            verbose("Warning: Window size %d exceeds total frame count %d, will analyze all available frames\n", 
                    window_size, total_frames);
        }
    }

    verbose("%d time window sizes.\n", episode_count);

    for (int t = 0; t < tau_count - 1; t++) {
        conditionAssert((tau_vector[t] >= 0), "Tau values should be positive", true);
        conditionAssert((tau_vector[t] < tau_vector[t + 1]), "Tau vector should be ascending order", true);
    }

    for (int q = 0; q < lambda_count - 1; q++) {
        conditionAssert((lambda_arr[q] >= 0), "q-vector values should be positive", true);
        conditionAssert((lambda_arr[q] < lambda_arr[q + 1]), "q-vector vector should be ascending order", true);
    }

    conditionAssert(mask_tolerance < 10 && mask_tolerance > 1.0,
            "mask_tolerance is likely undesired value, refer to README for more information");

    conditionAssert(lambda_arr[lambda_count-1] <= scale_vector[scale_count-1],
            "The largest q-vector should be smaller than the smallest scale.", true);

    verbose("%d tau-values.\n", tau_count);
    verbose("%d q-vector values.\n", lambda_count);

    verbose("Parameter Check Done.\n");
    //////////
    ///  Video Setup
    //////////

    // Web-cam alignment
    if (use_webcam) {
        cv::VideoCapture tmp_cap(webcam_idx);

        int main_scale = scale_vector[0];

        while(1) {
            cv::Mat tmp_frame;
            tmp_cap >> tmp_frame;

            for (int s = 0; s < scale_count; s++) {
                int scale = scale_vector[s];
                int tiles_per_side = (main_scale / scale);
                int tiles_per_frame = tiles_per_side * tiles_per_side;

                for (int t = 0; t < tiles_per_frame; t++) {
                    cv::Rect rect(x_offset + (t / tiles_per_side) * scale + s, y_offset + (t % tiles_per_side) * scale + s, scale, scale); // add s to each to help with readability
                    cv::rectangle(tmp_frame, rect, cv::Scalar(0, 255, 0));
                }
            }
            cv::imshow( "Web-cam", tmp_frame );

            char c=(char) cv::waitKey(25);
            if(c==27)
                break;
        }
        tmp_cap.release();
        cv::destroyAllWindows();
    }

    video_info_struct info;
    FILE *moviefile;
    cv::VideoCapture cap;

    if (benchmark_mode) {
    	info.w = scale_vector[0];
    	info.h = scale_vector[0];
    	info.bpp = 1;
    	info.fps = 1.0;
    } else if (use_moviefile) { // if we have a movie-file we use custom handler
        moviefile = fopen(file_in.c_str(), "rb");
        conditionAssert(moviefile != NULL, "couldn't open .movie file", true);
        info = initFile(moviefile, frame_offset);
    } else { // for other file types handle with OpenCV
        if (use_webcam) {
            cap = cv::VideoCapture(webcam_idx);
        } else {
            cap = cv::VideoCapture(file_in);
        }

        conditionAssert(cap.isOpened(), "error opening video file with openCV", true);

        info.w = cap.get(cv::CAP_PROP_FRAME_WIDTH);
        info.h = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
        info.fps = static_cast<float>(cap.get(cv::CAP_PROP_FPS)); // cast from double to float

        cv::Mat test_img;
        cap >> test_img;

        // Due to the difficulty in dealing with many image types, we only
        // deal with multi-channel data if image is CV_8U (i.e. uchar)
        int type = test_img.type();
        info.bpp = (type % 8) ? 1 : test_img.channels();

        if (!use_webcam)
            cap = cv::VideoCapture(file_in); // re-open so can view first frame again

        // Offset the video by frame_offset frames
        for (int i = 0; i < frame_offset; i++) {
        	cap >> test_img;
        }
    }


    if (use_index_fps) { // if flag to use frame indices as frame-rate (same as setting FPS to 1)
        info.fps = 1.0;
    }
    if (use_explicit_fps) { // if flag to explicitly specify the video frame-rate
    	info.fps = explicit_fps;
    }

    info.x_off = x_offset;
    info.y_off = y_offset;

    verbose("Video Setup Done.\n");
    //////////
    ///  Parameter check
    //////////

    conditionAssert((scale_vector[0] + info.x_off <= info.w && scale_vector[0] + info.y_off <= info.h),
            "the specified out dimensions must be smaller than actual image size", true);

    conditionAssert((tau_vector[tau_count - 1] <= chunk_frame_count),
            "the largest tau value must be smaller than number frames in a chunk", true);

    //////////
    ///  Initialise variables
    //////////

    const int buffer_frame_count = chunk_frame_count * 3;
    const int main_scale = scale_vector[0];
    int chunks_already_parsed = 0;

    verbose("[Video info - (%d x %d), %d Frames (offset %d), %.4f FPS]\n", info.w, info.h, total_frames, frame_offset, info.fps);

    // streams

    cudaStream_t stream_1, stream_2;

    if (multistream) {
        cudaStreamCreate(&stream_1);
        cudaStreamCreate(&stream_2);
    } else {
        cudaStreamCreate(&stream_1);
    }


    verbose("Initialise Variables Done.\n");
    //////////
    ///  Memory Allocations
    //////////

    size_t total_host_memory   = 0;
    size_t total_device_memory = 0;

    // main device buffer
    size_t buffer_size  = sizeof(unsigned char) * buffer_frame_count * info.bpp * info.w * info.h;

    unsigned char *d_buffer;
    gpuErrorCheck(cudaMalloc((void** )&d_buffer, buffer_size));

    total_device_memory += buffer_size;

    // host buffer (multi-stream)
    size_t chunk_size  = sizeof(unsigned char) * chunk_frame_count * info.bpp * info.w * info.h;

    unsigned char *h_chunk_1;
    unsigned char *h_chunk_2;

    if (multistream) {
        gpuErrorCheck(cudaHostAlloc((void **) &h_chunk_1, chunk_size, cudaHostAllocDefault));
        gpuErrorCheck(cudaHostAlloc((void **) &h_chunk_2, chunk_size, cudaHostAllocDefault));
        total_host_memory += 2 * chunk_size;
    } else {
        gpuErrorCheck(cudaHostAlloc((void **) &h_chunk_1, chunk_size, cudaHostAllocDefault));
        h_chunk_2 = h_chunk_1;
        total_host_memory += 1 * chunk_size;
    }


    if (benchmark_mode) {
    	verbose("Benchmark mode - filling host buffer with random data.\n");
    	for (int i = 0; i < info.bpp * info.w * info.h; i++) {
    		h_chunk_1[i] = static_cast<unsigned char>(rand() % 255);
    		h_chunk_2[i] = static_cast<unsigned char>(rand() % 255);
    	}
    }
    // work space (multi-stream)
    size_t workspace_size = sizeof(float) * chunk_frame_count * main_scale * main_scale;

    float *d_workspace_1;
    float *d_workspace_2;

    if (multistream) {
        gpuErrorCheck(cudaMalloc((void** ) &d_workspace_1, workspace_size));
        gpuErrorCheck(cudaMalloc((void** ) &d_workspace_2, workspace_size));
        total_device_memory += 2 * workspace_size;
    } else {
        gpuErrorCheck(cudaMalloc((void** ) &d_workspace_1, workspace_size));
        d_workspace_2= d_workspace_1;
        total_device_memory += 1 * workspace_size;
    }

    // FFT buffer
    size_t fft_buffer_size = 0;
    for (int s = 0; s < scale_count; s++) {
        int scale = scale_vector[s];
        int tiles_per_frame = (main_scale / scale) * (main_scale / scale);
        int tile_size = (scale / 2 + 1) * scale;

        fft_buffer_size += sizeof(cufftComplex) * tile_size * tiles_per_frame * buffer_frame_count;
    }

    cufftComplex *d_fft_buffer;

    gpuErrorCheck(cudaMalloc((void** ) &d_fft_buffer, fft_buffer_size));

    total_device_memory += fft_buffer_size;

    // FFT intensity accumulator (multi-stream) - initial values set to zero
    size_t accum_size = 0;
    for (int s = 0; s < scale_count; s++) {
        int scale = scale_vector[s];
        int tiles_per_frame = (main_scale / scale) * (main_scale / scale);
        accum_size += sizeof(float) * (scale / 2 + 1) * scale * tiles_per_frame * tau_count;
    }

    float *d_accum_1;
    float *d_accum_2;

    if (multistream) {
        gpuErrorCheck(cudaMalloc((void** ) &d_accum_1, accum_size));
        gpuErrorCheck(cudaMalloc((void** ) &d_accum_2, accum_size));

        gpuErrorCheck(cudaMemset(d_accum_1, 0, accum_size));
        gpuErrorCheck(cudaMemset(d_accum_2, 0, accum_size));

        total_device_memory += 2 * accum_size; 
    } else {
        gpuErrorCheck(cudaMalloc((void** ) &d_accum_1, accum_size));

        gpuErrorCheck(cudaMemset(d_accum_1, 0, accum_size));
        d_accum_2 = d_accum_1;

        total_device_memory += 1 * accum_size;
    }

    size_t free_memory = 0;
    size_t total_memory = 0;
    gpuErrorCheck(cudaMemGetInfo(&free_memory, &total_memory));

    // tau-vector
    int *d_tau_vector;
    gpuErrorCheck(cudaMalloc((void** ) &d_tau_vector, tau_count * sizeof(int)));
    gpuErrorCheck(cudaMemcpy(d_tau_vector, tau_vector, tau_count * sizeof(int), cudaMemcpyHostToDevice));
    total_device_memory += sizeof(int) * tau_count;

    verbose("Memory Allocations Done.\n"
            "Total memory allocated\n"
            "Device:\n\tExplictly allocated:\t %f GB \n\tTotal allocated:\t %f GB\n\tFree memory remaining:\t %f GB\n"
            "Host:\t %f GB\n",
            total_device_memory          / (float) 1073741824,
            (total_memory - free_memory) / (float) 1073741824,
            free_memory                  / (float) 1073741824,
            total_host_memory            / (float) 1073741824);

    //////////
    ///  FFT Plan
    //////////

    cufftHandle *FFT_plan_list = new cufftHandle[scale_count];

    int rank = 2;
    int istride = 1;
    int ostride = 1;

    for (int s = 0; s < scale_count; s++) {
        int scale = scale_vector[s];

        int tiles_per_frame = (main_scale / scale) * (main_scale / scale);
        int batch_count = chunk_frame_count * tiles_per_frame;
        int n[2] = {scale, scale};

        int idist = scale * scale;
        int odist = scale * (scale/2+1);

        int inembed[] = {scale, scale};
        int onembed[] = {scale, scale/2+1};

        size_t mem_usage;

        verbose("FFT Plan Info:\n");
        verbose("\tn: (%d, %d), inembed: (%d, %d), onembed: (%d, %d), idist, odist: (%d, %d), batch: %d\n", n[0], n[1], inembed[0], inembed[1], onembed[0], onembed[1], idist, odist, batch_count);

        int plan_code = cufftPlanMany(&FFT_plan_list[s], rank, n, inembed, istride, idist, onembed, ostride, odist, CUFFT_R2C, batch_count);
        int esti_code = cufftEstimateMany(rank, n, inembed, istride, idist, onembed, ostride, odist, CUFFT_R2C, batch_count, &mem_usage);

        conditionAssert(plan_code == CUFFT_SUCCESS, "main cuFFT plan failure", true);
        conditionAssert(esti_code == CUFFT_SUCCESS, "error estimating cuFFT plan memory usage", true);
    }

    verbose("FFT Plan Done.\n");

    //////////
    ///  Pointer Allocation
    //////////

    // FFT buffer & FFT intensity accumulator are scale dependent so we define a array to hold values for each scale

    float **d_accum_list_1 = new float*[scale_count];
    float **d_accum_list_2 = new float*[scale_count];
    cufftComplex **d_fft_buffer_list    = new cufftComplex*[scale_count];

    d_accum_list_1[0] = d_accum_1;
    d_accum_list_2[0] = d_accum_2;
    d_fft_buffer_list[0] = d_fft_buffer;

    for (int s = 0; s < scale_count - 1; s++) {
        int scale = scale_vector[s];

        int tile_size = (scale/2 + 1) * scale;
        int tiles_per_frame = (main_scale / scale) * (main_scale / scale);

        d_accum_list_1[s+1] = d_accum_list_1[s] + tiles_per_frame * tile_size * tau_count;
        d_accum_list_2[s+1] = d_accum_list_2[s] + tiles_per_frame * tile_size * tau_count;

        d_fft_buffer_list[s+1] = d_fft_buffer_list[s] + tiles_per_frame * tile_size * buffer_frame_count;
    }

    cufftComplex **d_start_list = new cufftComplex*[scale_count];
    cufftComplex **d_end_list   = new cufftComplex*[scale_count];
    cufftComplex **d_junk_list  = new cufftComplex*[scale_count];

    for (int s = 0; s < scale_count; s++) {
        int tiles_per_frame = (main_scale / scale_vector[s]) * (main_scale / scale_vector[s]);
        int tile_size  = (scale_vector[s]/2 + 1) * scale_vector[s];

        d_start_list[s]  = d_fft_buffer_list[s];
        d_end_list[s]    = d_fft_buffer_list[s] + 1 * tiles_per_frame * tile_size * chunk_frame_count;
        d_junk_list[s]   = d_fft_buffer_list[s] + 2 * tiles_per_frame * tile_size * chunk_frame_count;
    }

    unsigned char *d_idle  = d_buffer;
    unsigned char *d_ready = d_buffer + 1 * chunk_frame_count * info.bpp * info.w * info.h;
    unsigned char *d_used  = d_buffer + 2 * chunk_frame_count * info.bpp * info.w * info.h;

    // pointers to shuffle with stream

    float *d_workspace_cur = d_workspace_1;
    float *d_workspace_nxt = d_workspace_2;

    float **d_accum_list_cur = d_accum_list_1;
    float **d_accum_list_nxt = d_accum_list_2;

    unsigned char *h_chunk_cur = h_chunk_1;
    unsigned char *h_chunk_nxt = h_chunk_2;

    cudaStream_t *stream_cur = &stream_1;
    cudaStream_t *stream_nxt = &stream_2;

    if (!multistream) {
        d_workspace_nxt = d_workspace_cur;
        d_accum_list_nxt = d_accum_list_cur;
        h_chunk_nxt = h_chunk_cur;
        stream_nxt = stream_cur;
    }

    verbose("Pointer Allocations Done\n");

    //////////
    ///  Main Loop 
    //////////
    
    verbose("Main loop start.\n");

    for (int e = 0; e < episode_count; e++) {
        int window_size = episode_vector[e];
        //printf("\t%d\n", window_size);
        conditionAssert(window_size >= 0, "Window size should be non-negative", true);
        
        if (e < episode_count - 1) {
            conditionAssert(window_size < episode_vector[e + 1], "Window sizes should be in ascending order", true);
        }
        
        if (window_size == 0) {
            verbose("Warning: Window with size 0 at position %d will be skipped\n", e+1);
        } else if (window_size > total_frames) {
            verbose("Warning: Window size %d exceeds total frame count %d, will analyze all available frames\n", 
                    window_size, total_frames);
        }
        
        verbose("\n[Processing analysis for time window size=%d frames (%d out of total %d)]\n", 
               window_size, e+1, episode_count);
        
        // Calculate how many windows we need to process for this episode
        int window_count = (total_frames + window_size - 1) / window_size;
        
        verbose("[Total %d windows to process]\n", window_count);
        
        for (int w = 0; w < window_count; w++) {
            // Calculate the starting frame of the current window
            int window_start = w * window_size;
            // Calculate actual frames in this window (handles edge case at the end of video)
            int frames_in_window = std::min(window_size, total_frames - w * window_size);
            
            verbose("\n[Processing window %d out of total %d: frame range %d-%d (total %d frames)]\n", 
                   w+1, window_count, window_start, window_start + frames_in_window - 1, frames_in_window);
            
            // Position video to the start of current window
            if (!benchmark_mode) {
                if (use_moviefile) {
                    fseek(moviefile, 0, SEEK_SET);
                    video_info_struct tmp_info = initFile(moviefile, window_start);
                    verbose("  Positioned movie file to frame %d\n", window_start);
                } else {
                    cap.set(cv::CAP_PROP_POS_FRAMES, window_start);
                    verbose("  Positioned video to frame %d\n", window_start);
                }
            }
            
            // Calculate number of chunks needed for this window
            int window_chunks = (frames_in_window + chunk_frame_count - 1) / chunk_frame_count;
            
            // === Initialize triple-buffer system for each window ===
            // Preload first two chunks of data to initialize the buffer system
            int first_chunk_frames = std::min(chunk_frame_count, frames_in_window);
            loadVideoToHost(use_moviefile, moviefile, cap, h_chunk_nxt, info, first_chunk_frames, benchmark_mode);
            loadVideoToHost(use_moviefile, moviefile, cap, h_chunk_cur, info, first_chunk_frames, benchmark_mode);
            
            // Pre-process the first buffer to initialize the start_list
            // This is crucial for the triple-buffer pattern to work correctly
            gpuErrorCheck(cudaMemcpyAsync(d_idle, h_chunk_nxt, chunk_size, cudaMemcpyHostToDevice, *stream_cur));
            parseChunk(d_idle, d_start_list, d_workspace_cur, scale_vector, scale_count, first_chunk_frames, 
                      info, FFT_plan_list, *stream_cur);
            gpuErrorCheck(cudaStreamSynchronize(*stream_cur));
            
            // Process all chunks in the current window
            for (int chunk_index = 0; chunk_index < window_chunks; chunk_index++) {
                // Calculate actual frames in this chunk (handles edge case at the end of window)
                int frames_in_chunk = std::min(chunk_frame_count, frames_in_window - chunk_index * chunk_frame_count);
                
                verbose("  [Processing chunk %d out of total %d (window frames %d-%d)]\n", 
                       chunk_index + 1, window_chunks, 
                       chunk_index * chunk_frame_count,
                       chunk_index * chunk_frame_count + frames_in_chunk - 1);
                
                // Process current chunk data - copy to device and perform FFT
                gpuErrorCheck(cudaMemcpyAsync(d_ready, h_chunk_cur, chunk_size, cudaMemcpyHostToDevice, *stream_cur));
                parseChunk(d_ready, d_end_list, d_workspace_cur, scale_vector, scale_count, frames_in_chunk, 
                          info, FFT_plan_list, *stream_cur);

                // Analyze each frame in the chunk - compare with previous frames to calculate ISFs
                for (int frame_offset = 0; frame_offset < frames_in_chunk; frame_offset++) {
                    analyseChunk(d_start_list, d_end_list, d_accum_list_cur, scale_count, scale_vector,
                                frames_in_chunk, chunk_frame_count, frame_offset, 
                                tau_count, tau_vector, *stream_cur);
                }

                // Ensure next stream operations don't start until current operations complete
                // This prevents data races in the triple-buffer system
                gpuErrorCheck(cudaStreamSynchronize(*stream_nxt));

                // Preload next chunk if needed
                if (window_chunks - chunk_index > 2) {
                    // Normal case - load next full chunk
                    int next_frames = std::min(chunk_frame_count, frames_in_window - (chunk_index + 2) * chunk_frame_count);
                    loadVideoToHost(use_moviefile, moviefile, cap, h_chunk_nxt, info, next_frames, benchmark_mode);
                } else if (chunk_index + 2 == window_chunks && frames_in_window % chunk_frame_count != 0) {
                    // Edge case - load remaining frames for the last partial chunk
                    int remaining_frames = frames_in_window % chunk_frame_count;
                    loadVideoToHost(use_moviefile, moviefile, cap, h_chunk_nxt, info, remaining_frames, benchmark_mode);
                }

                // Rotate pointers for triple-buffer pattern
                // This swaps current and next pointers for host and device buffers
                swap<unsigned char>(h_chunk_cur, h_chunk_nxt);
                swap<float>(d_workspace_cur, d_workspace_nxt);
                swap<float*>(d_accum_list_cur, d_accum_list_nxt);
                swap<cudaStream_t>(stream_cur, stream_nxt);

                // Rotate the three-pointer circular buffers for FFT data and raw frame data
                rotateThreePtr<cufftComplex*>(d_junk_list, d_start_list, d_end_list);
                rotateThreePtr<unsigned char>(d_used, d_ready, d_idle);

                verbose("  [Chunk %d out of total %d completed]\n", chunk_index + 1, window_chunks);

                // Periodic accumulator processing if enabled
                if (dump_accum_after != 0 && chunk_index != 0 && chunk_index % dump_accum_after == 0) {
                    verbose("[Parsing Accumulator]\n");
                    cudaDeviceSynchronize();
                    if (multistream) {
                        combineAccumulators(d_accum_list_cur, d_accum_list_nxt, scale_vector, scale_count, tau_count);
                    }

                    int tmp_frame_count = chunk_frame_count * dump_accum_after;
                    std::string tmp_name = file_out + "_t" + std::to_string(chunks_already_parsed / dump_accum_after) + "_";

                    analyse_accums(scale_vector, scale_count, lambda_arr, lambda_count, 
                                 tau_vector, tau_count, tmp_frame_count, mask_tolerance, 
                                 tmp_name, d_accum_list_cur, info.fps, window_size, w,
                                 enable_angle_analysis, angle_count);

                    verbose("  [Clearing accumulators for next batch processing]\n");
                    if (multistream) {
                        gpuErrorCheck(cudaMemset(d_accum_1, 0, accum_size));
                        gpuErrorCheck(cudaMemset(d_accum_2, 0, accum_size));
                    } else {
                        gpuErrorCheck(cudaMemset(d_accum_1, 0, accum_size));
                    }

                    chunks_already_parsed += dump_accum_after;
                }
            }
            
            // Process the accumulated data for this window
            analyse_accums(scale_vector, scale_count, lambda_arr, lambda_count, 
                         tau_vector, tau_count, frames_in_window, mask_tolerance, 
                         file_out, d_accum_list_cur, info.fps, window_size, w,
                         enable_angle_analysis, angle_count);
            
            // Reset accumulators for the next window
            if (multistream) {
                gpuErrorCheck(cudaMemset(d_accum_1, 0, accum_size));
                gpuErrorCheck(cudaMemset(d_accum_2, 0, accum_size));
            } else {
                gpuErrorCheck(cudaMemset(d_accum_1, 0, accum_size));
            }

            verbose("[Window %d out of total %d processing completed]\n", w+1, window_count);
        }

        verbose("\n[Completed analysis for time window size=%d frames]\n\n", window_size);
    }

    
    cudaDeviceSynchronize();
    auto end_main = std::chrono::high_resolution_clock::now();

    for (int s = 0; s < scale_count; s++) {
        cufftDestroy(FFT_plan_list[s]);
    }

    // Free memory locations we no longer need

    cudaFree(h_chunk_1);
    cudaFree(h_chunk_2);
    cudaFree(d_buffer);
    cudaFree(d_fft_buffer);
    cudaFree(d_workspace_1);
    cudaFree(d_workspace_2);

    if (multistream) {
        combineAccumulators(d_accum_list_cur, d_accum_list_nxt, scale_vector, scale_count, tau_count);
        //cudaFree(d_accum_2);
    }

    //////////
    ///  Analysis
    //////////
    verbose("Analysis.\n");

    auto end_out = std::chrono::high_resolution_clock::now();

    auto duration1 = std::chrono::duration_cast < std::chrono::microseconds
            > (end_main - start_time).count();
    auto duration2 = std::chrono::duration_cast < std::chrono::microseconds
            > (end_out - end_main).count();

    printf("[Time elapsed] "
           "\n\tMain:\t\t%f s, "
           "\n\tRadial Average:\t%f s,"
           "\n\tTotal\t\t%f s,\t(%f frame / second)\n",
           (float) duration1 / 1e6,
           (float) duration2 / 1e6,
           ((float) duration1 + (float) duration2) / 1e6,
           (float) (total_frames * 1e6) / ((float) duration1 + (float) duration2));
}
