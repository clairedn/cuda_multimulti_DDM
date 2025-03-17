//////////////////////////////////////
//  Reduction code is based heavily on the reduction_example from Nvidia's CUDA SDK examples
//  See "Optimizing parallel reduction in CUDA" - M. Harris for more details
//  some tweaks in regard to adding Boolean mask made
//////////////////////////////////////

#include <string>
#include <iostream>
#include <fstream>
#include <algorithm> 
#include <vector>

#include "constants.hpp"
#include "debug.hpp"
#include "azimuthal_average_kernel.cuh"

inline unsigned int nextPow2(unsigned int x) {
  --x;
  x |= x >> 1;
  x |= x >> 2;
  x |= x >> 4;
  x |= x >> 8;
  x |= x >> 16;
  return ++x;
}

///////////////////////////////////////////////////////
//	Writes ISF(lambda, tau) to file. 
//  When angle analysis is enabled, it writes separate
//  ISF data for each angular segment, including angle
//  information in the output file.
///////////////////////////////////////////////////////
void writeIqtToFile(std::string filename,
                    float *ISF,
                    float *lambda_arr, int lambda_count,
                    int   *tau_arr,    int tau_count,
                    float fps,
                    bool enable_angle_analysis,
                    int angle_count) {

    std::ofstream out_file(filename); 

    if (out_file.is_open()) {
        // lambda - values
        for (int lidx = 0; lidx < lambda_count; lidx++) {
            out_file << lambda_arr[lidx] << " ";
        }
        out_file << "\n";

        // tau - values
        for (int ti = 0; ti < tau_count; ti++) {
            out_file << static_cast<float>(tau_arr[ti]) / fps << " ";
        }
        out_file << "\n";

        if (enable_angle_analysis) {
            int full_angle_count = 2 * angle_count;  
            for (int angle_idx = 0; angle_idx < full_angle_count; angle_idx++) {
                
                float angle_width = 180.0 / angle_count;
                out_file << "# Angle section " + std::to_string(angle_idx) + 
                    " (center angle " + std::to_string((angle_idx * angle_width - 90.0) + angle_width/2) + 
                    " degrees, range: " + std::to_string(angle_idx * angle_width - 90.0) + "-" + 
                    std::to_string((angle_idx + 1) * angle_width - 90.0) + " degrees)\n";

                // I(lambda, tau) - values
                for (int li = 0; li < lambda_count; li++) {
                    for (int ti = 0; ti < tau_count; ti++) {
                        int idx = (angle_idx < angle_count) 
                                  ? (li * angle_count + angle_idx) * tau_count + ti
                                  : (li * angle_count + (angle_idx + full_angle_count / 2) % full_angle_count) * tau_count + ti;
                        out_file << ISF[idx] << " ";
                    }
                    out_file << "\n";
                }
                out_file << "\n";
            }
        } else {
            for (int li = 0; li < lambda_count; li++) {
                for (int ti = 0; ti < tau_count; ti++) {
                    out_file << ISF[li * tau_count + ti] << " ";
                }
                out_file << "\n";
            }
        }

        out_file.close();
        verbose("I(lambda, tau) written to %s\n", filename.c_str());
    } else {
        fprintf(stderr, "[Out Error] Unable to open %s.\n", filename.c_str());
        exit(EXIT_FAILURE);
    }
}


// Device analysis

///////////////////////////////////////////////////////
//	This function builds azimuthal boolean pixel masks
//	based on given input parameters. When angle analysis
//  is enabled, it creates separate masks for different
//  angular segments at each q-value. Masks are built on
//	host and copied to device memory location.
///////////////////////////////////////////////////////
void buildAzimuthMask(bool *d_mask_out,
                      int *h_pixel_counts,
                      float *q_arr, int q_count,
                      float q_tolerance,
                      int w, int h,
                      bool enable_angle_analysis,
                      int angle_count) {
    
    int total_masks = q_count * (enable_angle_analysis ? angle_count : 1);
    
    float q2_arr[q_count]; // array containing squared q-values
    for (int i = 0; i < q_count; i++)
        q2_arr[i] = q_arr[i] * q_arr[i];
    
    int element_count = (w/2 + 1) * h; // number of elements in mask (right half of FFT)
    bool *h_mask = new bool[element_count * total_masks];
    memset(h_mask, 0, sizeof(bool) * element_count * total_masks);
    
    // pre-calc some values
    float tol2 = q_tolerance * q_tolerance;
    int half_w = w / 2;
    int half_h = h / 2;
    
    int x_shift, y_shift;
    float r2, r2q2_ratio;

    bool px;
    float angle_step = M_PI / angle_count;  // Size of each angular segment
    
    // Initialize pixel counts for all masks to zero
    for (int mask_idx = 0; mask_idx < total_masks; mask_idx++) {
        h_pixel_counts[mask_idx] = 0;
    }
    
    // Create masks for each q-value
    for (int q_idx = 0; q_idx < q_count; q_idx++) {
        // Iterate over each pixel in the right half of the image
        for (int x = 0; x < (w/2 + 1); x++) {
            for (int y = 0; y < h; y++) {
                // Calculate pixel offset from center (FFT shift)
                x_shift = (x + half_w) % w - half_w;
                y_shift = (y + half_h) % h - half_h;
                
                r2 = x_shift * x_shift + y_shift * y_shift;
                r2q2_ratio = r2 / q2_arr[q_idx];
                
                // Check if pixel is within the annular region for this q-value
                px = (1 <= r2q2_ratio) && (r2q2_ratio <= tol2);
                
                if (px) {  // If pixel is within the annular region
                    if (enable_angle_analysis) {
                        // Calculate pixel angle (-π to π range)
                        float angle = atan2(y_shift, x_shift);
                        
                        // Normalize angle to 0-1 range (mapping -π/2 to 0, π/2 to 1)
                        float normalized_angle = (angle + M_PI/2) / M_PI;
                        
                        // Determine which angular segment the pixel belongs to
                        int angle_idx = std::min((int)(normalized_angle * angle_count), angle_count - 1);
                        
                        // Calculate mask index for this q-value and angle segment
                        int mask_idx = q_idx * angle_count + angle_idx;
                        
                        // Update corresponding mask and count
                        h_mask[mask_idx * element_count + y * (w/2 + 1) + x] = true;
                        h_pixel_counts[mask_idx]++;
                    } else {
                        // No angle analysis - just count pixels for this q-value
                        h_pixel_counts[q_idx]++;
                        h_mask[q_idx * element_count + y * (w/2 + 1) + x] = px;
                    }
                } else if (!enable_angle_analysis) {
                    // When angle analysis is disabled and pixel is not in range, set mask to false
                    h_mask[q_idx * element_count + y * (w/2 + 1) + x] = false;
                }
            }
        }
    }

    // Check if each mask has pixels meeting the criteria
    if (enable_angle_analysis) {
        for (int q_idx = 0; q_idx < q_count; q_idx++) {
            for (int angle_idx = 0; angle_idx < angle_count; angle_idx++) {
                int mask_idx = q_idx * angle_count + angle_idx;
                if (h_pixel_counts[mask_idx] == 0) {
                    verbose("[Mask Generation] q: %f, (#q: %d, angle: %d) has zero mask pixels for scale %d x %d\n", 
                           q_arr[q_idx], q_idx, angle_idx, w, h);
                }
            }
        }
    } else {
        // No angle analysis - check masks for each q-value only
        for (int q_idx = 0; q_idx < q_count; q_idx++) {
            if (h_pixel_counts[q_idx] == 0) {
                verbose("[Mask Generation] q: %f, (#q: %d) has zero mask pixels for scale %d x %d\n", q_arr[q_idx], q_idx, w, h);
            }
        }
    }

    // Copy mask onto GPU
    gpuErrorCheck(cudaMemcpy(d_mask_out, h_mask, sizeof(bool) * element_count * total_masks, cudaMemcpyHostToDevice));

    delete[] h_mask;  
}

///////////////////////////////////////////////////////
// Code to perform masked (GPU) reduction of ISF
// Analyzes FFT data using boolean masks to compute the ISF
// When angle analysis is enabled, processes separate masks
// for different angular segments at each q-value
// For optimal performance with reduction operations,
// future work could consider performing two separate reductions
// on (w/2)*(h/2) blocks which are more likely to be powers of 2
///////////////////////////////////////////////////////
float * analyseFFTDevice(float *d_data_in,
                        bool *d_mask,
                        int *h_px_count,
                        float norm_factor,
                        int tau_count,
                        int q_count,
                        int tile_count,
                        int w, int h,
                        bool enable_angle_analysis,
                        int angle_count) {

    // Total elements in the right half of the FFT
    int n = (w / 2 + 1) * h;

    // Compute the number of threads and blocks for the reduction kernel
    // For small datasets, use power of 2 sized threadblocks scaled to data size
    // For larger datasets, use fixed BLOCKSIZE (defined in header)
    int threads = (n < BLOCKSIZE * 2) ? nextPow2((n + 1) / 2) : BLOCKSIZE;
    int blocks = (n + (threads * 2 - 1)) / (threads * 2);

    // Limit maximum blocks to 64 for optimal GPU scheduling
    // This is a performance tuning parameter that may vary by GPU architecture
    blocks = (64 < blocks) ? 64 : blocks;

    // Allocate device and host memory for intermediate reduction results
    float *d_intermediateSums;
    float *h_intermediateSums = new float[blocks];

    gpuErrorCheck(cudaMalloc((void **)&d_intermediateSums, sizeof(float) * blocks));

    // Calculate total number of q-values to process
    // For angle analysis, this is q_count * angle_count (covering half-circle)
    int total_q = q_count;
    if (enable_angle_analysis) {
        total_q = q_count * angle_count; // angle_count represents half-circle segments
    }
    
    // Allocate and initialize ISF result array (tau_count values for each q-angle combination)
    float * ISF = new float[tau_count * total_q]();
    
    // Process each tau value
    for (int tau_idx = 0; tau_idx < tau_count; tau_idx++) {
        if (enable_angle_analysis) {
            // With angle analysis: process each q-value and angle segment combination
            for (int q_idx = 0; q_idx < q_count; q_idx++) {
                for (int angle_idx = 0; angle_idx < angle_count; angle_idx++) {
                    int mask_idx = q_idx * angle_count + angle_idx;
                    float val = 0;
                    
                    // Only process if the mask contains pixels
                    if (h_px_count[mask_idx] != 0) {
                        // Execute the reduction kernel for this mask and tau value
                        maskReduce<float>(n, threads, blocks, d_data_in + n*tau_idx*tile_count, 
                                         d_mask + n*mask_idx, d_intermediateSums);
                        
                        // Copy partial results from device to host
                        gpuErrorCheck(cudaMemcpy(h_intermediateSums, d_intermediateSums, blocks * sizeof(float), cudaMemcpyDeviceToHost));

                        // Combine partial sums from each block on CPU
                        for (int i = 0; i < blocks; i++) {
                            val += h_intermediateSums[i];
                        }

                        // Normalize by pixel count and apply normalization factor
                        val /= static_cast<float>(h_px_count[mask_idx]);
                        val *= norm_factor;
                    }
                    
                    // Store result in ISF array
                    ISF[mask_idx * tau_count + tau_idx] = val;
                }
            }
        } else {
            // Without angle analysis: process each q-value only
            for (int q_idx = 0; q_idx < q_count; q_idx++) {
                float val = 0;
                
                // Only process if the mask contains pixels
                if (h_px_count[q_idx] != 0) {
                    // Execute the reduction kernel for this mask and tau value
                    maskReduce<float>(n, threads, blocks, d_data_in + n*tau_idx*tile_count, 
                                     d_mask + n*q_idx, d_intermediateSums);
                    
                    // Copy partial results from device to host
                    gpuErrorCheck(cudaMemcpy(h_intermediateSums, d_intermediateSums, blocks * sizeof(float), cudaMemcpyDeviceToHost));

                    // Combine partial sums from each block on CPU
                    for (int i = 0; i < blocks; i++) {
                        val += h_intermediateSums[i];
                    }
                    
                    // Multiply by 2 to account for FFT symmetry (only processing right half)
                    val *= 2;
                    // Normalize by pixel count and apply normalization factor
                    val /= static_cast<float>(h_px_count[q_idx]);
                    val *= norm_factor;
                }
                
                // Store result in ISF array
                ISF[q_idx * tau_count + tau_idx] = val;
            }
        }
    }

    // Free temporary device and host memory to prevent memory leaks
    cudaFree(d_intermediateSums);
    delete[] h_intermediateSums;

    // Ensure all GPU operations are complete before returning
    cudaDeviceSynchronize();

    // Return the computed ISF array (caller is responsible for freeing this memory)
    return ISF;
}
