#include <string>

#ifndef _AZIMUTHAL_AVERAGE_
#define _AZIMUTHAL_AVERAGE_

void buildAzimuthMask(bool *d_mask_out,
					  int *h_pixel_counts,
					  float *q_arr, int q_count,
					  float q_tolerance,
					  int w, int h,
					  bool enable_angle_analysis,
					  int angle_count);

float * analyseFFTDevice(float *d_data_in,
                        bool *d_mask,
                        int *h_px_count,
                        float norm_factor,
                        int tau_count,
                        int q_count,
                        int tile_count,
                        int w, int h,
                        bool enable_angle_analysis,
                        int angle_count);

void writeIqtToFile(std::string filename,
					float *ISF,
					float *lambda_arr, int lambda_count,
					int   *tau_arr,	   int tau_count,
					float fps,
					bool enable_angle_analysis,
					int angle_count) ;

#endif