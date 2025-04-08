#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>

#include <string>
#include <iostream>
#include <fstream>

#include "debug.hpp"

struct DDMparams {
	std::string     file_in;
	std::string     file_out;
	std::string     q_file_name;	// file-path for q-vector
	std::string     t_file_name;	// file-path for tau-vector
	std::string		s_file_name;	// file-path for scale-vector
	std::string     e_file_name;    // file-path for episode-vector

	int    frame_count;				// number of frames to analyse
	int    frame_offset     = 0;    // number of frames to skip at start
	int    x_off 			= 0;    // number of pixels to offset x=0 by in frame
	int    y_off 			= 0;	// number of pixels to offset y=0 by in frame
	int    chunk_length		= 30;   // number of frames in frame buffer, default is 30
	int    rolling_purge	= 0;    // purge and analyse accumulators after number of frames

	bool   use_webcam 		= false;
	int    webcam_idx 		= 0;
	bool   use_movie_file	= false;
	bool   use_index_fps 	= false; // if flag set to false, use frame-indcies not frame-rate
	bool   use_explicit_fps = false;
	float  explicit_fps 	= 1.0;
	bool   multi_stream 	= true;
	float  q_tolerance		= 1.2;  // tolerance factor for q-vector mask: values between q and q*tolerance are included in the mask
	bool   benchmark_mode 	= false;
    bool   use_episodes = false;            // Whether to use time windows from episode file
	bool enable_angle_analysis = false;   // Whether to enable angle sector analysis, disabled by default
	int angle_count = 8;                 // Number of angle sections, default is 8
} params;

// forward declare main DDM function
void runDDM(std::string file_in,
            std::string file_out,
            int *tau_arr,
            int tau_count,
            float *lambda_arr,
            int lambda_count,
            int *scale_arr,
            int scale_count,
            int x_off,
            int y_off,
            int *episode_vector,
            int episode_count,
            int frame_count,
            int frame_offset,
            int chunk_frame_count,
            bool multistream,
            bool use_webcam,
            int webcam_idx,
            float q_tolerance,
            bool use_movie_file,
            bool use_index_fps,
            bool use_explicit_fps,
            float explicit_fps,
            int dump_accum_after,
            bool benchmark_mode,
            bool enable_angle_analysis,
            int angle_count);

void printHelp() {
    fprintf(stderr,
    		"\n ~~ multiscale DDM - CUDA - HELP ~~ \n"
    		"\n"
			"  Usage ./ddm [OPTION]..\n"
			"  -h           Print out this help.\n"
			"   REQUIRED ARGS\n"
			"  -o PATH      Output file-path.\n"
			"  -N INT       Number of frames to analyse.\n"
			"  -Q PATH      Specify path to lambda-value file (line separated).\n"
			"  -T PATH 		Specify path to tau-value file (line separated). \n"
			"  -S PATH 		Specify path to scale-value file (line separated). \n"
			"  -E PATH      Specify path to episode-value file (line separated).\n"

    		"   INPUT ARGS\n"
			"  -f PATH      Specify path to input video (either -f or -W option must be given).\n"
			"  -W INT       Use web-camera as input video, (web-camera number can be supplied, defaults to first connected camera).\n"
    		"  -B 			Benchmark mode, will perform analysis on random data.\n"

			"   OPTIONAL ARGS\n"
    		"  -s OFFSET	Set first frame offset (default 0).\n"
			"  -x OFFSET    Set x-offset (default 0).\n"
			"  -y OFFSET    Set y-offset (default 0).\n"
			"  -I           Use frame indices for tau-labels not real time.\n"
			"  -v			Verbose mode on.\n"
            "  -Z           Turn off multi-steam (smaller memory footprint - slower execution time).\n"
			"  -t INT       Set the q-vector mask tolerance - percent (integer only) (default 20 i.e. radial mask (1 - 1.2) * q).\n"
			"  -C INT	    Set main chunk frame count, a buffer 3x chunk frame count will be allocated in memory (default 30 frames).\n"
			"  -G SIZE      Sub-divide analysis, buffer will be output and purged every SIZE chunks\n"
    		"  -M			Set if using movie-file format.\n"
    		"  -F FPS 		Force the analysis to assume a specific frame-rate, over-rides other options.\n"
            "  -A           Enable angle analysis\n"
            "  -n INT       Set angle count\n"
            );
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv) {

	printf("DDM Start\n");

    // Flags
    bool input_specified = false;

    for (;;) {
        switch (getopt(argc, argv, "ho:N:s:x:y:Q:T:S:E:If:W::vZt:C:MG:F:BAn:")) {
            case '?':
            case 'h':
                printHelp();
                return -1;

            case 'o':
                params.file_out = optarg;
                continue;

            case 'N':
                params.frame_count = atoi(optarg);
                continue;

            case 'x':
                 params.x_off = atoi(optarg);
                 continue;

             case 'y':
                 params.y_off = atoi(optarg);
                 continue;

             case 'Q':
                 params.q_file_name = optarg;
                 continue;

             case 'T':
                 params.t_file_name = optarg;
                 continue;

             case 'S':
                 params.s_file_name = optarg;
                 continue;

             case 'E':
                 params.e_file_name = optarg;
                 continue;

             case 'I':
                 params.use_index_fps = true;
                 continue;

             case 's':
            	 params.frame_offset = atoi(optarg);
            	 continue;

            case 'f':
            	{
				conditionAssert(!input_specified, "Cannot use both in-filepath and web-cam option at same time.", true);

				params.file_in = optarg;
				input_specified = true;
            	}
                continue;

            case 'W':
            	{
				conditionAssert(!input_specified, "Cannot use both in-filepath and web-cam option at same time.", true);

				params.use_webcam = true;
				if (optarg != NULL) {
					params.webcam_idx = atoi(optarg);
				}
				input_specified = true;
            	}
                continue;

            case 'B':
            	{
                params.benchmark_mode = true;
                input_specified = true;
                }
                continue;

            case 'v':
                setVerbose(true);
                continue;

             case 't':
                 params.q_tolerance = 1.0 + static_cast<float>(atoi(optarg)) / 100;
                 continue;

             case 'C':
                 params.chunk_length = atoi(optarg);
                 continue;

             case 'M':
                 params.use_movie_file = true;
                 continue;

             case 'G':
                 params.rolling_purge = atoi(optarg);
                 continue;

             case 'F':
                 params.use_explicit_fps = true;
                 params.explicit_fps = atof(optarg);
                 continue;
                 
             case 'A':
                 params.enable_angle_analysis = true;
                 continue;
                 
             case 'n':
                 params.angle_count = atoi(optarg);
                  continue;
        }
        break;
    }

    if (optind != argc) {
        printHelp();
        conditionAssert(false, "An unexpected option was found.", true);
    }

    conditionAssert(input_specified, "Must specify input.", true);

    // Angle parameter conversion (input full circle angle count, output half circle angle count for processing)
    // params.angle_count = params.enable_angle_analysis ? (params.angle_count + 1) / 2 : params.angle_count;
    
    /////////////////////////////////////////////////////////////////////////////

    std::ifstream q_file(params.q_file_name);
    std::ifstream t_file(params.t_file_name);
    std::ifstream s_file(params.s_file_name);
    std::ifstream e_file(params.e_file_name);

    conditionAssert(q_file.is_open(), "cannot open lambda-file.", true);
    conditionAssert(t_file.is_open(), "cannot open tau-file", true);
    conditionAssert(s_file.is_open(), "cannot open scales-file.", true);
    conditionAssert(e_file.is_open(), "cannot open episode-file.", true);
    
    /// Read scale, lambda, tau and episode values

    // First count number of elements in each file
    
    int tau_count = 0;
    int lambda_count = 0;
    int scale_count = 0;
    int episode_count = 0;

    float tmp_q;
    while (q_file >> tmp_q) {
        lambda_count++;
    }
    
    int tmp_tau;
    while (t_file >> tmp_tau) {
        tau_count++;
    }
    
    int tmp_scale;
    while (s_file >> tmp_scale) {
        scale_count++;
    }
    
    int tmp_episode;
    while (e_file >> tmp_episode) {
        episode_count++;
    }
    
    // Seek back to the beginning of the file and read the values
    
    q_file.clear();
    q_file.seekg(0, std::ios::beg);
    t_file.clear();
    t_file.seekg(0, std::ios::beg);
    s_file.clear();
    s_file.seekg(0, std::ios::beg);
    e_file.clear();
    e_file.seekg(0, std::ios::beg);
    
    float lambda_arr[lambda_count];
    int tau_arr[tau_count];
    int scale_arr[scale_count];
    int episode_arr[episode_count];

    int idx;

    idx = 0;
    while (q_file >> tmp_q) {
        lambda_arr[idx++] = tmp_q;
    }
    
    idx = 0;
    while (t_file >> tmp_tau) {
        tau_arr[idx++] = tmp_tau;
    }
    
    idx = 0;
    while (s_file >> tmp_scale) {
        scale_arr[idx++] = tmp_scale;
    }
    
    idx = 0;
    while (e_file >> tmp_episode) {
        episode_arr[idx++] = tmp_episode;
    }
    
    runDDM(params.file_in,
           params.file_out,
           tau_arr,
           tau_count,
           lambda_arr,
           lambda_count,
           scale_arr,
           scale_count,
           params.x_off,
           params.y_off,
           episode_arr,
           episode_count,
           params.frame_count,
           params.frame_offset,
           params.chunk_length,
           params.multi_stream,
           params.use_webcam,
           params.webcam_idx,
           params.q_tolerance,
           params.use_movie_file,
           params.use_index_fps,
           params.use_explicit_fps,
           params.explicit_fps,
           params.rolling_purge,
           params.benchmark_mode,
           params.enable_angle_analysis,
           params.angle_count);
    

    printf("DDM End\n");
    
    q_file.close();
    t_file.close();
    s_file.close();
    e_file.close();

    return 0;
}


