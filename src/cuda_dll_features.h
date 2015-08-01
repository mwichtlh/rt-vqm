/**
 * @file cuda_dll_features.h
 * @author Gregor Wicklein
 * @date  Oct 11, 2014
 * @brief These functions produce the feature-sets from a video representation which are necessary to calculate the VQM general model.
 */

#ifndef CUDA_DLL_FEATURES_H_
#define CUDA_DLL_FEATURES_H_

#include <stdio.h>
#include <stdlib.h>


#include <cuda_runtime.h>
#include <cuda.h>
//#include <ctime>
#include <math.h>

#include "cuda_read_video.h"
#include "cuda_dll_model.h"

#include "cuda_format.h"
#include "compare_feature.h"
#include "calculate_roi.h"


/**
 * Initialize all parameters for feature calculation, e.g. offsets, roi and with of the video slices.
 * @param path path to the raw video (YUV)
 * @param dataW horizontal resolution of the video
 * @param dataH vertical resolution of the video
 * @param frameRate framerate of the video
 * @return
 */
VQMVideo initialize_video(char* path, int dataW, int dataH, int frameRate, int offset, int length, temporal_scaling scaling);

/**
 * Allocate memory for features to produce
 * @param video the current video
 * @return video the updated video
 */
VQMVideo allocate_memory(VQMVideo video);

/**
 * Calculate features (for the general model) for the given video
 * @param path path path to the raw video (YUV)
 * @param dataW dataW horizontal resolution of the video
 * @param dataH dataH vertical resolution of the video
 * @param frameRate framerate of the video
 * @param orig_dataW horizontal resolution which this video should be scaled up
 * @param orig_dataH vertical resolution which this video should be scaled up
 * @param orig_frameRate framerate which this video should be scaled up
 * @return
 */
VQMVideo run_feature_calculation(char* path, int dataW, int dataH, int frameRate, int orig_dataW, int orig_dataH, int orig_frameRate, int offset, int length, temporal_scaling scaling);

/**
 * Actual calculation of the features
 * @param video current video
 */
void calculate_feature(VQMVideo video);

/**
 * Free memory after feature calculation
 * @param video current video
 */
void free_memory(VQMVideo video);


VQMVideo run_feature_calculation(VQMVideo input);

#endif /* CUDA_DLL_FEATURES_H_ */
