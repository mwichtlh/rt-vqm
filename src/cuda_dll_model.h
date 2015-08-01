/**
 * @file cuda_dll_model.h
 * @author Gregor Wicklein
 * @date  Jul 31, 2014
 * @brief Performs filtering and feature extraction on a single slice of the video
 */


#ifndef CUDA_DLL_MODEL_H_
#define CUDA_DLL_MODEL_H_

#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <stdio.h>
#include <unistd.h>

#include "cuda_filter.h"
#include "cuda_block_static.h"
#include "cuda_filter_ati.h"
#include "cuda_dll_model.h"
#include "cuda_format.h"
#include "cuda_read_video.h"



#include "cuda_format.h"


/**
 * Initializes CUDA related resources for feature extraction (e.g. device memory)
 * @param video current video
 * @param sobel_kernel the edge enhancement filter kernel
 * @param OneKernel the smooth filter kernel
 * @param rmin angle paramteter for HV and HVB calculation
 * @param ratio_threshold threshold vor HV and HVB calculation
 * @return
 */
cuda_state initialize(VQMVideo video, FLOAT* sobel_kernel, FLOAT* OneKernel,double rmin,
		double ratio_threshold);

/**
 * Extract features from a single video slice
 * @param video the current video
 * @param state the current CUDA state
 * @param t_slice index of the video slice to process
 */
void extract_features(VQMVideo video , cuda_state *state, int t_slice);


#endif /* CUDA_DLL_MODEL_H_ */
