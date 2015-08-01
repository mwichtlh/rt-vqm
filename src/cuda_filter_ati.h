/**
 * @file cuda_filter_ati.h
 * @author Gregor Wicklein
 * @date  Jul 21, 2014
 * @brief Contains functions for the temporal filtering, necessary of the ati feature-set.
 */

#ifndef CUDA_FILTER_ATI_H
#define CUDA_FILTER_ATI_H

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <math.h>

#include "cuda_format.h"

//CUDA block-dimensions for the temporal filter
#define DIM_ATI_X 16
#define DIM_ATI_Y 16

/**
 * Performs temporal filtering by calculating the difference between each two consecutive frames.
 * The first frame in a video slice is compared to the last frame from the prior frame.
 * @param video the video
 * @param state the CUDA state withe the current video slice loaded in
 */
void temporal_filter(VQMVideo video, cuda_state *state);


#ifdef MATLAB
void cuda_filter_ati(FLOAT *result, FLOAT *data, FLOAT *priorFrame, int dataW, int dataH, int dataT);
#endif

#endif /* CUDA_FILTER_ATI_H */
