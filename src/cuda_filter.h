/**
 * @file cuda_filter.h
 * @author Gregor Wicklein
 * @date  Jul 9, 2014
 * @brief Functions to apply edge enhancement and smooth filtering, necessary for si, hv and hvb features
 */


#ifndef CUDA_FILTER_H_
#define CUDA_FILTER_H_

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <math.h>
#ifdef MATLAB
#include "mex.h"
#endif
#include "cuda_format.h"

//Radius of the filter kernel
#define KERNEL_RADIUS 6

//CUDA block-dimensions for the edge and smooth filter
#define DIM_FILTER_X 16
#define DIM_FILTER_Y 16

//CUDA block-dimensions for the SI and HV calculations
#define DIM_SQRT_X 8
#define DIM_SQRT_Y 8

/**
 * This function applies edge enhancing and smooth filtering on the Y plane and produces si, hv and hvb data
 * @param current video
 * @param CUDA state with current video slice loaded in
 */
void filter_video(VQMVideo video, cuda_state *state);

#ifdef MATLAB
void filter_si_hv(FLOAT *si, FLOAT *hv, FLOAT *hvb, FLOAT *video,
		int dataW, int dataH, int size, FLOAT *filter, FLOAT rmin, FLOAT ratio_threshold);
#endif



#endif /* CUDA_FILTER_H_ */
