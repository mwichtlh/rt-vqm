/**
 * @file cuda_block_static.h
 * @author Gregor Wicklein
 * @date  Jul 16, 2014
 * @brief These file contains block_statistic functions. They are necessary to collapse ST-regions from the YUV video into feature values.
 */


#ifndef CUDA_BLOCK_STATIC_H_
#define CUDA_BLOCK_STATIC_H_
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <math.h>

#include "cuda_format.h"

/**
 * Calculate feature value by the mean of all pixel values in a ST-Region
 * @param result The resulting feature-set (transferred to host memory)
 * @param d_data The raw video material (in device memory)
 * @param sliceDim Dimension of the video slice to process
 * @param featureDim Dimension of the ST-Region
 * @param offset Offset on the border of the frames, which should be ignored
 * @param stream CUDA stream on which the calculations should be processed
 * @param state Current CUDA State
 */
void block_statistic_mean(FLOAT *result, FLOAT *d_data, VQMDim3 sliceDim,
		VQMDim2 featureDim, VQMDim2 offset, cudaStream_t stream, cuda_state *state);

/**
 * Calculate feature value by the standard deviation of all pixel values in a ST-Region.
 * @param result The resulting feature-set (transferred to host memory)
 * @param d_data The raw video material (in device memory)
 * @param sliceDim Dimension of the video slice to process
 * @param featureDim Dimension of the ST-Region
 * @param offset Offset on the border of the frames, which should be ignored
 * @param stream CUDA stream on which the calculations should be processed
 * @param state Current CUDA State
 */
void block_statistic_std(FLOAT *result, FLOAT *d_data, VQMDim3 sliceDim,
		VQMDim2 featureDim, VQMDim2 offset, cudaStream_t stream, cuda_state *state);

/**
 * Calculate feature value by the mean of all pixel values in a ST-Region. These ST-Region
 * in this function does span over only one frame and will extract features from all frames
 * separately
 * @param result The resulting feature-set (transferred to host memory)
 * @param d_data The raw video material (in device memory)
 * @param sliceDim Dimension of the video slice to process
 * @param featureDim Dimension of the ST-Region
 * @param offset Offset on the border of the frames, which should be ignored
 * @param stream CUDA stream on which the calculations should be processed
 * @param state Current CUDA State
 */
void block_statistic_mean_sliced(FLOAT *result, FLOAT *d_data, VQMDim3 sliceDim,
		VQMDim2 featureDim, VQMDim2 offset, cudaStream_t stream, cuda_state *state);


//void block_statistic_mean(FLOAT *result, FLOAT *data, int dataW, int dataH, int dataT, int vsize, int hsize, int mode, int vOffset, int hOffset);


#endif /* CUDA_BLOCK_STATIC_H_ */
