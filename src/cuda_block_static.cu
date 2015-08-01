#include "cuda_block_static.h"

#ifdef MATLAB
#include "mex.h"
#endif



#define TILE_W  8
#define TILE_H  8
//#define TILE_W_SQRT 8
//#define TILE_H_SQRT 8
#define KERNEL_RADIUS 6
#define BLOCK_DIM_X 8
#define BLOCK_DIM_Y 8

__global__ void mean_cuda(FLOAT *d_Result, FLOAT *d_Data, VQMDim3 dataDim, VQMDim2 featureDim, VQMDim2 offset);
__global__ void std_cuda(FLOAT *d_Result, FLOAT *d_Data, VQMDim3 dataDim, VQMDim2 featureDim, VQMDim2 offset);
__global__ void mean_sliced_cuda(FLOAT *d_Result, FLOAT *d_Data, VQMDim3 dataDim, VQMDim2 featureDim, VQMDim2 offset);


void block_statistic_mean(FLOAT *result, FLOAT *d_data, VQMDim3 sliceDim,
		VQMDim2 featureDim, VQMDim2 offset, cudaStream_t stream, cuda_state *state){



	int dataW = sliceDim.width - 2 * offset.width;
	int dataH = sliceDim.height - 2 * offset.height;

	int gridDimX = (dataW + featureDim.width * BLOCK_DIM_X - 1) / (featureDim.width * BLOCK_DIM_X);
	int gridDimY = (dataH + BLOCK_DIM_Y - 1) / BLOCK_DIM_Y;



	dim3 block(BLOCK_DIM_X, BLOCK_DIM_Y, 1);
	dim3 grid(gridDimX, gridDimY, 1);

	//size has to be a power of 8
	int resultSize = ((dataW - dataW % 8) / featureDim.width)
			* ((dataH - dataH % 8) / featureDim.height);

	mean_cuda<<<grid, block,0,stream>>>(state->d_result, d_data, sliceDim ,featureDim, offset);
	cudaThreadSynchronize();
	cudaMemcpyAsync(result, state->d_result, sizeof(FLOAT) * resultSize,
			cudaMemcpyDeviceToHost,stream);

}

void block_statistic_std(FLOAT *result, FLOAT *d_data, VQMDim3 sliceDim,
		VQMDim2 featureDim, VQMDim2 offset, cudaStream_t stream, cuda_state *state){



	int dataW = sliceDim.width - 2 * offset.width;
	int dataH = sliceDim.height - 2 * offset.height;
	int gridDimX = (dataW + featureDim.width * BLOCK_DIM_X - 1) / (featureDim.width * BLOCK_DIM_X);
	int gridDimY = (dataH + BLOCK_DIM_Y - 1) / BLOCK_DIM_Y;

	dim3 block(BLOCK_DIM_X, BLOCK_DIM_Y, 1);
	dim3 grid(gridDimX, gridDimY, 1);

	//size has to be a power of 8
	int resultSize = ((dataW - dataW % 8) / featureDim.width)
			* ((dataH - dataH % 8) / featureDim.height);

	std_cuda<<<grid, block,0,stream>>>(state->d_result, d_data, sliceDim , featureDim, offset);
	cudaThreadSynchronize();
	cudaMemcpyAsync(result, state->d_result, sizeof(FLOAT) * resultSize,
			cudaMemcpyDeviceToHost,stream);
	//cudaThreadSynchronize();
}

void block_statistic_mean_sliced(FLOAT *result, FLOAT *d_data, VQMDim3 sliceDim,
		VQMDim2 featureDim, VQMDim2 offset, cudaStream_t stream, cuda_state *state){


	int dataW = sliceDim.width - 2 * offset.width;
	int dataH = sliceDim.height - 2 * offset.height;
	int dataT = sliceDim.depth;

	int gridDimX = (dataW + featureDim.width * 8 - 1) / (featureDim.width * 8);
	int gridDimY = (dataH + featureDim.height * 8 - 1) / (featureDim.height * 8);

	dim3 block(BLOCK_DIM_Y, BLOCK_DIM_Y, dataT);
	dim3 grid(gridDimX, gridDimY, 1);

	//size has to be a power of 8
	int resultSize = ((dataW - dataW % 8) / featureDim.width)
			* ((dataH - dataH % 8) / featureDim.height) * dataT;
	mean_sliced_cuda<<<grid, block,0,stream>>>(state->d_result, d_data, sliceDim ,featureDim, offset);
	cudaThreadSynchronize();
	cudaMemcpyAsync(result, state->d_result, sizeof(FLOAT) * resultSize,
			cudaMemcpyDeviceToHost,stream);
	cudaThreadSynchronize();
}


__global__ void mean_cuda(FLOAT *d_Result, FLOAT *d_Data, VQMDim3 dataDim , VQMDim2 featureDim, VQMDim2 offset) {

		int dataW = dataDim.width - 2 * offset.width;
		int dataH = dataDim.height- 2 * offset.height;
		int dataT = dataDim.depth;

	__shared__ FLOAT data[TILE_W][TILE_H];

	const int x = threadIdx.x + blockIdx.x * blockDim.x;
	const int y = threadIdx.y + blockIdx.y * blockDim.y;

	int size = dataDim.width * dataDim.height;
	int gLoc = (y+offset.height) *  dataDim.width + x * featureDim.width + offset.width   ;

	FLOAT sum = 0;

	//size has to be a power of 8
	int dataW_cut = (dataW-dataW%8)/featureDim.width;
	int dataH_cut = (dataH-dataH%8)/featureDim.height;

	//printf("x: %d, y:%d\n",dataW_cut,dataH_cut);

	int i;
	int j;
	int position = gLoc;
	if (x  < dataW_cut  && y < dataH_cut * featureDim.height) {
		for (j = 0; j < dataT; j++){
			for (i = 0; i < featureDim.width; i++)  {
				sum += d_Data[position + i];

			}
			position+=size;
		}
		data[threadIdx.x][threadIdx.y] = sum;

		__syncthreads();

		if (threadIdx.y % featureDim.height == 0) {
			sum = 0;
			for (i = 0; i < featureDim.height; i++)
				sum += data[threadIdx.x][threadIdx.y + i];
			sum = sum / (featureDim.height * featureDim.width * dataT);
			int rLoc = y / featureDim.height * dataW_cut + x;
			d_Result[rLoc] = sum;
		}
	}

}



__global__ void mean_sliced_cuda(FLOAT *d_Result, FLOAT *d_Data, VQMDim3 dataDim , VQMDim2 featureDim, VQMDim2 offset) {

	int dataW = dataDim.width - 2 * offset.width;
	int dataH = dataDim.height- 2 * offset.height;

	int dataW_cut = (dataW-dataW%featureDim.width)/featureDim.width;
	int dataH_cut = (dataH-dataH%featureDim.height)/featureDim.height;

	const int x = threadIdx.x + blockIdx.x * blockDim.x;
	const int y = threadIdx.y + blockIdx.y * blockDim.y;
	const int z = threadIdx.z;



	int size = dataDim.width*dataDim.height;
	int gLoc = (featureDim.height * y+offset.height) * dataDim.width + x * featureDim.width + offset.width + z * size  ;

	FLOAT sum = 0;


	int i;
	int j;
	int position = gLoc;
	if (x   < dataW_cut  && y  < dataH_cut ) {
		for (j = 0; j < featureDim.height; j++)
		{
			for (i = 0; i < featureDim.width; i++) {
				sum += d_Data[position + i];
		}
			position+=dataDim.width;
		}


		sum = sum / (featureDim.height * featureDim.width);
		gLoc = y / featureDim.height * dataW_cut + x + dataW_cut * dataH_cut * z;
		gLoc = y * dataW_cut + x + dataW_cut  * dataH_cut  * z;
		d_Result[gLoc] = sum;


	}

}

__global__ void std_cuda(FLOAT *d_Result, FLOAT *d_Data, VQMDim3 dataDim , VQMDim2 featureDim, VQMDim2 offset) {

	__shared__ FLOAT data[TILE_W][TILE_H];
	__shared__ FLOAT data_sqare[TILE_W][TILE_H];

	int dataW = dataDim.width - 2 * offset.width;
	int dataH = dataDim.height- 2 * offset.height;
	int dataT = dataDim.depth;

	const int x = threadIdx.x + blockIdx.x * blockDim.x;
	const int y = threadIdx.y + blockIdx.y * blockDim.y;
	int size = dataDim.width*dataDim.height;

	int gLoc = (y+offset.height) * dataDim.width + x * featureDim.width + offset.width   ;

	//size has to be a power of 8
	int dataW_cut = (dataW-dataW%8)/featureDim.width;
	int dataH_cut = (dataH-dataH%8)/featureDim.height;


	FLOAT sum = 0;
	FLOAT sum_squre = 0;
	FLOAT temp;

	int i;
	int j;
	int position = gLoc;
	if (x  < dataW_cut && y < dataH_cut * featureDim.height) {
		for (j = 0; j < dataT; j++){
			for  (i = 0; i < featureDim.width; i++) {
				temp = d_Data[position + i];
				sum += temp;
				sum_squre += temp * temp;
			}
			position+=size;
		}
		data[threadIdx.x][threadIdx.y] = sum;
		data_sqare[threadIdx.x][threadIdx.y] = sum_squre;

		__syncthreads();

		if (threadIdx.y % featureDim.height == 0) {
			sum = 0;
			sum_squre = 0;
			for (i = 0; i < featureDim.height; i++) {
				sum += data[threadIdx.x][threadIdx.y + i];
				sum_squre += data_sqare[threadIdx.x][threadIdx.y + i];
			}
			sum = sum / (featureDim.height * featureDim.width * dataT);
			sum_squre = sum_squre / (featureDim.height * featureDim.width * dataT);
			int rLoc = y / featureDim.height * dataW_cut  + x;
			d_Result[rLoc] = sqrt(fmax(sum_squre - sum * sum, (FLOAT) 0));

		}
	}

}



#ifdef MATLAB
void block_statistic_mean(FLOAT *result, FLOAT *data, int dataW, int dataH,
		int dataT, int vsize, int hsize, int mode, int vOffset, int hOffset ) {

	int blockDimX = 16;
	int blockDimY = 16;

	int size = dataW * dataH * dataT;

	dataW = dataW-2 * hOffset;
	dataH = dataH-2 * vOffset;


	int gridDimX = (dataW + hsize * blockDimX - 1) / (hsize * blockDimX);
	int gridDimY = (dataH + blockDimY - 1) / blockDimY;


	int resultSize = dataW / hsize * dataH / vsize;

	FLOAT *d_video = NULL;
	cudaMalloc((void**) &d_video, sizeof(FLOAT) * size);
	cudaMemcpy(d_video, data, sizeof(FLOAT) * size, cudaMemcpyHostToDevice);

	FLOAT *d_result = NULL;
	cudaMalloc((void**) &d_result, sizeof(FLOAT) * resultSize);

	dim3 block(blockDimX, blockDimY, 1);
	dim3 grid(gridDimX, gridDimY, 1);

	if (mode == MEAN)
		mean_cuda<<<grid, block>>>(d_result, d_video, dataW, dataH, dataT ,vsize, hsize, hOffset, vOffset);
		else
		std_cuda<<<grid, block>>>(d_result, d_video, dataW, dataH, dataT ,vsize, hsize, hOffset, vOffset);

	cudaThreadSynchronize();	// Wait for the GPU launched work to complete

	cudaMemcpy(result, d_result, sizeof(FLOAT) * resultSize,
			cudaMemcpyDeviceToHost);

	cudaFree((void*) d_result);
	cudaFree((void*) d_video);

	cudaError_t last_error = cudaGetLastError();
	if (last_error != cudaSuccess) {
		// print the CUDA error message and exit
		printf("CUDA error: %s\n", cudaGetErrorString(last_error));
		exit(-1);
	}

	cudaDeviceReset();
}




/* The gateway function */
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {

	FLOAT *data = mxGetPr(prhs[0]);
	FLOAT vsize = mxGetScalar(prhs[1]);
	FLOAT hsize = mxGetScalar(prhs[2]);
	FLOAT mode = mxGetScalar(prhs[3]);

	const mwSize *dataDim = mxGetDimensions(prhs[0]);

	const mwSize dataW = dataDim[0];
	const mwSize dataH = dataDim[1];
	const mwSize dataT = dataDim[2];

	int resultDataW = (dataW-12) / hsize;
	int resultDataH = (dataH-12) / vsize;

	/* create the output matrix */
	plhs[0] = mxCreateDoubleMatrix(resultDataW, resultDataH, mxREAL);

	FLOAT *result = mxGetPr(plhs[0]);

	/* call the computational routine */
	block_statistic_mean(result, data, dataW, dataH, dataT, (int) vsize,
			(int) hsize, (int)mode, 6,6);
}
#endif
