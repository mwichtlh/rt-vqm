#include "cuda_filter_ati.h"


__global__ void filter_ati(FLOAT *d_Result, FLOAT *d_Data, int dataW, int dataH, int dataT);

void temporal_filter(VQMVideo video, cuda_state *state){

	int dataW = video.dataDim.height;
	int dataH = video.dataDim.width;

	int ati_gridDimX = (dataW + DIM_ATI_X - 1) / DIM_ATI_X;
	int ati_gridDimY = (dataH + DIM_ATI_Y - 1) / DIM_ATI_Y;

	dim3 ati_block(DIM_ATI_X, DIM_ATI_Y, 1);
	dim3 ati_grid(ati_gridDimX, ati_gridDimY, 1);

	if (state->has_Prior_Frame)
	{
		state->dataT_ati = state->dataT;
		filter_ati<<<ati_grid, ati_block>>>(state->d_ati, state->d_ati_video, state->dataW, state->dataH, state->dataT+1);
	}
	else
	{
		state->dataT_ati = state->dataT - 1;
		filter_ati<<<ati_grid, ati_block>>>(state->d_ati, state->d_y, state->dataW, state->dataH, state->dataT);
	}
}




__global__ void filter_ati(FLOAT *d_Result, FLOAT *d_Data, int dataW, int dataH, int dataT)
		 {

	const int x = threadIdx.x + blockIdx.x * blockDim.x;
	const int y = threadIdx.y + blockIdx.y * blockDim.y;

	int position = y*dataW + x;
	int frameSize = dataW * dataH;

	FLOAT sum = d_Data[position];
	FLOAT temp;

	int i;
	if (x < dataW && y < dataH ) {
		for (i = 1; i < dataT; i++) {
				temp = d_Data[position + i * frameSize];
				d_Result[position + (i-1) * frameSize] = abs(sum-temp);
				sum = temp;
			}
		}
}





#ifdef MATLAB

void cuda_filter_ati(FLOAT *result, FLOAT *data, FLOAT *priorFrame, int dataW, int dataH,
		int dataT) {

	int blockDimX = 16;
	int blockDimY = 16;

	if(priorFrame != 0)
		dataT++;

	int size = dataW * dataH * dataT;
	int frameSize = dataW*dataH;

	int gridDimX = (dataW + blockDimX - 1) / blockDimX;
	int gridDimY = (dataH + blockDimY - 1) / blockDimY;

	int resultSize = dataW * dataH * (dataT -1);

	FLOAT *d_video = NULL;
	cudaMalloc((void**) &d_video, sizeof(FLOAT) * size);

	if(priorFrame != 0)
	{
		cudaMemcpy(d_video, priorFrame, sizeof(FLOAT) * frameSize, cudaMemcpyHostToDevice);
		cudaMemcpy(d_video+frameSize, data, sizeof(FLOAT) * (size-frameSize), cudaMemcpyHostToDevice);
	}
	else
		cudaMemcpy(d_video, data, sizeof(FLOAT) * size, cudaMemcpyHostToDevice);

	FLOAT *d_result = NULL;
	cudaMalloc((void**) &d_result, sizeof(FLOAT) * resultSize);

	dim3 block(blockDimX, blockDimY, 1);
	dim3 grid(gridDimX, gridDimY, 1);

	filter_ati<<<grid, block>>>(d_result, d_video, dataW,  dataH, dataT);

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


	const mwSize *dataDim = mxGetDimensions(prhs[0]);

	const mwSize dataW = dataDim[0];
	const mwSize dataH = dataDim[1];
	const mwSize dataT = dataDim[2];

	int resultDataW = dataW;
	int resultDataH = dataH;
	int resultDataT = dataT;

	/* create the output matrix */
	//plhs[0] = mxCreateDoubleMatrix(resultDataW, resultDataH, mxREAL);





	/* call the computational routine */
	if(nrhs == 1){
		mwSize dims[3] = {resultDataW,resultDataH,resultDataT-1};
		mwSize ndim = 3;
		plhs[0] =  mxCreateNumericArray(ndim, dims, mxDOUBLE_CLASS, mxREAL);
		FLOAT *result = mxGetPr(plhs[0]);
		cuda_filter_ati(result, data, 0,dataW, dataH, dataT);
	}
	else
	{
		mwSize dims[3] = {resultDataW,resultDataH,resultDataT};
		mwSize ndim = 3;
		plhs[0] =  mxCreateNumericArray(ndim, dims, mxDOUBLE_CLASS, mxREAL);
		FLOAT *result = mxGetPr(plhs[0]);
		FLOAT *priorFrame = mxGetPr(prhs[1]);
		cuda_filter_ati(result, data, priorFrame,dataW, dataH, dataT);
	}

}

#endif
