#include "cuda_filter.h"

#define max2(a,b) (((a)>(b))?(a):(b))
#define min2(a,b) (((a)<(b))?(a):(b))

// parts of this code have been adapted from http://www.evl.uic.edu/sjames/cs525/final.html

__global__ void convolutionRowGPU(FLOAT *d_Result, FLOAT *d_Data, int dataW, int dataH);

__global__ void filterRowGPU(FLOAT *d_Result, FLOAT *d_kernel,
		FLOAT *d_Data, int dataW, int dataH);

__global__ void convolutionColGPU(FLOAT *d_Result, FLOAT *d_Data, int dataW, int dataH);

__global__ void filterColGPU(FLOAT *d_Result, FLOAT *d_kernel,
		FLOAT *d_Data, int dataW, int dataH);

__global__ void convolutionSqrtGPU(FLOAT *d_result, FLOAT *d_col, FLOAT *d_row,
		int dataW, int dataH, FLOAT rmin, FLOAT ratio_threshold);



void filter_video(VQMVideo video, cuda_state *state) {

	FLOAT rmin = state->rmin;
	FLOAT ratio_threshold = state->ratio_threshold;

	int dataW = video.dataDim.height;
	int dataHT = video.dataDim.width * video.dataDim.depth;

	const int grid_filter_x = (dataW + (DIM_FILTER_X - 1)) / DIM_FILTER_X;
	const int grid_filter_y = (dataHT + (DIM_FILTER_Y - 1)) / DIM_FILTER_Y;
	const int grid_sqrt_x = (dataW + (DIM_SQRT_X - 1)) / DIM_SQRT_X;
	const int grid_sqrt_y = (dataHT + (DIM_SQRT_Y - 1)) / DIM_SQRT_Y;

	dim3 block(DIM_FILTER_X, DIM_FILTER_Y, 1);
	dim3 grid(grid_filter_x, grid_filter_y, 1);

	dim3 block_sqrt(DIM_SQRT_X, DIM_SQRT_Y, 1);
	dim3 grid_sqrt(grid_sqrt_x, grid_sqrt_y, 1);

	//Edge enhancement filtering
	filterRowGPU<<<grid, block,0,state->stream1>>>(state->d_row_1, state->d_sobel_Kernel, state->d_y, dataW, dataHT);
	filterColGPU<<<grid, block,0,state->stream2>>>(state->d_col_1, state->d_sobel_Kernel, state->d_y, dataW, dataHT);

	//Smooth filtering
	convolutionColGPU<<<grid, block,0,state->stream1>>>(state->d_row_2, state->d_row_1, dataW, dataHT);
	convolutionRowGPU<<<grid, block,0,state->stream2>>>(state->d_col_2, state->d_col_1, dataW, dataHT);

	cudaThreadSynchronize();

	//Calculate si, hv, and hvb
	convolutionSqrtGPU<<<grid_sqrt, block_sqrt, 0,state->stream1>>>(state->d_si , state->d_row_2, state->d_col_2, dataW, dataHT,rmin,ratio_threshold);
	//cudaThreadSynchronize();
}

// parts of this code have been adapted from http://www.evl.uic.edu/sjames/cs525/final.html
__global__ void convolutionRowGPU(FLOAT *d_Result, FLOAT *d_Data, int dataW,
		int dataH) {

	__shared__ FLOAT data[(DIM_FILTER_X + KERNEL_RADIUS * 2) * DIM_FILTER_Y];

	const int gLoc = threadIdx.x + blockIdx.x * blockDim.x + threadIdx.y * dataW
			+ blockIdx.y * blockDim.y * dataW;
	int x;

	const int x0 = threadIdx.x + blockIdx.x * blockDim.x;
	const int y0 = threadIdx.y + blockIdx.y * blockDim.y;
	const int shift = threadIdx.y * (DIM_FILTER_X + KERNEL_RADIUS * 2);

	if (y0 < dataH && x0 < dataW + 2 * KERNEL_RADIUS) {
		x = x0 - KERNEL_RADIUS;
		if (x < 0 || x > dataW - 1)
			data[threadIdx.x + shift] = 0;
		else
			data[threadIdx.x + shift] = d_Data[gLoc - KERNEL_RADIUS];

		x = x0 + KERNEL_RADIUS;
		if (x > dataW - 1)
			data[threadIdx.x + 2 * KERNEL_RADIUS + shift] = 0;
		else
			data[threadIdx.x + 2 * KERNEL_RADIUS + shift] = d_Data[gLoc
					+ KERNEL_RADIUS];

	}
	__syncthreads();

	if (x0 < dataW && y0 < dataH) {
		FLOAT sum = 0;
		x = KERNEL_RADIUS + threadIdx.x;

		for (int i = -KERNEL_RADIUS; i <= KERNEL_RADIUS; i++)
			sum += data[x + i + shift];
		d_Result[gLoc] = sum;
	}

}

__global__ void filterRowGPU(FLOAT *d_Result, FLOAT *d_kernel, FLOAT *d_Data,
		int dataW, int dataH) {

	__shared__ FLOAT Kernel[KERNEL_RADIUS * 2 + 1];
	__shared__ FLOAT data[(DIM_FILTER_X + KERNEL_RADIUS * 2) * DIM_FILTER_Y];

	const int gLoc = threadIdx.x + blockIdx.x * blockDim.x + threadIdx.y * dataW
			+ blockIdx.y * blockDim.y * dataW;

	int x;
	const int x0 = threadIdx.x + blockIdx.x * blockDim.x;
	const int y0 = threadIdx.y + blockIdx.y * blockDim.y;
	const int shift = threadIdx.y * (DIM_FILTER_X + KERNEL_RADIUS * 2);

	Kernel[x0 % 13] = d_kernel[x0 % 13];

	if (y0 < dataH && x0 < dataW + 2 * KERNEL_RADIUS) {
		x = x0 - KERNEL_RADIUS;
		if (x < 0 || x > dataW - 1)
			data[threadIdx.x + shift] = 0;
		else
			data[threadIdx.x + shift] = d_Data[gLoc - KERNEL_RADIUS];

		x = x0 + KERNEL_RADIUS;
		if (x > dataW - 1)
			data[threadIdx.x + 2 * KERNEL_RADIUS + shift] = 0;
		else
			data[threadIdx.x + 2 * KERNEL_RADIUS + shift] = d_Data[gLoc
					+ KERNEL_RADIUS];

	}
	__syncthreads();

	if (x0 < dataW && y0 < dataH) {
		FLOAT sum = 0;
		x = KERNEL_RADIUS + threadIdx.x;

		for (int i = -KERNEL_RADIUS; i <= KERNEL_RADIUS; i++)
			sum += data[x + i + shift] * Kernel[KERNEL_RADIUS - i];
		d_Result[gLoc] = sum;
	}
}

__global__ void convolutionColGPU(FLOAT *d_Result, FLOAT *d_Data, int dataW,
		int dataH) {

	__shared__ FLOAT data[DIM_FILTER_X * (DIM_FILTER_Y + KERNEL_RADIUS * 2)];

	const int gLoc = threadIdx.x + blockIdx.x * blockDim.x + threadIdx.y * dataW
			+ blockIdx.y * blockDim.y * dataW;

	const int x0 = threadIdx.x + blockIdx.x * blockDim.x;
	const int y0 = threadIdx.y + blockIdx.y * blockDim.y;

	int y;
	const int shift = threadIdx.y * (DIM_FILTER_X);

	if (y0 < dataH + 2 * KERNEL_RADIUS && x0 < dataW) {
		y = y0 - KERNEL_RADIUS;
		if (y < 0 || y > dataH - 1)
			data[threadIdx.x + shift] = 0;

		else
			data[threadIdx.x + shift] = d_Data[gLoc - dataW * KERNEL_RADIUS];

		const int shift1 = shift + 2 * KERNEL_RADIUS * DIM_FILTER_X;
		y = y0 + KERNEL_RADIUS;
		if (y > dataH - 1)
			data[threadIdx.x + shift1] = 0;

		else
			data[threadIdx.x + shift1] = d_Data[gLoc + dataW * KERNEL_RADIUS];

	}

	__syncthreads();
	if (y0 < dataH && x0 < dataW) {
		FLOAT sum = 0;
		y = KERNEL_RADIUS + threadIdx.y;
		for (int i = -KERNEL_RADIUS; i <= KERNEL_RADIUS; i++)
			sum += data[threadIdx.x + (y + i) * DIM_FILTER_X];
		d_Result[gLoc] = sum;

	}

}

__global__ void filterColGPU(FLOAT *d_Result, FLOAT *d_kernel, FLOAT *d_Data,
		int dataW, int dataH) {

	__shared__ FLOAT Kernel[KERNEL_RADIUS * 2 + 1];
	__shared__ FLOAT data[DIM_FILTER_X * (DIM_FILTER_Y + KERNEL_RADIUS * 2)];

	const int gLoc = threadIdx.x + blockIdx.x * blockDim.x + threadIdx.y * dataW
			+ blockIdx.y * blockDim.y * dataW;

	const int x0 = threadIdx.x + blockIdx.x * blockDim.x;
	const int y0 = threadIdx.y + blockIdx.y * blockDim.y;

	const int shift = threadIdx.y * (DIM_FILTER_X);

	Kernel[x0 % 13] = d_kernel[x0 % 13];

	int y;

	if (y0 < dataH + 2 * KERNEL_RADIUS && x0 < dataW) {
		y = y0 - KERNEL_RADIUS;
		if (y < 0 || y > dataH - 1)
			data[threadIdx.x + shift] = 0;

		else
			data[threadIdx.x + shift] = d_Data[gLoc - dataW * KERNEL_RADIUS];

		const int shift1 = shift + 2 * KERNEL_RADIUS * DIM_FILTER_X;
		y = y0 + KERNEL_RADIUS;
		if (y > dataH - 1)
			data[threadIdx.x + shift1] = 0;

		else
			data[threadIdx.x + shift1] = d_Data[gLoc + dataW * KERNEL_RADIUS];

	}

	__syncthreads();

	if (y0 < dataH && x0 < dataW) {
		FLOAT sum = 0;
		y = KERNEL_RADIUS + threadIdx.y;
		for (int i = -KERNEL_RADIUS; i <= KERNEL_RADIUS; i++)
			sum += data[threadIdx.x + (y + i) * DIM_FILTER_X]
					* Kernel[KERNEL_RADIUS - i];
		d_Result[gLoc] = sum;

	}

}

__global__ void convolutionSqrtGPU(FLOAT *d_result, FLOAT *d_col, FLOAT *d_row,
		int dataW, int dataH, FLOAT rmin, FLOAT ratio_threshold) {

	const int gLoc = threadIdx.x + blockIdx.x * blockDim.x + threadIdx.y * dataW
			+ blockIdx.y * blockDim.y * dataW;

	const int x0 = threadIdx.x + blockIdx.x * blockDim.x;
	const int y0 = threadIdx.y + blockIdx.y * blockDim.y;

	if (x0 < dataW && y0 < dataH) {
		FLOAT data_col = abs(d_col[gLoc]);
		FLOAT data_row = abs(d_row[gLoc]);
		FLOAT ratio = min2(data_col, data_row) / max2(data_col, data_row);
		FLOAT hv = 0;
		FLOAT hvb = 0;
		FLOAT si = sqrt(data_col * data_col + data_row * data_row);
		if (si > rmin) {
			if (ratio > ratio_threshold)
				hv = si;
			else
				hvb = si;
		}
		d_result[gLoc] = si;
		d_col[gLoc] = hv;
		d_row[gLoc] = hvb;
	}
}

#ifdef MATLAB
void catchError()
{
	cudaError_t last_error = cudaGetLastError();
	if (last_error != cudaSuccess) {
		// print the CUDA error message and exit
		printf("CUDA error: %s\n", cudaGetErrorString(last_error));
		exit(-1);
	}
}

void filter_si_hv(FLOAT *si, FLOAT *hv, FLOAT *hvb, FLOAT *video,
		int dataW, int dataH, int size, FLOAT *filter, FLOAT rmin, FLOAT ratio_threshold) {

//	feature_printMatrix(video, dataW, dataH, 10, 10, 0,  0);

	FLOAT OneKernel[] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};

	int DIMx = 16;
	int DIMy = 16;
	int DIMx_sqrt = 8;
	int DIMy_sqrt = 8;

	const int GRIDx = (dataW + (DIMx - 1)) / DIMx;
	const int GRIDy = (dataH + (DIMy - 1)) / DIMy;

	const int GRIDx_sqrt= (dataW + (DIMx_sqrt - 1)) / DIMx_sqrt;
	const int GRIDy_sqrt = (dataH + (DIMy_sqrt - 1)) / DIMy_sqrt;

	dim3 block(DIMx, DIMy, 1);
	dim3 grid(GRIDx, GRIDy, 1);

	dim3 block_sqrt(DIMx_sqrt, DIMy_sqrt, 1);
	dim3 grid_sqrt(GRIDx_sqrt, GRIDy_sqrt, 1);

	FLOAT *d_video = NULL;
	cudaMalloc((void**) &d_video, sizeof(FLOAT) * size);
	cudaMemcpy(d_video, video, sizeof(FLOAT) * size, cudaMemcpyHostToDevice);

//	d_feature_printMatrix(d_video, dataW, dataH, 10, 10, 0,  0);

	FLOAT *d_sobel_Kernel = NULL;
	cudaMalloc((void**) &d_sobel_Kernel, sizeof(FLOAT) * 13);
	cudaMemcpy(d_sobel_Kernel, filter, sizeof(FLOAT) * 13, cudaMemcpyHostToDevice);

	FLOAT *d_Ones = NULL;
	cudaMalloc((void**) &d_Ones, sizeof(FLOAT) * 13);
	cudaMemcpy(d_Ones, OneKernel, sizeof(FLOAT) * 13, cudaMemcpyHostToDevice);

	FLOAT *d_row_1 = NULL;
	cudaMalloc((void**) &d_row_1, sizeof(FLOAT) * size);

	FLOAT *d_col_1 = NULL;
	cudaMalloc((void**) &d_col_1, sizeof(FLOAT) * size);

	cudaStream_t stream1, stream2;
	cudaStreamCreate(&stream1);
	cudaStreamCreate(&stream2);

	filterRowGPU<<<grid, block,0,stream1>>>(d_row_1, d_sobel_Kernel, d_video, dataW, dataH);
	filterColGPU<<<grid, block,0,stream2>>>(d_col_1, d_sobel_Kernel, d_video, dataW, dataH);

	cudaThreadSynchronize();

//	cudaError_t last_error = cudaGetLastError();
//	if (last_error != cudaSuccess) {
//		// print the CUDA error message and exit
//		printf("CUDA error: %s\n", cudaGetErrorString(last_error));
//	}

//	d_feature_printMatrix(d_row_1, dataW, dataH, 10, 10, 0,  0);
//	d_feature_printMatrix(d_col_1, dataW, dataH, 10, 10,  0, 0);

//	cudaFree((void*) d_video);
//	cudaFree((void*) d_sobel_Kernel);

	FLOAT *d_row_2 = NULL;
	cudaMalloc((void**) &d_row_2, sizeof(FLOAT) * size);

	FLOAT *d_col_2 = NULL;
	cudaMalloc((void**) &d_col_2, sizeof(FLOAT) * size);

	convolutionColGPU<<<grid, block,0,stream1>>>(d_row_2, d_row_1, dataW, dataH);
	convolutionRowGPU<<<grid, block,0,stream2>>>(d_col_2, d_col_1, dataW, dataH);

//	cudaFree((void*) d_row_1);
//	cudaFree((void*) d_col_1);
//	cudaFree((void*) d_Ones);

	FLOAT *d_si = NULL;
	cudaMalloc((void**) &d_si, sizeof(FLOAT) * size);

	convolutionSqrtGPU<<<grid_sqrt, block_sqrt>>>(d_si , d_row_2, d_col_2, dataW, dataH,rmin,ratio_threshold);

	cudaThreadSynchronize();// Wait for the GPU launched work to complete

//	catchError();
//
	cudaMemcpy(si, d_si, sizeof(FLOAT) * size, cudaMemcpyDeviceToHost);
	cudaMemcpy(hv, d_row_2, sizeof(FLOAT) * size, cudaMemcpyDeviceToHost);
	cudaMemcpy(hvb, d_col_2, sizeof(FLOAT) * size, cudaMemcpyDeviceToHost);
//
//
	cudaFree((void*) d_si);
	cudaFree((void*) d_row_2);
	//cudaFree((void*) d_col_2);
	//cudaDeviceReset();

}

/* The gateway function */
void mexFunction( int nlhs, mxArray *plhs[],
		int nrhs, const mxArray *prhs[])
{

	FLOAT *video = mxGetPr(prhs[0]);
	FLOAT *filter = mxGetPr(prhs[1]);
	FLOAT rmin = mxGetScalar(prhs[2]);
	FLOAT ratio_threshold = mxGetScalar(prhs[3]);

	size_t imageX = mxGetM(prhs[0]);
	size_t imageY = mxGetN(prhs[0]);

	/* create the output matrix */
	plhs[0] = mxCreateDoubleMatrix(imageX,imageY,mxREAL);
	plhs[1] = mxCreateDoubleMatrix(imageX,imageY,mxREAL);
	plhs[2] = mxCreateDoubleMatrix(imageX,imageY,mxREAL);

	/* get a pointer to the real data in the output matrix */
	FLOAT *si = mxGetPr(plhs[0]);
	FLOAT *hv = mxGetPr(plhs[1]);
	FLOAT *hvb = mxGetPr(plhs[2]);

	/* call the computational routine */

	filter_si_hv(si, hv, hvb, video, imageX, imageY, imageX*imageY, filter, rmin, ratio_threshold);
}

#endif
