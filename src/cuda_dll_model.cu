#include "cuda_dll_model.h"

#ifdef MATLAB
#include "mex.h"
#endif


cuda_state initialize(VQMVideo video, FLOAT* sobel_kernel, FLOAT* OneKernel,
		double rmin, double ratio_threshold) {
	cuda_state state;

	state.dataW = video.dataDim.height;
	state.dataH = video.dataDim.width;
	state.dataT = video.dataDim.depth;

	int dataW = video.dataDim.height;
	int dataH = video.dataDim.width;
	int dataT = video.dataDim.depth;

	state.rmin = rmin;
	state.ratio_threshold = ratio_threshold;

	int frameSize = dataW * dataH;
	int size = frameSize * dataT;

	int slice_size = slice_size = video.videoDim.width * video.videoDim.height
			* video.dataDim.depth;
	int video_offset = video_offset = slice_size * 1.5;

	cudaExtent volumeSize_Y = make_cudaExtent(video.raw_video.dim.width,
			video.raw_video.dim.height, video.raw_video.t_slice);
	cudaExtent volumeSize_Cr = make_cudaExtent(video.raw_video.dim.width / 2,
			video.raw_video.dim.height / 2, video.raw_video.t_slice);
	cudaExtent volumeSize_Cb = make_cudaExtent(video.raw_video.dim.width / 2,
			video.raw_video.dim.height / 2, video.raw_video.t_slice);

	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<unsigned char>();



	cudaMalloc3DArray(&state.d_raw_y, &channelDesc, volumeSize_Y);
	cudaMalloc3DArray(&state.d_raw_cb, &channelDesc, volumeSize_Cb);
	cudaMalloc3DArray(&state.d_raw_cr, &channelDesc, volumeSize_Cr);

	cudaMalloc((void**) &state.d_ati, sizeof(FLOAT) * size);
	cudaMalloc((void**) &state.d_si, sizeof(FLOAT) * size);
	cudaMalloc((void**) &state.d_row_1, sizeof(FLOAT) * size);
	cudaMalloc((void**) &state.d_col_1, sizeof(FLOAT) * size);
	cudaMalloc((void**) &state.d_row_2, sizeof(FLOAT) * size);
	cudaMalloc((void**) &state.d_col_2, sizeof(FLOAT) * size);
	cudaMalloc((void**) &state.d_cb, sizeof(FLOAT) * size);
	cudaMalloc((void**) &state.d_cr, sizeof(FLOAT) * size);
	cudaMalloc((void**) &state.d_Ones, sizeof(FLOAT) * 13);
	cudaMalloc((void**) &state.d_sobel_Kernel, sizeof(FLOAT) * 13);
	cudaMalloc((void**) &state.d_ati_video, sizeof(FLOAT) * (size + frameSize));
	state.d_y = state.d_ati_video + frameSize;
	cudaMalloc((void**) &state.d_result,
			sizeof(FLOAT) * frameSize / 16 * dataT);
	cudaMalloc((void**) &state.d_raw_video,
			sizeof(unsigned char) * video_offset);

	cudaMemcpy(state.d_sobel_Kernel, sobel_kernel, sizeof(FLOAT) * 13,
			cudaMemcpyHostToDevice);

	cudaMemcpy(state.d_Ones, OneKernel, sizeof(FLOAT) * 13,
			cudaMemcpyHostToDevice);

	cudaStreamCreate(&state.stream1);
	cudaStreamCreate(&state.stream2);


	return state;

}




void extract_features(VQMVideo video, cuda_state *state,
		int slice) {


	int size_y_std = DATA_2D_SIZE(video.y_std.dataDim);
	int size_hvb_mean = DATA_2D_SIZE(video.hvb_mean.dataDim);
	int size_hv_mean = DATA_2D_SIZE(video.hv_mean.dataDim);
	int size_si_std = DATA_2D_SIZE(video.si_std.dataDim);
	int size_cb_mean = DATA_2D_SIZE(video.cb_mean.dataDim) * video.dataDim.depth;
	int size_cr_mean = DATA_2D_SIZE(video.cr_mean.dataDim) * video.dataDim.depth;
	int size_ati_std = DATA_2D_SIZE(video.ati_std.dataDim);


	FLOAT* y_std = video.y_std.array + slice * size_y_std;
	FLOAT* hvb_mean = video.hvb_mean.array + slice * size_hvb_mean;
	FLOAT* hv_mean = video.hv_mean.array + slice * size_hv_mean;
	FLOAT* si_std = video.si_std.array + slice * size_si_std;
	FLOAT* cb_mean = video.cb_mean.array + slice * size_cb_mean;
	FLOAT* cr_mean = video.cr_mean.array + slice * size_cr_mean;
	FLOAT* ati_std = video.ati_std.array + slice * size_ati_std;


	FLOAT *si = state->d_si;
	FLOAT *hv = state->d_col_2;
	FLOAT *hvb = state->d_row_2;


	VQMDim3 dataDim;
	dataDim.width = video.dataDim.height;
	dataDim.height = video.dataDim.width;
	dataDim.depth = video.dataDim.depth;

	VQMDim2 offset;
	offset.height = 6;
	offset.width = 6;

	VQMDim2 featureDim8;
	featureDim8.height = 8;
	featureDim8.width = 8;

	VQMDim2 featureDim4;
	featureDim4.height = 4;
	featureDim4.width = 4;




	temporal_filter(video, state);

	int dataW = video.dataDim.width;
	int dataH = video.dataDim.height;
	int dataT = video.dataDim.depth;


	//copy last frame of the video-slice and add it to the next slice (for ati filtering)
	cudaMemcpyAsync(state->d_ati_video, state->d_y+dataH*dataW*(dataT-1), dataH*dataW*sizeof(FLOAT),cudaMemcpyDeviceToDevice,state->stream2);
		state->has_Prior_Frame = true;


	block_statistic_mean_sliced(cr_mean, state->d_cr, dataDim, featureDim8, offset,
			state->stream1, state);

	block_statistic_mean_sliced(cb_mean, state->d_cb, dataDim, featureDim8, offset,
			state->stream2, state);

	block_statistic_std(y_std, state->d_y, dataDim, featureDim4, offset,
			 state->stream1, state);

	dataDim.depth = state->dataT_ati;
	block_statistic_std(ati_std, state->d_ati, dataDim, featureDim4, offset,
			state->stream2, state);
	dataDim.depth = video.dataDim.depth;


	filter_video(video, state);


	if(slice<video.t_slice_total)
		prepare_next_slice(video , state, slice+1);

	block_statistic_std(si_std, si, dataDim, featureDim8, offset,
			state->stream2, state);

	block_statistic_mean(hv_mean, hv, dataDim, featureDim8, offset,
			state->stream1, state);

	block_statistic_mean(hvb_mean, hvb, dataDim, featureDim8, offset,
			state->stream2, state);

	checkCUDAerror();

	//cudaThreadSynchronize();




}

#ifdef MATLAB
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {

	FLOAT *y = mxGetPr(prhs[0]);
	FLOAT *cb = mxGetPr(prhs[1]);
	FLOAT *cr = mxGetPr(prhs[2]);
	FLOAT *filter = mxGetPr(prhs[3]);
	FLOAT rmin = mxGetScalar(prhs[4]);
	FLOAT ratio_threshold = mxGetScalar(prhs[5]);

	FLOAT *priorFrame = 0;
	if(nrhs>6)
	priorFrame = mxGetPr(prhs[6]);

	const mwSize *yDim = mxGetDimensions(prhs[0]);

	const mwSize dataW = yDim[0];
	const mwSize dataH = yDim[1];
	const mwSize dataT = yDim[2];

//	int resultDataW = dataW;
//	int resultDataH = dataH;
//	int resultDataT = dataT;

	/* create the output matrix */
	//plhs[0] = mxCreateDoubleMatrix(resultDataW, resultDataH, mxREAL);
	int imageX = dataW-12;
	int imageY = dataH-12;

	int imageX8 = imageX/8;
	int imageY8 = imageY/8;

	int imageX4 = (imageX-imageX%8)/4;
	int imageY4 = (imageY-imageY%8)/4;

	plhs[0] = mxCreateNumericMatrix(imageX4,imageY4,MATLAB_FLOAT,mxREAL);
	plhs[1] = mxCreateNumericMatrix(imageX8,imageY8,MATLAB_FLOAT,mxREAL);
	plhs[2] = mxCreateNumericMatrix(imageX8,imageY8,MATLAB_FLOAT,mxREAL);
	plhs[3] = mxCreateNumericMatrix(imageX8,imageY8,MATLAB_FLOAT,mxREAL);

	mwSize dims[3] = {imageX8,imageY8,dataT};
	mwSize ndim = 3;

	plhs[4] = mxCreateNumericArray(ndim, dims, MATLAB_FLOAT, mxREAL);
	plhs[5] = mxCreateNumericArray(ndim, dims, MATLAB_FLOAT, mxREAL);

	plhs[6] = mxCreateDoubleMatrix(imageX4,imageY4,mxREAL);

	FLOAT *y_std = mxGetPr(plhs[0]);
	FLOAT *hvb_mean = mxGetPr(plhs[1]);
	FLOAT *hv_mean = mxGetPr(plhs[2]);
	FLOAT *si_std = mxGetPr(plhs[3]);
	FLOAT *cb_mean = mxGetPr(plhs[4]);
	FLOAT *cr_mean = mxGetPr(plhs[5]);
	FLOAT *ati_std = mxGetPr(plhs[6]);

	//printMatrix(cb, dataW, dataH, 0,0);

	printf("dataW: %d\n",dataW);
	printf("dataH: %d\n",dataH);
	printf("dataT: %d\n",dataT);
	printf("rmin: %.2f\n",rmin);
	printf("ratio_threshold: %.2f\n",ratio_threshold);

	cuda_state state = initialize(dataW,dataH,dataT);
	cuda_model_copy_on_device(y, cb, cr, filter, priorFrame , &state);

	//void cuda_model_run_Callback_feature_general(VQMVideo video , cuda_state *state, int t_slice)

	extract_features(y_std, hvb_mean, hv_mean, si_std, cb_mean, cr_mean, ati_std, &state ,y, cb, cr, filter, dataW, dataH, dataT, rmin, ratio_threshold,priorFrame);

	//printMatrix(y_std, imageX4, imageY4, 0,0);

	cudaDeviceReset();

}

#endif
