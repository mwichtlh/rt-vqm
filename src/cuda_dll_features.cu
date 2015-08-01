#include "cuda_dll_features.h"

#ifdef MATLAB
#include "mex.h"
#endif

using namespace std;

clock_t __begin;
clock_t __end;
double __elapsed_secs;

FLOAT rmin = 20;
FLOAT ratio_threshold = 0.2288753702;

cuda_state* state = 0;

//FLOAT filter[] = { 0.005262480978744, 0.017344596686535, 0.042740095161241,
//		0.076896118758078, 0.095773908674672, 0.069675107433038, 0,
//		-0.069675107433038, -0.095773908674672, -0.076896118758078,
//		-0.042740095161241, -0.017344596686535, -0.005262480978744 };

//FLOAT filter[] = {
//
//		0.005262480978743695400612345736135466722771525382995605,
//		0.04274009516124132462833173917715612333267927169799805,
//		0.07689611875807839114216335474338848143815994262695312,
//		0.09577390867467201751583871782713686116039752960205078,
//		0.06967510743303773068646478350274264812469482421875,
//		0,
//		-0.06967510743303773068646478350274264812469482421875,
//		-0.09577390867467201751583871782713686116039752960205078,
//		-0.07689611875807839114216335474338848143815994262695312,
//		-0.04274009516124132462833173917715612333267927169799805,
//		-0.01734459668653457603548773136026284191757440567016602,
//		-0.005262480978743695400612345736135466722771525382995605 };
//

FLOAT OneKernel[] = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };

void feature_printMatrix(FLOAT *matrix, int dataW, int dataH, int offsetX,
		int offsetY, int cutX, int cutY) {
	printf("\n");

	int dataH_off = cutX + offsetX;
	int dataW_off = cutY + offsetY;

	int i, j;

	printf("\t ");
	for (i = cutY; i < dataH_off; i++)
		printf("%d\t ", i);
	printf("\n");

	for (j = cutX; j < dataW_off; j++) {
		printf("%d\t ", j);
		for (i = cutX; i < dataH_off; i++)
			printf("%.0f\t", matrix[i * dataW + j]);
		printf(" -%d\t ", j);
		printf("\n");
	}

}

void d_feature_printMatrix(FLOAT *d_matrix, int dataW, int dataH, int offsetX,
		int offsetY, int cutX, int cutY) {

	FLOAT* matrix = (FLOAT*) malloc(dataW * dataH * sizeof(FLOAT));

	cudaMemcpy(matrix, d_matrix, sizeof(FLOAT) * dataW * dataH,
			cudaMemcpyDeviceToHost);
	feature_printMatrix(matrix, dataW, dataH, offsetX, offsetY, cutX, cutY);

}


VQMVideo initialize_video(char* path, int dataW, int dataH, int frameRate,
		int orig_dataW, int orig_dataH, int orig_frameRate, int offset, int length, temporal_scaling scaling) {

	VQMVideo video;

	video.scaling = scaling;

	int fps;
	if(scaling == NONE)
		fps = orig_frameRate;
	else
		fps = frameRate;

	video.raw_video.dim.width = orig_dataW;
	video.raw_video.dim.height = orig_dataH;
	video.raw_video.video_size = get_video_size(path);
	video.raw_video.fps = orig_frameRate;
	video.offset = round(offset*((FLOAT)orig_frameRate/(FLOAT)fps));
	if(length != 0){
		video.raw_video.frames = round(length*(FLOAT)(orig_frameRate/(FLOAT)fps));
		video.raw_video.video_size = video.raw_video.dim.width * video.raw_video.dim.height * 1.5 * video.raw_video.frames * sizeof(unsigned char);
	}
	else
		video.raw_video.frames = video.raw_video.video_size
			/ (orig_dataW * orig_dataH * 1.5);
	video.raw_video.t_slice = floor(video.raw_video.fps * 0.2);

	video.videoDim.width = dataW;
	video.videoDim.height = dataH;
	video.fps = fps;
	video.path = path;
	video.frames = video.raw_video.frames * (video.fps / video.raw_video.fps);

	video.dataDim.depth = floor(video.fps * 0.2);
	video.t_slice_total = video.frames / video.dataDim.depth;

	video = calculate_roi(video);
	video.dataDim.width = video.right - video.left + 1;
	video.dataDim.height = video.bottom - video.top + 1;

	cudaMallocHost((void**) &video.slice_buffer,
			video.dataDim.width * video.dataDim.height * 1.5 * 6 * sizeof(unsigned char)*2);

	video.runtime_read = (float*) malloc(sizeof(float));

	return video;

}

VQMVideo initialize_video(VQMVideo video) {

	video.raw_video.video_size = get_video_size(video.path);
	video.raw_video.frames = video.raw_video.video_size
			/ (DATA_2D_SIZE(video.raw_video.dim) * 1.5);
	video.raw_video.t_slice = video.raw_video.fps * 0.2;

	video.frames = video.raw_video.frames * (video.fps / video.raw_video.fps);

	video.dataDim.depth = video.fps * 0.2;
	video.t_slice_total = video.frames / video.dataDim.depth;

	video = calculate_roi(video);
	video.dataDim.width = video.right - video.left + 1;
	video.dataDim.height = video.bottom - video.top + 1;

	cudaMallocHost((void**) &video.slice_buffer,
			video.dataDim.width * video.dataDim.height * 1.5 * 6 * sizeof(unsigned char));

	video.runtime_read = (float*) malloc(sizeof(float));

	return video;

}

VQMVideo allocate_memory(VQMVideo video) {

	int imageX = video.dataDim.width - 12;
	int imageY = video.dataDim.height - 12;

	int size_y_std = (imageX / 4) * (imageY / 4);
	int size_hvb_mean = (imageX / 8) * (imageY / 8);
	int size_hv_mean = (imageX / 8) * (imageY / 8);
	int size_si_std = (imageX / 8) * (imageY / 8);
	int size_cb_mean = (imageX / 8) * (imageY / 8) * video.dataDim.depth;
	int size_cr_mean = (imageX / 8) * (imageY / 8) * video.dataDim.depth;
	int size_ati_std = (imageX / 4) * (imageY / 4);

	//FLOAT *y_std;
	cudaMallocHost(&video.y_std.array, size_y_std * video.t_slice_total * sizeof(FLOAT));
	//video.y_std.array = (FLOAT*)calloc(size_y_std * 75 , sizeof(FLOAT));
	video.y_std.dataDim.width = (imageX / 4);
	video.y_std.dataDim.height = (imageY / 4);
	video.y_std.dataDim.depth = video.t_slice_total;

	//FLOAT *hvb_mean;
	cudaMallocHost(&video.hvb_mean.array, size_hvb_mean * video.t_slice_total * sizeof(FLOAT));
	//video.hvb_mean.array = (FLOAT*)calloc(size_hvb_mean * 75 , sizeof(FLOAT));
	video.hvb_mean.dataDim.width = (imageX / 8);
	video.hvb_mean.dataDim.height = (imageY / 8);
	video.hvb_mean.dataDim.depth = video.t_slice_total;

	//FLOAT *hv_mean;
	cudaMallocHost(&video.hv_mean.array, size_hv_mean * video.t_slice_total * sizeof(FLOAT));
	//video.hv_mean.array = (FLOAT*)calloc(size_hv_mean * 75 , sizeof(FLOAT));
	video.hv_mean.dataDim.width = (imageX / 8);
	video.hv_mean.dataDim.height = (imageY / 8);
	video.hv_mean.dataDim.depth = video.t_slice_total;

	//FLOAT *si_std;
	cudaMallocHost(&video.si_std.array, size_si_std * video.t_slice_total * sizeof(FLOAT));
	//video.si_std.array = (FLOAT*)calloc(size_si_std * 75 , sizeof(FLOAT));
	video.si_std.dataDim.width = (imageX / 8);
	video.si_std.dataDim.height = (imageY / 8);
	video.si_std.dataDim.depth = video.t_slice_total;

	//FLOAT *cr_mean;
	cudaMallocHost(&video.cr_mean.array, size_cr_mean * video.t_slice_total * sizeof(FLOAT));
	//video.cr_mean.array = (FLOAT*)calloc(size_cr_mean * 75 , sizeof(FLOAT));
	video.cr_mean.dataDim.width = (imageX / 8);
	video.cr_mean.dataDim.height = (imageY / 8);
	video.cr_mean.dataDim.depth = video.dataDim.depth * video.t_slice_total;

	//FLOAT *cb_mean;
	cudaMallocHost(&video.cb_mean.array, size_cb_mean * video.t_slice_total * sizeof(FLOAT));
	//video.cb_mean.array = (FLOAT*)calloc(size_cb_mean * 75, sizeof(FLOAT));
	video.cb_mean.dataDim.width = (imageX / 8);
	video.cb_mean.dataDim.height = (imageY / 8);
	video.cb_mean.dataDim.depth = video.dataDim.depth * video.t_slice_total;

	//FLOAT *ati_std;
	cudaMallocHost(&video.ati_std.array, size_ati_std * video.t_slice_total * sizeof(FLOAT));
	//video.ati_std.array = (FLOAT*)calloc(size_ati_std * 75, sizeof(FLOAT));
	video.ati_std.dataDim.width = (imageX / 4);
	video.ati_std.dataDim.height = (imageY / 4);
	video.ati_std.dataDim.depth = video.t_slice_total;

	return video;
}

void free_memory(VQMVideo video) {
	//FLOAT *y_std;
	free(video.y_std.array);
	//FLOAT *hvb_mean;
	free(video.hvb_mean.array);
	//FLOAT *hv_mean;
	free(video.hv_mean.array);
	//FLOAT *si_std;
	free(video.si_std.array);
	//FLOAT *cb_mean;
	free(video.cb_mean.array);
	//FLOAT *cr_mean;
	free(video.cr_mean.array);
	//FLOAT *ati_std;
	free(video.ati_std.array);
}

VQMVideo run_feature_calculation(char* path, int dataW, int dataH,
		int frameRate, int orig_dataW, int orig_dataH, int orig_frameRate, int offset, int length, temporal_scaling scaling) {

	VQMVideo result = initialize_video(path, dataW, dataH, frameRate,
			orig_dataW, orig_dataH, orig_frameRate, offset, length, scaling);
	//print_VQMVideo(result);
	result = allocate_memory(result);
	calculate_feature(result);
	return result;
}


VQMVideo run_feature_calculation(VQMVideo input) {

	VQMVideo result = initialize_video(input);
	//print_VQMVideo(result);
	result = allocate_memory(result);
	calculate_feature(result);
	return result;
}


void calculate_feature(VQMVideo video) {

	FLOAT *filter2 = (FLOAT*) malloc(13 * sizeof(FLOAT));
	int x;
	for (x = -6; x <= 6; x++) {
		filter2[x + 6] = (x * 0.5) * exp(-0.5 * (x * 0.5) * (x * 0.5));
	}

	//generate filter kernel for edge enhancement
	FLOAT sum = 0;
	for (x = 7; x <= 13; x++)
		sum += filter2[x];
	for (x = -6; x <= 6; x++)
		filter2[x + 6] = (filter2[x + 6] / (13 * sum)) * 4;

	//cudaDeviceReset();

	clock_t _start;
	clock_t _end;
	_start = clock();



	//read raw video from media
	//read_video(&video.raw_video.array, video.path, frame_size*video.offset*sizeof(unsigned char),
	//		video.raw_video.video_size);
	read_video(&video);

	_end = clock();
	float _elapsed_secs = double(_end - _start) / CLOCKS_PER_SEC;
	*(video.runtime_read) = _elapsed_secs;

	//initalize and allocate CUDA state
	cuda_state init = initialize(video, filter2, OneKernel, rmin,
				ratio_threshold);
	state = &init;

	//state->has_Prior_Frame = false;


	int slice;

	//calculate features for the whole video
	for (slice = 0; slice < video.t_slice_total; slice++) {
		//sample video-slice
		if (slice == 0)
			prepare_next_slice(video, state, slice);

		//calculate features from video-slice
		extract_features(video, state, slice);

		//if(slice<video.t_slice_total)
			//	prepare_next_slice(video , state, slice+1);

		//copy last frame of the video-slice and add it to the next slice (for ati filtering)
		//cudaMemcpyAsync(state->d_ati_video, state->d_y+dataH*dataW*(dataT-1), dataH*dataW*sizeof(FLOAT),cudaMemcpyDeviceToDevice);
		//state->has_Prior_Frame = true;

	}
	checkCUDAerror();

	//clear memory
	//cudaDeviceReset();
	free(video.raw_video.array);
}

#ifdef MATLAB
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {

	const mxArray *path_Data = prhs[0];
	int path_length;
	//char *path;

	path_length = mxGetN(path_Data)+1;
	//path = mxCalloc(path_length, sizeof(char));
	path = (char*)malloc(path_length * sizeof(char));
	mxGetString(path_Data,path,path_length);

	FLOAT *sobel_Kernel = mxGetPr(prhs[1]);
	//FLOAT *sobel_Kernel = 0;

	int dataW = 1244;
	int dataH = 708;
	int dataT = 6;

	int slices = 75;

	int imageX = dataW-12;
	int imageY = dataH-12;

	int imageX8 = imageX/8;
	int imageY8 = imageY/8;

	int imageX4 = (imageX-imageX%8)/4;
	int imageY4 = (imageY-imageY%8)/4;

	mwSize dims4[3] = {imageY4,imageX4,slices};
	mwSize ndim = 3;

	plhs[0] = mxCreateNumericArray(ndim, dims4, MATLAB_FLOAT, mxREAL);
	plhs[6] = mxCreateNumericArray(ndim, dims4, MATLAB_FLOAT, mxREAL);

	mwSize dims8[3] = {imageY8,imageX8,slices};

	plhs[1] = mxCreateNumericArray(ndim, dims8, MATLAB_FLOAT, mxREAL);
	plhs[2] = mxCreateNumericArray(ndim, dims8, MATLAB_FLOAT, mxREAL);
	plhs[3] = mxCreateNumericArray(ndim, dims8, MATLAB_FLOAT, mxREAL);

	mwSize dims450[3] = {imageY8,imageX8,dataT*slices};

	plhs[4] = mxCreateNumericArray(ndim, dims450, MATLAB_FLOAT, mxREAL);
	plhs[5] = mxCreateNumericArray(ndim, dims450, MATLAB_FLOAT, mxREAL);

	FLOAT *y_std = (FLOAT*)mxGetPr(plhs[0]);
	FLOAT *hvb_mean = (FLOAT*)mxGetPr(plhs[1]);
	FLOAT *hv_mean = (FLOAT*)mxGetPr(plhs[2]);
	FLOAT *si_std = (FLOAT*)mxGetPr(plhs[3]);
	FLOAT *cb_mean = (FLOAT*)mxGetPr(plhs[4]);
	FLOAT *cr_mean = (FLOAT*)mxGetPr(plhs[5]);
	FLOAT *ati_std = (FLOAT*)mxGetPr(plhs[6]);

	//printMatrix(cb, dataW, dataH, 0,0);

	printf("dataW: %d\n",dataW);
	printf("dataH: %d\n",dataH);
	printf("dataT: %d\n",dataT);
	printf("rmin: %.2f\n",rmin);
	printf("ratio_threshold: %.2f\n",ratio_threshold);

	run_feature_calculation(y_std, hvb_mean, hv_mean,si_std, cb_mean, cr_mean, ati_std, sobel_Kernel);

	cudaDeviceReset();

}

#endif
