#include "cuda_format.h"


void checkCUDAerror() {
	cudaError_t last_error = cudaGetLastError();
	if (last_error != cudaSuccess) {

		printf("CUDA error: %s\n", cudaGetErrorString(last_error));
	}
}

void printMatrix(FLOAT *matrix, int dataW, int dataH, int offsetX,
		int offsetY) {
	printf("\n");

	int i, j;
	for (i = 0; i < offsetY; i++) {
		for (j = 0; j < offsetX; j++)
			printf("%.2f\t", matrix[i * dataW + j]);
		printf("\n");
	}

}


void char_printMatrix(unsigned char *matrix, int dataW, int dataH, int offsetX,
		int offsetY) {
	printf("\n");

	int i, j;
	for (i = 0; i < offsetY; i++) {
		for (j = 0; j < offsetX; j++)
			printf("%d\t", matrix[i * dataW + j]);
		printf("\n");
	}

}

void d_printMatrix(FLOAT *d_matrix, int dataW, int dataH, int offsetX,
		int offsetY) {

	FLOAT* matrix = (FLOAT*) malloc(dataW * dataH * sizeof(FLOAT));

	cudaMemcpy(matrix, d_matrix, sizeof(FLOAT) * dataW * dataH,
			cudaMemcpyDeviceToHost);
	printMatrix(matrix, dataW, dataH, offsetX, offsetY);

}


void print_features(VQMFeature features, int rows, int cols, int time)
{
	int r,c,t;
	int offset;

	printf("dataW: %d\n", features.dataDim.width);
	printf("dataH: %d\n", features.dataDim.height);
	printf("dataT: %d\n", features.dataDim.depth);

	for(t=0;t<time;t++)
	{
		for(r=0;r<rows;r++)
		{
			for(c=0;c<cols;c++)
			{
				offset = t*features.dataDim.depth*features.dataDim.width + r*features.dataDim.height + c;
				printf("%f\t", features.array[offset]);
			}
			printf("\n");
		}
		printf("\n");
	}

}

void print_VQMVideo(VQMVideo video){
	printf("path:\t%s\n", video.path);

	printf("videoW:\t%d\n", video.videoDim.width);
	printf("videoH:\t%d\n", video.videoDim.height);
	printf("frameRate:\t%d\n", video.fps);
	printf("frames:\t%d\n", video.frames);
	printf("offset:\t%d\n", video.offset);
	printf("t_slice:\t%d\n", video.dataDim.depth);
	printf("t_slice_total:\t%d\n", video.t_slice_total);
	printf("video_size:\t%ld\n", video.raw_video.video_size);
	printf("\n");

	printf("left:\t%d\n", video.left);
	printf("right:\t%d\n", video.right);
	printf("top:\t%d\n", video.top);
	printf("bottom:\t%d\n", video.bottom);
	printf("\n");

	printf("dataW:\t%d\n", video.dataDim.width);
	printf("dataH:\t%d\n", video.dataDim.height);
	printf("dataT:\t%d\n", video.dataDim.depth);

}

void print_VQMVideo_features(VQMVideo video){


	printf("si_std:\t\t%d x %d x %d\n", video.si_std.dataDim.width,video.si_std.dataDim.height,video.si_std.dataDim.depth);
	printf("hv_mean:\t%d x %d x %d\n", video.hv_mean.dataDim.width,video.hv_mean.dataDim.height,video.hv_mean.dataDim.depth);
	printf("hvb_mean:\t%d x %d x %d\n", video.hvb_mean.dataDim.width,video.hvb_mean.dataDim.height,video.hvb_mean.dataDim.depth);
	printf("ati_std:\t%d x %d x %d\n", video.ati_std.dataDim.width,video.ati_std.dataDim.height,video.ati_std.dataDim.depth);
	printf("y_std:\t\t%d x %d x %d\n", video.y_std.dataDim.width,video.y_std.dataDim.height,video.y_std.dataDim.depth);
	printf("cr_mean:\t%d x %d x %d\n", video.cr_mean.dataDim.width,video.cr_mean.dataDim.height,video.cr_mean.dataDim.depth);
	printf("cb_mean:\t%d x %d x %d\n", video.cb_mean.dataDim.width,video.cb_mean.dataDim.height,video.cb_mean.dataDim.depth);
	printf("\n");
}


VQMFeature create_features(VQMDim3 dataDim){

	VQMFeature feature;
	feature.dataDim.width = dataDim.width;
	feature.dataDim.height = dataDim.height;
	feature.dataDim.depth = dataDim.depth;
	feature.array = (FLOAT*)malloc(feature.dataDim.width*feature.dataDim.height*feature.dataDim.depth*sizeof(FLOAT));

	return feature;
}

VQMFeature create_features(VQMDim3 dataDim, FLOAT* array){

	VQMFeature feature;
	feature.dataDim.width = dataDim.width;
	feature.dataDim.height = dataDim.height;
	feature.dataDim.depth = dataDim.depth;
	feature.array = array;
	return feature;
}

void print_VQMResult(VQMResult result){

	printf("si_loss\t\t%.8f\n", result.si_loss);
	printf("hv_loss\t\t%.8f\n", result.hv_loss);
	printf("hv_gain\t\t%.8f\n", result.hv_gain);
	printf("color1\t\t%.8f\n", result.color1);
	printf("si_gain\t\t%.8f\n", result.si_gain);
	printf("contati\t\t%.8f\n", result.contati);
	printf("color2\t\t%.8f\n", result.color2);
	printf("\nVQM value\t%.16f\n\n", result.VQM);

}

VQMVideo create_video(char* path,VQMDim2 rawResolution , int rawFPS, VQMDim2 targetResolution, int targetFPS)
{
	VQMVideo video;

	video.path = path;
	video.raw_video.dim = rawResolution;
	video.raw_video.fps = rawFPS;

	video.videoDim = targetResolution;
	video.fps = targetFPS;

	return video;
}
