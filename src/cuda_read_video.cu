#include "cuda_read_video.h"

#ifdef MATLAB
#include "mex.h"
#endif



//textures to sample pixel from raw video
texture<unsigned char, 3, cudaReadModeNormalizedFloat> tex_Y;   // Y textur
texture<unsigned char, 3, cudaReadModeNormalizedFloat> tex_Cr;  // Cb textur
texture<unsigned char, 3, cudaReadModeNormalizedFloat> tex_Cb;  // Cr textur

using namespace std;

clock_t begin;
clock_t end;
double elapsed_secs;



__global__ void cuda_convert_video_matlab(FLOAT *y, FLOAT *cr, FLOAT *cb,
		unsigned char *video, int dataW, int dataH, int dataT, int offset_left,
		int offset_top, int matrixW, int matrixH) {

	const int x0 = threadIdx.x + blockIdx.x * blockDim.x;
	const int y0 = threadIdx.y + blockIdx.y * blockDim.y;
	const int z0 = threadIdx.z;

	int frame_dim = dataW * dataH;
	int frame_size = frame_dim * 1.5;
	int matrix_size = matrixW * matrixH;

	if (x0 < matrixW && y0 < matrixH) {
		unsigned char y_pixel = video[(y0 + offset_left)
				+ (x0 + offset_top) * dataH + frame_size * z0];
		unsigned char cr_pixel = video[(y0 + offset_left) / 2
				+ (x0 + offset_top) / 2 * dataH / 2 + frame_dim
				+ frame_size * z0];
		unsigned char cb_pixel = video[(y0 + offset_left) / 2
				+ (x0 + offset_top) / 2 * dataH / 2 + frame_dim + frame_dim / 4
				+ frame_size * z0];

		y[x0 + y0 * matrixW + z0 * matrix_size] = y_pixel;
		//Subtract 128 from all Cb and Cr values, according to "128" flag in read_avi.m
		cr[x0 + y0 * matrixW + z0 * matrix_size] = cr_pixel-128;
		cb[x0 + y0 * matrixW + z0 * matrix_size] = cb_pixel-128;

	}

}


__global__ void cuda_convert_video_texture_framedub(FLOAT *y, FLOAT *cr, FLOAT *cb,
		unsigned char *video, VQMScale scale, int offset_left,
		int offset_top, VQMDim3 dataDim) {


	const int x0 = threadIdx.x + blockIdx.x * blockDim.x;
	const int y0 = threadIdx.y + blockIdx.y * blockDim.y;
	const int z0 = threadIdx.z;

	int matrix_size = dataDim.width * dataDim.height;

	int y_y = y0 + offset_left;
	int x_y = x0 + offset_top;
	int y_c = (y0 + offset_left) / 2;
	int x_c = (x0 + offset_top) / 2;

	float testOffset = 0;
	if(!z0%2 && scale.temporal != 1)
		testOffset = 0;
	if(z0%2 && scale.temporal != 1)
		testOffset = 1;

	//testOffset = 0;

	if (x0 < dataDim.height && y0 < dataDim.width) {
		FLOAT y_pixel = 255 * tex3D(tex_Y, scale.horizonal*y_y+0.5f, scale.vertical*x_y+0.5f,scale.temporal* (z0 -testOffset)+ 0.5f);

		FLOAT cr_pixel = 255 * tex3D(tex_Cr, scale.horizonal*y_c+0.5f, scale.vertical*x_c+0.5f, scale.temporal* (z0 -testOffset)+ 0.5f);

		FLOAT cb_pixel = 255 *  tex3D(tex_Cb, scale.horizonal*y_c+0.5f, scale.vertical*x_c+0.5f, scale.temporal* (z0 -testOffset)+ 0.5f);

		y[x0 + y0 * dataDim.height + z0 * matrix_size] = y_pixel;
		//Subtract 128 from all Cb and Cr values, according to "128" flag in read_avi.m
		cr[x0 + y0 * dataDim.height + z0 * matrix_size] = cb_pixel-128;
		cb[x0 + y0 * dataDim.height + z0 * matrix_size] = cr_pixel-128;


	}


}

__global__ void cuda_convert_video_texture_interpolated(FLOAT *y, FLOAT *cr, FLOAT *cb,
		unsigned char *video, VQMScale scale, int offset_left,
		int offset_top, VQMDim3 dataDim) {


	const int x0 = threadIdx.x + blockIdx.x * blockDim.x;
	const int y0 = threadIdx.y + blockIdx.y * blockDim.y;
	const int z0 = threadIdx.z;

	int matrix_size = dataDim.width * dataDim.height;

	int y_y = y0 + offset_left;
	int x_y = x0 + offset_top;
	int y_c = (y0 + offset_left) / 2;
	int x_c = (x0 + offset_top) / 2;



	//testOffset = 0;

	if (x0 < dataDim.height && y0 < dataDim.width) {
		FLOAT y_pixel = 255 * tex3D(tex_Y, scale.horizonal*y_y+0.5f, scale.vertical*x_y+0.5f,scale.temporal* z0 + 0.5f);

		FLOAT cr_pixel = 255 * tex3D(tex_Cr, scale.horizonal*y_c+0.5f, scale.vertical*x_c+0.5f, scale.temporal* z0 + 0.5f);

		FLOAT cb_pixel = 255 *  tex3D(tex_Cb, scale.horizonal*y_c+0.5f, scale.vertical*x_c+0.5f, scale.temporal* z0 + 0.5f);

		y[x0 + y0 * dataDim.height + z0 * matrix_size] = y_pixel;
		//Subtract 128 from all Cb and Cr values, according to "128" flag in read_avi.m
		cr[x0 + y0 * dataDim.height + z0 * matrix_size] = cb_pixel-128;
		cb[x0 + y0 * dataDim.height + z0 * matrix_size] = cr_pixel-128;

	}


}




void copy_frame(cudaArray* d_buffer, unsigned char* h_buffer, int dataW, int dataH, int offset, cudaStream_t stream){

	cudaMemcpy3DParms copyParams = {0};
    copyParams.srcPtr   = make_cudaPitchedPtr((void *)h_buffer, dataW*sizeof(unsigned char), dataW, dataH);
    copyParams.dstArray = d_buffer;
    copyParams.extent   = make_cudaExtent(dataW, dataH, 1);
    copyParams.dstPos = make_cudaPos(0,0,offset);
    copyParams.kind     = cudaMemcpyHostToDevice;

    cudaMemcpy3DAsync(&copyParams,stream);

}


void prepare_next_slice(VQMVideo video , cuda_state *state, int slice){

	//Prepare CUDA textures
	tex_Y.normalized = false;                     	//initialize Y texture
	tex_Y.filterMode = cudaFilterModeLinear;
	tex_Y.addressMode[0] = cudaAddressModeWrap;
	tex_Y.addressMode[1] = cudaAddressModeWrap;
	tex_Y.addressMode[2] = cudaAddressModeWrap;

	tex_Cr.normalized = false;                      //initialize Cr texture
	tex_Cr.filterMode = cudaFilterModeLinear;
	tex_Cr.addressMode[0] = cudaAddressModeWrap;
	tex_Cr.addressMode[1] = cudaAddressModeWrap;
	tex_Cr.addressMode[2] = cudaAddressModeWrap;

	tex_Cb.normalized = false;                      //initialize Cb texture
	tex_Cb.filterMode = cudaFilterModeLinear;
	tex_Cb.addressMode[0] = cudaAddressModeWrap;
	tex_Cb.addressMode[1] = cudaAddressModeWrap;
	tex_Cb.addressMode[2] = cudaAddressModeWrap;


	//calculate size of the YUV (YCrCb 4:2:0) planes.
	int y_size = DATA_2D_SIZE(video.raw_video.dim);
	int cb_size = y_size/4;
	int cr_size = y_size/4;

	//complete size of a frame
	int frame_size = y_size+ cb_size+cr_size;


	//calculate scale between original video resolution and sample resolution
	VQMScale scale;
	scale.horizonal = (FLOAT)video.raw_video.dim.width / (FLOAT)video.videoDim.width;
	scale.vertical  = (FLOAT)video.raw_video.dim.height / (FLOAT)video.videoDim.height;
	scale.temporal = (FLOAT)video.raw_video.fps / (FLOAT)video.fps;
	//scale.temporal = 1;

	//currennt position in the video
	unsigned char* video_position;
	if(video.raw_video.fps != video.fps)
		video_position = video.raw_video.array + frame_size*(slice*video.raw_video.t_slice) ;
	else
		video_position = video.raw_video.array + frame_size*(slice*video.raw_video.t_slice) ;

	if(video.raw_video.fps == 15)
		video_position = video.raw_video.array + frame_size*(slice*video.raw_video.t_slice) ;

	//copy raw video frames to CUDA textures
	int frame=0;
	int raw_frame;
	int rawVideoDimW = video.raw_video.dim.width;
	int rawVideoDimH = video.raw_video.dim.height;

	int FramesToCopy = video.dataDim.depth*scale.temporal;
//	if(video.raw_video.fps != video.fps)
//		FramesToCopy++;

	if(slice==0)
		memcpy(video.slice_buffer,video_position, FramesToCopy*frame_size);

	for(frame=0; frame< FramesToCopy; frame++)
	{
//		raw_frame = (frame *  (FLOAT)video.raw_video.fps / (FLOAT)video.fps);
//		if(frame%2 &&  video.raw_video.fps != video.fps)
//			raw_frame++;
		raw_frame = frame;

		copy_frame(state->d_raw_cb, video.slice_buffer+raw_frame*frame_size+y_size, rawVideoDimW/2, rawVideoDimH/2, frame, state->stream2);
		copy_frame(state->d_raw_cr, video.slice_buffer+raw_frame*frame_size+y_size+cb_size, rawVideoDimW/2, rawVideoDimH/2, frame, state->stream2);
		copy_frame(state->d_raw_y, video.slice_buffer+raw_frame*frame_size, rawVideoDimW, rawVideoDimH, frame, state->stream2);

	}

	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<unsigned char>();

	 cudaBindTextureToArray(tex_Y, state->d_raw_y, channelDesc);
	 cudaBindTextureToArray(tex_Cb, state->d_raw_cb, channelDesc);
	 cudaBindTextureToArray(tex_Cr, state->d_raw_cr, channelDesc);

	//Sample video
	int GRIDx = (video.videoDim.width + (DIM_SAMPLE_X - 1)) / DIM_SAMPLE_X;
	int GRIDy = (video.videoDim.height + (DIM_SAMPLE_Y - 1)) / DIM_SAMPLE_Y;

	dim3 block(DIM_SAMPLE_X, DIM_SAMPLE_Y, video.dataDim.depth);
	dim3 grid(GRIDy, GRIDx, 1);


	cudaThreadSynchronize();
	if(video.scaling == FRAME_INTERPOL)
		cuda_convert_video_texture_interpolated<<<grid, block>>>(state->d_y, state->d_cr, state->d_cb,
				state->d_raw_video, scale,  video.left-1, video.top-1, video.dataDim);
	else
		cuda_convert_video_texture_framedub<<<grid, block>>>(state->d_y, state->d_cr, state->d_cb,
				state->d_raw_video, scale,  video.left-1, video.top-1, video.dataDim);
	 if(slice<video.t_slice_total-1)
	 	memcpy(video.slice_buffer,video_position  + frame_size*video.raw_video.t_slice , FramesToCopy*frame_size);
	// cudaThreadSynchronize();



}


long int get_video_size(char *path) {

	FILE *fp;
	fp = fopen(path, "r");

	fseek(fp, 0, SEEK_END);
	long int size = ftell(fp);
	fclose (fp);

	return size;
}

void read_video(unsigned char **video, char *path, long int offset, int size) {


	FILE *fp;
	fp = fopen(path, "r");
	fseek(fp, 0, SEEK_SET);
	//cudaMallocHost((void**)video, size);
	*video = (unsigned char*) malloc(size * sizeof(unsigned char));

	fseek(fp, offset, SEEK_SET);
	size_t file_size = fread(*video, size, 1, fp);


}


void read_video(VQMVideo* video) {


	//calculate size of the YUV (YCrCb 4:2:0) planes.
	int y_size = DATA_2D_SIZE(video->raw_video.dim);
	int cb_size = y_size/4;
	int cr_size = y_size/4;

	//complete size of a frame
	int frame_size = y_size+ cb_size+cr_size;

	long int offset = (long int)frame_size*(long int)video->offset*(long int)sizeof(unsigned char);

	FILE *fp;
	fp = fopen(video->path, "r");
	fseek(fp, 0, SEEK_SET);
	video->raw_video.array = (unsigned char*) malloc(video->raw_video.video_size * sizeof(unsigned char));


	fseek(fp, offset, SEEK_SET);
	size_t file_size = fread(video->raw_video.array, video->raw_video.video_size, 1, fp);
	fclose(fp);


}

#ifdef MATLAB
void convert_video(FLOAT *y, FLOAT *cr, FLOAT *cb, unsigned char *video,
		int dataW, int dataH, int dataT) {
	int w;
	int h;
	int t;
	unsigned char cr_pixel;
	unsigned char cb_pixel;

	int VALID_REGION_TOP = 0;
	int VALID_REGION_LEFT = 0;
	int VALID_REGION_BOTTOM = 720;
	int VALID_REAGION_RIGHT = 1280;

	int dataW_cut = VALID_REAGION_RIGHT - VALID_REGION_LEFT;
	int dataH_cut = VALID_REGION_BOTTOM - VALID_REGION_TOP;

	int frame_dim = dataW * dataH;
	int frame_size = frame_dim * 1.5;

	for (t = 0; t < dataT; t++) {

		for (h = VALID_REGION_TOP; h < VALID_REGION_BOTTOM; h++)
			for (w = VALID_REGION_LEFT; w < VALID_REAGION_RIGHT; w++) {
				y[w - VALID_REGION_LEFT + (h - VALID_REGION_TOP) * dataW_cut
						+ frame_dim * t] = (FLOAT) (video[(h * dataW + w
						+ t * frame_size)]);
			}

		for (h = VALID_REGION_TOP; h < VALID_REGION_BOTTOM / 2; h++)
			for (w = VALID_REGION_LEFT; w < VALID_REAGION_RIGHT / 2; w++) {
				cr_pixel = video[(h) * dataW / 2 + w + frame_dim
						+ t * frame_size];
				cb_pixel = video[(h) * dataW / 2 + w + frame_dim + frame_dim / 4
						+ t * frame_size];
				cr[(h * 2) * dataW + w * 2 + frame_dim * t] = (FLOAT) ((cr_pixel
						- 128) % 256);
				cb[(h * 2) * dataW + w * 2 + frame_dim * t] = (FLOAT) ((cb_pixel
						+ 128) % 256);

				cr[(h * 2) * dataW + w * 2 + 1 + frame_dim * t] =
						(FLOAT) ((cr_pixel - 128) % 256);
				cb[(h * 2) * dataW + w * 2 + 1 + frame_dim * t] =
						(FLOAT) ((cb_pixel + 128) % 256);

				cr[((h * 2) + 1) * dataW + w * 2 + frame_dim * t] =
						(FLOAT) ((cr_pixel - 128) % 256);
				cb[((h * 2) + 1) * dataW + w * 2 + frame_dim * t] =
						(FLOAT) ((cb_pixel + 128) % 256);

				cr[((h * 2) + 1) * dataW + w * 2 + 1 + frame_dim * t] =
						(FLOAT) ((cr_pixel - 128) % 256);
				cb[((h * 2) + 1) * dataW + w * 2 + 1 + frame_dim * t] =
						(FLOAT) ((cb_pixel + 128) % 256);
			}

	}

}

//matrices in matlab have different orientation
void convert_video_matlab(FLOAT *y, FLOAT *cr, FLOAT *cb, unsigned char *video,
		int dataW, int dataH, int dataT) {
	int w;
	int h;
	int t;
	unsigned char cr_pixel;
	unsigned char cb_pixel;



	int frame_dim = dataW * dataH;
	int frame_size = frame_dim * 1.5;

	for (t = 0; t < dataT; t++) {

		for (h = 0; h < dataW; h++)
			for (w = 0; w < dataH; w++) {
				y[h + w * dataW + frame_dim * t] = (FLOAT) (video[(h * dataH + w
						+ t * frame_size)]);
			}

		for (h = 0; h < dataW / 2; h++) {
			for (w = 0; w < dataH / 2; w++) {
				cr_pixel =
						video[h * dataH / 2 + w + frame_dim + t * frame_size];
				cb_pixel = video[h * dataH / 2 + w + frame_dim + frame_dim / 4
						+ t * frame_size];

				//format issue, replace with clean solution
				FLOAT cr_value = (FLOAT) (cr_pixel - 128);
//				if(cr_value > 255)
//					cr_value = 0;
				FLOAT cb_value = (FLOAT) (cb_pixel - 128);
//				if(cb_value < -255)
//					cb_value = 0;

				cr[(w * 2) * dataW + h * 2 + frame_dim * t] = cr_value;
				cb[(w * 2) * dataW + h * 2 + frame_dim * t] = cb_value;

				cr[(w * 2) * dataW + h * 2 + 1 + frame_dim * t] = cr_value;
				cb[(w * 2) * dataW + h * 2 + 1 + frame_dim * t] = cb_value;

				cr[((w * 2) + 1) * dataW + h * 2 + frame_dim * t] = cr_value;
				cb[((w * 2) + 1) * dataW + h * 2 + frame_dim * t] = cb_value;

				cr[((w * 2) + 1) * dataW + h * 2 + 1 + frame_dim * t] =
						cr_value;
				cb[((w * 2) + 1) * dataW + h * 2 + 1 + frame_dim * t] =
						cb_value;

			}
		}

	}

}

void transform_video(FLOAT *y, FLOAT *cr, FLOAT *cb, unsigned char *video, int size, int currentFrame, int rows, int cols, int offset_left, int offset_right, int matrixW, int matrixH)
{

	int frame_dim = rows*cols;
	int frame_size = frame_dim*1.5;

	int matrix_dim = matrixW * matrixH;

	//int offset = frame_size*(currentFrame-1);

	unsigned char *d_video = NULL;
	cudaMalloc((void**) &d_video, sizeof(char) * frame_size*size);

	FLOAT *d_y = NULL;
	FLOAT *cr_y = NULL;
	FLOAT *cb_y = NULL;

	cudaMalloc((void**) &d_y, sizeof(FLOAT) * matrix_dim*size);
	cudaMalloc((void**) &cr_y, sizeof(FLOAT) * matrix_dim*size);
	cudaMalloc((void**) &cb_y, sizeof(FLOAT) * matrix_dim*size);


	int DIMx = 8;
	int DIMy = 8;

	const int GRIDx = (matrixW + (DIMx - 1)) / DIMx;
	const int GRIDy = (matrixH + (DIMy - 1)) / DIMy;

	dim3 block(DIMx, DIMy, size);
	dim3 grid(GRIDx, GRIDy, 1);




	cudaMemcpy(d_video, video, sizeof(char) * frame_size * size,
		cudaMemcpyHostToDevice);

	cuda_convert_video_matlab<<<grid, block>>>(d_y, cr_y, cb_y,
			d_video, rows, cols, size, offset_left, offset_right, matrixW, matrixH);

	cudaThreadSynchronize();

	cudaMemcpy(y, d_y, sizeof(FLOAT) * matrix_dim*size, cudaMemcpyDeviceToHost);
	cudaMemcpy(cr, cr_y, sizeof(FLOAT) * matrix_dim*size, cudaMemcpyDeviceToHost);
	cudaMemcpy(cb, cb_y, sizeof(FLOAT) * matrix_dim*size, cudaMemcpyDeviceToHost);

	cudaError_t last_error = cudaGetLastError();
	if (last_error != cudaSuccess) {
		// print the CUDA error message and exit
		printf("CUDA error: %s\n", cudaGetErrorString(last_error));
		cudaDeviceReset();
	}


}

//called in dll_video.m - line 305
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {

	const mxArray *path_Data = prhs[0];
	int path_length;
	char *path;

	path_length = mxGetN(path_Data)+1;
	//path = mxCalloc(path_length, sizeof(char));
	path = (char*)malloc(path_length * sizeof(char));
	mxGetString(path_Data,path,path_length);

	int currentFrame = (int)mxGetScalar(prhs[1]);
	int size = (int)mxGetScalar(prhs[2]);
	int rows = (int)mxGetScalar(prhs[3]);
	int cols = (int)mxGetScalar(prhs[4]);
	int offset_left = (int)mxGetScalar(prhs[5]);
	int offset_right = (int)mxGetScalar(prhs[6]);
	int matrixW = (int)mxGetScalar(prhs[7]);
	int matrixH = (int)mxGetScalar(prhs[8]);

	mwSize dims[3] = {matrixW,matrixH,size};
	mwSize ndim = 3;

	plhs[0] = mxCreateNumericArray(ndim, dims, mxDOUBLE_CLASS, mxREAL);
	plhs[1] = mxCreateNumericArray(ndim, dims, mxDOUBLE_CLASS, mxREAL);
	plhs[2] = mxCreateNumericArray(ndim, dims, mxDOUBLE_CLASS, mxREAL);

	FLOAT *y = mxGetPr(plhs[0]);
	FLOAT *cr = mxGetPr(plhs[1]);
	FLOAT *cb = mxGetPr(plhs[2]);

	int frame_dim = rows*cols;
	int frame_size = frame_dim*1.5;

	int matrix_dim = matrixW * matrixH;

	int offset = frame_size*(currentFrame-1);

//	printf("size :%d\n ",size);
//	printf("rows :%d\n ",rows);
//	printf("cols :%d\n ",cols);
//	printf("offset_left :%d\n ",offset_left);
//	printf("offset_right :%d\n ",offset_right);
//	printf("matrixW :%d\n ",matrixW);
//	printf("matrixH :%d\n ",matrixH);

	char path_yuv[1000];
	strcpy (path_yuv, path);
	strcat (path_yuv, ".yuv");

	unsigned char *video;
	read_video(&video, path_yuv,offset,frame_size*size);

//	read_video(&video, "/media/gregor/HardDrive/LIVE_database/bf_org.yuv", 0,
//			frame_size * size );

	printf("size: %d\n",size);
	printf("rows: %d\n",rows);
	printf("cols: %d\n",cols);
	printf("offset_left: %d\n",offset_left);
	printf("offset_right: %d\n",offset_right);
	printf("matrixW: %d\n",matrixW);
	printf("matrixH: %d\n",matrixH);

	transform_video(y, cr, cb, video, size, currentFrame, rows, cols, offset_left, offset_right, matrixW, matrixH);


	//convert_video_matlab(y, cr, cb, video, rows, cols, size);

//	for (int j=0;j<30;j++){
//	for (int i=0;i<30;i++){
//		printf("%.0f\t", cr[j+720*i]);
//	}
//	printf("\n ");
//	}
	free(video);
	cudaDeviceReset();


}

#endif
