/**
 * @file cuda_format.h
 * @author Gregor Wicklein
 * @date Aug 24, 2014
 * @brief Definition of container structures for videos, feature-matrices and result vectors
 */


#ifndef CUDA_FORMAT_H_
#define CUDA_FORMAT_H_
#include <stdio.h>
#include <stdlib.h>

//textures to sample pixel from raw video
//texture<unsigned char, 3, cudaReadModeNormalizedFloat> tex_Y;   // Y textur
//texture<unsigned char, 3, cudaReadModeNormalizedFloat> tex_Cr;  // Cb textur
//texture<unsigned char, 3, cudaReadModeNormalizedFloat> tex_Cb;  // Cr textur

//#include <ctime>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <stdio.h>
#include <unistd.h>

#define FLOAT double
//#define FLOAT float

//#define MATLAB_FLOAT mxDOUBLE_CLASS
#define MATLAB_FLOAT mxSINGLE_CLASS

#define FEATURE_SIZE(x) x.width*x.height*x.depth
#define DATA_2D_SIZE(x) x.width*x.height
#define DATA_3D_SIZE(x) x.width*x.height*x.depth

typedef enum {FRAME_DUB, FRAME_INTERPOL, NONE} temporal_scaling;

typedef enum {
	MEAN, STD, MEAN_SLICED
} feature_mode;

typedef struct {
	int width;
	int height;
} VQMDim2;

typedef struct {
	int width;
	int height;
	int depth;
} VQMDim3;

/** @struct VQMScale
 *  @brief Scaling factor of a video
 *  @var VQMFeature::horizontal
 *  Horizontal scaling factor
 *  @var VQMFeature::vertical
 *  Vertical scaling factor
 *  @var VQMFeature::temporal
 *  Temporal scaling factor
 */
typedef struct {
	FLOAT horizonal;
	FLOAT vertical;
	FLOAT temporal;
} VQMScale;


typedef struct {
	VQMDim2 dim;
	int frames;
	int fps;
	int t_slice;
	unsigned char* array;
	long int video_size;
} VQMRawVideo;

/** @struct VQMFeature
 *  @brief 3 dimensional matrix of VQM features
 *  @var VQMFeature::dataDim
 *  Dimension of the matrix
 *  @var VQMFeature::array
 *  Value array, consecutive lines in a frame, consecutive frames
 */
typedef struct {
	VQMDim3 dataDim;
	FLOAT* array;
} VQMFeature;

/** @struct VQMVideo
 *  @brief This structure represents meta-data of a video, including configuration parameters e.g. region of interest and feature vectors
 *  @var VQMVideo::path
 * 	Local file-path to the video (uncompressed, headerless YUV 420p)
 *  @var VQMVideo::videoW
 *  Horizontal resolution of the video
 *  @var VQMVideo::videoH
 *  Vertical resolution of the video
 *  @var VQMVideo::frames
 *  Number of frames in the video
 *  @var VQMVideo::frameRate
 *  Framerate in the video
 *  @var VQMVideo::t_slice
 *  Width of a temporal slice
 *  @var VQMVideo::t_slice_total
 *  Number of a temporal slices in the video
 *  @var VQMVideo::video_size
 *  Size of the Video (in bytes)
 *  @var VQMVideo::left
 *  Region of interest: left border offset
 *  @var VQMVideo::right
 *  Region of interest: right border offset
 *  @var VQMVideo::top
 *  Region of interest: top border offset
 *  @var VQMVideo::bottom
 *  Region of interest: bottom border offset
 *  @var VQMVideo::dataW
 *  Horizontal width of the ROI
 *  @var VQMVideo::dataH
 * 	Vertical width of the ROI
 *  @var VQMVideo::dataT
 *  Temporal width of the ROI per temporal slice
 *  @var VQMVideo::y_std
 *  y feature matrix
 *  @var VQMVideo::hvb_mean
 *  hvb feature matrix
 *  @var VQMVideo::hv_mean
 *  hv feature matrix
 *  @var VQMVideo::si_std
 *  si feature matrix
 *  @var VQMVideo::cb_mean
 *  cb feature matrix
 *  @var VQMVideo::cr_mean
 *  cr feature matrix
 *  @var VQMVideo::ati_std
 *  ati feature matrix
 */
typedef struct {
	char* path;
	VQMDim2 videoDim;
	int frames;
	int fps;
	int t_slice_total;

	int left;
	int right;
	int top;
	int bottom;

	temporal_scaling scaling;

	VQMDim3 dataDim;
	VQMDim3 dataTDim;

	VQMRawVideo raw_video;

	VQMFeature y_std;
	VQMFeature hvb_mean;
	VQMFeature hv_mean;
	VQMFeature si_std;
	VQMFeature cb_mean;
	VQMFeature cr_mean;
	VQMFeature ati_std;

	unsigned char* slice_buffer;

	//for debug purposes
	float* runtime_read;

	int offset;

} VQMVideo;

/** @struct VQMResult
 *	@brief Result vector for VQM for the general model. For a precise definition see NTIA Report 02-392 (Chapter 6.3)
 *  @var VQMResult::si_loss
 *  -0.2097 * Y_si13_8x8_6F_std_12_ratio_loss_below5%_10%
 *  @var VQMResult::hv_loss
 *  +0.5969 * Y_hv13_angle0.225_rmin20_8x8_6F_mean_3_ratio_loss_below5%_mean_square_clip_0.06
 *  @var VQMResult::hv_gain
 *  0.2483 * Y_hv13_angle0.225_rmin20_8x8_6F_mean_3_log_gain_above95%_mean
 *  @var VQMResult::color1
 *  +0.0192 * color_coher_color_8x8_1F_mean_euclid_std_10%_clip_0.6
 *  @var VQMResult::si_gain
 *  -2.3416 * [Y_si13_8x8_6F_std_8_log_gain_mean_mean_clip_0.004 | 0.14 ]
 *  @var VQMResult::contati
 *  +0.0431 * Y_contrast_ati_4x4_6F_std_3_ratio_gain_mean_10%
 *  @var VQMResult::color2
 *  +0.0076 * color_coher_color_8x8_1F_mean_euclid_above99%tail_std} | 0.0
 *  @var VQMResult::VQM
 *  The actual VQM value
 */
typedef struct {
	FLOAT si_loss;FLOAT hv_loss;FLOAT hv_gain;FLOAT color1;FLOAT si_gain;FLOAT contati;FLOAT color2;FLOAT VQM;

} VQMResult;



/**
 * @struct cuda_state
 * @brief Contains CUDA related information which are necessary to process a video slice.
 * This contains the dimensions of the video slice as well as pointers to the array on the
 * CUDA device. After initialization, this state should be reused over all video slices
 * to avoid overhead due to allocating and freeing memory.
 * @var cuda_state::dataW
 * width of the video slice to process
 * @var cuda_state::dataH
 * height of the video slice to process
 * @var cuda_state::dataT
 * depth (number of frames) of the video slice to process
 * @var cuda_state::dataT_ati
 * depth (number of frames) of the video slice including for ati_feature
 * @var cuda_state::d_raw_video
 * array to store raw video data
 * @var cuda_state::d_raw_y
 * array to store raw Y component of a video slice
 * @var cuda_state::d_raw_cb
 * array to store raw Cb component of a video slice
 * @var cuda_state::d_raw_cr
 * array to store raw Cr component of a video slice
 * @var cuda_state::d_y
 * array to store sampled Y component of a video slice
 * @var cuda_state::d_cb
 * array to store sampled Cb component of a video slice
 * @var cuda_state::d_cr
 * array to store sampled Cr component of a video slice
 * @var cuda_state::d_ati_video
 * array to store sampled Y component of a video slice plus the prior frame for ati feature calculation
 * @var cuda_state::d_ati
 * array to store ati features
 * @var cuda_state::d_si
 * array to store si features
 * @var cuda_state::d_row_1
 * array to store interim results from filtering
 * @var cuda_state::d_col_1
 * array to store interim results from filtering
 * @var cuda_state::d_row_2
 * array to store interim results from filtering
 * @var cuda_state::d_col_2
 * array to store interim results from filtering
 * @var cuda_state::d_sobel_Kernel
 * array to store the sobel filter kernel
 * @var cuda_state::d_Ones
 * array to store the smooth filter kernel
 * @var cuda_state::d_result
 * array to store the calculated feature before transferring to host memory
 * @var cuda_state::rmin
 * the rmin value for the hv and hvb features
 * @var cuda_state:: ratio_threshold
 * the ratio_threshold value for the hv and hvb features
 * @var cuda_state::has_Prior_Frame
 * flag which indicates whether the current videoslice has a prior frame or not (because it is the first slice in the video)
 * @var cuda_state::stream1
 * first CUDA stream on which the calculations are executed
 * @var cuda_state::stream2
 * second CUDA stream on which the calculations are executed
 */
typedef struct {
	int dataW;
	int dataH;
	int dataT;
	int dataT_ati;
	unsigned char *d_raw_video;
	cudaArray *d_raw_y;
	cudaArray *d_raw_cb;
	cudaArray *d_raw_cr;
	FLOAT *d_y;
	FLOAT *d_ati_video;
	FLOAT *d_cb;
	FLOAT *d_cr;
	FLOAT *d_ati;
	FLOAT *d_si;
	FLOAT *d_row_1;
	FLOAT *d_col_1;
	FLOAT *d_row_2;
	FLOAT *d_col_2;

	FLOAT *d_Ones;
	FLOAT *d_sobel_Kernel;
	FLOAT *d_result;
	FLOAT rmin;
	FLOAT ratio_threshold;
	bool has_Prior_Frame;
	cudaStream_t stream1, stream2;
} cuda_state;

/**
 * @brief Create and allocate space for  a feature matrix
 * @param Horizontal with of the matrix
 * @param Vertical with of the matrix
 * @param Temporal with of the matrix
 * @return
 */
VQMFeature create_features(VQMDim3 dataDim);

/**
 * @brief Create a feature matrix and fill it with array data
 * @param Horizontal with of the matrix
 * @param Vertical with of the matrix
 * @param Temporal with of the matrix
 * @return
 */
VQMFeature create_features(VQMDim3 dataDim, FLOAT* array);

/**
 * @brief Print the content of a feature vector to the console
 * @param Feature matrix to print
 * @param horizontal limitation for output
 * @param vertical limitation for output
 * @param temporal limitation for output
 */
void print_features(VQMFeature features, int rows, int cols, int time);

/**
 * @brief Print video information to the console
 * @param video to print
 */
void print_VQMVideo(VQMVideo video);


void print_VQMVideo_features(VQMVideo video);


/**
 * @brief vqm result information to the console
 * @param results to print
 */
void print_VQMResult(VQMResult result);

/**
 * @brief Decode and print last CUDA error
 */
void checkCUDAerror();

/**
 * @print a feature or sample matrix (in host memory) to the console
 * @param matrix pointer to the data
 * @param dataW width of the matrix
 * @param dataH height of the matrix
 * @param offsetX number of pixel in horizontal direction which should be printed
 * @param offsetY number of pixel in vertical direction which should be printed
 */
void printMatrix(FLOAT *matrix, int dataW, int dataH, int offsetX,
		int offsetY);

/**
 * @print a feature or sample matrix (in device memory) to the console
 * @param matrix pointer to the data
 * @param dataW width of the matrix
 * @param dataH height of the matrix
 * @param offsetX number of pixel in horizontal direction which should be printed
 * @param offsetY number of pixel in vertical direction which should be printed
 */
void d_printMatrix(FLOAT *d_matrix, int dataW, int dataH, int offsetX,
		int offsetY);

VQMVideo create_video(char* path,VQMDim2 rawResolution , int rawFPS, VQMDim2 targetResolution, int targetFPS);

#endif /* CUDA_FORMAT_H_ */

