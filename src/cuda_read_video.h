/**
 * @file cuda_read_video.h
 * @author Gregor Wicklein
 * @date  Aug 16, 2014
 * @brief Contains functions to read raw video material (YUV) from media and convert it into a floating point representation (sampling)
 */

#ifndef CUDA_READ_VIDEO_H_
#define CUDA_READ_VIDEO_H_

#include "cuda_format.h"


#define DIM_SAMPLE_X 8
#define DIM_SAMPLE_Y 8

/**
 * Returns the size (byte) of a file
 * @param path Path to the file
 * @return size of the file (byte)
 */
long int get_video_size(char *path);

/**
 * Read a given file to an char array (inc. memory allocation)
 * @param video destination pointer for char array
 * @param path path Path to the file
 * @param offset offset from file-head to start reading
 * @param size number of bytes to read from file
 */
void read_video(unsigned char **video, char *path, long int offset, int size);

/**
 * This video samples a temporal slice of the video for further processing
 * @param video video to read slice from
 * @param state current CUDA state
 * @param t_slice number (position) of the slice to read
 */


void read_video(VQMVideo* video);


void prepare_next_slice(VQMVideo video , cuda_state *state, int t_slice);



void char_printMatrix(unsigned char *matrix, int dataW, int dataH, int offsetX,
		int offsetY);

void printMatrix(FLOAT *matrix, int dataW, int dataH, int offsetX,
		int offsetY);


#ifdef MATLAB
void convert_video(FLOAT *y, FLOAT *cr, FLOAT *cb, unsigned char *video, int dataW, int dataH, int dataT);


//matrices in matlab have different orientation
void convert_video_matlab(FLOAT *y, FLOAT *cr, FLOAT *cb, unsigned char *video, int dataW, int dataH, int dataT);
#endif





#endif /* CUDA_READ_VIDEO_H_ */
