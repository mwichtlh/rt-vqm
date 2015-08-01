/**
 * @file calculate_roi.h
 * @author Gregor Wicklein
 * @date  14 Oct 2014
 * @brief Methods for calculating the region of interest in a video frame
 *
 */


#ifndef CALCULATE_ROI_H_
#define CALCULATE_ROI_H_

#include <stdio.h>
#include <stdlib.h>
#include "cuda_format.h"


/**
 * Calculates the region of interest (ROI) for a
 * given video according to the VQM standard
 * @param video The original video
 * @return The video with updated ROI
 */
VQMVideo calculate_roi(VQMVideo video);

/**
 * Selects default valid region for different resolutions
 * Valid regions are defined in Matlab Code: dll_default_vr.m
 * @param video video to be updated
 * @return
 */
VQMVideo format_roi(VQMVideo video);


/**
 *Adjust the requested Spatial Region of Interest (SROI) as specified
 *See VQM MATLAB code: adjust_requested_sroi.m for specification
 * @param video video to update
 * @param hv_size Divisor. Width and height must evenly divide by this value
 * @param extra Necessary ROI border width
 * @return updated VQMVideo
 */
VQMVideo adjust_requested_sroi(VQMVideo video, int hv_size, int extra);


/**
 * Extends ROI border by border value
 * @param video video to be updated
 * @param border border width extension
 * @return
 */
VQMVideo add_border(VQMVideo video, int border);


#endif /* CALCULATE_ROI_H_ */
