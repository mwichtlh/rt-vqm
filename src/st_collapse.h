/**
 * @file st_collapse.h
 * @author Gregor Wicklein
 * @date  Oct 11, 2014
 * @brief These functions perform the collapsing of features to produce a single quality parameter value.
 *  The collapsing functions are defined in the NTIA Report 02-392, Chapter 5.3
 */

#ifndef ST_COLLAPSE_H_
#define ST_COLLAPSE_H_

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "cuda_format.h"

/**
 * Collapse features frame-wise (1 result per frame).
 * Sort features ascending and return the mean of the lower 5%.
 * @param orig original features
 * @return collapsed features
 */
VQMFeature* below_5(VQMFeature* orig);

/**
 * Collapse all features (1 overall result).
 * Sort features ascending and return the value at 10%.
 * @param orig original features
 * @return collapsed features
 */
VQMFeature* _10_percent(VQMFeature* orig);

/**
 * Collapse all features (1 overall result).
 * Return the average of all features
 * @param orig original features
 * @return collapsed features
 */
VQMFeature* mean(VQMFeature* orig);

/**
 * Collapse features frame-wise (1 result per frame).
 * Return the average of all features on a frame.
 * @param orig original features
 * @return collapsed features
 */
VQMFeature* meanFrames(VQMFeature* orig);

/**
 * Collapse features frame-wise (1 result per frame).
 * Return the standard deviation of all features on a frame.
 * @param orig original features
 * @return collapsed features
 */
VQMFeature* std_dev(VQMFeature* orig);

/**
 * Collapse features frame-wise (1 result per frame).
 * Sort features ascending and return the mean of the upper 5%.
 * @param orig original features
 * @return collapsed features
 */
VQMFeature* above_95(VQMFeature* orig);

/**
 * Collapse features frame-wise (1 result per frame).
 * Sort features ascending, calculate the mean of the upper 1% and subtract the feature value at 99%
 * @param orig original features
 * @return collapsed features
 */
VQMFeature* above_99tail(VQMFeature* orig);

/**
 * Duplicate a feature-set
 * @param orig original feature-set
 * @return duplicated feature set
 */
VQMFeature copy_VQMFeature(VQMFeature orig);

#endif /* ST_COLLAPSE_H_ */
