/**
 * @file compare_dual_feature.h
 * @author Gregor Wicklein
 * @date  Oct 12, 2014
 * @brief This methods provide the comparison functions to combine
 * 		  four feature sets, two from original and two from processed video representation.
 */

#ifndef COMPARE_DUAL_FEATURE_H_
#define COMPARE_DUAL_FEATURE_H_

#include "compare_feature.h"

#define max2(a,b) (((a)>(b))?(a):(b))
#define min2(a,b) (((a)<(b))?(a):(b))

/*! Defines comparison function for dual features.
 **/
typedef enum {
	euclid, 		/*!<  sqrt(proc^2 + orig^2)*/
	divide, 		/*!<  positive part of ((proc - orig)/orig)*/
	multiply 		/*!<  positive part of ((proc - orig)/orig)*/
} dual_Comparison;


/**
 * Compares four feature sets (two by each video representation) by a given function and applies thresholding or weighting
 * @param first features from original video representation
 * @param second features from processed video representation
 * @param first features from original video representation
 * @param second features from processed video representation
 * @param comparison function
 * @param thresholding/weighting function
 * @param thresholding/weighting value
 * @return The resulting compared features
 */
VQMFeature compare_dual_feature(VQMFeature orig1, VQMFeature orig2, VQMFeature proc1, VQMFeature proc2, dual_Comparison dcomparison, Comparison comparison,
		Property property, FLOAT property_value);


#endif /* COMPARE_DUAL_FEATURE_H_ */
