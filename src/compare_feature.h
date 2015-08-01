/**
 *
 * @file compare_feature.h
 * @author Gregor Wicklein
 * @date  Oct 9, 2014
 * @brief This methods provide the comparison functions to combine
 * 		  features from original and processed video representation.
 */

#ifndef COMPARE_FEATURE_H_
#define COMPARE_FEATURE_H_

#define max2(a,b) (((a)>(b))?(a):(b))
#define min2(a,b) (((a)<(b))?(a):(b))



#include "cuda_format.h"

//

/*! Defines comparison function.
 * comments and enums are adapted from compare_feature.m in VQM code.
 **/
typedef enum {
	ratio_gain,    	/*!<  positive part of ((proc - orig)/orig)*/
	ratio_loss,     /*!<  negative part of ((proc - orig)/orig)*/
	log_gain,       /*!<  positive part of log10(proc/orig)*/
	log_loss,       /*!<  negative part of log10(proc/orig)*/
	minus_gain,     /*!<  positive part of (proc - orig)*/
	minus_loss,     /*!<  negative part of (proc - orig)*/
	none            /*!<  no comparison */
} Comparison;

/*! Defines thresholding and weighting modes
 * */
typedef enum{
	MinThreshold,    /*!<Clip 'orig' and 'proc' at the given minimum
					 threshold value before computing the requested
					 comparison function.*/
	MaxThreshold,     /*!<Clip 'orig' and 'proc' at the given maximum
					  threshold value before computing the requested
					  comparison function.*/
	NoThreshold,	  /*!<Perform no thresholding.*/

	Weight	 		 /*!<Weight particular features (multiply with weighting value).*/

}  Property;

/**
 * Compares two Feature sets by a given function and applies thresholding
 * @param features from original video representation
 * @param features from processed video representation
 * @param comparison function
 * @param thresholding function
 * @param thresholding value
 * @return The resulting compared features
 */
VQMFeature compare_feature(VQMFeature orig, VQMFeature proc, Comparison comparison,
		Property property, FLOAT property_value);

#endif /* COMPARE_FEATURE_H_ */
