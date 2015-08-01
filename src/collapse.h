/**
 * @file collapse.h
 * @author Gregor Wicklein
 * @date  Oct 20, 2014
 * @brief Methods contain functions to perform the comparison and collapsing
 * 		  of features and calculating of quality parameters. The quality parameters
 * 		  are configured according to the VQM General Model.
 *
 */



#ifndef COLLAPSE_H_
#define COLLAPSE_H_
#include <stdio.h>
#include <stdlib.h>
#include "cuda_dll_features.h"
#include "st_collapse.h"
#include "cuda_format.h"
#include "compare_dual_feature.h"

/**
 * Calculates the si_loss quality parameter
 * @param si_orig si features from the original video
 * @param si_orig si features from the processed video
 * @return quality parameter value for si_loss
 */
FLOAT si_loss(VQMFeature si_orig, VQMFeature si_proc);

/**
 * Calculates the hv_loss quality parameter
 * @param hv_mean_orig hv features from the original video
 * @param hvb_mean_orig hvb features from the original video
 * @param hv_mean_proc hv features from the process video
 * @param hvb_mean_proc hvb features from the process video
 * @return quality parameter value for hv_loss
 */
FLOAT hv_loss(VQMFeature hv_mean_orig, VQMFeature hvb_mean_orig,
		VQMFeature hv_mean_proc, VQMFeature hvb_mean_proc);

/**
 * Calculates the hv_gain quality parameter
 * @param hv_mean_orig hv features from the original video
 * @param hvb_mean_orig hvb features from the original video
 * @param hv_mean_proc hv features from the process video
 * @param hvb_mean_proc hvb features from the process video
 * @return quality parameter value for hv_gain
 */
FLOAT hv_gain(VQMFeature hv_mean_orig, VQMFeature hvb_mean_orig,
		VQMFeature hv_mean_proc, VQMFeature hvb_mean_proc);

/**
 * Calculates the color1 quality parameter
 * @param collor_collapse combined color feature
 * @return quality parameter value for color1
 */
FLOAT color1(VQMFeature* collor_collapse);

/**
 * Calculates the si_gain quality parameter
 * @param si_orig si features from the original video
 * @param si_proc si features from the processed video
 * @return quality parameter value for si_gain
 */
FLOAT si_gain(VQMFeature si_orig, VQMFeature si_proc);

/**
 * Calculates the contati quality parameter
 * @param cont_feature features from the original video
 * @param ati_feature  features from the original video
 * @param cont_feature2  features from the process video
 * @param ati_feature2  features from the process video
 * @return quality parameter value for contati
 */
FLOAT contati(VQMFeature cont_feature, VQMFeature ati_feature,
		VQMFeature cont_feature2, VQMFeature ati_feature2);

/**
 * Calculates the color2 quality parameter
 * @param collor_collapse combined color feature
 * @return quality parameter value for color2
 */
FLOAT color2(VQMFeature* collor_collapse);

/**
 * Calculates the collapse quality parameter
 * @param orig_video features from the original video
 * @param proc_video features from the processed video
 * @return quality parameter value for collapse
 */
VQMResult collapse(VQMVideo orig_video, VQMVideo proc_video);

#endif /* COLLAPSE_H_ */
