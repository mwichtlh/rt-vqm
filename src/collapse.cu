
#include "collapse.h"


VQMResult collapse(VQMVideo orig_video, VQMVideo proc_video) {

	//printMatrix(orig_video.si_std.array,orig_video.si_std.dataDim.width,orig_video.si_std.dataDim.height,20,20);
	//printMatrix(proc_video.si_std.array,proc_video.si_std.dataDim.width,proc_video.si_std.dataDim.height,20,20);

	VQMResult result;

	//SI_loss
	result.si_loss = si_loss(orig_video.si_std, proc_video.si_std);

	//HV_loss
	result.hv_loss = hv_loss(orig_video.hv_mean, orig_video.hvb_mean,
			proc_video.hv_mean, proc_video.hvb_mean);

	//HV_gain
	result.hv_gain = hv_gain(orig_video.hv_mean, orig_video.hvb_mean,
			proc_video.hv_mean, proc_video.hvb_mean);

	//Color collapse;
	VQMFeature color_collapse = compare_dual_feature(orig_video.cr_mean,
			orig_video.cb_mean, proc_video.cr_mean, proc_video.cb_mean, euclid,
			none, Weight, 1.5);

	VQMFeature color_collapse2 = copy_VQMFeature(color_collapse);

	//Color1
	result.color1 = color1(&color_collapse);

	//si_gain
	result.si_gain = si_gain(orig_video.si_std, proc_video.si_std);
	//contati
	result.contati = contati(orig_video.y_std, orig_video.ati_std,
			proc_video.y_std, proc_video.ati_std);
	//color2
	result.color2 = color2(&color_collapse2);

	//VQM value
	result.VQM = result.si_loss + result.hv_loss + result.hv_gain
			+ result.color1 + result.si_gain + result.contati + result.color2;

	return result;

}

FLOAT si_loss(VQMFeature si_orig, VQMFeature si_proc) {

	VQMFeature _features = compare_feature(si_orig, si_proc, ratio_loss,
			MinThreshold, 12);

	VQMFeature* features = &_features;
	VQMFeature* below5 = below_5(features);
	VQMFeature* result = _10_percent(below5);
	FLOAT return_value = -0.2097 * result->array[0];

	return return_value;
}

FLOAT hv_loss(VQMFeature hv_mean_orig, VQMFeature hvb_mean_orig,
		VQMFeature hv_mean_proc, VQMFeature hvb_mean_proc) {

	VQMFeature _hv_loss_feature = compare_dual_feature(hv_mean_orig,
			hvb_mean_orig, hv_mean_proc, hvb_mean_proc, divide, ratio_loss,
			MinThreshold, 3);

	VQMFeature*  hv_loss_feature = &_hv_loss_feature;
	VQMFeature* below5 = below_5(hv_loss_feature);
	VQMFeature* mean_feature = mean(below5);
	FLOAT mean_value = mean_feature->array[0];
	mean_value = mean_value * mean_value;
	FLOAT hv_loss = -0.06 + max2(mean_value, 0.06);
	hv_loss = 0.5969 * hv_loss;

	return hv_loss;
}

FLOAT hv_gain(VQMFeature hv_mean_orig, VQMFeature hvb_mean_orig,
		VQMFeature hv_mean_proc, VQMFeature hvb_mean_proc) {
	VQMFeature _hv_gain_feature = compare_dual_feature(hv_mean_orig,
			hvb_mean_orig, hv_mean_proc, hvb_mean_proc, divide, log_gain,
			MinThreshold, 3);

	VQMFeature* hv_gain_feature = &_hv_gain_feature;

	VQMFeature* above95 = above_95(hv_gain_feature);
	VQMFeature* gain_mean_feature = mean(above95);
	FLOAT hv_gain = gain_mean_feature->array[0];
	hv_gain = 0.2483 * hv_gain;

	return hv_gain;
}

FLOAT color1(VQMFeature* collor_collapse) {

	VQMFeature* std_dev_features = std_dev(collor_collapse);
	VQMFeature* _10percent = _10_percent(std_dev_features);
	FLOAT color1 = _10percent->array[0];
	color1 = -0.6 + max2(color1, 0.6);
	color1 = color1 * 0.0192;

	return color1;
}

FLOAT si_gain(VQMFeature si_orig, VQMFeature si_proc) {

	VQMFeature _si_gain_feature = compare_feature(si_orig, si_proc, log_gain,
			MinThreshold, 8);

	VQMFeature* si_gain_feature = &_si_gain_feature;

	si_gain_feature = mean(si_gain_feature);
	FLOAT si_gain_value = si_gain_feature->array[0];
	si_gain_value = min2(0.14,-0.004 + max(0.004,si_gain_value));
	si_gain_value = si_gain_value * -2.3416;

	return si_gain_value;
}

FLOAT contati(VQMFeature cont_feature, VQMFeature ati_feature,
		VQMFeature cont_feature2, VQMFeature ati_feature2) {


	VQMFeature _contati_feature = compare_dual_feature(cont_feature, ati_feature,
			cont_feature2, ati_feature2, multiply, ratio_gain, MinThreshold, 3);

	VQMFeature* contati_feature = &_contati_feature;

	contati_feature = meanFrames(contati_feature);
	contati_feature = _10_percent(contati_feature);
	FLOAT contati_value = contati_feature->array[0];
	contati_value = contati_value * 0.0431;

	return contati_value;
}

FLOAT color2(VQMFeature* collor_colapse) {

	VQMFeature* color2_feature = above_99tail(collor_colapse);
	color2_feature->dataDim.height = color2_feature->dataDim.depth;
	color2_feature->dataDim.depth = 1;

	color2_feature = std_dev(color2_feature);

	FLOAT color2_value = color2_feature->array[0] * 0.0076;

	return color2_value;
}
