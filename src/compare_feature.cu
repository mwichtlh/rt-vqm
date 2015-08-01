#include "compare_feature.h"

void ratio_loss_func(VQMFeature orig, VQMFeature proc, VQMFeature comp,
		Property property, FLOAT property_value);

void log_gain_func(VQMFeature orig, VQMFeature proc, VQMFeature comp,
		Property property, FLOAT property_value);

void ratio_gain_func(VQMFeature orig, VQMFeature proc, VQMFeature comp,
		Property property, FLOAT property_value);

VQMFeature compare_feature(VQMFeature orig, VQMFeature proc,
		Comparison comparison, Property property, FLOAT property_value) {

	VQMFeature compared = create_features(orig.dataDim);

	if (comparison == ratio_loss)
		ratio_loss_func(orig, proc, compared, property, property_value);

	if (comparison == log_gain)
		log_gain_func(orig, proc, compared, property, property_value);

	if (comparison == ratio_gain)
		ratio_gain_func(orig, proc, compared, property, property_value);

	return compared;

}

//ratio_loss_comparison
void ratio_loss_func(VQMFeature orig, VQMFeature proc, VQMFeature comp,
		Property property, FLOAT property_value) {
	int i,i2;


	FLOAT orig_value;
	FLOAT proc_value;

	//FLOAT scale = proc.dataDim.depth/orig.dataDim.depth;
	FLOAT scale = 1;

	int w, h, d;
	for (w = 0; w < orig.dataDim.width; w++) {
		for (h = 0; h < orig.dataDim.height; h++) {
			for (d = 0; d < orig.dataDim.depth; d++) {
				i = d * orig.dataDim.width * orig.dataDim.height
						+ h * orig.dataDim.width + w;
				i2 = d*scale * orig.dataDim.width * orig.dataDim.height
						+ h * orig.dataDim.width + w;

				if (property == MinThreshold) {
					orig_value = max2(orig.array[i], property_value);
					proc_value = max2(proc.array[i2], property_value);
				}
				if (property == MaxThreshold) {
					orig_value = min(orig.array[i], property_value);
					proc_value = min(proc.array[i2], property_value);
				}
				if (property == NoThreshold) {
					orig_value = orig.array[i];
					proc_value = proc.array[i2];
				}
				comp.array[i] = min2((proc_value - orig_value) / orig_value, 0);
			}
		}
	}

}

//ratio_gain_comparison
void ratio_gain_func(VQMFeature orig, VQMFeature proc, VQMFeature comp,
		Property property, FLOAT property_value) {
	int i,i2;
	FLOAT orig_value;
	FLOAT proc_value;

	//FLOAT scale = proc.dataDim.depth/orig.dataDim.depth;
	FLOAT scale = 1;

	int w, h, d;
	for (w = 0; w < orig.dataDim.width; w++) {
		for (h = 0; h < orig.dataDim.height; h++) {
			for (d = 0; d < orig.dataDim.depth; d++) {
				i = d * orig.dataDim.width * orig.dataDim.height
						+ h * orig.dataDim.width + w;
				i2 = d*scale * orig.dataDim.width * orig.dataDim.height
						+ h * orig.dataDim.width + w;


				if (property == MinThreshold) {
					orig_value = max2(orig.array[i], property_value);
					proc_value = max2(proc.array[i2], property_value);
				}
				if (property == MaxThreshold) {
					orig_value = min2(orig.array[i], property_value);
					proc_value = min2(proc.array[i2], property_value);
				}
				if (property == NoThreshold) {
					orig_value = orig.array[i];
					proc_value = proc.array[i2];
				}

				comp.array[i] = max2((proc_value - orig_value) / orig_value, 0);
			}
		}
	}

}

//log_gain_comparison
void log_gain_func(VQMFeature orig, VQMFeature proc, VQMFeature comp,
		Property property, FLOAT property_value) {
	int i,i2;

	FLOAT orig_value;
	FLOAT proc_value;

	//FLOAT scale = proc.dataDim.depth/orig.dataDim.depth;
	FLOAT scale = 1;

	int w, h, d;
	for (w = 0; w < orig.dataDim.width; w++) {
		for (h = 0; h < orig.dataDim.height; h++) {
			for (d = 0; d < orig.dataDim.depth; d++) {
				i = d * orig.dataDim.width * orig.dataDim.height
						+ h * orig.dataDim.width + w;
				i2 = d*scale * orig.dataDim.width * orig.dataDim.height
						+ h * orig.dataDim.width + w;
				if (property == MinThreshold) {
					orig_value = max2(orig.array[i], property_value);
					proc_value = max2(proc.array[i2], property_value);
				}
				if (property == MaxThreshold) {
					orig_value = min2(orig.array[i], property_value);
					proc_value = min2(proc.array[i2], property_value);
				}
				if (property == NoThreshold) {
					orig_value = orig.array[i];
					proc_value = proc.array[i2];
				}

				comp.array[i] = max2(log10(proc_value / orig_value), 0);

			}
		}
	}

}
