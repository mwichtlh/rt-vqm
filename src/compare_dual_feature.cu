#include "compare_dual_feature.h"

VQMFeature divide_func(VQMFeature orig1, VQMFeature orig2, FLOAT threshold);
VQMFeature weight_func(VQMFeature orig, FLOAT weight);
VQMFeature euclid_func(VQMFeature orig1, VQMFeature orig2, VQMFeature proc1,
		VQMFeature proc2);
VQMFeature multiply_func(VQMFeature orig1, VQMFeature orig2, FLOAT threshold);

VQMFeature compare_dual_feature(VQMFeature orig1, VQMFeature orig2,
		VQMFeature proc1, VQMFeature proc2, dual_Comparison dcomparison,
		Comparison comparison, Property property, FLOAT property_value) {

	VQMFeature orig;
	VQMFeature proc;
	VQMFeature result;

	if (dcomparison == multiply) {
		FLOAT threshold = property_value;
		orig = multiply_func(orig1, orig2, threshold);
		proc = multiply_func(proc1, proc2, threshold);
		result = compare_feature(orig, proc, comparison, NoThreshold, 3);
	}

	if (dcomparison == divide) {
		FLOAT threshold = property_value;
		proc = divide_func(proc1, proc2, threshold);
		orig = divide_func(orig1, orig2, threshold);
		result = compare_feature(orig, proc, comparison, NoThreshold, 3);
	}

	if (dcomparison == euclid) {

		if (property == Weight) {
			FLOAT weight = property_value;
			orig2 = weight_func(orig2, weight);
			proc2 = weight_func(proc2, weight);
		}

		result = euclid_func(orig1, orig2, proc1, proc2);
	}

	return result;
}

//division function
VQMFeature divide_func(VQMFeature orig1, VQMFeature orig2, FLOAT threshold) {

	VQMFeature orig = create_features(orig1.dataDim);
	int size = FEATURE_SIZE(orig1.dataDim);
	int i;
	for (i = 0; i < size; i++)
		orig.array[i] = max2(orig1.array[i],threshold)
				/ max2(orig2.array[i],threshold);

	return orig;
}

//multiplication function
VQMFeature multiply_func(VQMFeature orig1, VQMFeature orig2, FLOAT threshold) {

	VQMFeature orig = create_features(orig1.dataDim);
	int size = FEATURE_SIZE(orig1.dataDim);

	int i;
	for (i = 0; i < size; i++)
		orig.array[i] = max2(orig1.array[i],threshold)
				* max2(orig2.array[i],threshold);

	return orig;
}

//weight feature set by a given weight
VQMFeature weight_func(VQMFeature orig, FLOAT weight) {

	VQMFeature result = create_features(orig.dataDim);
	int size = FEATURE_SIZE(orig.dataDim);
	int i;
	for (i = 0; i < size; i++)
		result.array[i] = orig.array[i] * weight;

	return result;
}

//euclidean comparison function
VQMFeature euclid_func(VQMFeature orig1, VQMFeature orig2, VQMFeature proc1,
		VQMFeature proc2) {

	VQMFeature result = create_features(orig1.dataDim);
	int i, i2;

	FLOAT proc_depth = proc1.dataDim.depth;
	FLOAT orig_depth = orig1.dataDim.depth;
	FLOAT scale = proc_depth/orig_depth;

	FLOAT x, y;
	int w, h, d;
	for (w = 0; w < orig1.dataDim.width; w++) {
		for (h = 0; h < orig1.dataDim.height; h++) {
			for (d = 0; d < orig1.dataDim.depth; d++) {
				i = d * orig1.dataDim.width * orig1.dataDim.height
						+ h * orig1.dataDim.width + w;
				i2 = d * scale * orig1.dataDim.width * orig1.dataDim.height
						+ h * orig1.dataDim.width + w;


				//int d2 = d*scale;
				//if(h == 1 && w == 1)
				//	printf("%d - %d - %f\n", d,d2,scale);

				x = orig1.array[i] - proc1.array[i2];
				y = orig2.array[i] - proc2.array[i2];
				x = pow(x, 2);
				y = pow(y, 2);
				result.array[i] = sqrt(x + y);
			}
		}
	}

//		FLOAT x, y;
//		for (i = 0; i < size; i++){
//					x = orig1.array[i] - proc1.array[i];
//					y = orig2.array[i] - proc2.array[i];
//					x = pow(x, 2);
//					y = pow(y, 2);
//					result.array[i] = sqrt(x + y);
//		}

	return result;
}

