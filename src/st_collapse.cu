#include "st_collapse.h"

bool debug_output = true;

int desc(const void* value1, const void* value2) {
	if (*(const FLOAT*) value1 < *(const FLOAT*) value2)
		return -1;
	return *(const FLOAT*) value1 > *(const FLOAT*) value2;
}

int asc(const void* value1, const void* value2) {
	if (*(const FLOAT*) value1 > *(const FLOAT*) value2)
		return -1;
	return *(const FLOAT*) value1 < *(const FLOAT*) value2;
}

FLOAT average(FLOAT* values, int length) {

	FLOAT average = 0;
	int i;
	for (i = 0; i < length; i++) {
		average += values[i];
	}
	return average / length;
}



VQMFeature copy_VQMFeature(VQMFeature orig) {
	VQMFeature returnValue;

	returnValue.dataDim.width = orig.dataDim.width;
	returnValue.dataDim.height = orig.dataDim.height;
	returnValue.dataDim.depth = orig.dataDim.depth;

	int size = orig.dataDim.width* orig.dataDim.height * orig.dataDim.depth;

	returnValue.array = (FLOAT*) malloc(size * sizeof(FLOAT));
	memcpy(returnValue.array, orig.array, size * sizeof(FLOAT));

	return returnValue;
}

VQMFeature* below_5(VQMFeature* orig) {

	int size = orig->dataDim.width* orig->dataDim.height;

	FLOAT percentile = 0.05;
	int want = 1 + roundf((size-1) * percentile);

	int i;
	for (i = 0; i < orig->dataDim.depth; i++) {
		qsort(orig->array + i * size, size, sizeof(FLOAT), desc);
		orig->array[i] = average(orig->array + i * size, want);
	}

	orig->dataDim.width = 1;
	orig->dataDim.height = 1;

	return orig;
}

VQMFeature* _10_percent(VQMFeature* orig) {

	int size = orig->dataDim.depth;

	FLOAT percentile = 0.1;
	int want = 1 + roundf((orig->dataDim.depth-1) * percentile);


	qsort(orig->array, size, sizeof(FLOAT), desc);

	orig->array[0] = orig->array[want-1];

	orig->dataDim.width = 1;
	orig->dataDim.height = 1;
	orig->dataDim.depth = 1;

	return orig;
}

VQMFeature* mean(VQMFeature* orig) {

	int size = orig->dataDim.width* orig->dataDim.height * orig->dataDim.depth;

	orig->array[0] = average(orig->array, size);

	orig->dataDim.width = 1;
	orig->dataDim.height = 1;
	orig->dataDim.depth = 1;

	return orig;
}

VQMFeature* meanFrames(VQMFeature* orig) {

	int size = orig->dataDim.width* orig->dataDim.height;

	orig->dataDim.width = 1;
	orig->dataDim.height = 1;

	int i;
	for(i=0;i<orig->dataDim.depth;i++)
	{
		orig->array[i] = average(orig->array+size*i, size);
	}

	return orig;
}


VQMFeature* std_dev(VQMFeature* orig) {


	int size = orig->dataDim.width* orig->dataDim.height;
	int t, i;
	FLOAT m, sum_s, s;

	orig->dataDim.width = 1;
	orig->dataDim.height = 1;

	for (t = 0; t < orig->dataDim.depth; t++) {
		sum_s = 0;
		m = average(orig->array+(t * size), size);
		for (i = 0; i < size; i++) {
			s = orig->array[t * size + i] - m;
			sum_s += s * s;
		}
		orig->array[t] = sqrt(sum_s/(size-1));
	}
	return orig;
}




VQMFeature* above_95(VQMFeature* orig) {

	int size = orig->dataDim.width* orig->dataDim.height;

	orig->dataDim.width = 1;
	orig->dataDim.height = 1;
	FLOAT percentile = 0.95;
	int want = 1 + roundf((size-1) * percentile);


	int i;
	for (i = 0; i < orig->dataDim.depth; i++) {
		qsort(orig->array + i * size, size, sizeof(FLOAT), desc);
		orig->array[i] = average(orig->array + i * size + want - 1,
				size - want + 1);
	}

	return orig;

}

VQMFeature* above_99tail(VQMFeature* orig) {



	int size = orig->dataDim.width* orig->dataDim.height;

	orig->dataDim.width = 1;
	orig->dataDim.height = 1;

	FLOAT percentile = 0.99;
	int want = 1 + roundf((size-1) * percentile);


	int i;
	for (i = 0; i < orig->dataDim.depth; i++) {
		qsort(orig->array + i * size, size, sizeof(FLOAT), desc);
		orig->array[i] = average(orig->array + i * size + want - 1,
				size - want + 1) - orig->array[want+ i * size-1];

	}

	return orig;

}
