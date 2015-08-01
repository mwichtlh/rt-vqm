#include "main.h"


double print_time(int format);

clock_t _begin;
clock_t _now;
clock_t _end;
double _elapsed_secs;
double _total_elapsed_secs;

char* orig_path;
char* proc_path;
int orig_x_res;
int orig_y_res;
int proc_x_res;
int proc_y_res;
int orig_fps;
int proc_fps;
int format = 1;

double runtime_video1;
double runtime_video2;
double runtime_collapse;

const char* orig_name = "original";
const char* proc_name = "processed";

int offset = 0;
int length = 0;
int scaling_parameter=0;

int parse_input_arguments(int argc, char *argv[]);

int main(int argc, char *argv[]) {

	format = 1;
	if(parse_input_arguments(argc, argv))
		return 0;


	if (format==0)
		printf("start\n");
	_begin = clock();

	VQMVideo orig_video;
	VQMVideo proc_video;
	VQMResult result;

	temporal_scaling scaling = NONE;

	if(scaling_parameter == 1)
		scaling = FRAME_DUB;
	if(scaling_parameter == 2)
		scaling = FRAME_INTERPOL;


	if (format==0)
		printf("start original\n");
	orig_video = run_feature_calculation(orig_path, orig_x_res, orig_y_res, orig_fps, orig_x_res, orig_y_res, orig_fps,offset,length, scaling);
	if (format==0){
		print_VQMVideo(orig_video);
	}
	runtime_video1 = print_time(format);


	if (format==0)
		printf("start processed\n");
	proc_video = run_feature_calculation(proc_path, orig_x_res, orig_y_res, orig_fps,  proc_x_res, proc_y_res, proc_fps,offset,length, scaling);
	if (format==0){
			print_VQMVideo(proc_video);
			printf("start collapsing\n");
		}
	runtime_video2 = print_time(format);

	//print_VQMVideo_features(orig_video);
	//print_VQMVideo_features(proc_video);

	result = collapse(orig_video, proc_video);
	runtime_collapse = print_time(format);



	float overall_runtime = runtime_video1 + runtime_video2 + runtime_collapse;

	if (format==0){
		print_VQMResult(result);
	}

	if (format==1)
		printf("%s,%s,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f\n", orig_name,
				proc_name, result.si_loss, result.hv_loss, result.hv_gain,
				result.color1, result.si_gain, result.contati, result.color2,
				result.VQM);
	if (format == 2)
		printf("%s,%s,%.12f,%.12f,%.12f,%.12f,%.12f,%.12f,Runtime\n", orig_name,
				proc_name, *orig_video.runtime_read, runtime_video1-*orig_video.runtime_read, *proc_video.runtime_read ,
				runtime_video2-*proc_video.runtime_read, runtime_collapse, overall_runtime);
	if (format == 3)
		printf("%s\t%s\t%.8f\t%.8f\t%.8f\t%.8f\t%.8f\t%.8f\t%.8f\t%.8f\n", orig_name,
				proc_name, result.si_loss, result.hv_loss, result.hv_gain,
				result.color1, result.si_gain, result.contati, result.color2,
				result.VQM);



	return 0;
}






double print_time(int print) {
	_now = clock();
	_elapsed_secs = double(_now - _end) / CLOCKS_PER_SEC;
	if(print==0)
		printf("time: %.2f sec.\n", _elapsed_secs);
	_end = _now;
	_total_elapsed_secs = double(_end - _begin) / CLOCKS_PER_SEC;
	if(print==0)
		printf("total time: %.2f sec.\n\n", _total_elapsed_secs);
	return _elapsed_secs;
}


int parse_input_arguments(int argc, char *argv[])
{
	if(argc < 9)
	{
		printf("\n");
		printf("\tToo few Arguments. The call has to be the following format:\n");
		printf("\tcuda_vqm ORIG_PATH PROC_PATH ORIG_WIDTH ORIG_HEIGHT PROC_WIDTH PROC_HEIGHT ORIG_FPS PROC_FPS\n");
		printf("\twith:\n");
		printf("\tORIG_PATH:\tPath to the original video (YUV 4:2:0)\n");
		printf("\tPROC_PATH:\tPath to the processed video (YUV 4:2:0)\n");
		printf("\tORIG_WIDTH:\tHorizontal resolution of the original video \n");
		printf("\tORIG_HEIGHT:\tVertical resolution of the original video \n");
		printf("\tPROC_WIDTH:\tHorizontal resolution of the processed video \n");
		printf("\tPROC_HEIGHT:\tVertical resolution of the processed video \n");
		printf("\tORIG_FPS:\tFramerate (fps) of the original video\n");
		printf("\tPROC_FPS:\tFramerate (fps) of the processed video\n");
		printf("\n");
		return 1;
	}


	orig_path = argv[1];
	proc_path = argv[2];
	orig_x_res = atoi(argv[3]);
	orig_y_res = atoi(argv[4]);
	proc_x_res = atoi(argv[5]);
	proc_y_res = atoi(argv[6]);
	orig_fps = atoi(argv[7]);
	proc_fps = atoi(argv[8]);
	if(argc >= 10)
		format = atoi(argv[9]);
	if(argc >= 11)
		orig_name = argv[10];
	if(argc >= 12)
		proc_name = argv[11];
	if(argc >= 13)
		scaling_parameter = atoi(argv[12]);
	if(argc >= 14)
		offset = atoi(argv[13]);
	if(argc >= 15)
		length = atoi(argv[14]);




	return 0;
}


