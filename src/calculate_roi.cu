#include "calculate_roi.h"


VQMVideo calculate_roi(VQMVideo video)
{
	VQMVideo roi;
	roi = format_roi(video);
	roi = adjust_requested_sroi(roi, 8, 6);
	roi = add_border(roi, 6);

	return roi;

}


VQMVideo adjust_requested_sroi(VQMVideo video, int hv_size, int extra){

	video.top += extra;
	video.left += extra;
	video.bottom -= extra;
	video.right -= extra;

	while( (video.bottom - video.top + 1)%hv_size != 0)
	{
	    if (video.top < video.videoDim.height - video.bottom)
	    	video.top++;
	    else
	    	video.bottom--;
	}

	while( (video.right - video.left + 1)%hv_size != 0)
	{
	    if (video.left < video.videoDim.width - video.right)
	    	video.left++;
	    else
	    	video.right--;
	}

	return video;

}

//MATLAB: dll_calib_video - 128
VQMVideo add_border(VQMVideo video, int border)
{
	video.top -= border;
	video.left -= border;
	video.bottom += border;
	video.right += border;

	return video;
}



VQMVideo format_roi(VQMVideo video){

	video.top = 0;
	video.left = 0;
	video.bottom = video.videoDim.height;
	video.right = video.videoDim.width;

	if ((video.videoDim.height == 486 || video.videoDim.height == 480) && video.videoDim.width == 720)
	{
	    // NTSC / 525-line
		video.top = 19;
		video.left = 23;
		video.bottom = video.videoDim.width - 18;
		video.right = video.videoDim.width - 22;
	}
	if ( video.videoDim.height == 576 && video.videoDim.width == 720)
	{
	   // PAL / 625-line
		video.top = 15;
		video.left = 23;
		video.bottom = video.videoDim.height - 14;
		video.right = video.videoDim.width - 22;
	}
	if ( video.videoDim.height == 720 && video.videoDim.width == 1280)
	{
		// 720p
		video.top = 7;
		video.left = 17;
		video.bottom = video.videoDim.height - 6;
		video.right = video.videoDim.width - 16;

	}
	if ( video.videoDim.height == 1080 && video.videoDim.width == 1920)
	{
		// 1080p
		video.top = 7;
		video.left = 17;
		video.bottom = video.videoDim.height - 6;
		video.right = video.videoDim.width - 16;
	}





	return video;

}
