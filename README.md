# RT-VQM - a CUDA-accelerated implementation of the NTIA VQM Algorithm

This is an implementation of the Video Quality Metric Algorithm by the NTIA.
The algorithm was accelerated by performing runtime critical operations on a GPU (CUDA).

## License Information

Please note that VQM is open source, but not under GPL. However, it may be used for commercial and non-commercial purposes.
For details on VQM in general, see: http://www.its.bldrdoc.gov/resources/video-quality-research/video-home.aspx

> THE NATIONAL TELECOMMUNICATIONS AND INFORMATION ADMINISTRATION, INSTITUTE FOR TELECOMMUNICATION SCIENCES ("NTIA/ITS") DOES NOT MAKE ANY WARRANTY OF ANY KIND, EXPRESS, IMPLIED OR STATUTORY, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY.  THIS SOFTWARE IS PROVIDED "AS IS."  NTIA/ITS does not warrant or make any representations regarding the use of the software or the results thereof, including but not limited to the correctness, accuracy, reliability or usefulness of the software or the results.  You can use, copy, modify, and redistribute the NTIA/ITS developed software upon your acceptance of these terms and conditions and upon your express agreement to provide appropriate acknowledgments of NTIA's ownership of and development of the software by keeping this exact text present in any copied or derivative works.
> The user of this Software ("Collaborator") agrees to hold the U.S. Government harmless and indemnifies the U.S. Government for all liabilities, demands, damages, expenses, and losses arising out of the use by the Collaborator, or any party acting on its behalf, of NTIA/ITS' Software, or out of any use, sale, or other disposition by the Collaborator, or others acting on its behalf, of products made by the use of NTIA/ITS' Software.
>
> The algorithms contained within the IVQM Software are covered by two U.S. Patents.
> U.S. Patent Number 5,446,492, entitled "A Perception-based Video Quality Measurement System," issued on August 29, 1995.
> U.S. Patent Number 6,496,221, entitled "In-Service Video Quality Measurement System Utilizing an Arbitrary Bandwidth Ancillary Data Channel," issued December 17, 2002.

Please note that there is no warranty for the program, to the extent permitted by applicable law. The program ist provided "as is" and there is no warranty of any kind. 

## Support

This program is a research prototype, not a software project intended to be used in productive systems. Consequently, we cannot provide support for it. However, feel free to fork the code for your own purposes.

## Citing

If you use this code for publications, please cite the following publication.

> M. Wichtlhuber, G. Wicklein, S. Wilk, W. Effelsberg, D. Hausheer: "RT-VQM: Real-Time Video Quality Assessment for Adaptive Video Streaming Using GPUs", ACM Multimedia Systems Conference, 2016.

Please note, that the original authors of VQM also want to be cited.

> Pinson, M., and Wolf, S.: "A New Standardized Method for Objectively Measuring Video Quality", IEEE Transactions on Broadcasting 50, 3 (2004), 312–322.

> Pinson, M., and Wolf, S.: "Application of the NTIA General Video Quality Metric (VQM) to HDTV Quality Monitoring". Workshop on Video Processing and Quality Metrics for Consumer Electronics (VPQM), 2007.

## Requirements

This software has been tested under: Linux (Ubuntu 14.04 LTS) with CUDA 6 and 6.5.

Running the software requires a CUDA-capable graphics card (with compute capability 2.1 or higher)

## Build
Requires Cmake 2.8 or higher. To build the software, run:

`cmake ./`

`make`

# Usage

This implementation requires uncompressed, planar YUV420 (I420) files, without any header information. 
According to the recommendation from the VQM authors, video sequences should be between 5 and 15 seconds.
The LIVE and LIVE MOBILE database provide appropriate material for testing, see: http://live.ece.utexas.edu/research/quality/live_mobile_video.html

To perform a VQM measurement, run:

`./RT-VQM ORIG_PATH PROC_PATH ORIG_WIDTH ORIG_HEIGHT PROC_WIDTH PROC_HEIGHT ORIG_FPS PROC_FPS`

with:

    ORIG_PATH:		Path to the original video (YUV 4:2:0)
    PROC_PATH:		Path to the processed video (YUV 4:2:0)
    ORIG_WIDTH:		Horizontal resolution of the original video 
    ORIG_HEIGHT:		Vertical resolution of the original video 
    PROC_WIDTH:		Horizontal resolution of the processed video 
    PROC_HEIGHT:		Vertical resolution of the processed video 
    ORIG_FPS:		Framerate (fps) of the original video
    PROC_FPS:		Framerate (fps) of the processed video

    (Optional)FORMAT	Output Format:
	    			0: (default) Verbose output
	    			1: CSV output	
	    			2: runtime output (for debugging)
    (Optional)ORIG_NAME	String which will be added to the csv output	
    (Optional)PROC_NAME	Second string which will be added to the csv output	
    (Optional)SCALING	Temporal scaling method for videos with odd framerates:
	    			0: (default) feature duplication
	    			1: frame duplication
	    			2: frame interpolation	
    (Optional)OFFSET	Offset (in frames) where the measurement should be started (default: 0)
    (Optional)LENGTH	Length (in frames) of material which should be analyzed (default: whole video)

An example could be:

`./RT-VQM ./original.yuv ./processed.yuv 1280 720 1280 720 30 30 1 original processed 0 30 600`

Unequal framerates are only supported if they differ by the power of 2 (e.g. 15fps and 30 fps)
Optional parameters are not necessary but if they are applied they have to be placed in the correct order.
