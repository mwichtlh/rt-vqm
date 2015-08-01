cmake_minimum_required(VERSION 2.8)

project(RT-VQM)


find_package(CUDA)

if (CUDA_FOUND)
	set(CUDASRCS 
	src/calculate_roi.cu	 
	src/compare_dual_feature.h  
	src/cuda_dll_features.h  
	src/cuda_filter.h	  
	src/main.h
	src/calculate_roi.h		 
	src/compare_feature.cu	 
	src/cuda_dll_model.cu    
	src/cuda_format.cu	  
	src/st_collapse.cu
	src/compare_feature.h	 
	src/cuda_dll_model.h     
	src/cuda_format.h	  
	src/st_collapse.h
	src/collapse.cu		 
	src/cuda_block_static.cu	 
	src/cuda_filter_ati.cu   
	src/cuda_read_video.cu
	src/collapse.h		 
	src/cuda_block_static.h	 
	src/cuda_filter_ati.h    
	src/cuda_read_video.h
	src/compare_dual_feature.cu 
	src/cuda_dll_features.cu	 
	src/cuda_filter.cu       
	src/main.cu 
	)
	set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} --gpu-architecture sm_20 -O3)
	set(CMAKE_CXX_FLAG "-g -Wall")
	CUDA_ADD_EXECUTABLE(RT-VQM ${CUDASRCS})
else()
	message("CUDA not found, please install CUDA SDK")
endif()


