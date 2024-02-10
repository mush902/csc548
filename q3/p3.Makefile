lake: lakegpu.cu lake.cu
	nvcc lakegpu.cu lake.cu -o lake -O3 -lm -Wno-deprecated-gpu-targets -I${CUDA_HOME}/include -L${CUDA_HOME}/lib64

