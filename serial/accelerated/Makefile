CUDA_HOME ?= /usr/local/cuda
all:
	nvfortran -O3 -Minfo=accel -Mpreprocess -acc -cuda -DUSE_GPU -DUSE_CUDA_DATA wallclock.c nvtx.f90 modules.f90 main.f90 -Mcudalib=cublas -L${CUDA_HOME}/lib64 -lnvToolsExt -o main_gpu
	nvfortran -O3 -Minfo=accel -Mpreprocess -acc -cuda -DUSE_BLAS -DUSE_GPU -DUSE_CUDA_DATA wallclock.c nvtx.f90 modules.f90 main.f90 -Mcudalib=cublas -L${CUDA_HOME}/lib64 -lnvToolsExt -o main_gpu_blas
clean:
	rm *.o *.mod main_gpu main_gpu_blas

