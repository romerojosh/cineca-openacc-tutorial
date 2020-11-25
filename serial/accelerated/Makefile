CUDA_HOME ?= /usr/local/cuda
all:
	nvfortran -O3 -DUSE_BLAS -Mpreprocess wallclock.c nvtx.f90 modules.f90 main.f90 -L${ESSL_LIB} -lesslsmp -L$(XL_F_LIB) -lxl -L${CUDA_HOME}/lib64 -lnvToolsExt -o main_cpu_blas
	nvfortran -O3 -Minfo=accel -Mpreprocess -acc -gpu=cuda10.2 wallclock.c nvtx.f90 modules.f90 main.f90 -L${CUDA_HOME}/lib64 -lnvToolsExt -o main_gpu_v1
	nvfortran -O3 -Minfo=accel -Mpreprocess -acc -cuda -gpu=cuda10.2 -DUSE_CUDA_DATA wallclock.c nvtx.f90 modules.f90 main.f90 -L${CUDA_HOME}/lib64 -lnvToolsExt -o main_gpu_v2
	nvfortran -O3 -Minfo=accel -Mpreprocess -acc -cuda -gpu=cuda10.2 -DUSE_CUBLAS -DUSE_CUDA_DATA wallclock.c nvtx.f90 modules.f90 main.f90 -Mcudalib=cublas -L${CUDA_HOME}/lib64 -lnvToolsExt -o main_gpu_v2_blas
clean:
	rm *.o *.mod main_cpu_blas main_gpu_v1 main_gpu_v2 main_gpu_v2_blas

