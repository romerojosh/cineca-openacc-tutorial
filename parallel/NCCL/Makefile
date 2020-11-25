CUDA_HOME ?= /usr/local/cuda
all:
	mpif90 -O3 -Minfo=accel -Mpreprocess -acc -cuda -gpu=cuda10.2 -DUSE_GPU -DUSE_CUDA_DATA wallclock.c nvtx.f90 modules.f90 main.f90 -L${CUDA_HOME}/lib64 -lnvToolsExt  -cudalib=nccl -o main_gpu_nccl
clean:
	rm *.o *.mod main_gpu_nccl

