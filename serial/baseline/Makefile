all:
	nvfortran -O3 -Mpreprocess wallclock.c modules.f90 main.f90 -o main_cpu
	nvfortran -O3 -DUSE_BLAS -Mpreprocess wallclock.c modules.f90 main.f90 -L${ESSL_LIB} -lesslsmp -L$(XL_F_LIB) -lxl -o main_cpu_blas
clean:
	rm *.o *.mod main_cpu main_cpu_blas
