# cineca-openacc-tutorial
## Introduction
This repository contains example code using OpenACC and CUDA Fortran to port a simple CG solver from CPU to GPU, in serial and parallel. Features covered are OpenACC, CUDA Fortran, CUBLAS, CUDA-aware MPI, and NCCL.

## Getting started on the Marconi 100 system
To start, get an interactive node on the Marconi 100 system using the following command:
```
srun -t 00:30:00 --ntasks-per-node=4 --nodes=1 --gres=gpu:4 --cpus-per-task 32 --mem=230000MB -A <account> -p m100_usr_prod --pty bash
```
This will request a single node with 4 GPUs with each task bound to a fair subset of CPUs.

Also, load required modules for this example using the following command:
```
module load hpc-sdk cuda/10.2 xl essl
```

## Serial Example
### Baseline CPU code
Navigate to the `serial/baseline` directory to view our initial CG code for CPU. `main.f90` contains the allocation of `A`, `b`, and `x` arrays and timing loop. The initialization and CG solver routines are located in `modules.f90`. The main code of interest for this example are all the subroutines/functions in the `CG_routines` module: `symmatvec`, `dot`, and `RunCG`. 

Building the code using `make` will produce two executables: `main_cpu` and `main_cpu_blas`. `main_cpu` compiles using hand-coded loops for `symmatvec` and `dot` while `main_cpu_blas` links in BLAS libraries. In this case, we link in the multi-threaded ESSL library.

Running these on the node produces the following results:
```
$ ./main_cpu
 Running Warmup....
 Iteration             1  residual:     11.63451702489179
 Iteration             2  residual:    1.9845558837315612E-002
 Iteration             3  residual:    3.1534373809098950E-005
 Iteration             4  residual:    5.0153211704420045E-008
 Iteration             5  residual:    7.9933529706510668E-011
 Iteration             6  residual:    1.2717273831867764E-013
 Iteration             7  residual:    2.9319987577518820E-016
 Running Test....
 Iteration             1  residual:     11.63451702489179
 Iteration             2  residual:    1.9845558837315612E-002
 Iteration             3  residual:    3.1534373809098950E-005
 Iteration             4  residual:    5.0153211704420045E-008
 Iteration             5  residual:    7.9933529706510668E-011
 Iteration             6  residual:    1.2717273831867764E-013
 Iteration             7  residual:    2.9319987577518820E-016
 Wall time:     1.783379793167114
 x:   -1.0226403871012114E-005   5.0902512156439134E-005
   0.3334710494981544
```
Using the BLAS build with ESSL and 4 threads gives a much better CPU result.
```
[jromero0@r215n02 baseline]$ OMP_NUM_THREADS=4 ./main_cpu_blas
 Running Warmup....
 Iteration             1  residual:     11.63451702489179
 Iteration             2  residual:    1.9845558837315768E-002
 Iteration             3  residual:    3.1534373809098719E-005
 Iteration             4  residual:    5.0153211704417312E-008
 Iteration             5  residual:    7.9933529729027946E-011
 Iteration             6  residual:    1.2717627790231455E-013
 Iteration             7  residual:    5.5767086895964766E-016
 Running Test....
 Iteration             1  residual:     11.63451702489179
 Iteration             2  residual:    1.9845558837315768E-002
 Iteration             3  residual:    3.1534373809098719E-005
 Iteration             4  residual:    5.0153211704417312E-008
 Iteration             5  residual:    7.9933529729027946E-011
 Iteration             6  residual:    1.2717627790231455E-013
 Iteration             7  residual:    5.5767086895964766E-016
 Wall time:    0.1647520065307617
 x:   -1.0226403871012148E-005   5.0902512156439032E-005
   0.3334710494981543
```

Continue on to the next section to see how to approach porting this program to GPUs.

### Porting to GPU
In this section, we will cover a sequence of steps to port this simple code to GPU with OpenACC and CUDA Fortran. While this code is simple, this sequence of steps is fairly general, and can be applied to other more complex code porting efforts. The final ported code is in the `serial/accelerated` directory if you want to skip forward and try that out.

#### Step 1: Profile
Profiling is a useful tool to guide porting and optimization efforts by enabling you to get a better understanding of where time is being spent in your code and set priorities for what to port. In complex codes, you may not be able to port everything to GPU so it is good to spend development time porting the most costly sections first. 
Nsight Systems is a profiling tool provided by NVIDIA that can be used for this task. While it is a profiling tool for GPUs, it can be used to initially profile CPU codes with the use of NVTX ranges. NVTX ranges can be added to the code to annotate segments of runtime with a name, and these ranges will show up in the resulting view of the profile.
To add NVTX ranges to the baseline CPU code, copy the `nvtx.f90` file from the `accelerated` directory and use it to add ranges to the CPU code with `nvtxStartRange` and `nvtxEndRange` subroutines. Look for examples in the `accelerated` directory source to see how this is done. When you have ranges added to the baseline code, recompile. You will need to add a link to the `libnvToolsExt.so` library, see the `Makefile` in `accelerated` for an example of how this is done. and run using the profiler using the following command:
```
$ OMP_NUM_THREADS=4 nsys profile -o cpu_blas_profile ./main_cpu_blas
```
You can copy the resulting `cpu_blas_profile.qdrep` file to your local system and visualize using the Nsight System GUI, available for download on NVIDIA's website.

To use our sample solution, go to the `accelerated` directory and type `make`. Run the `main_cpu_blas` executable in that directory to generate a CPU profile with NVTX ranges.

#### Step 2: Port loops using OpenACC
The next step is to start adding some OpenACC directives to port loops to GPU. In this example, there are a total of 5 loops to port (3 in `RunCG`, 1 in `symmatvec` and 1 in `dot`). See if you can apply OpenACC parallel loop directives to port these loops. When you are done, recompile with code with the `-acc` compiler flags to enable OpenACC directives. Optionally, add the `-Minfo=accel` flag to get more verbose output about how the compiler is applying OpenACC to the ported loops. 

To use our sample solution for this step, go to the `accelerated` directory and run the `main_gpu_v1` binary. Running this on the node should generate the following output:
```
$ ./main_gpu_v1
 Running Warmup....
 Iteration             1  residual:     11.63451702489180
 Iteration             2  residual:    1.9845558837315817E-002
 Iteration             3  residual:    3.1534373809098760E-005
 Iteration             4  residual:    5.0153211704417186E-008
 Iteration             5  residual:    7.9933529702152848E-011
 Iteration             6  residual:    1.2717205406485772E-013
 Iteration             7  residual:    2.0605670154693845E-016
 Running Test....
 Iteration             1  residual:     11.63451702489180
 Iteration             2  residual:    1.9845558837315817E-002
 Iteration             3  residual:    3.1534373809098760E-005
 Iteration             4  residual:    5.0153211704417186E-008
 Iteration             5  residual:    7.9933529702152848E-011
 Iteration             6  residual:    1.2717205406485772E-013
 Iteration             7  residual:    2.0605670154693845E-016
 Wall time:     1.582125902175903
 x:   -1.0226403871012107E-005   5.0902512156439025E-005
   0.3334710494981544
```
We see that with just applying OpenACC to the loops only with no other changes, the wall time is no better than the serial CPU baseline without BLAS and significantly worse than the baseline with BLAS! Run Nsight Systems to generate a profile of this case to investigate what is going on. What you will see is a lot of data movement between the GPU and CPU taking up a significant portion of the runtime.

#### Step 3: Data Management
With the loops ported to run on the GPU, the next step is to deal with memory movement. Without any explicit instructions, OpenACC will generate many host-to-device and device-to-host memory transfers, at worse before and after every OpenACC kernel. We can handle data in two ways:
1. Using OpenACC Data directives
2. CUDA Fortran array attributes (device and/or managed)
Pick a method and try to add code to explictly deal with memory movement in your code. The main arrays you need to consider are `A`, `b`, `x` that are allocated in `main.f90`, as well as the temporary arrays created in `modules.f90` in `RunCG`: `Ax`, `r`, and `p`.
After making your changes, recompile and run the code and see if the performance improves. If you choose to use CUDA Fortran arrays, you will need to add the `-cuda` flag to your compilation line to enable CUDA Fortran features.

To use our sample solution for this step, go to the `accelerated` directory and run the `main_gpu_v2` binary. This binary was compiled using CUDA Fortran managed memory to handle data transfers (search for `USE_CUDA_DATA` macros in the source to see the lines that were changed to enable this). To use OpenACC Data directives, compile the sample with `-DUSE_ACC_DATA` instead of `-DUSE_CUDA_DATA` (and check out related lines in the source). Running this binary on the node should generate the following output:
```
$ ./main_gpu_v2
 Running Warmup....
 Iteration             1  residual:     11.63451702489180
 Iteration             2  residual:    1.9845558837315817E-002
 Iteration             3  residual:    3.1534373809098760E-005
 Iteration             4  residual:    5.0153211704417186E-008
 Iteration             5  residual:    7.9933529702152848E-011
 Iteration             6  residual:    1.2717205406485772E-013
 Iteration             7  residual:    2.0605670154693845E-016
 Running Test....
 Iteration             1  residual:     11.63451702489180
 Iteration             2  residual:    1.9845558837315817E-002
 Iteration             3  residual:    3.1534373809098760E-005
 Iteration             4  residual:    5.0153211704417186E-008
 Iteration             5  residual:    7.9933529702152848E-011
 Iteration             6  residual:    1.2717205406485772E-013
 Iteration             7  residual:    2.0605670154693845E-016
 Wall time:    2.0910978317260742E-002
 x:   -1.0226403871012107E-005   5.0902512156439025E-005
   0.3334710494981544
``` 

With improved data management, we achieve a much faster result, with good speedup on the GPU for this problem over the CPU results.

#### Step 4: Using libraries
Now we will look into porting the code path using BLAS by replacing BLAS calls with cuBLAS, the GPU-accelerated BLAS library provided by NVIDIA. Luckily, the `nvfortran` compiler provides a simple to use `cublas` module with overloaded interfaces for common BLAS routines, like `dgemv` and `ddot` used in this sample. All that is required is to add the line `use cublas` to the subroutine/function using the BLAS call, and ensuring that GPU buffers are passed as arguments to the routine. These can either be `device` or `managed` CUDA Fortran arrays or GPU pointers passed to the routine in OpenACC using the `acc host_data use_device(...)` directive. 
To use our sample solution for this step, go to the `accelerated` directory and run the `main_gpu_v2_blas` binary. Look at the `symmatvec` and `dot` subroutines to see the required changes. Running this binary on the node should generate the following output:
```
$ ./main_gpu_v2_blas
 Running Warmup....
 Iteration             1  residual:     11.63451702489180
 Iteration             2  residual:    1.9845558837315827E-002
 Iteration             3  residual:    3.1534373809098875E-005
 Iteration             4  residual:    5.0153211704417180E-008
 Iteration             5  residual:    7.9933529702319292E-011
 Iteration             6  residual:    1.2717208028678500E-013
 Iteration             7  residual:    2.1006346441582599E-016
 Running Test....
 Iteration             1  residual:     11.63451702489180
 Iteration             2  residual:    1.9845558837315827E-002
 Iteration             3  residual:    3.1534373809098875E-005
 Iteration             4  residual:    5.0153211704417180E-008
 Iteration             5  residual:    7.9933529702319292E-011
 Iteration             6  residual:    1.2717208028678500E-013
 Iteration             7  residual:    2.1006346441582599E-016
 Wall time:    2.0750045776367188E-002
 x:   -1.0226403871012100E-005   5.0902512156439012E-005
   0.3334710494981544
```
Using cuBLAS doesn't have a huge impact on performance, indicating that our OpenACC loops are performing as well as native libraries for the input matrix and arrays sizes in our sample. In general however, it is good to look for existing GPU library implementations of existing computations in your code before implementing things yourself.

## Parallel Example
Inside the `parallel` directory, there are examples of parallel versions of this simple CG code. To distribute the problem, the `A` matrix is split between processes into row-slabs (which can also be interpreted as column slabs in row-major data layout due to the symmetry of A). Then, when a matrix-vector multiplication is performed, each rank multiplies its set of matrix columns by a segment of the vector, producing a partial result. These partial results are then combined using an allreduce operation. As such, this allreduce operation is the only communucation routine we need to consider in this example.

The `parallel` folder contains two subdirectories, `MPI` and `NCCL`, which contain implementations using MPI and NCCL respectively. 

### Using MPI
Navigate into the `MPI` directory and take a look at code in `main.f90` and `RunCG` in `modules.f90`. In `main.f90`, you will find initialization code related to MPI that you will be familiar with from CPU programs, but also a section of code related to setting the GPU device. In this section, we assign GPU devices to ranks using `cudaSetDevice` and also `acc_set_device_num`, which covers the device that will be used by both CUDA Fortran and also OpenACC. In this example, we set the device for each process to the `local_rank`, which is a fairly typical set up. 

After this, we can take a look at the `MPI_Allreduce` calls in `RunCG` to combine the partial matrix-vector sums across ranks. Since we are using a CUDA-aware MPI implementation, we can simply pass GPU arrays directly to the `MPI_Allreduce` call. For OpenACC, this is achieved by using the `acc host_data use_device(Ax)` directive, but here we are using `managed` arrays from CUDA Fortran. 

Type `make` to compile the MPI parallelized program, `main_gpu_mpi`. To run on the M100 node, use the following command:
```
$ mpirun -np <N>  --map-by ppr:2:socket ./main_gpu_mpi
```
where `N` can be 1, 2, or 4. The `--map-by` argument is useful to ensure that processes are bound to the appropriate CPU socket/cores for best performance. This setting is requesting 2 processes per socket.

Running for `N` = 1, 2 and 4 results in the following:
```
$ mpirun -np 1 --map-by ppr:2:socket ./main_gpu_mpi
 Running parallel version on            1 ranks...
 rank            0 using GPU            0
 Running Warmup....
 Iteration             1  residual:     11.63451702489180
 Iteration             2  residual:    1.9845558837315817E-002
 Iteration             3  residual:    3.1534373809098760E-005
 Iteration             4  residual:    5.0153211704417186E-008
 Iteration             5  residual:    7.9933529702152848E-011
 Iteration             6  residual:    1.2717205406485772E-013
 Iteration             7  residual:    2.0605670154693845E-016
 Running Test....
 Iteration             1  residual:     11.63451702489180
 Iteration             2  residual:    1.9845558837315817E-002
 Iteration             3  residual:    3.1534373809098760E-005
 Iteration             4  residual:    5.0153211704417186E-008
 Iteration             5  residual:    7.9933529702152848E-011
 Iteration             6  residual:    1.2717205406485772E-013
 Iteration             7  residual:    2.0605670154693845E-016
 Wall time:    2.1522998809814453E-002
 x:   -1.0226403871012107E-005   5.0902512156439025E-005
   0.3334710494981550

$ mpirun -np 2 --map-by ppr:2:socket ./main_gpu_mpi
 Running parallel version on            2 ranks...
 rank            0 using GPU            0
 rank            1 using GPU            1
 Running Warmup....
 Iteration             1  residual:     11.63451702489180
 Iteration             2  residual:    1.9845558837315827E-002
 Iteration             3  residual:    3.1534373809098828E-005
 Iteration             4  residual:    5.0153211704417147E-008
 Iteration             5  residual:    7.9933529702654869E-011
 Iteration             6  residual:    1.2717213308682699E-013
 Iteration             7  residual:    2.1790798387780098E-016
 Running Test....
 Iteration             1  residual:     11.63451702489180
 Iteration             2  residual:    1.9845558837315827E-002
 Iteration             3  residual:    3.1534373809098828E-005
 Iteration             4  residual:    5.0153211704417147E-008
 Iteration             5  residual:    7.9933529702654869E-011
 Iteration             6  residual:    1.2717213308682699E-013
 Iteration             7  residual:    2.1790798387780098E-016
 Wall time:    1.2560844421386719E-002
 x:   -1.0226403871012103E-005   5.0902512156439019E-005
   0.3334710494981550

$ mpirun -np 4 --map-by ppr:2:socket ./main_gpu_mpi
 Running parallel version on            4 ranks...
 rank            0 using GPU            0
 rank            1 using GPU            1
 rank            2 using GPU            2
 rank            3 using GPU            3
 Running Warmup....
 Iteration             1  residual:     11.63451702489180
 Iteration             2  residual:    1.9845558837315834E-002
 Iteration             3  residual:    3.1534373809098753E-005
 Iteration             4  residual:    5.0153211704416981E-008
 Iteration             5  residual:    7.9933529702365899E-011
 Iteration             6  residual:    1.2717208770659891E-013
 Iteration             7  residual:    2.1118342771089487E-016
 Running Test....
 Iteration             1  residual:     11.63451702489180
 Iteration             2  residual:    1.9845558837315834E-002
 Iteration             3  residual:    3.1534373809098753E-005
 Iteration             4  residual:    5.0153211704416981E-008
 Iteration             5  residual:    7.9933529702365899E-011
 Iteration             6  residual:    1.2717208770659891E-013
 Iteration             7  residual:    2.1118342771089487E-016
 Wall time:    8.5079669952392578E-003
 x:   -1.0226403871012100E-005   5.0902512156439019E-005
```

Comparing the wall times across the runs, we see that the strong scaling efficiency drops quite a bit at 4 GPUs. To understand why, we can profile this case using Nsight Systems using the following command:
```
$ mpirun -np 4 --map-by ppr:2:socket nsys profile -o gpu_mpi.rank%q{OMPI_COMM_WORLD_RANK} ./main_gpu_mpi
```
This command will generate a profile output for each rank separately, with the `%q{...}` notation in the output string used to add an exported environment variable to the name. Download one of the resulting profiles to to a local system, load it in the Nsight Systems GUI, and look for the `Allreduce` NVTX ranges we've added to show the allreduce time. What you'll find is that the `MPI_Allreduce` is generating a device-to-host copy, followed by a CPU allreduce, followed by a host-to-device copy, which is inefficient and limiting strong scaling.
```

### Using NCCL
Instead of using MPI for the allreduce, we can try to use functionality from the NVIDIA Collective Commuincation Library (NCCL) instead. NCCL has highly optimized GPU collectives (e.g. allreduce) as well as support for point-to-point messaging (e.g. send/recv). Replacing `MPI_Allreduce` with `ncclAllReduce` is straightforward as the APIs are very similar. 

First, look at the MPI setup code in `main.f90`. To use NCCL, some additional initialization is required to create a NCCL communicator, and that code is added after the device assignment. Besides this small additional set up, the `MPI_Allreduce` calls in `RunCG` are replaced with calls `ncclAllReduce`.

Type `make` to compile the NCCL parallelized program, `main_gpu_nccl`. To run on the M100 node, use the following command:
```
$ mpirun -np <N>  --map-by ppr:2:socket ./main_gpu_nccl
```
where `N` can be 1, 2, or 4.

Running for `N` = 1, 2 and 4 results in the following:
```
$ mpirun -np 1 --map-by ppr:2:socket ./main_gpu_nccl
 Running parallel version on            1 ranks...
 rank            0 using GPU            0
 Running Warmup....
 Iteration             1  residual:     11.63451702489180
 Iteration             2  residual:    1.9845558837315817E-002
 Iteration             3  residual:    3.1534373809098760E-005
 Iteration             4  residual:    5.0153211704417186E-008
 Iteration             5  residual:    7.9933529702152848E-011
 Iteration             6  residual:    1.2717205406485772E-013
 Iteration             7  residual:    2.0605670154693845E-016
 Running Test....
 Iteration             1  residual:     11.63451702489180
 Iteration             2  residual:    1.9845558837315817E-002
 Iteration             3  residual:    3.1534373809098760E-005
 Iteration             4  residual:    5.0153211704417186E-008
 Iteration             5  residual:    7.9933529702152848E-011
 Iteration             6  residual:    1.2717205406485772E-013
 Iteration             7  residual:    2.0605670154693845E-016
 Wall time:    2.1223068237304688E-002
 x:   -1.0226403871012107E-005   5.0902512156439025E-005
   0.3334710494981550

$ mpirun -np 2 --map-by ppr:2:socket ./main_gpu_nccl
 Running parallel version on            2 ranks...
 rank            0 using GPU            0
 rank            1 using GPU            1
 Running Warmup....
 Iteration             1  residual:     11.63451702489180
 Iteration             2  residual:    1.9845558837315827E-002
 Iteration             3  residual:    3.1534373809098828E-005
 Iteration             4  residual:    5.0153211704417147E-008
 Iteration             5  residual:    7.9933529702654869E-011
 Iteration             6  residual:    1.2717213308682699E-013
 Iteration             7  residual:    2.1790798387780098E-016
 Running Test....
 Iteration             1  residual:     11.63451702489180
 Iteration             2  residual:    1.9845558837315827E-002
 Iteration             3  residual:    3.1534373809098828E-005
 Iteration             4  residual:    5.0153211704417147E-008
 Iteration             5  residual:    7.9933529702654869E-011
 Iteration             6  residual:    1.2717213308682699E-013
 Iteration             7  residual:    2.1790798387780098E-016
 Wall time:    1.1735916137695313E-002
 x:   -1.0226403871012103E-005   5.0902512156439019E-005
   0.3334710494981550

$ mpirun -np 4 --map-by ppr:2:socket ./main_gpu_nccl
 Running parallel version on            4 ranks...
 rank            0 using GPU            0
 rank            1 using GPU            1
 rank            2 using GPU            2
 rank            3 using GPU            3
 Running Warmup....
 Iteration             1  residual:     11.63451702489180
 Iteration             2  residual:    1.9845558837315827E-002
 Iteration             3  residual:    3.1534373809098713E-005
 Iteration             4  residual:    5.0153211704416955E-008
 Iteration             5  residual:    7.9933529702372219E-011
 Iteration             6  residual:    1.2717208872618989E-013
 Iteration             7  residual:    2.1133686326639035E-016
 Running Test....
 Iteration             1  residual:     11.63451702489180
 Iteration             2  residual:    1.9845558837315827E-002
 Iteration             3  residual:    3.1534373809098713E-005
 Iteration             4  residual:    5.0153211704416955E-008
 Iteration             5  residual:    7.9933529702372219E-011
 Iteration             6  residual:    1.2717208872618989E-013
 Iteration             7  residual:    2.1133686326639035E-016
 Wall time:    6.9658756256103516E-003
 x:   -1.0226403871012102E-005   5.0902512156439019E-005
   0.3334710494981550
```

Using `ncclAllReduce` in place of `MPI_Allreduce` provides a boost to strong scaling efficiency. To understand why, run a profile with the NCCL parallelized program like with the MPI program and look at the allreduce NVTX range. Unlike `MPI_Allreduce`, `ncclAllReduce` avoids any copies to/from the host and performs the operation on the GPU, resulting in more efficient scaling for this sample code.
