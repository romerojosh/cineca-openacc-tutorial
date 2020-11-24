program CG
  use constants
  use util
  use CG_routines
  use nvtx
  use parallel
#if USE_GPU
  use cudafor
  use openacc
  use nccl
#endif
  implicit none

  integer, parameter :: N = 16384
  real(WP), allocatable :: A(:,:)
  real(WP), allocatable :: b(:)
  real(WP), allocatable :: x(:)
  real(WP) :: ts, te, wallclock

#ifdef USE_CUDA_DATA
  attributes(managed) :: A, b, x
#endif

  call MPI_Init(ierr)
  call MPI_Comm_rank(MPI_COMM_WORLD, rank, ierr)
  call MPI_Comm_size(MPI_COMM_WORLD, nranks, ierr)
  call MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, local_comm, ierr)  
  call MPI_Comm_rank(local_comm, local_rank, ierr)

  if (rank == 0) print*, "Running parallel version on", nranks, "ranks..."
  call MPI_Barrier(MPI_COMM_WORLD, ierr)
#if USE_GPU
  print*, "rank", rank, "using GPU", local_rank
  ierr = cudaSetDevice(local_rank)
  call acc_init(acc_get_device_type())
  call acc_set_device_num(local_rank, acc_get_device_type()) 

  if (rank == 0) then
    ! Rank 0 generates unique id
    nccl_result = ncclGetUniqueId(nccl_id)
  endif
  
  ! Broadcast unique id to all ranks
  call MPI_Bcast(nccl_id%internal, sizeof(nccl_id%internal), &
                 MPI_CHAR, 0, MPI_COMM_WORLD, ierr)
  
  ! Initialize NCCL communicator
  nccl_result = ncclCommInitRank(nccl_comm, nranks, nccl_id, rank)
#endif

  if (mod(N, nranks) .ne. 0) then
    print*, "ERROR: N must be evenly divisble by nranks. Aborting.."
    call exit(1)
  endif
  allocate(A(N/nranks, N)) ! Row slab of A only
  allocate(b(N))
  allocate(x(N))

  call nvtxStartRange("GenerateCGInput", 1)
  call GenerateCGInput(N, A, b)
  call nvtxEndRange()

#ifdef USE_ACC_DATA
!$acc enter data copyin(A, b) create(x)
#endif
  ! Warmup
  if (rank == 0) print*, "Running Warmup...."
  call nvtxStartRange("WARMUP", 2)
  x = 0.d0
#ifdef USE_ACC_DATA
!$acc update device(x)
#endif
  call RunCG(N, A, b, x, 1d-15, 10000)
  call nvtxEndRange()

  if (rank == 0) print*, "Running Test...."
  call nvtxStartRange("TEST", 3)
  x = 0.d0
#ifdef USE_ACC_DATA
!$acc update device(x)
#endif
  ts = wallclock()
  call RunCG(N, A, b, x, 1d-15, 10000)
  te = wallclock()
#ifdef USE_ACC_DATA
!$acc exit data delete(A,b) copyout(x)
#endif
  call nvtxEndRange()

  if (rank == 0) then
    print*, "Wall time: ", te - ts
    print*, "x: ", minval(x), maxval(x), sum(x)
  endif
  call MPI_Finalize(ierr)
  
end program
