program CG
  use constants
  use util
  use CG_routines
  use nvtx
  implicit none

  integer, parameter :: N = 16384
  real(WP), allocatable :: A(:,:)
  real(WP), allocatable :: b(:)
  real(WP), allocatable :: x(:)
  real(WP) :: ts, te, wallclock

#ifdef USE_CUDA_DATA
  attributes(managed) :: A, b, x
#endif

  allocate(A(N, N))
  allocate(b(N))
  allocate(x(N))

  call nvtxStartRange("GenerateCGInput", 1)
  call GenerateCGInput(N, A, b)
  call nvtxEndRange()

#ifdef USE_ACC_DATA
!$acc enter data copyin(A, b) create(x)
#endif
  ! Warmup
  print*, "Running Warmup...."
  call nvtxStartRange("WARMUP", 2)
  x = 0.d0
#ifdef USE_ACC_DATA
!$acc update device(x)
#endif
  call RunCG(N, A, b, x, 1d-15, 10000)
  call nvtxEndRange()

  print*, "Running Test...."
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

  print*, "Wall time: ", te - ts
  print*, "x: ", minval(x), maxval(x), sum(x)
  
end program
