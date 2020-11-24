program CG
  use constants
  use util
  use CG_routines
  implicit none

  integer, parameter :: N = 16384
  real(WP), allocatable :: A(:,:)
  real(WP), allocatable :: b(:)
  real(WP), allocatable :: x(:)
  real(WP) :: ts, te, wallclock

  allocate(A(N, N))
  allocate(b(N))
  allocate(x(N))

  call GenerateCGInput(N, A, b)

  ! Warmup
  print*, "Running Warmup...."
  x = 0.d0
  call RunCG(N, A, b, x, 1d-15, 10000)

  print*, "Running Test...."
  x = 0.d0
  ts = wallclock()
  call RunCG(N, A, b, x, 1d-15, 10000)
  te = wallclock()

  print*, "Wall time: ", te - ts
  print*, "x: ", minval(x), maxval(x), sum(x)
  
end program
