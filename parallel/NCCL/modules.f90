
module constants
  integer, parameter :: WP = 8
end module

module parallel
#ifdef USE_GPU
  use nccl
#endif
  use mpi
  integer :: ierr, rank, nranks
  integer :: local_comm, local_rank
#ifdef USE_GPU
  type(ncclUniqueId) :: nccl_id
  type(ncclResult) :: nccl_result
  type(ncclComm) :: nccl_comm 
#endif
end module

module util
  use constants
  implicit none
  contains
    subroutine GenerateCGInput(N, A, b)
      use constants
      use parallel
      integer, intent(in) :: N
      real(WP), intent(out) :: A(N/nranks, N)
      real(WP), intent(out) :: b(N)
      integer :: i, j
      real(WP), allocatable :: tmp(:,:)

      call srand(42)

      ! Generate complete random SPD matrix A
      allocate(tmp(N,N))
      call random_number(tmp)

      ! Copy row slab (Note: this is exactly the transposed A we need for the matvec)
      do j = 1, N
        do i = 1, N/nranks
          A(i,j) = 0.5d0 * (tmp(i + (N/nranks) * rank, j) + tmp(j, i + (N/nranks) * rank))
          if (i + (N/nranks) * rank == j) A(i,j) = A(i,j) + N
        end do
      end do
      deallocate(tmp)

      ! Generate random RHS vector b
      call random_number(b)

    end subroutine GenerateCGInput
end module

module CG_routines
  use nvtx
#ifdef USE_CUBLAS
  use cublas
#endif
  use constants
  contains
  subroutine symmatvec(M, N, AT, x, Ax)
    implicit none
    integer, intent(in) :: M, N
    real(WP), intent(in) :: AT(N, M), x(N)
    real(WP), intent(out) :: Ax(M)

#ifdef USE_CUDA_DATA
    attributes(managed) :: AT, x, Ax
#endif

    integer :: i, j
    real(WP) :: s

    call nvtxStartRange("symmatvec", 22)
#if !defined(USE_BLAS) && !defined(USE_CUBLAS)
    ! Note: Since A is symmetric, we can use the "transpose"
    ! for better memory access here
    !$acc parallel loop gang private(s)
    do i = 1, M
      s = 0.d0
      !$acc loop vector reduction(+:s)
      do j = 1, N
        s = s + AT(j,i) * x(j)
      end do
      Ax(i) = s
    end do
#else
#ifdef USE_ACC_DATA
    !$acc host_data use_device(AT, x, Ax)
#endif
    call dgemv('T', N, M, 1.d0, AT, N, x, 1, 0.d0, Ax, 1)
#ifdef USE_ACC_DATA
    !$acc end host_data
#endif
#endif
    call nvtxEndRange

  end subroutine

  function dot(N, x, y) result(r)
#ifdef USE_CUBLAS
  use cublas
#endif
    implicit none
    integer, intent(in) :: N
    real(WP), intent(in) :: x(N), y(N)
#ifdef USE_CUDA_DATA
    attributes(managed) :: x, y
#endif
#ifdef USE_BLAS
    real(WP) :: ddot
#endif
    real(WP) :: r
    integer :: i

    call nvtxStartRange("dot", 22)
#if !defined(USE_BLAS) && !defined(USE_CUBLAS)
    r = 0.d0
    !$acc parallel loop
    do i = 1, N
      r = r + x(i) * y(i)
    enddo
#else
#ifdef USE_ACC_DATA
    !$acc host_data use_device(x, y)
#endif
    r = ddot(N, x, 1, y, 1)
#ifdef USE_ACC_DATA
    !$acc end host_data
#endif
#endif
    call nvtxEndRange
  end function dot

  subroutine RunCG(N, A, b, x, tol, max_iter)
    use parallel
    implicit none
    integer, intent(in) :: N, max_iter
    real(WP), intent(in) :: A(N/nranks, N), b(N), tol
    real(WP), intent(inout) :: x(N)

    real(WP) :: alpha, rr0, rr
    real(WP), allocatable :: Ax(:), r(:), p(:)
    integer :: it, i

#ifdef USE_CUDA_DATA
    attributes(managed) :: A, b, x, Ax, r, p
#endif

    call nvtxStartRange("RunCG", 11)
    allocate(Ax(N), r(N), p(N))

#ifdef USE_ACC_DATA
!$acc enter data create(Ax, r, p)
#endif
    call symmatvec(N, N/nranks, A, x(rank * (N/nranks) + 1), Ax)
    call nvtxStartRange("Allreduce", 13)
#ifdef USE_ACC_DATA
    !$acc host_data use_device(Ax)
#endif
#ifdef USE_GPU
    nccl_result = ncclAllReduce(Ax, Ax, N, ncclDouble, ncclSum, nccl_comm, 0)
#else
    call MPI_Allreduce(MPI_IN_PLACE, Ax, N, MPI_DOUBLE_PRECISION, MPI_SUM, MPI_COMM_WORLD, ierr)
#endif
#ifdef USE_ACC_DATA
    !$acc end host_data
#endif
    call nvtxEndRange()
    !$acc parallel loop
    do i = 1, N
      r(i) = b(i) - Ax(i)
      p(i) = r(i)
    enddo
    rr0 = dot(N, r, r)

    do it = 1, max_iter
      call nvtxStartRange("CG Iter", 12)
      call symmatvec(N, N/nranks, A, p(rank * (N/nranks) + 1), Ax)
      call nvtxStartRange("Allreduce", 13)
#ifdef USE_ACC_DATA
    !$acc host_data use_device(Ax)
#endif
#ifdef USE_GPU
      nccl_result = ncclAllReduce(Ax, Ax, N, ncclDouble, ncclSum, nccl_comm, 0)
#else
      call MPI_Allreduce(MPI_IN_PLACE, Ax, N, MPI_DOUBLE_PRECISION, MPI_SUM, MPI_COMM_WORLD, ierr)
#endif
#ifdef USE_ACC_DATA
    !$acc end host_data
#endif
      call nvtxEndRange()
      alpha = rr0 / dot(N, p, Ax)

      !$acc parallel loop
      do i = 1, N
        x(i) = x(i) + alpha * p(i)
        r(i) = r(i) - alpha * Ax(i)
      enddo

      rr = dot(N, r, r)

      if (rank == 0) print*, "Iteration ", it, " residual: ", sqrt(rr)
      if (sqrt(rr) <= tol) then
        call nvtxEndRange()
#ifdef USE_ACC_DATA
        !$acc exit data delete(Ax, r, p) 
#endif
        deallocate(Ax, r, p)
        call nvtxEndRange()
        return
      endif
      !$acc parallel loop
      do i = 1, N
        p(i) = r(i) + (rr / rr0) * p(i)
      enddo
      rr0 = rr
      call nvtxEndRange()
    enddo

#ifdef USE_ACC_DATA
    !$acc exit data delete(Ax, r, p) 
#endif
    deallocate(Ax, r, p)
    call nvtxEndRange()
    
  end subroutine RunCG
end module
