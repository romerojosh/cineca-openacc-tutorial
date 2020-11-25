
module constants
  integer, parameter :: WP = 8
end module

module util
  use constants
  implicit none
  contains
    subroutine GenerateCGInput(N, A, b)
      use constants
      integer, intent(in) :: N
      real(WP), intent(out) :: A(N, N)
      real(WP), intent(out) :: b(N)
      integer :: i, j
      real(WP), allocatable :: tmp(:,:)

      call srand(42)

      ! Generate complete random SPD matrix A
      allocate(tmp(N,N))
      call random_number(tmp)

      do j = 1, N
        do i = 1, N
          A(i,j) = 0.5d0 * (tmp(i, j) + tmp(j, i))
          if (i == j) A(i,j) = A(i,j) + N
        end do
      end do
      deallocate(tmp)

      ! Generate random RHS vector b
      call random_number(b)

    end subroutine GenerateCGInput
end module

module CG_routines
  use constants
  contains
  subroutine symmatvec(M, N, AT, x, Ax)
    implicit none
    integer, intent(in) :: M, N
    real(WP), intent(in) :: AT(N, M), x(N)
    real(WP), intent(out) :: Ax(M)

    integer :: i, j
    real(WP) :: s

#ifndef USE_BLAS
    ! Note: Since A is symmetric, we can use the "transpose"
    ! for better memory access here
    do i = 1, M
      s = 0.d0
      do j = 1, N
        s = s + AT(j,i) * x(j)
      end do
      Ax(i) = s
    end do
#else
    call dgemv('T', N, M, 1.d0, AT, N, x, 1, 0.d0, Ax, 1)
#endif

  end subroutine

  function dot(N, x, y) result(r)
    implicit none
    integer, intent(in) :: N
    real(WP), intent(in) :: x(N), y(N)
    integer :: i
    real(WP) :: ddot
    real(WP) :: r

#ifndef USE_BLAS
    r = 0.d0
    do i = 1, N
      r = r + x(i) * y(i)
    enddo
#else
    r = ddot(N, x, 1, y, 1)
#endif
  end function dot

  subroutine RunCG(N, A, b, x, tol, max_iter)
    implicit none
    integer, intent(in) :: N, max_iter
    real(WP), intent(in) :: A(N, N), b(N), tol
    real(WP), intent(inout) :: x(N)

    real(WP) :: alpha, rr0, rr
    real(WP), allocatable :: Ax(:), r(:), p(:)
    integer :: it, i

    allocate(Ax(N), r(N), p(N))

    call symmatvec(N, N, A, x, Ax)
    do i = 1, N
      r(i) = b(i) - Ax(i)
      p(i) = r(i)
    enddo
    rr0 = dot(N, r, r)

    do it = 1, max_iter
      call symmatvec(N, N, A, p, Ax)
      alpha = rr0 / dot(N, p, Ax)

      do i = 1, N
        x(i) = x(i) + alpha * p(i)
        r(i) = r(i) - alpha * Ax(i)
      enddo

      rr = dot(N, r, r)

      print*, "Iteration ", it, " residual: ", sqrt(rr)
      if (sqrt(rr) <= tol) then
        deallocate(Ax, r, p)
        return
      endif
      do i = 1, N
        p(i) = r(i) + (rr / rr0) * p(i)
      enddo
      rr0 = rr
    enddo

    deallocate(Ax, r, p)
    
  end subroutine RunCG
end module
