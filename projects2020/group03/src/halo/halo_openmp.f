module m_halo_openmp
  implicit none
  private

  public :: update_halo
  contains
    subroutine update_halo(field, num_halo)
      !DIR$ INLINEALWAYS update_halo
      use, intrinsic :: iso_fortran_env, only: REAL32

      real(kind = REAL32), intent(inout) :: field(:, :, :)
      integer, intent(in) :: num_halo

      integer :: nx
      integer :: ny
      integer :: nz
      integer :: i
      integer :: j
      integer :: k

      nx = size(field, 1) - 2 * num_halo
      ny = size(field, 2) - 2 * num_halo
      nz = size(field, 3)

      !$omp single

      ! north
      !$omp task
      do k = 1, nz
        do j = 1, num_halo
          do i = 1 + num_halo, nx + num_halo
            field(i, j, k) = field(i, j + ny, k)
          end do
        end do
      end do
      !$omp end task

      ! south
      !$omp task
      do k = 1, nz
        do j = ny + num_halo + 1, ny + 2 * num_halo
          do i = 1 + num_halo, nx + num_halo
            field(i, j, k) = field(i, j - ny, k)
          end do
        end do
      end do
      !$omp end task

      ! east
      !$omp task
      do k = 1, nz
        do j = 1 + num_halo, ny + num_halo
          do i = 1, num_halo
            field(i, j, k) = field(i + nx, j, k)
          end do
        end do
      end do
      !$omp end task

      ! west
      !$omp task
      do k = 1, nz
        do j = 1 + num_halo, ny + num_halo
          do i = nx + num_halo + 1, nx + 2 * num_halo
            field(i, j, k) = field(i - nx, j, k)
          end do
        end do
      end do
      !$omp end task

      ! northeast
      !$omp task
      do k = 1, nz
        do j = 1, num_halo
          do i = 1, num_halo
            field(i, j, k) = field(i + nx, j, k)
          end do
        end do
      end do
      !$omp end task

      ! northwest
      !$omp task
      do k = 1, nz
        do j = ny + num_halo + 1, ny + 2 * num_halo
          do i = 1, num_halo
            field(i, j, k) = field(i + nx, j, k)
          end do
        end do
      end do
      !$omp end task

      ! southeast
      !$omp task
      do k = 1, nz
        do j = 1, num_halo
          do i = nx + num_halo + 1, nx + 2 * num_halo
            field(i, j, k) = field(i - nx, j, k)
          end do
        end do
      end do
      !$omp end task

      ! southwest
      !$omp task
      do k = 1, nz
        do j = ny + num_halo + 1, ny + 2 * num_halo
          do i = nx + num_halo + 1, nx + 2 * num_halo
            field(i, j, k) = field(i - nx, j, k)
          end do
        end do
      end do
      !$omp end task

      !$omp taskwait
      !$omp end single
    end subroutine
end module

! vim: set filetype=fortran expandtab tabstop=2 softtabstop=2 :
