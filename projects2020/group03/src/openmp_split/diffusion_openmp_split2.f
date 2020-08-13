module m_diffusion_openmp_split2
  implicit none
  private

  public :: apply_diffusion
  contains
    subroutine apply_diffusion(in_field, out_field, num_halo, alpha, p, num_iter, z_slices_on_cpu)
      use, intrinsic :: iso_fortran_env, only: REAL64
      use m_partitioner, only: Partitioner

      real(kind = REAL64), intent(inout) :: in_field(:, :, :)
      real(kind = REAL64), intent(inout) :: out_field(:, :, :)
      integer, intent(in) :: num_halo
      real(kind = REAL64), intent(in) :: alpha
      type(Partitioner), intent(in) :: p
      integer, intent(in) :: num_iter
      integer, intent(in) :: z_slices_on_cpu

      integer :: iter
      integer :: i
      integer :: j
      integer :: k
      integer :: k0
      real(kind = REAL64) :: alpha_20
      real(kind = REAL64) :: alpha_08
      real(kind = REAL64) :: alpha_02
      real(kind = REAL64) :: alpha_01
      integer :: nx
      integer :: ny
      integer :: nz

      nx = size(in_field, 1) - 2 * num_halo
      ny = size(in_field, 2) - 2 * num_halo
      nz = size(in_field, 3)

      k0 = nz - z_slices_on_cpu

      alpha_20 = -20 * alpha + 1
      alpha_08 =   8 * alpha
      alpha_02 =  -2 * alpha
      alpha_01 =  -1 * alpha

      !$omp target data &
      !$omp   map(to: in_field(:, :, 1:k0)) &
      !$omp   map(from: out_field(:, :, 1:k0))
      do iter = 1, num_iter
        !$omp single
        ! GPU
        !$omp target teams nowait
        call update_halo_gpu(in_field, num_halo, z_slices_on_cpu)
        !$omp distribute
        do k = 1, k0
          !$omp parallel do simd collapse(2) &
          !$omp   default(none) &
          !$omp   shared(nx, ny, num_halo, in_field, out_field, alpha_20, alpha_08, alpha_02, alpha_01, k) &
          !$omp   lastprivate(i, j)
          do j = 1 + num_halo, ny + num_halo
            do i = 1 + num_halo, nx + num_halo
              out_field(i, j, k) = &
                + alpha_20 * in_field(i,     j,     k) &
                + alpha_08 * in_field(i - 1, j,     k) &
                + alpha_08 * in_field(i + 1, j,     k) &
                + alpha_08 * in_field(i,     j - 1, k) &
                + alpha_08 * in_field(i,     j + 1, k) &
                + alpha_02 * in_field(i - 1, j - 1, k) &
                + alpha_02 * in_field(i - 1, j + 1, k) &
                + alpha_02 * in_field(i + 1, j - 1, k) &
                + alpha_02 * in_field(i + 1, j + 1, k) &
                + alpha_01 * in_field(i - 2, j,     k) &
                + alpha_01 * in_field(i + 2, j,     k) &
                + alpha_01 * in_field(i,     j - 2, k) &
                + alpha_01 * in_field(i,     j + 2, k)
            end do
          end do

          if (iter /= num_iter) then
              !$omp parallel do simd collapse(2) &
              !$omp   default(none) &
              !$omp   shared(nx, ny, num_halo, in_field, out_field, k) &
              !$omp   lastprivate(i, j)
              do j = 1 + num_halo, ny + num_halo
                do i = 1 + num_halo, nx + num_halo
                  in_field(i, j, k) = out_field(i, j, k)
                end do
              end do
          end if
        end do

        if (iter == num_iter) then
          call update_halo_gpu(out_field, num_halo, z_slices_on_cpu)
        end if
        !$omp end target teams
        !$omp end single nowait

        ! CPU
        !$omp parallel &
        !$omp   default(none) &
        !$omp   shared(nx, ny, nz, num_halo, num_iter, in_field, out_field, alpha_20, alpha_08, alpha_02, alpha_01, p, z_slices_on_cpu, k0) &
        !$omp   private(iter, i, j, k)
        call update_halo_cpu(in_field, num_halo, z_slices_on_cpu)

        !$omp do schedule(dynamic)
        do k = 1 + k0, nz
          !$omp simd collapse(2)
          do j = 1 + num_halo, ny + num_halo
            do i = 1 + num_halo, nx + num_halo
              out_field(i, j, k) = &
                + alpha_20 * in_field(i,     j,     k) &
                + alpha_08 * in_field(i - 1, j,     k) &
                + alpha_08 * in_field(i + 1, j,     k) &
                + alpha_08 * in_field(i,     j - 1, k) &
                + alpha_08 * in_field(i,     j + 1, k) &
                + alpha_02 * in_field(i - 1, j - 1, k) &
                + alpha_02 * in_field(i - 1, j + 1, k) &
                + alpha_02 * in_field(i + 1, j - 1, k) &
                + alpha_02 * in_field(i + 1, j + 1, k) &
                + alpha_01 * in_field(i - 2, j,     k) &
                + alpha_01 * in_field(i + 2, j,     k) &
                + alpha_01 * in_field(i,     j - 2, k) &
                + alpha_01 * in_field(i,     j + 2, k)
            end do
          end do

          if (iter /= num_iter) then
            !$omp simd collapse(2)
            do j = 1 + num_halo, ny + num_halo
              do i = 1 + num_halo, nx + num_halo
                in_field(i, j, k) = out_field(i, j, k)
              end do
            end do
          end if
        end do
        !$omp end do

        if (iter == num_iter) then
          call update_halo_cpu(out_field, num_halo, z_slices_on_cpu)
        end if
        !$omp end parallel
        !$omp barrier
      end do
      !$omp end target data
    end subroutine

    subroutine update_halo_cpu(field, num_halo, z_slices_on_cpu)
#ifdef _CRAYC
      !DIR$ INLINEALWAYS update_halo
#endif
#ifdef __INTEL_COMPILER
      !DIR$ ATTRIBUTES FORCEINLINE :: update_halo
#endif
      use, intrinsic :: iso_fortran_env, only: REAL64

      real(kind = REAL64), intent(inout) :: field(:, :, :)
      integer, intent(in) :: num_halo

      integer :: nx
      integer :: ny
      integer :: nz
      integer :: z_slices_on_cpu
      integer :: i
      integer :: j
      integer :: k
      integer :: k0

      nx = size(field, 1) - 2 * num_halo
      ny = size(field, 2) - 2 * num_halo
      nz = size(field, 3)

      k0 = nz - z_slices_on_cpu

      !$omp do
      do k = 1 + k0, nz
        ! north
        !$omp simd collapse(2)
        do j = 1, num_halo
          do i = 1 + num_halo, nx + num_halo
            field(i, j, k) = field(i, j + ny, k)
          end do
        end do

        ! south
        !$omp simd collapse(2)
        do j = ny + num_halo + 1, ny + 2 * num_halo
          do i = 1 + num_halo, nx + num_halo
            field(i, j, k) = field(i, j - ny, k)
          end do
        end do

        ! east
        !$omp simd collapse(2)
        do j = 1 + num_halo, ny + num_halo
          do i = 1, num_halo
            field(i, j, k) = field(i + nx, j, k)
          end do
        end do

        ! west
        !$omp simd collapse(2)
        do j = 1 + num_halo, ny + num_halo
          do i = nx + num_halo + 1, nx + 2 * num_halo
            field(i, j, k) = field(i - nx, j, k)
          end do
        end do

        ! northeast
        !$omp simd collapse(2)
        do j = 1, num_halo
          do i = 1, num_halo
            field(i, j, k) = field(i + nx, j, k)
          end do
        end do

        ! northwest
        !$omp simd collapse(2)
        do j = ny + num_halo + 1, ny + 2 * num_halo
          do i = 1, num_halo
            field(i, j, k) = field(i + nx, j, k)
          end do
        end do

        ! southeast
        !$omp simd collapse(2)
        do j = 1, num_halo
          do i = nx + num_halo + 1, nx + 2 * num_halo
            field(i, j, k) = field(i - nx, j, k)
          end do
        end do

        ! southwest
        !$omp simd collapse(2)
        do j = ny + num_halo + 1, ny + 2 * num_halo
          do i = nx + num_halo + 1, nx + 2 * num_halo
            field(i, j, k) = field(i - nx, j, k)
          end do
        end do
      end do
    end subroutine

    subroutine update_halo_gpu(field, num_halo, z_slices_on_cpu)
#ifdef _CRAYC
      !DIR$ INLINEALWAYS update_halo
#endif
#ifdef __INTEL_COMPILER
      !DIR$ ATTRIBUTES FORCEINLINE :: update_halo
#endif
      use, intrinsic :: iso_fortran_env, only: REAL64

      real(kind = REAL64), intent(inout) :: field(:, :, :)
      integer, intent(in) :: num_halo

      integer :: nx
      integer :: ny
      integer :: nz
      integer :: z_slices_on_cpu
      integer :: i
      integer :: j
      integer :: k
      integer :: k0

      !$omp declare target

      nx = size(field, 1) - 2 * num_halo
      ny = size(field, 2) - 2 * num_halo
      nz = size(field, 3)

      k0 = nz - z_slices_on_cpu

      !$omp distribute
      do k = 1, k0
        ! north
        !$omp parallel do simd collapse(2) &
        !$omp   default(none) &
        !$omp   shared(nx, ny, num_halo, field, k) &
        !$omp   lastprivate(i, j)
        do j = 1, num_halo
          do i = 1 + num_halo, nx + num_halo
            field(i, j, k) = field(i, j + ny, k)
          end do
        end do

        ! south
        !$omp parallel do simd collapse(2) &
        !$omp   default(none) &
        !$omp   shared(nx, ny, num_halo, field, k) &
        !$omp   lastprivate(i, j)
        do j = ny + num_halo + 1, ny + 2 * num_halo
          do i = 1 + num_halo, nx + num_halo
            field(i, j, k) = field(i, j - ny, k)
          end do
        end do

        ! east
        !$omp parallel do simd collapse(2) &
        !$omp   default(none) &
        !$omp   shared(nx, ny, num_halo, field, k) &
        !$omp   lastprivate(i, j)
        do j = 1 + num_halo, ny + num_halo
          do i = 1, num_halo
            field(i, j, k) = field(i + nx, j, k)
          end do
        end do

        ! west
        !$omp parallel do simd collapse(2) &
        !$omp   default(none) &
        !$omp   shared(nx, ny, num_halo, field, k) &
        !$omp   lastprivate(i, j)
        do j = 1 + num_halo, ny + num_halo
          do i = nx + num_halo + 1, nx + 2 * num_halo
            field(i, j, k) = field(i - nx, j, k)
          end do
        end do

        ! northeast
        !$omp parallel do simd collapse(2) &
        !$omp   default(none) &
        !$omp   shared(nx, ny, num_halo, field, k) &
        !$omp   lastprivate(i, j)
        do j = 1, num_halo
          do i = 1, num_halo
            field(i, j, k) = field(i + nx, j, k)
          end do
        end do

        ! northwest
        !$omp parallel do simd collapse(2) &
        !$omp   default(none) &
        !$omp   shared(nx, ny, num_halo, field, k) &
        !$omp   lastprivate(i, j)
        do j = ny + num_halo + 1, ny + 2 * num_halo
          do i = 1, num_halo
            field(i, j, k) = field(i + nx, j, k)
          end do
        end do

        ! southeast
        !$omp parallel do simd collapse(2) &
        !$omp   default(none) &
        !$omp   shared(nx, ny, num_halo, field, k) &
        !$omp   lastprivate(i, j)
        do j = 1, num_halo
          do i = nx + num_halo + 1, nx + 2 * num_halo
            field(i, j, k) = field(i - nx, j, k)
          end do
        end do

        ! southwest
        !$omp parallel do simd collapse(2) &
        !$omp   default(none) &
        !$omp   shared(nx, ny, num_halo, field, k) &
        !$omp   lastprivate(i, j)
        do j = ny + num_halo + 1, ny + 2 * num_halo
          do i = nx + num_halo + 1, nx + 2 * num_halo
            field(i, j, k) = field(i - nx, j, k)
          end do
        end do
      end do
      !$omp end distribute
    end subroutine
end module

! vim: set filetype=fortran expandtab tabstop=2 softtabstop=2 :
