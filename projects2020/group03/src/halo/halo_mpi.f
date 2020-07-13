module m_halo_mpi
  implicit none
  private

  public :: update_halo
  contains
    subroutine update_halo(field, num_halo, p)
#ifdef _CRAYC
      !DIR$ INLINEALWAYS update_halo
#endif
#ifdef __INTEL_COMPILER
      !DIR$ ATTRIBUTES FORCEINLINE :: update_halo
#endif
      use, intrinsic :: iso_fortran_env, only: REAL32
      use mpi, only: &
        MPI_SUCCESS, MPI_FLOAT, MPI_STATUS_IGNORE
        ! MPI_Isend, MPI_Irecv
      use m_utils, only: error
      use m_partitioner, only: Partitioner

      real(kind = REAL32), intent(inout) :: field(:, :, :)
      integer, intent(in) :: num_halo
      type(Partitioner), intent(in) :: p

      integer :: nx
      integer :: ny
      integer :: nz
      integer :: ierror

      integer :: s_send_req
      integer :: s_recv_req
      real(kind = REAL32), allocatable :: s_send_buf(:, :, :)
      real(kind = REAL32), allocatable :: s_recv_buf(:, :, :)

      integer :: n_send_req
      integer :: n_recv_req
      real(kind = REAL32), allocatable :: n_send_buf(:, :, :)
      real(kind = REAL32), allocatable :: n_recv_buf(:, :, :)

      integer :: w_send_req
      integer :: w_recv_req
      real(kind = REAL32), allocatable :: w_send_buf(:, :, :)
      real(kind = REAL32), allocatable :: w_recv_buf(:, :, :)

      integer :: e_send_req
      integer :: e_recv_req
      real(kind = REAL32), allocatable :: e_send_buf(:, :, :)
      real(kind = REAL32), allocatable :: e_recv_buf(:, :, :)

      nx = size(field, 1) - 2 * num_halo
      ny = size(field, 2) - 2 * num_halo
      nz = size(field, 3)

      ! allocate and prepost recieves
      allocate(n_recv_buf(nx, num_halo, nz))
      allocate(s_recv_buf(nx, num_halo, nz))
      allocate(e_recv_buf(num_halo, ny + 2 * num_halo, nz))
      allocate(w_recv_buf(num_halo, ny + 2 * num_halo, nz))

      !$omp critical
      call MPI_Irecv(n_recv_buf, size(n_recv_buf), MPI_FLOAT, p%top(), 0, p%comm(), n_recv_req, ierror)
      call error(ierror /= MPI_SUCCESS, 'Problem with MPI_Irecv', code = ierror)
      call MPI_Irecv(s_recv_buf, size(s_recv_buf), MPI_FLOAT, p%bottom(), 0, p%comm(), s_recv_req, ierror)
      call error(ierror /= MPI_SUCCESS, 'Problem with MPI_Irecv', code = ierror)
      call MPI_Irecv(e_recv_buf, size(e_recv_buf), MPI_FLOAT, p%right(), 0, p%comm(), e_recv_req, ierror)
      call error(ierror /= MPI_SUCCESS, 'Problem with MPI_Irecv', code = ierror)
      call MPI_Irecv(w_recv_buf, size(w_recv_buf), MPI_FLOAT, p%left(), 0, p%comm(), w_recv_req, ierror)
      call error(ierror /= MPI_SUCCESS, 'Problem with MPI_Irecv', code = ierror)
      !$omp end critical

      ! allocate send buffers
      allocate(n_send_buf(nx, num_halo, nz))
      allocate(s_send_buf(nx, num_halo, nz))
      allocate(e_send_buf(num_halo, ny + 2 * num_halo, nz))
      allocate(w_send_buf(num_halo, ny + 2 * num_halo, nz))

      ! send north and south
      n_send_buf(:, :, :) = field(1 + num_halo:nx + num_halo, 1 + num_halo:2 * num_halo, :)
      !$omp critical
      call MPI_Isend(n_send_buf, size(n_send_buf), MPI_FLOAT, p%bottom(), 0, p%comm(), n_send_req, ierror)
      call error(ierror /= MPI_SUCCESS, 'Problem with MPI_Isend', code = ierror)
      s_send_buf(:, :, :) = field(1 + num_halo:nx + num_halo, 1 + ny:ny + num_halo, :)
      call MPI_Isend(s_send_buf, size(s_send_buf), MPI_FLOAT, p%top(), 0, p%comm(), s_send_req, ierror)
      call error(ierror /= MPI_SUCCESS, 'Problem with MPI_Isend', code = ierror)
      !$omp end critical

      ! unpack north and south
      !$omp critical
      call MPI_Wait(n_recv_req, MPI_STATUS_IGNORE, ierror)
      call error(ierror /= MPI_SUCCESS, 'Problem with MPI_Wait', code = ierror)
      call MPI_Wait(s_recv_req, MPI_STATUS_IGNORE, ierror)
      call error(ierror /= MPI_SUCCESS, 'Problem with MPI_Wait', code = ierror)
      !$omp end critical
      field(1 + num_halo:nx + num_halo, 1 + ny + num_halo:ny + 2 * num_halo, :) = n_recv_buf(:, :, :)
      field(1 + num_halo:nx + num_halo, 1:num_halo, :) = s_recv_buf(:, :, :)

      ! send east and west
      e_send_buf(:, :, :) = field(1 + num_halo:2 * num_halo, :, :)
      !$omp critical
      call MPI_Isend(e_send_buf, size(e_send_buf), MPI_FLOAT, p%left(), 0, p%comm(), e_send_req, ierror)
      call error(ierror /= MPI_SUCCESS, 'Problem with MPI_Isend', code = ierror)
      w_send_buf(:, :, :) = field(1 + nx:nx + num_halo, :, :)
      call MPI_Isend(w_send_buf, size(w_send_buf), MPI_FLOAT, p%right(), 0, p%comm(), w_send_req, ierror)
      call error(ierror /= MPI_SUCCESS, 'Problem with MPI_Isend', code = ierror)

      ! unpack east and west
      !$omp critical
      call MPI_Wait(e_recv_req, MPI_STATUS_IGNORE, ierror)
      call error(ierror /= MPI_SUCCESS, 'Problem with MPI_Wait', ierror)
      call MPI_Wait(w_recv_req, MPI_STATUS_IGNORE, ierror)
      call error(ierror /= MPI_SUCCESS, 'Problem with MPI_Wait', code = ierror)
      !$omp end critical
      field(1 + nx + num_halo:nx + 2 * num_halo, :, :) = e_recv_buf(:, :, :)
      field(1:num_halo, :, :) = w_recv_buf(:, :, :)

      ! await sends
      !$omp critical
      call MPI_Wait(n_send_req, MPI_STATUS_IGNORE, ierror)
      call error(ierror /= MPI_SUCCESS, 'Problem with MPI_Wait', code = ierror)
      call MPI_Wait(s_send_req, MPI_STATUS_IGNORE, ierror)
      call error(ierror /= MPI_SUCCESS, 'Problem with MPI_Wait', code = ierror)
      call MPI_Wait(e_send_req, MPI_STATUS_IGNORE, ierror)
      call error(ierror /= MPI_SUCCESS, 'Problem with MPI_Wait', code = ierror)
      call MPI_Wait(w_send_req, MPI_STATUS_IGNORE, ierror)
      call error(ierror /= MPI_SUCCESS, 'Problem with MPI_Wait', code = ierror)
      !$omp end critical
    end subroutine
end module

! vim: set filetype=fortran expandtab tabstop=2 softtabstop=2 :
