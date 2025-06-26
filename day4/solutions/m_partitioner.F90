!> @brief Partitioner class
!>
!> Ported from partioner.py
!>
!> @author Michal Sudwoj
!> @author Oliver Fuhrer
!> @date 2020-06-15
module m_partitioner
  use, intrinsic :: iso_fortran_env, only: REAL32, REAL64, error_unit
  use mpi, only: &
    MPI_FLOAT, MPI_DOUBLE, MPI_SUCCESS, &
    MPI_Comm_Rank, MPI_Comm_Size, MPI_Barrier
    ! MPI_Scatter, MPI_Gather, MPI_Allgather
    ! see https://github.com/pmodels/mpich/issues/3568
  use m_utils, only: error

  implicit none
  private

  logical, parameter :: debug = .false.

  !> @brief 2-dimensional domain decomposition of a 3-dimensional computational grid among MPI ranks on a communicator.
  type, public :: Partitioner
    private

    integer              :: comm_
    integer              :: rank_
    integer              :: num_ranks_
    integer              :: num_halo_
    integer              :: size_(2)
    integer, allocatable :: domains_(:, :)
    integer, allocatable :: shapes_(:, :)
    integer              :: domain_(4)
    integer              :: shape_(3)
    integer              :: max_shape_(3)
    logical              :: periodic_(2)
    integer              :: global_shape_(3)
  contains
    procedure, public :: comm
    procedure, public :: num_halo
    procedure, public :: periodic
    procedure, public :: rank
    procedure, public :: num_ranks
    procedure, public :: shape => shape_f
    procedure, public :: global_shape
    procedure, public :: size => size_f
    procedure, public :: position
    procedure, public :: left
    procedure, public :: right
    procedure, public :: top
    procedure, public :: bottom
    generic,   public :: scatter => scatter_f32, scatter_f64
    generic,   public :: gather => gather_f32, gather_f64
    procedure, public :: compute_domain

    procedure         :: scatter_f32
    procedure         :: scatter_f64
    procedure         :: gather_f32
    procedure         :: gather_f64
    procedure         :: setup_grid
    procedure         :: get_neighbor_rank
    procedure, nopass :: cyclic_offset
    procedure         :: setup_domain
    procedure, nopass :: distribute_to_bins
    procedure, nopass :: cumsum
    procedure, nopass :: find_max_shape
    procedure         :: rank_to_position
    procedure         :: position_to_rank
  end type Partitioner

  interface Partitioner
    module procedure constructor
  end interface

contains

  type(Partitioner) function constructor(comm, domain, num_halo, periodic) result(this)
    integer, intent(in) :: comm
    integer, intent(in) :: domain(3)
    integer, intent(in) :: num_halo
    logical, intent(in), optional :: periodic(2)

    integer :: ierror
    logical :: periodic_(2)
    integer :: rank

    if (present(periodic)) then
      periodic_ = periodic
    else
      periodic_ = [.true., .true.]
    end if

    call error(.not. (domain(1) > 0 .and. domain(2) > 0 .and. domain(3) > 0), 'Invalid domain specification (negative size)')
    call error(.not. (num_halo >= 0), 'Number of halo points must be zero or positive')

    this%comm_ = comm
    this%num_halo_ = num_halo
    this%periodic_ = periodic_

    call MPI_Comm_Rank(this%comm_, this%rank_, ierror)
    call error(ierror /= MPI_SUCCESS, 'Problem with MPI_Comm_Rank', code = ierror)

    call MPI_Comm_Size(this%comm_, this%num_ranks_, ierror)
    call error(ierror /= MPI_SUCCESS, 'Problem with MPI_Comm_Size', code = ierror)

    this%global_shape_(1) = domain(1) + 2 * num_halo
    this%global_shape_(2) = domain(2) + 2 * num_halo
    this%global_shape_(3) = domain(3)

    call this%setup_grid()
    call this%setup_domain(domain, num_halo)

    if (debug) then
      call MPI_Barrier(this%comm_, ierror)

      if (this%rank_ == 0) then
        write(error_unit, '(a33, /, a30, 3(i6), /, a30, 1(i6), /, a30, 2(l6))') &
          '====== Global information =======', &
          'Domain size (nx,ny,nz): ', this%global_shape_ - [2 * num_halo, 2 * num_halo, 0], &
          'Halo width: ', this%num_halo_, &
          'Periodicity (x,y): ', this%periodic_
      end if
      call flush(error_unit)
      call MPI_Barrier(this%comm_, ierror)

      do rank = 0, this%num_ranks_ - 1
        if (this%rank_ == rank) then
          call error(ierror /= MPI_SUCCESS, 'Problem with MPI_Barrier', code = ierror)
          write(error_unit, '(a12, i13, a8, /, a30, 2(i6), /, a30, 3(i6), /, a30, 4(i6), / a30, 4(i6))') &
          '====== Rank ', rank, ' =======', &
          'Position (x,y): ', this%position(), &
          'Subdomain size (nx,ny,nz): ', this%shape() - [2 * num_halo, 2 * num_halo, 0], &
          'Position on global grid: ', this%compute_domain(), &
          'Neighbors (trbl): ', this%top(), this%right(), this%bottom(), this%left()
        end if
        call flush(error_unit)
        call MPI_Barrier(this%comm_, ierror)
      end do

      if (this%rank_ == 0) then
        write(error_unit, '(a33)') '================================='
      end if
      call flush(error_unit)
      call MPI_Barrier(this%comm_, ierror)
    end if

end function

  !> @brief Returns the MPI communicator used to setup the partioner
  integer pure function comm(this)
    class(Partitioner), intent(in) :: this

    comm = this%comm_
  end function

  !> @brief Returns the number of halo points
  integer pure function num_halo(this)
    class(Partitioner), intent(in) :: this

    num_halo = this%num_halo_
  end function

  !> @brief Returns the periodicity of all dimensions
  pure function periodic(this)
    class(Partitioner), intent(in) :: this
    logical :: periodic(2)

    periodic = this%periodic_
  end function

  !> @brief Returns the rank of the current MPI worker
  integer pure function rank(this)
    class(Partitioner), intent(in) :: this

    rank = this%rank_
  end function

  !> @brief Returns the numer of ranks that have been distributed by this partitioner
  integer pure function num_ranks(this)
    class(Partitioner), intent(in) :: this

    num_ranks = this%num_ranks_
  end function

  !> @brief Returns the shape of a local field (including halo points)
  pure function shape_f(this)
    class(Partitioner), intent(in) :: this
    integer :: shape_f(3)

    shape_f = this%shape_
  end function

  !> @brief Returns the shape of a global field (including halo points)
  pure function global_shape(this)
    class(Partitioner), intent(in) :: this
    integer :: global_shape(3)

    global_shape = this%global_shape_
  end function

  !> @brief Dimensions of the two-dimensional worker grid
  pure function size_f(this) result(size)
    class(Partitioner), intent(in) :: this
    integer :: size(2)

    size = this%size_
  end function

  !> @brief Position of the current rank on two-dimensional worker grid.
  pure function position(this)
    class(Partitioner), intent(in) :: this
    integer :: position(2)

    position = this%rank_to_position(this%rank_)
  end function

  !> @brief Returns the rank of the left neighbor
  integer pure function left(this) result(rank)
    class(Partitioner), intent(in) :: this
    integer, parameter :: position(2) = [-1, 0]

    rank = this%get_neighbor_rank(position)
  end function

  !> @brief Returns the rank of the right neighbor
  integer pure function right(this) result(rank)
    class(Partitioner), intent(in) :: this
    integer, parameter :: position(2) = [+1, 0]

    rank = this%get_neighbor_rank(position)
  end function

  !> @brief Returns the rank of the top neighbor
  integer pure function top(this) result(rank)
    class(Partitioner), intent(in) :: this
    integer, parameter :: position(2) = [0, +1]

    rank = this%get_neighbor_rank(position)
  end function

  !> @brief Returns the rank of the bottom neighbor
  integer pure function bottom(this) result(rank)
    class(Partitioner), intent(in) :: this
    integer, parameter :: position(2) = [0, -1]

    rank = this%get_neighbor_rank(position)
  end function

  !> @brief Scatter a global field from a root rank to the workers (f32)
  function scatter_f32(this, field, root) result(r)
    class(Partitioner), intent(in) :: this
    real(REAL32), intent(in) :: field(:, :, :)
    integer, optional :: root
    real(REAL32), allocatable :: r(:, :, :)

    integer :: root_
    real(REAL32), allocatable :: sendbuf(:, :, :, :)
    integer :: rank
    integer :: j_start
    integer :: i_start
    integer :: j_end
    integer :: i_end
    real(REAL32), allocatable :: recvbuf(:, :, :)
    integer :: ierror

    if (present(root)) then
      root_ = root
    else
      root_ = 0
    end if

    if (this%rank_ == root_) then
      call error(any(shape(field) /= this%global_shape_), 'Field does not have the correct shape')
    end if
    call error(.not. (0 <= root_ .and. root_ < this%num_ranks_), 'Root processor must be a valid rank')

    if (this%num_ranks_ == 1) then
      r = field
      return
    end if

    if (this%rank_ == root_) then
      allocate(sendbuf(this%max_shape_(1), this%max_shape_(2), this%max_shape_(3), 0:this%num_ranks_ - 1))

      do rank = 0, this%num_ranks_ - 1
        i_start = this%domains_(rank, 1) + 1
        j_start = this%domains_(rank, 2) + 1
        i_end   = this%domains_(rank, 3)
        j_end   = this%domains_(rank, 4)

        sendbuf(:i_end - i_start + 1, :j_end - j_start + 1, :, rank) = field(i_start:i_end, j_start:j_end, :)
      end do
    else
      allocate(sendbuf(0, 0, 0, 0))
    end if

    allocate(recvbuf(this%max_shape_(1), this%max_shape_(2), this%max_shape_(3)))
    call MPI_Scatter( &
      sendbuf, size(recvbuf), MPI_FLOAT, &
      recvbuf, size(recvbuf), MPI_FLOAT, &
      root_, this%comm_, ierror &
    )
    call error(ierror /= 0, 'Problem with MPI_Scatter', code = ierror)

    i_start = this%domain_(1)
    j_start = this%domain_(2)
    i_end   = this%domain_(3)
    j_end   = this%domain_(4)

    r = recvbuf(:i_end - i_start + 1, :j_end - j_start + 1, :)
  end function

  !> @brief Scatter a global field from a root rank to the workers (f64)
  function scatter_f64(this, field, root) result(r)
    class(Partitioner), intent(in) :: this
    real(REAL64), intent(in) :: field(:, :, :)
    integer, optional :: root
    real(REAL64), allocatable :: r(:, :, :)

    integer :: root_
    real(REAL64), allocatable :: sendbuf(:, :, :, :)
    integer :: rank
    integer :: j_start
    integer :: i_start
    integer :: j_end
    integer :: i_end
    real(REAL64), allocatable :: recvbuf(:, :, :)
    integer :: ierror

    if (present(root)) then
      root_ = root
    else
      root_ = 0
    end if

    if (this%rank_ == root_) then
      call error(any(shape(field) /= this%global_shape_), 'Field does not have the correct shape')
    end if
    call error(.not. (0 <= root_ .and. root_ < this%num_ranks_), 'Root processor must be a valid rank')

    if (this%num_ranks_ == 1) then
      r = field
      return
    end if

    if (this%rank_ == root_) then
      allocate(sendbuf(this%max_shape_(1), this%max_shape_(2), this%max_shape_(3), 0:this%num_ranks_ - 1))

      do rank = 0, this%num_ranks_ - 1
        i_start = this%domains_(rank, 1) + 1
        j_start = this%domains_(rank, 2) + 1
        i_end   = this%domains_(rank, 3)
        j_end   = this%domains_(rank, 4)

        sendbuf(:i_end - i_start + 1, :j_end - j_start + 1, :, rank) = field(i_start:i_end, j_start:j_end, :)
      end do
    else
      allocate(sendbuf(0, 0, 0, 0))
    end if

    allocate(recvbuf(this%max_shape_(1), this%max_shape_(2), this%max_shape_(3)))
    call MPI_Scatter( &
      sendbuf, size(recvbuf), MPI_DOUBLE, &
      recvbuf, size(recvbuf), MPI_DOUBLE, &
      root_, this%comm_, ierror &
    )
    call error(ierror /= 0, 'Problem with MPI_Scatter', code = ierror)

    i_start = this%domain_(1)
    j_start = this%domain_(2)
    i_end   = this%domain_(3)
    j_end   = this%domain_(4)

    r = recvbuf(:i_end - i_start + 1, :j_end - j_start + 1, :)
  end function

  !> @brief Gather a distributed field from workers to a single global field on a root rank (f32)
  function gather_f32(this, field, root) result(r)
    class(Partitioner), intent(in) :: this
    real(REAL32), intent(in) :: field(:, :, :)
    integer, optional :: root
    real(REAL32), allocatable :: r(:, :, :)

    integer :: root_
    integer :: j_start
    integer :: i_start
    integer :: j_end
    integer :: i_end
    real(REAL32), allocatable :: sendbuf(:, :, :)
    real(REAL32), allocatable :: recvbuf(:, :, :, :)
    integer :: ierror
    real(REAL32), allocatable :: global_field(:, :, :)
    integer :: rank

    if (present(root)) then
      root_ = root
    else
      root_ = 0
    end if

    call error(any(shape(field) /= this%shape_), 'Field does not have the correct shape')
    call error(.not. (-1 <= root_ .and. root_ < this%num_ranks_), 'Root processor must be -1 (all) or a valid rank')

    if (this%num_ranks_ == 1) then
      r = field
      return
    end if

    i_start = this%domain_(1)
    j_start = this%domain_(2)
    i_end   = this%domain_(3)
    j_end   = this%domain_(4)

    allocate(sendbuf(this%max_shape_(1), this%max_shape_(2), this%max_shape_(3)))
    sendbuf(:i_end - i_start + 1, :j_end - j_start + 1, :) = field

    if (this%rank_ == root_ .or. root_ == -1) then
      allocate(recvbuf(this%max_shape_(1), this%max_shape_(2), this%max_shape_(3), 0:this%num_ranks_ - 1))
    else
      allocate(recvbuf(0, 0, 0, 0))
    end if

    if (root_ > -1) then
      call MPI_Gather( &
        sendbuf, size(sendbuf), MPI_FLOAT, &
        recvbuf, size(sendbuf), MPI_FLOAT, &
        root_, this%comm_, ierror &
      )
      call error(ierror /= MPI_SUCCESS, 'Problem with MPI_Gather', code = ierror)
    else
      call MPI_Allgather( &
        sendbuf, size(sendbuf), MPI_FLOAT, &
        recvbuf, size(sendbuf), MPI_FLOAT, &
        this%comm_, ierror &
      )
      call error(ierror /= MPI_SUCCESS, 'Problem with MPI_Allgather', code = ierror)
    end if

    if (this%rank_ == root_ .or. root_ == -1) then
      allocate(global_field(this%global_shape_(1), this%global_shape_(2), this%global_shape_(3)))
      do rank = 0, this%num_ranks_ - 1
        i_start = this%domains_(rank, 1) + 1
        j_start = this%domains_(rank, 2) + 1
        i_end   = this%domains_(rank, 3)
        j_end   = this%domains_(rank, 4)

        global_field(i_start:i_end, j_start:j_end, :) = recvbuf(:i_end - i_start + 1, :j_end - j_start + 1, :, rank)
      end do

      r = global_field
    else
      allocate(r(0, 0, 0))
    end if
  end function

  !> @brief Gather a distributed field from workers to a single global field on a root rank (f64)
  function gather_f64(this, field, root) result(r)
    class(Partitioner), intent(in) :: this
    real(REAL64), intent(in) :: field(:, :, :)
    integer, optional :: root
    real(REAL64), allocatable :: r(:, :, :)

    integer :: root_
    integer :: j_start
    integer :: i_start
    integer :: j_end
    integer :: i_end
    real(REAL64), allocatable :: sendbuf(:, :, :)
    real(REAL64), allocatable :: recvbuf(:, :, :, :)
    integer :: ierror
    real(REAL64), allocatable :: global_field(:, :, :)
    integer :: rank

    if (present(root)) then
      root_ = root
    else
      root_ = 0
    end if

    call error(any(shape(field) /= this%shape_), 'Field does not have the correct shape')
    call error(.not. (-1 <= root_ .and. root_ < this%num_ranks_), 'Root processor must be -1 (all) or a valid rank')

    if (this%num_ranks_ == 1) then
      r = field
      return
    end if

    i_start = this%domain_(1)
    j_start = this%domain_(2)
    i_end   = this%domain_(3)
    j_end   = this%domain_(4)

    allocate(sendbuf(this%max_shape_(1), this%max_shape_(2), this%max_shape_(3)))
    sendbuf(:i_end - i_start + 1, :j_end - j_start + 1, :) = field

    if (this%rank_ == root_ .or. root_ == -1) then
      allocate(recvbuf(this%max_shape_(1), this%max_shape_(2), this%max_shape_(3), 0:this%num_ranks_ - 1))
    else
      allocate(recvbuf(0, 0, 0, 0))
    end if

    if (root_ > -1) then
      call MPI_Gather( &
        sendbuf, size(sendbuf), MPI_DOUBLE, &
        recvbuf, size(sendbuf), MPI_DOUBLE, &
        root_, this%comm_, ierror &
      )
      call error(ierror /= MPI_SUCCESS, 'Problem with MPI_Gather', code = ierror)
    else
      call MPI_Allgather( &
        sendbuf, size(sendbuf), MPI_DOUBLE, &
        recvbuf, size(sendbuf), MPI_DOUBLE, &
        this%comm_, ierror &
      )
      call error(ierror /= MPI_SUCCESS, 'Problem with MPI_Allgather', code = ierror)
    end if

    if (this%rank_ == root_ .or. root_ == -1) then
      allocate(global_field(this%global_shape_(1), this%global_shape_(2), this%global_shape_(3)))
      do rank = 0, this%num_ranks_ - 1
        i_start = this%domains_(rank, 1) + 1
        j_start = this%domains_(rank, 2) + 1
        i_end   = this%domains_(rank, 3)
        j_end   = this%domains_(rank, 4)

        global_field(i_start:i_end, j_start:j_end, :) = recvbuf(:i_end - i_start + 1, :j_end - j_start + 1, :, rank)
      end do

      r = global_field
    else
      allocate(r(0, 0, 0))
    end if
  end function

  !> @brief Return position of subdomain withoug halo on the global domain
  pure function compute_domain(this) result(subdomain)
    class(Partitioner), intent(in) :: this
    integer :: subdomain(4)

    subdomain(1) = this%domain_(1) + this%num_halo_
    subdomain(2) = this%domain_(2) + this%num_halo_
    subdomain(3) = this%domain_(3) - this%num_halo_
    subdomain(4) = this%domain_(4) - this%num_halo_
  end function

  !> @brief Distribute ranks onto a Cartesion grid of workers
  subroutine setup_grid(this)
    class(Partitioner), intent(inout) :: this

    integer :: ranks_x

    do ranks_x = floor(sqrt(1.0 * this%num_ranks_)), 1, -1
      if (mod(this%num_ranks_, ranks_x) == 0) then
        exit
      end if
    end do

    this%size_(1) = ranks_x
    this%size_(2) = this%num_ranks_ / ranks_x
  end subroutine

  !> @brief Get the rank ID of a neighboring rank at a certain offset relative to the current rank
  integer pure function get_neighbor_rank(this, offset) result(rank)
    class(Partitioner), intent(in) :: this
    integer, intent(in) :: offset(2)

    integer :: pos(2)
    integer :: pos_offset(2)

    pos = this%rank_to_position(this%rank_)
    pos_offset(1) = this%cyclic_offset(pos(1), offset(1), this%size_(1), this%periodic_(1))
    pos_offset(2) = this%cyclic_offset(pos(2), offset(2), this%size_(2), this%periodic_(2))

    rank = this%position_to_rank(pos_offset)
  end function

  !> @brief Add offset with cyclic boundary conditions
  integer pure function cyclic_offset(pos, offset, size, periodic) result(p)
    integer, intent(in) :: pos
    integer, intent(in) :: offset
    integer, intent(in) :: size
    logical, intent(in) :: periodic

    p = pos + offset
    if (periodic) then
      do while (p < 1)
        p = p + size
      end do
      do while (p > size)
        p = p - size
      end do
    end if

    if (p < 1 .or. p > size) then
      p = -1
    end if

  end function

  !> @brief Distribute the points of the computational grid onto the Cartesion grid of workers
  subroutine setup_domain(this, shape, num_halo)
    class(Partitioner), intent(inout) :: this
    integer, intent(in) :: shape(3)
    integer, intent(in) :: num_halo

    integer :: size_z
    integer, allocatable :: size_y(:)
    integer, allocatable :: size_x(:)
    integer, allocatable :: pos_y(:)
    integer, allocatable :: pos_x(:)
    integer :: pos(2)
    integer :: rank

    size_x = this%distribute_to_bins(shape(1), this%size_(1))
    size_y = this%distribute_to_bins(shape(2), this%size_(2))
    size_z = shape(3)

    pos_x = this%cumsum(size_x, 1 + num_halo)
    pos_y = this%cumsum(size_y, 1 + num_halo)

    allocate(this%domains_(0:this%num_ranks_ - 1, 4))
    allocate(this%shapes_(0:this%num_ranks_ - 1, 3))

    do rank = 0, this%num_ranks_ - 1
      pos = this%rank_to_position(rank)
      this%domains_(rank, 1) = pos_x(pos(1)) - num_halo
      this%domains_(rank, 2) = pos_y(pos(2)) - num_halo
      this%domains_(rank, 3) = pos_x(pos(1) + 1) + num_halo - 1
      this%domains_(rank, 4) = pos_y(pos(2) + 1) + num_halo - 1

      this%shapes_(rank, 1) = this%domains_(rank, 3) - this%domains_(rank, 1) + 1
      this%shapes_(rank, 2) = this%domains_(rank, 4) - this%domains_(rank, 2) + 1
      this%shapes_(rank, 3) = size_z
    end do

    this%domain_ = this%domains_(this%rank_, :)
    this%shape_  = this%shapes_(this%rank_, :)

    this%max_shape_ = this%find_max_shape(this%shapes_)
  end subroutine

  !> @brief Distribute a number of elements to a number of bins
  pure function distribute_to_bins(num, bins) result(bin_size)
    integer, intent(in) :: num
    integer, intent(in) :: bins
    integer, allocatable :: bin_size(:)

    integer :: n
    integer :: extend
    integer :: start_extend

    n = num / bins
    allocate(bin_size(bins))
    bin_size = n
    extend = num - n * bins
    if (extend > 0) then
      start_extend = bins / 2 - extend / 2 + 1
      bin_size(start_extend:start_extend + extend - 1) = bin_size(start_extend:start_extend + extend - 1) + 1
    end if
  end function

  !> @brief Cumulative sum with an optional initial value
  pure function cumsum(array, initial_value)
    integer, intent(in) :: array(:)
    integer, intent(in), optional :: initial_value
    integer, allocatable :: cumsum(:)

    integer :: n, i
    integer :: initial_value_

    if (present(initial_value)) then
      initial_value_ = initial_value
    else
      initial_value_ = 0
    end if

    n = size(array)
    allocate(cumsum(n + 1))
    cumsum(1) = initial_value_
    do i = 1, n
      cumsum(i + 1) = cumsum(i) + array(i)
    end do
  end function

  !> @brief Find maximum dimensions of subdomains across all ranks
  function find_max_shape(shapes) result(max_shape)
    integer, intent(in) :: shapes(:, :)
    integer :: max_shape(3)

    integer :: shape

    call error(size(shapes, 2) /= 3, 'Wrong shapes size')

    max_shape = shapes(1, :)
    do shape = 2, size(shapes, 1)
      max_shape(1) = max(max_shape(1), shapes(shape, 1))
      max_shape(2) = max(max_shape(2), shapes(shape, 2))
      max_shape(3) = max(max_shape(3), shapes(shape, 3))
    end do
  end function

  !> @brief Find position of rank on worker grid
  pure function rank_to_position(this, rank) result(pos)
    class(Partitioner), intent(in) :: this
    integer, intent(in) :: rank
    integer :: pos(2)

    pos(1) = mod(rank, this%size_(1)) + 1
    pos(2) = rank / this%size_(1) + 1
  end function

  !> @brief Find rank given a position on the worker grid
  integer pure function position_to_rank(this, pos) result(rank)
    class(Partitioner), intent(in) :: this
    integer, intent(in) :: pos(2)

    rank = (pos(2) - 1) * this%size_(1) + (pos(1) - 1)
  end function

end module m_partitioner

! vim: set filetype=fortran expandtab tabstop=2 softtabstop=2 :
