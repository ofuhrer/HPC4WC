
! ******************************************************
!     Program: stencil2d
!      Author: Oliver Fuhrer
!       Email: oliverf@vulcan.com
!        Date: 20.05.2020
! Description: Simple stencil example (4th-order diffusion)
! ******************************************************

! Driver for apply_diffusion() that sets up fields and does timings
program stencil2d
    use mpi
    use openacc
    implicit none

    integer, parameter :: wp = kind(1.0d0)
    integer :: nx, ny, nz, num_iter
    integer :: num_halo = 2
    real(wp) :: alpha = 1.0_wp / 32.0_wp

    real(wp), allocatable :: in_field(:,:,:), out_field(:,:,:)
    integer :: local_nx, local_ny, local_nz
    integer :: rank, num_procs, ierr
    integer :: cart_comm, dims(3)
    logical :: periods(3)
    integer :: coords(3)
    integer :: left, right, up, down
    real(wp) :: start_time, end_time

    call MPI_Init(ierr)
    call MPI_Comm_rank(MPI_COMM_WORLD, rank, ierr)
    call MPI_Comm_size(MPI_COMM_WORLD, num_procs, ierr)

    call read_input()
    call setup_domain_decomposition()
    call allocate_fields()
    call initialize_fields()

    start_time = MPI_Wtime()

    call apply_diffusion(in_field, out_field, alpha, num_iter)

    end_time = MPI_Wtime()

    if (rank == 0) then
        print *, "Total time: ", end_time - start_time, " seconds"
    end if

    call cleanup()
    call MPI_Finalize(ierr)

contains

    ! Reads input parameters from command line arguments.
    ! Parameters:
    !   nx: Integer, grid size in x-direction
    !   ny: Integer, grid size in y-direction
    !   nz: Integer, grid size in z-direction
    !   num_iter: Integer, number of iterations
    ! Broadcasts the input parameters to all processes.
    subroutine read_input()
        character(len=32) :: arg
        if (rank == 0) then
            if (command_argument_count() /= 8) then
                print *, "Usage: ./stencil2d-openacc.x --nx <nx> --ny <ny> --nz <nz> --num_iter <num_iter>"
                call MPI_Abort(MPI_COMM_WORLD, 1, ierr)
            end if
            call get_command_argument(2, arg)
            read(arg, *) nx
            call get_command_argument(4, arg)
            read(arg, *) ny
            call get_command_argument(6, arg)
            read(arg, *) nz
            call get_command_argument(8, arg)
            read(arg, *) num_iter
            print *, "Input parameters:"
            print *, "nx =", nx, "ny =", ny, "nz =", nz, "num_iter =", num_iter
        end if
        call MPI_Bcast(nx, 1, MPI_INTEGER, 0, MPI_COMM_WORLD, ierr)
        call MPI_Bcast(ny, 1, MPI_INTEGER, 0, MPI_COMM_WORLD, ierr)
        call MPI_Bcast(nz, 1, MPI_INTEGER, 0, MPI_COMM_WORLD, ierr)
        call MPI_Bcast(num_iter, 1, MPI_INTEGER, 0, MPI_COMM_WORLD, ierr)
    end subroutine read_input

    ! Sets up the domain decomposition for parallel processing.
    ! Creates a 3D Cartesian communicator and calculates local domain sizes.
    ! Determines neighboring processes for halo exchange.
    subroutine setup_domain_decomposition()
        dims = 0
        periods = [.false., .false., .false.]
        call MPI_Dims_create(num_procs, 3, dims, ierr)
        call MPI_Cart_create(MPI_COMM_WORLD, 3, dims, periods, .true., cart_comm, ierr)
        call MPI_Cart_coords(cart_comm, rank, 3, coords, ierr)
        call MPI_Cart_shift(cart_comm, 0, 1, left, right, ierr)
        call MPI_Cart_shift(cart_comm, 1, 1, down, up, ierr)

        local_nx = nx / dims(1)
        local_ny = ny / dims(2)
        local_nz = nz / dims(3)

        if (coords(1) < mod(nx, dims(1))) local_nx = local_nx + 1
        if (coords(2) < mod(ny, dims(2))) local_ny = local_ny + 1
        if (coords(3) < mod(nz, dims(3))) local_nz = local_nz + 1
    end subroutine setup_domain_decomposition
    
    ! Allocates memory for input and output fields.
    ! Includes halo regions in the allocation.
    subroutine allocate_fields()
        allocate(in_field(0:local_nx+2*num_halo-1, 0:local_ny+2*num_halo-1, 0:local_nz+2*num_halo-1))
        allocate(out_field(0:local_nx+2*num_halo-1, 0:local_ny+2*num_halo-1, 0:local_nz+2*num_halo-1))
    end subroutine allocate_fields

    ! Initializes the input and output fields.
    ! Sets initial values for the simulation domain.
    subroutine initialize_fields()
        integer :: i, j, k
        in_field = 0.0_wp
        do k = num_halo, local_nz+num_halo-1
            do j = num_halo, local_ny+num_halo-1
                do i = num_halo, local_nx+num_halo-1
                    if (i+coords(1)*local_nx >= nx/4 .and. i+coords(1)*local_nx < 3*nx/4 .and. &
                        j+coords(2)*local_ny >= ny/4 .and. j+coords(2)*local_ny < 3*ny/4 .and. &
                        k+coords(3)*local_nz >= nz/4 .and. k+coords(3)*local_nz < 3*nz/4) then
                        in_field(i,j,k) = 1.0_wp
                    end if
                end do
            end do
        end do
        out_field = in_field
    end subroutine initialize_fields

    ! Integrate 4th-order diffusion equation by a certain number of iterations.
    !
    !  in_field          -- input field (nx x ny x nz with halo in x- and y-direction)
    !  out_field         -- result (must be same size as in_field)
    !  alpha             -- diffusion coefficient (dimensionless)
    !  num_iter          -- number of iterations to execute
    !
    subroutine apply_diffusion(in_field, out_field, alpha, num_iter)
        real(wp), intent(inout) :: in_field(0:, 0:, 0:), out_field(0:, 0:, 0:)
        real(wp), intent(in) :: alpha
        integer, intent(in) :: num_iter
        real(wp), allocatable :: tmp_field(:,:,:)
        integer :: iter, i, j, k

        allocate(tmp_field(0:local_nx+2*num_halo-1, 0:local_ny+2*num_halo-1, 0:local_nz+2*num_halo-1))

        !$acc data copy(in_field, out_field) create(tmp_field)
        do iter = 1, num_iter
            call exchange_halos(in_field)

            !$acc parallel loop collapse(3)
            do k = num_halo, local_nz+num_halo-1
                do j = num_halo, local_ny+num_halo-1
                    do i = num_halo, local_nx+num_halo-1
                        tmp_field(i,j,k) = -4.0_wp * in_field(i,j,k) + &
                                           in_field(i-1,j,k) + in_field(i+1,j,k) + &
                                           in_field(i,j-1,k) + in_field(i,j+1,k)
                    end do
                end do
            end do

            !$acc parallel loop collapse(3)
            do k = num_halo, local_nz+num_halo-1
                do j = num_halo, local_ny+num_halo-1
                    do i = num_halo, local_nx+num_halo-1
                        out_field(i,j,k) = in_field(i,j,k) - alpha * tmp_field(i,j,k)
                    end do
                end do
            end do

            !$acc parallel
            in_field = out_field
            !$acc end parallel
        end do
        !$acc end data

        deallocate(tmp_field)
    end subroutine apply_diffusion

    ! Exchanges halo regions between neighboring processes.
    ! Parameters:
    !   field: Real array, the field to exchange halos for
    ! Uses non-blocking MPI communication for efficiency.
    subroutine exchange_halos(field)
        real(wp), intent(inout) :: field(0:, 0:, 0:)
        integer :: requests(12), statuses(MPI_STATUS_SIZE,12)
        integer :: count, datatype
        integer :: sizes(3), subsizes(3), starts(3)
        integer :: req_count

        req_count = 0

        ! X direction
        count = (local_ny + 2*num_halo) * (local_nz + 2*num_halo)
        if (left /= MPI_PROC_NULL) then
            req_count = req_count + 1
            call MPI_Isend(field(num_halo,0,0), count, MPI_DOUBLE_PRECISION, left, 0, cart_comm, requests(req_count), ierr)
        end if
        if (right /= MPI_PROC_NULL) then
            req_count = req_count + 1
            call MPI_Irecv(field(local_nx+num_halo,0,0), count, MPI_DOUBLE_PRECISION, right, 0, cart_comm, requests(req_count), ierr)
        end if
        if (right /= MPI_PROC_NULL) then
            req_count = req_count + 1
            call MPI_Isend(field(local_nx+num_halo-1,0,0), count, MPI_DOUBLE_PRECISION, right, 1, cart_comm, requests(req_count), ierr)
        end if
        if (left /= MPI_PROC_NULL) then
            req_count = req_count + 1
            call MPI_Irecv(field(0,0,0), count, MPI_DOUBLE_PRECISION, left, 1, cart_comm, requests(req_count), ierr)
        end if

        ! Y direction
        sizes = [local_nx+2*num_halo, local_ny+2*num_halo, local_nz+2*num_halo]
        subsizes = [local_nx+2*num_halo, num_halo, local_nz+2*num_halo]
        starts = [0, 0, 0]
        call MPI_Type_create_subarray(3, sizes, subsizes, starts, MPI_ORDER_FORTRAN, MPI_DOUBLE_PRECISION, datatype, ierr)
        call MPI_Type_commit(datatype, ierr)

        if (down /= MPI_PROC_NULL) then
            req_count = req_count + 1
            call MPI_Isend(field(0,num_halo,0), 1, datatype, down, 2, cart_comm, requests(req_count), ierr)
        end if
        if (up /= MPI_PROC_NULL) then
            req_count = req_count + 1
            call MPI_Irecv(field(0,local_ny+num_halo,0), 1, datatype, up, 2, cart_comm, requests(req_count), ierr)
        end if
        if (up /= MPI_PROC_NULL) then
            req_count = req_count + 1
            call MPI_Isend(field(0,local_ny+num_halo-1,0), 1, datatype, up, 3, cart_comm, requests(req_count), ierr)
        end if
        if (down /= MPI_PROC_NULL) then
            req_count = req_count + 1
            call MPI_Irecv(field(0,0,0), 1, datatype, down, 3, cart_comm, requests(req_count), ierr)
        end if

        call MPI_Type_free(datatype, ierr)

        ! Z direction
        sizes = [local_nx+2*num_halo, local_ny+2*num_halo, local_nz+2*num_halo]
        subsizes = [local_nx+2*num_halo, local_ny+2*num_halo, num_halo]
        starts = [0, 0, 0]
        call MPI_Type_create_subarray(3, sizes, subsizes, starts, MPI_ORDER_FORTRAN, MPI_DOUBLE_PRECISION, datatype, ierr)
        call MPI_Type_commit(datatype, ierr)

        if (coords(3) > 0) then
            req_count = req_count + 1
            call MPI_Isend(field(0,0,num_halo), 1, datatype, coords(3)-1, 4, cart_comm, requests(req_count), ierr)
        end if
        if (coords(3) < dims(3)-1) then
            req_count = req_count + 1
            call MPI_Irecv(field(0,0,local_nz+num_halo), 1, datatype, coords(3)+1, 4, cart_comm, requests(req_count), ierr)
        end if
        if (coords(3) < dims(3)-1) then
            req_count = req_count + 1
            call MPI_Isend(field(0,0,local_nz+num_halo-1), 1, datatype, coords(3)+1, 5, cart_comm, requests(req_count), ierr)
        end if
        if (coords(3) > 0) then
            req_count = req_count + 1
            call MPI_Irecv(field(0,0,0), 1, datatype, coords(3)-1, 5, cart_comm, requests(req_count), ierr)
        end if

        call MPI_Waitall(req_count, requests, statuses, ierr)
        call MPI_Type_free(datatype, ierr)
    end subroutine exchange_halos

    ! Deallocates dynamically allocated memory.
    subroutine cleanup()
        deallocate(in_field, out_field)
    end subroutine cleanup

end program stencil2d
