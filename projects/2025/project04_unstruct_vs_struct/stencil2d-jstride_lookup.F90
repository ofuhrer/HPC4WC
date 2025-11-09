! ******************************************************
!     Program: stencil2d
!      Author: Oliver Fuhrer
!       Email: oliverf@vulcan.com
!        Date: 20.05.2020
! Description: Simple stencil example (4th-order diffusion)
! ******************************************************

! Driver for apply_diffusion() that sets up fields and does timings
program main
    use m_utils, only: timer_start, timer_end, timer_get, is_master, num_rank, write_field_to_file
    implicit none

    ! constants
    integer, parameter :: wp = 4
    
    ! local
    integer :: nx, ny, nz, num_iter
    logical :: scan
    
    integer :: num_halo = 2
    real (kind=wp) :: alpha = 1.0_wp / 32.0_wp

    ! Unstructured grid: fields are now (nx*ny, nz)
    real (kind=wp), allocatable :: in_field(:, :)
    real (kind=wp), allocatable :: out_field(:, :)
    real (kind=wp), allocatable :: cart_field(:,:,:)

    integer :: timer_work
    real (kind=8) :: runtime

    integer :: cur_setup, num_setups = 1
    integer :: nx_setups(7) = (/ 16, 32, 48, 64, 96, 128, 192 /)
    integer :: ny_setups(7) = (/ 16, 32, 48, 64, 96, 128, 192 /)

    integer, allocatable :: neighbor_lookup(:,:)

#ifdef CRAYPAT
    include "pat_apif.h"
    integer :: istat
    call PAT_record( PAT_STATE_OFF, istat )
#endif

    call init()

    if ( is_master() ) then
        write(*, '(a)') '# ranks nx ny nz num_iter time'
        write(*, '(a)') 'data = np.array( [ \'
    end if

    if ( scan ) num_setups = size(nx_setups) * size(ny_setups)
    do cur_setup = 0, num_setups - 1

        if ( scan ) then
            nx = nx_setups( modulo(cur_setup, size(ny_setups) ) + 1 )
            ny = ny_setups( cur_setup / size(ny_setups) + 1 )
        end if

        call setup()

        if ( .not. scan .and. is_master() ) then
            allocate(cart_field(nx, ny, nz))
            call unstructured_to_cartesian(in_field, cart_field, nx, ny, nz)
            call write_field_to_file(cart_field, num_halo, "in_field.dat")
            deallocate(cart_field)
        end if

        ! warmup caches
        call apply_diffusion( in_field, out_field, alpha, num_iter=1 )

        ! time the actual work
#ifdef CRAYPAT
        call PAT_record( PAT_STATE_ON, istat )
#endif
        timer_work = -999
        call timer_start('work', timer_work)

        call apply_diffusion( in_field, out_field, alpha, num_iter=num_iter )
        
        call timer_end( timer_work )
#ifdef CRAYPAT
        call PAT_record( PAT_STATE_OFF, istat )
#endif

        ! no update_halo in unstructured grid
        
        if ( .not. scan .and. is_master() ) then
            allocate(cart_field(nx, ny, nz))
            call unstructured_to_cartesian(out_field, cart_field, nx, ny, nz)
            call write_field_to_file(cart_field, num_halo, "out_field.dat")
            deallocate(cart_field)
        end if

        call cleanup()

        runtime = timer_get( timer_work )
        if ( is_master() ) &
            write(*, '(a, i5, a, i5, a, i5, a, i5, a, i8, a, e15.7, a)') &
                '[', num_rank(), ',', nx, ',', ny, ',', nz, ',', num_iter, ',', runtime, '], \'

    end do

    if ( is_master() ) then
        write(*, '(a)') '] )'
    end if

    call finalize()

contains


    ! Integrate 4th-order diffusion equation by a certain number of iterations.
    !
    !  in_field          -- input field (nx x ny x nz with halo in x- and y-direction)
    !  out_field         -- result (must be same size as in_field)
    !  alpha             -- diffusion coefficient (dimensionless)
    !  num_iter          -- number of iterations to execute
    !
    subroutine apply_diffusion( in_field, out_field, alpha, num_iter )
        implicit none
        
        ! arguments
        real (kind=wp), intent(inout) :: in_field(:, :)
        real (kind=wp), intent(inout) :: out_field(:, :)
        real (kind=wp), intent(in) :: alpha
        integer, intent(in) :: num_iter
        
        ! local
        real (kind=wp), allocatable :: tmp1_field(:,:)
        real (kind=wp) :: laplap
        integer :: iter, n, k, nL, nR, nD, nU
        
        if (allocated(tmp1_field)) deallocate(tmp1_field)
        allocate(tmp1_field(nx*ny, nz))
        tmp1_field = 0.0_wp

        do iter = 1, num_iter
        
            ! Step 1: 2D-Laplacian on in_field
            do k = 1, nz
                do n = 1, nx*ny
                    nL = neighbor_lookup(n, 1)
                    nR = neighbor_lookup(n, 2)
                    nD = neighbor_lookup(n, 3)
                    nU = neighbor_lookup(n, 4)
                    tmp1_field(n, k) = -4._wp * in_field(n, k) + in_field(nL, k) + in_field(nR, k) + in_field(nD, k) + in_field(nU, k)
                end do
            end do
            
            ! Step 2: 2D-Laplacian on tmp1_field
            do k = 1, nz
                do n = 1, nx*ny
                    nL = neighbor_lookup(n, 1)
                    nR = neighbor_lookup(n, 2)
                    nD = neighbor_lookup(n, 3)
                    nU = neighbor_lookup(n, 4)
                    laplap = -4._wp * tmp1_field(n, k) + tmp1_field(nL, k) + tmp1_field(nR, k) + tmp1_field(nD, k) + tmp1_field(nU, k)
                    if (iter == num_iter) then
                        out_field(n, k) = in_field(n, k) - alpha * laplap
                    else
                        in_field(n, k) = in_field(n, k) - alpha * laplap
                    end if
                end do
            end do
        end do
        if (allocated(tmp1_field)) deallocate(tmp1_field)
    end subroutine apply_diffusion


    
    ! initialize at program start
    ! (init MPI, read command line arguments)
    subroutine init()
        use mpi, only : MPI_INIT
        use m_utils, only : error
        implicit none

        ! local
        integer :: ierror

        ! initialize MPI environment
        call MPI_INIT(ierror)
        call error(ierror /= 0, 'Problem with MPI_INIT', code=ierror)

        call read_cmd_line_arguments()

    end subroutine init


    ! setup everything before work
    ! (init timers, allocate memory, initialize fields)
    subroutine setup()
        use m_utils, only : timer_init
        implicit none

        ! local
        integer :: n, k, i, j

        call timer_init()

        allocate( in_field(nx*ny, nz) )
        in_field = 0.0_wp
        do k = 1 + nz / 4, 3 * nz / 4
            do j = 1 + ny / 4, 3 * ny / 4
                do i = 1 + nx / 4, 3 * nx / 4
                    n = i + (j-1)*nx
                    in_field(n, k) = 1.0_wp
                end do
            end do
        end do
        allocate( out_field(nx*ny, nz) )
        out_field = in_field

        ! initialized lookup tables for neighbors
        if (allocated(neighbor_lookup)) deallocate(neighbor_lookup)
        allocate(neighbor_lookup(nx*ny, 4))
        do n = 1, nx*ny
            neighbor_lookup(n, 1) = neighbor_n(n, 1, nx, ny) ! left
            neighbor_lookup(n, 2) = neighbor_n(n, 2, nx, ny) ! right
            neighbor_lookup(n, 3) = neighbor_n(n, 3, nx, ny) ! down
            neighbor_lookup(n, 4) = neighbor_n(n, 4, nx, ny) ! up
        end do
    end subroutine setup


    ! read and parse the command line arguments
    ! (read values, convert type, ensure all required arguments are present,
    !  ensure values are reasonable)
    subroutine read_cmd_line_arguments()
        use m_utils, only : error
        implicit none

        ! local
        integer iarg, num_arg
        character(len=256) :: arg, arg_val

        ! setup defaults
        nx = -1
        ny = -1
        nz = -1
        num_iter = -1
        scan = .false.

        num_arg = command_argument_count()
        iarg = 1
        do while ( iarg <= num_arg )
            call get_command_argument(iarg, arg)
            select case (arg)
            case ("--nx")
                call error(iarg + 1 > num_arg, "Missing value for -nx argument")
                call get_command_argument(iarg + 1, arg_val)
                call error(arg_val(1:1) == "-", "Missing value for -nx argument")
                read(arg_val, *) nx
                iarg = iarg + 1
            case ("--ny")
                call error(iarg + 1 > num_arg, "Missing value for -ny argument")
                call get_command_argument(iarg + 1, arg_val)
                call error(arg_val(1:1) == "-", "Missing value for -ny argument")
                read(arg_val, *) ny
                iarg = iarg + 1
            case ("--nz")
                call error(iarg + 1 > num_arg, "Missing value for -nz argument")
                call get_command_argument(iarg + 1, arg_val)
                call error(arg_val(1:1) == "-", "Missing value for -nz argument")
                read(arg_val, *) nz
                iarg = iarg + 1
            case ("--num_iter")
                call error(iarg + 1 > num_arg, "Missing value for -num_iter argument")
                call get_command_argument(iarg + 1, arg_val)
                call error(arg_val(1:1) == "-", "Missing value for -num_iter argument")
                read(arg_val, *) num_iter
                iarg = iarg + 1
            case ("--scan")
                scan = .true.
            case default
                call error(.true., "Unknown command line argument encountered: " // trim(arg))
            end select
            iarg = iarg + 1
        end do

        ! make sure everything is set
        if (.not. scan) then
            call error(nx == -1, 'You have to specify nx')
            call error(ny == -1, 'You have to specify ny')
        end if
        call error(nz == -1, 'You have to specify nz')
        call error(num_iter == -1, 'You have to specify num_iter')

        ! check consistency of values
        if (.not. scan) then
            call error(nx < 0 .or. nx > 1024*1024, "Please provide a reasonable value of nx")
            call error(ny < 0 .or. ny > 1024*1024, "Please provide a reasonable value of ny")
        end if
        call error(nz < 0 .or. nz > 1024, "Please provide a reasonable value of nz")
        call error(num_iter < 1 .or. num_iter > 1024*1024, "Please provide a reasonable value of num_iter")

    end subroutine read_cmd_line_arguments


    ! cleanup at end of work
    ! (report timers, free memory)
    subroutine cleanup()
        implicit none
        
        deallocate(in_field, out_field)
        if (allocated(neighbor_lookup)) deallocate(neighbor_lookup)
    end subroutine cleanup


    ! finalize at end of program
    ! (finalize MPI)
    subroutine finalize()
        use mpi, only : MPI_FINALIZE
        use m_utils, only : error
        implicit none

        integer :: ierror

        call MPI_FINALIZE(ierror)
        call error(ierror /= 0, 'Problem with MPI_FINALIZE', code=ierror)

    end subroutine finalize



    ! --- Help functions for the unstructured grid (elemental, pure) ---

    ! function computes n based on i, j, and nx
    elemental pure function ij_to_n(i, j, nx) result(n)
        integer, intent(in) :: i, j, nx
        integer :: n
        n = i + (j-1)*nx
    end function ij_to_n

    ! function computes i based on n and nx
    elemental pure function n_to_i(n, nx) result(i)
        integer, intent(in) :: n, nx
        integer :: i
        i = modulo(n-1, nx) + 1
    end function n_to_i

    ! function computes j based on n and nx
    elemental pure function n_to_j(n, nx) result(j)
        integer, intent(in) :: n, nx
        integer :: j
        j = (n-1)/nx + 1
    end function n_to_j

    ! function computes neighboring cells 
    elemental pure function neighbor_n(n, dir, nx, ny) result(nbr)
        integer, intent(in) :: n, dir, nx, ny
        integer :: nbr, i, j
        i = n_to_i(n, nx)
        j = n_to_j(n, nx)
        select case(dir)
        case(1) ! left
            nbr = ij_to_n(modulo(i-2, nx)+1, j, nx)
        case(2) ! right
            nbr = ij_to_n(modulo(i, nx)+1, j, nx)
        case(3) ! down
            nbr = ij_to_n(i, modulo(j-2, ny)+1, nx)
        case(4) ! up
            nbr = ij_to_n(i, modulo(j, ny)+1, nx)
        end select
    end function neighbor_n

    ! subroutine to regrid the unstructured grid to a Cartesian grid
    subroutine unstructured_to_cartesian(unstruct_field, cart_field, nx, ny, nz)
        real(kind=wp), intent(in)  :: unstruct_field(nx*ny, nz)
        real(kind=wp), intent(out) :: cart_field(nx, ny, nz)
        integer, intent(in) :: nx, ny, nz
        integer :: n, k, i, j
        do k = 1, nz
            do n = 1, nx*ny
                i = n_to_i(n, nx)
                j = n_to_j(n, nx)
                cart_field(i, j, k) = unstruct_field(n, k)
            end do
        end do
    end subroutine unstructured_to_cartesian

end program main
