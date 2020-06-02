! ******************************************************
!     Program: stencil2d
!      Author: Oliver Fuhrer
!       Email: oliverf@vulcan.com
!        Date: 20.05.2020
! Description: Simple stencil example (4th-order diffusion)
! ******************************************************

! Driver for apply_diffusion() that sets up fields and does timings
program main
    use m_utils, only: timer_start, timer_end
    implicit none

    ! constants
    integer, parameter :: wp = 4
    
    ! local
    integer :: nx, ny, nz, num_iter
    integer :: num_halo = 2
    real (kind=wp) :: alpha = 1.0_wp / 32.0_wp

    real (kind=wp), allocatable :: in_field(:, :, :)
    real (kind=wp), allocatable :: out_field(:, :, :)

    integer :: timer_work = -999
    integer :: istat

#ifdef CRAYPAT
    include "pat_apif.h"
#endif

    call init()

    ! warmup caches
    call apply_diffusion( in_field, out_field, alpha, num_iter=1 )

    ! time the actual work
#ifdef CRAYPAT
    call PAT_region_begin(1, 'work', istat )
#endif
    call timer_start('work', timer_work)

    call apply_diffusion( in_field, out_field, alpha, num_iter=num_iter )
    
    call timer_end(timer_work)
#ifdef CRAYPAT
    call PAT_region_end(1, istat)
#endif

    call cleanup()

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
        real (kind=wp), intent(inout) :: in_field(:, :, :)
        real (kind=wp), intent(inout) :: out_field(:, :, :)
        real (kind=wp), intent(in) :: alpha
        integer, intent(in) :: num_iter
        
        ! local
        real (kind=wp), save, allocatable :: tmp1_field(:, :)
        real (kind=wp), save, allocatable :: tmp2_field(:, :)
        integer :: iter, i, j, k
        
        ! this is only done the first time this subroutine is called (warmup)
        if ( .not. allocated(tmp1_field) ) then
            allocate( tmp1_field(nx + 2 * num_halo, ny + 2 * num_halo) )
            tmp1_field = 0.0_wp
        end if
        if ( .not. allocated(tmp2_field) ) then
            allocate( tmp2_field(nx + 2 * num_halo, ny + 2 * num_halo) )
            tmp2_field = 0.0_wp
        end if
        
        do iter = 1, num_iter            
            do k = 1, nz

                call update_halo_2d( in_field(:, :, k) )            

                do j = 1 + num_halo - 1, ny + num_halo + 1
                do i = 1 + num_halo - 1, nx + num_halo + 1
                    tmp1_field(i, j) = -4._wp * in_field(i, j, k)        &
                        + in_field(i - 1, j, k) + in_field(i + 1, j, k)  &
                        + in_field(i, j - 1, k) + in_field(i, j + 1, k)
                end do
                end do

                do j = 1 + num_halo, ny + num_halo
                do i = 1 + num_halo, nx + num_halo
                
                    tmp2_field(i, j) = -4._wp * tmp1_field(i, j)       &
                        + tmp1_field(i - 1, j) + tmp1_field(i + 1, j)  &
                        + tmp1_field(i, j - 1) + tmp1_field(i, j + 1)
                        
                    if ( iter == num_iter ) then
                        out_field(i, j, k) = in_field(i, j, k) - alpha * tmp2_field(i, j)
                    else
                        in_field(i, j, k)  = in_field(i, j, k) - alpha * tmp2_field(i, j)
                    end if
                    
                end do
                end do

            end do
        end do
            
    end subroutine apply_diffusion


    ! Compute Laplacian using 2nd-order centered differences.
    !     
    !  in_field          -- input field (nx x ny x nz with halo in x- and y-direction)
    !  lap_field         -- result (must be same size as in_field)
    !  num_halo          -- number of halo points
    !  extend            -- extend computation into halo-zone by this number of points
    !
    subroutine laplacian_2d( field, lap, num_halo, extend )
        implicit none
            
        ! argument
        real (kind=wp), intent(in) :: field(:, :)
        real (kind=wp), intent(inout) :: lap(:, :)
        integer, intent(in) :: num_halo, extend
        
        ! local
        integer :: i, j
            
        do j = 1 + num_halo - extend, ny + num_halo + extend
        do i = 1 + num_halo - extend, nx + num_halo + extend
            lap(i, j) = -4._wp * field(i, j)      &
                + field(i - 1, j) + field(i + 1, j)  &
                + field(i, j - 1) + field(i, j + 1)
        end do
        end do

    end subroutine laplacian_2d

    ! Update the halo-zone using an up/down and left/right strategy.
    !    
    !  field             -- input/output field (nz x ny x nx with halo in x- and y-direction)
    !
    !  Note: corners are updated in the left/right phase of the halo-update
    !
    subroutine update_halo_2d( field )
        implicit none
            
        ! argument
        real (kind=wp), intent(inout) :: field(:, :)
        
        ! local
        integer :: i, j
            
        ! bottom edge (without corners)
        do j = 1, num_halo
        do i = 1 + num_halo, nx + num_halo
            field(i, j) = field(i, j + ny)
        end do
        end do
            
        ! top edge (without corners)
        do j = ny + num_halo + 1, ny + 2 * num_halo
        do i = 1 + num_halo, nx + num_halo
            field(i, j) = field(i, j - ny)
        end do
        end do
        
        ! left edge (including corners)
        do j = 1, ny + 2 * num_halo
        do i = 1, num_halo
            field(i, j) = field(i + nx, j)
        end do
        end do
                
        ! right edge (including corners)
        do j = 1, ny + 2 * num_halo
        do i = nx + num_halo + 1, nx + 2 * num_halo
            field(i, j) = field(i - nx, j)
        end do
        end do
        
    end subroutine update_halo_2d
        

    ! initialize at program start
    ! (init MPI, init timers, read command line arguments, 
    !  allocate memory, initialize fields)
    subroutine init()
        use mpi, only : MPI_INIT
        use m_utils, only : error, is_master, timer_init
        implicit none

        ! local
        integer :: ierror
        integer :: i, j, k

        ! initialize MPI environment
        call MPI_INIT(ierror)
        call error(ierror /= 0, 'Problem with MPI_INIT', code=ierror)

        call timer_init()

        call read_cmd_line_arguments()

        allocate( in_field(nx + 2 * num_halo, ny + 2 * num_halo, nz) )
        do j = 1 + num_halo + ny / 4, num_halo + 3 * ny / 4
        do i = 1 + num_halo + nx / 4, num_halo + 3 * nx / 4
            in_field(i, j, :) = 1.0_wp
        end do
        end do

        allocate( out_field(nx + 2 * num_halo, ny + 2 * num_halo, nz) )
        out_field = in_field

    end subroutine init


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

        num_arg = command_argument_count()
        iarg = 1
        do while ( iarg <= num_arg )
            call get_command_argument(iarg, arg)
            select case (arg)
            case ("-nx")
                call error(iarg + 1 > num_arg, "Missing value for -nx argument")
                call get_command_argument(iarg + 1, arg_val)
                call error(arg_val(1:1) == "-", "Missing value for -nx argument")
                read(arg_val, *) nx
                iarg = iarg + 1
            case ("-ny")
                call error(iarg + 1 > num_arg, "Missing value for -ny argument")
                call get_command_argument(iarg + 1, arg_val)
                call error(arg_val(1:1) == "-", "Missing value for -ny argument")
                read(arg_val, *) ny
                iarg = iarg + 1
            case ("-nz")
                call error(iarg + 1 > num_arg, "Missing value for -nz argument")
                call get_command_argument(iarg + 1, arg_val)
                call error(arg_val(1:1) == "-", "Missing value for -nz argument")
                read(arg_val, *) nz
                iarg = iarg + 1
            case ("-num_iter")
                call error(iarg + 1 > num_arg, "Missing value for -num_iter argument")
                call get_command_argument(iarg + 1, arg_val)
                call error(arg_val(1:1) == "-", "Missing value for -num_iter argument")
                read(arg_val, *) num_iter
                iarg = iarg + 1
            case default
                call error(.true., "Unknown command line argument encountered: " // trim(arg))
            end select
            iarg = iarg + 1
        end do

        ! make sure everything is set
        call error(nx == -1, 'You have to specify nx')
        call error(ny == -1, 'You have to specify ny')
        call error(nz == -1, 'You have to specify nz')
        call error(num_iter == -1, 'You have to specify num_iter')

        ! check consistency of values
        call error(nx < 0 .or. nx > 1024*1024, "Please provide a reasonable value of nx")
        call error(ny < 0 .or. ny > 1024*1024, "Please provide a reasonable value of ny")
        call error(nz < 0 .or. nz > 1024, "Please provide a reasonable value of nz")
        call error(num_iter < 1 .or. num_iter > 1024*1024, "Please provide a reasonable value of num_iter")

    end subroutine read_cmd_line_arguments


    ! cleanup at end of program
    ! (report timers, finalize MPI)
    subroutine cleanup()
        use mpi, only : MPI_FINALIZE, MPI_COMM_WORLD, MPI_DOUBLE_PRECISION, MPI_SUM
        use m_utils, only : error, timer_get, timer_print, is_master
        implicit none

        integer :: ierror
        
        call timer_print()

        call MPI_FINALIZE(ierror)
        call error(ierror /= 0, 'Problem with MPI_FINALIZE', code=ierror)

    end subroutine cleanup

end program main
