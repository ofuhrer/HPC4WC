! ******************************************************
!     Program: stencil_2d
!      Author: Oliver Fuhrer
!       Email: oliverf@vulcan.com
!        Date: 20.05.2020
! Description: Simple stencil example
! ******************************************************


program main
    use m_utils, only: timer_start, timer_end
    implicit none

    ! constants
    integer, parameter :: wp = 4
    
    ! local
    integer :: nx, ny, nz, num_iter
    integer :: num_halo = 2
    integer :: timer_work
    real (kind=wp), allocatable :: in_field(:, :, :)
    real (kind=wp), allocatable :: out_field(:, :, :)
    integer :: istat
    integer*8 :: flop_counter = 0, byte_counter = 0

#ifdef CRAYPAT
    include "pat_apif.h"
    !call PAT_record( PAT_STATE_OFF, istat )
#endif

    call init()

    ! warmup caches
    call work(num_iter=1, increase_counters=.false.)

#ifdef CRAYPAT
    !call PAT_record( PAT_STATE_ON, istat )
    call PAT_region_begin(1, 'work', istat )
#endif
    call timer_start('work', timer_work)

    call work(num_iter=num_iter, increase_counters=.true.)
    
    call timer_end(timer_work)
#ifdef CRAYPAT
    call PAT_region_end(1, istat)
    !call PAT_record( PAT_STATE_OFF, istat )
#endif

    call cleanup()

contains


    subroutine work(num_iter, increase_counters)
        implicit none
        
        ! arguments
        integer, intent(in) :: num_iter
        logical, intent(in) :: increase_counters
        
        ! local
        real (kind=wp), save, allocatable :: tmp_field(:, :, :)
        integer :: iter, i, j, k
        
        ! this is only done the first time this subroutine is called (warmup)
        if (.not. allocated(tmp_field) ) then
            allocate( tmp_field(nx + 2 * num_halo, ny + 2 * num_halo, nz) )
            tmp_field = 0.0_wp
        end if
        
        do iter = 1, num_iter
        
            call update_halo( in_field, increase_counters=increase_counters )
            
            call laplacian( in_field, tmp_field, num_halo, extend=1, increase_counters=increase_counters )
 
            call laplacian( tmp_field, out_field, num_halo, extend=0, increase_counters=increase_counters )
            
            if ( iter /= num_iter ) then
                do k = 1, nz
                do j = 1 + num_halo, ny + num_halo
                do i = 1 + num_halo, nx + num_halo
                    in_field(i, j, k) = out_field(i, j, k)
                    if (increase_counters) byte_counter = byte_counter + 2 * wp
                end do
                end do
                end do
            end if

        end do
            
    end subroutine work


    ! compute the Laplacian
    subroutine laplacian( in_field, lap_field, num_halo, extend, increase_counters )
        implicit none
            
        ! argument
        real (kind=wp), intent(in) :: in_field(:, :, :)
        real (kind=wp), intent(inout) :: lap_field(:, :, :)
        integer, intent(in) :: num_halo, extend
        logical, intent(in) :: increase_counters
        
        ! local
        integer :: i, j, k
            
        do k = 1, nz
        do j = 1 + num_halo - extend, ny + num_halo + extend
        do i = 1 + num_halo - extend, nx + num_halo + extend
            lap_field(i, j, k) = -4._wp * in_field(i, j, k)      &
                + in_field(i - 1, j, k) + in_field(i + 1, j, k)  &
                + in_field(i, j - 1, k) + in_field(i, j + 1, k)
            if (increase_counters) flop_counter = flop_counter + 5
            if (increase_counters) byte_counter = byte_counter + 6 * wp
        end do
        end do
        end do

    end subroutine laplacian


    ! implement periodic halo-updates
    subroutine update_halo( field, increase_counters )
        implicit none
            
        ! argument
        real (kind=wp), intent(inout) :: field(:, :, :)
        logical, intent(in) :: increase_counters
        
        ! local
        integer :: i, j, k
            
        ! left edge (including corners)
        do k = 1, nz
        do j = 1, ny
        do i = 1, num_halo
            field(i, j, k) = field(nx + i, j, k)
            if ( increase_counters ) byte_counter = byte_counter + 2 * wp
        end do
        end do
        end do
                
        ! right edge (including corners)
        do k = 1, nz
        do j = 1, ny
        do i = nx + num_halo + 1, nx + 2 * num_halo
            field(i, j, k) = field(i - nx, j, k))
            if ( increase_counters ) byte_counter = byte_counter + 2 * wp
        end do
        end do
        end do
        
        ! bottom edge (without corners)
        do k = 1, nz
        do j = 1, num_halo
        do i = 1 + num_halo, nx + num_halo
            field(i, j, k) = field(i, j + ny, k)
            if ( increase_counters ) byte_counter = byte_counter + 2 * wp
        end do
        end do
        end do
            
        ! top edge (without corners)
        do k = 1, nz
        do j = ny + num_halo + 1, ny + 2 * num_halo
        do i = 1 + num_halo, nx + num_halo
            field(i, j, k) = field(i, j - ny, k)
            if ( increase_counters ) byte_counter = byte_counter + 2 * wp
        end do
        end do
        end do
        
    end subroutine update_halo
        

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
        call random_number( in_field )

        allocate( out_field(nx + 2 * num_halo, ny + 2 * num_halo, nz) )
        out_field = 0.0_wp

    end subroutine init


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


    subroutine cleanup()
        use mpi, only : MPI_FINALIZE, MPI_COMM_WORLD, MPI_DOUBLE_PRECISION, MPI_SUM
        use m_utils, only : error, timer_print, is_master
        implicit none

        integer :: ierror
        real(kind=8) :: global_flop_counter, global_byte_counter

        call MPI_REDUCE(real(flop_counter, 8), global_flop_counter, 1, MPI_DOUBLE_PRECISION, MPI_SUM, 0, MPI_COMM_WORLD, ierror)
        call MPI_REDUCE(real(byte_counter, 8), global_byte_counter, 1, MPI_DOUBLE_PRECISION, MPI_SUM, 0, MPI_COMM_WORLD, ierror)
        if ( is_master() ) then
            write(*, *) 'Total number of GigaFLOP = ', &
                global_flop_counter / 1024.d0 / 1024.d0 / 1024.d0
            write(*, *) 'Total number of GByte transferred = ', &
                global_byte_counter / 1024.d0 / 1024.d0 / 1024.d0
        end if
        
        call timer_print()

        call MPI_FINALIZE(ierror)
        call error(ierror /= 0, 'Problem with MPI_FINALIZE', code=ierror)

    end subroutine cleanup

end program main
