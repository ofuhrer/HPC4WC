! ******************************************************
!     Program: stencil2d
!      Author: Oliver Fuhrer
!       Email: oliverf@vulcan.com
!        Date: 20.05.2020
! Description: Simple stencil example (4th-order diffusion)
! ******************************************************

! Driver for apply_diffusion() that sets up fields and does timings
program main
    use omp_lib
    use m_utils, only: timer_start, timer_end, timer_get, is_master, num_rank, write_field_to_file
    implicit none

    ! constants
    integer, parameter :: wp = 4
    
    ! local
    integer :: nx, ny, nz, num_iter
    logical :: scan
    
    integer :: num_halo = 2
    real (kind=wp) :: alpha = 1.0_wp / 32.0_wp

    real (kind=wp), allocatable :: in_field(:, :, :)
    real (kind=wp), allocatable :: out_field(:, :, :)

    integer :: timer_work
    real (kind=8) :: runtime

    integer :: cur_setup, num_setups = 1
    integer :: nx_setups(7) = (/ 16, 32, 48, 64, 96, 128, 192 /)
    integer :: ny_setups(7) = (/ 16, 32, 48, 64, 96, 128, 192 /)

#ifdef CRAYPAT
    include "pat_apif.h"
    integer :: istat
    call PAT_record( PAT_STATE_OFF, istat )
#endif

    call init()
    
    !$omp parallel
    !$omp master
    write(*,*) '#threads = ', omp_get_num_threads()
    !$omp end master
    !$omp end parallel

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

        if ( .not. scan .and. is_master() ) &
            call write_field_to_file( in_field, num_halo, "in_field.dat" )

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

        call update_halo( out_field )
        if ( .not. scan .and. is_master() ) &
            call write_field_to_file( out_field, num_halo, "out_field.dat" )

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
        real (kind=wp), intent(inout) :: in_field(:, :, :)
        real (kind=wp), intent(inout) :: out_field(:, :, :)
        real (kind=wp), intent(in) :: alpha
        integer, intent(in) :: num_iter
        
        ! local
        real (kind=wp), save, allocatable :: tmp1_field(:, :)
        real (kind=wp) :: laplap
        integer :: iter, i, j, k
        
        ! this is only done the first time this subroutine is called (warmup)
        ! or when the dimensions of the fields change
        if ( allocated(tmp1_field) .and. &
            any( shape(tmp1_field) /= (/nx + 2 * num_halo, ny + 2 * num_halo/) ) ) then
            deallocate( tmp1_field )
        end if
        if ( .not. allocated(tmp1_field) ) then
            allocate( tmp1_field(nx + 2 * num_halo, ny + 2 * num_halo) )
            tmp1_field = 0.0_wp
        end if
        
        do iter = 1, num_iter
                    
            call update_halo( in_field )

            !$omp parallel do default(none) private(tmp1_field, laplap) shared(nz, num_halo, ny, nx, in_field, num_iter, out_field, alpha, iter)
            do k = 1, nz
                do j = 1 + num_halo - 1, ny + num_halo + 1
                do i = 1 + num_halo - 1, nx + num_halo + 1
                    tmp1_field(i, j) = -4._wp * in_field(i, j, k)        &
                        + in_field(i - 1, j, k) + in_field(i + 1, j, k)  &
                        + in_field(i, j - 1, k) + in_field(i, j + 1, k)
                end do
                end do

                do j = 1 + num_halo, ny + num_halo
                do i = 1 + num_halo, nx + num_halo
                
                    laplap = -4._wp * tmp1_field(i, j)       &
                        + tmp1_field(i - 1, j) + tmp1_field(i + 1, j)  &
                        + tmp1_field(i, j - 1) + tmp1_field(i, j + 1)
                        
                    if ( iter == num_iter ) then
                        out_field(i, j, k) = in_field(i, j, k) - alpha * laplap
                    else
                        in_field(i, j, k)  = in_field(i, j, k) - alpha * laplap
                    end if
                    
                end do
                end do

            end do
            !$omp end parallel do
            
        end do
            
    end subroutine apply_diffusion


    
    ! Update the halo-zone using an up/down and left/right strategy.
    !    
    !  field             -- input/output field (nz x ny x nx with halo in x- and y-direction)
    !
    !  Note: corners are updated in the left/right phase of the halo-update
    !
    subroutine update_halo( field )
        implicit none
            
        ! argument
        real (kind=wp), intent(inout) :: field(:, :, :)
        
        ! local
        integer :: i, j, k
            
        ! bottom edge (without corners)
        do k = 1, nz
        do j = 1, num_halo
        do i = 1 + num_halo, nx + num_halo
            field(i, j, k) = field(i, j + ny, k)
        end do
        end do
        end do
            
        ! top edge (without corners)
        do k = 1, nz
        do j = ny + num_halo + 1, ny + 2 * num_halo
        do i = 1 + num_halo, nx + num_halo
            field(i, j, k) = field(i, j - ny, k)
        end do
        end do
        end do
        
        ! left edge (including corners)
        do k = 1, nz
        do j = 1, ny + 2 * num_halo
        do i = 1, num_halo
            field(i, j, k) = field(i + nx, j, k)
        end do
        end do
        end do
                
        ! right edge (including corners)
        do k = 1, nz
        do j = 1, ny + 2 * num_halo
        do i = nx + num_halo + 1, nx + 2 * num_halo
            field(i, j, k) = field(i - nx, j, k)
        end do
        end do
        end do
        
    end subroutine update_halo
        

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
        integer :: i, j, k

        call timer_init()

        allocate( in_field(nx + 2 * num_halo, ny + 2 * num_halo, nz) )
        in_field = 0.0_wp
        do k = 1 + nz / 4, 3 * nz / 4
        do j = 1 + num_halo + ny / 4, num_halo + 3 * ny / 4
        do i = 1 + num_halo + nx / 4, num_halo + 3 * nx / 4
            in_field(i, j, k) = 1.0_wp
        end do
        end do
        end do

        allocate( out_field(nx + 2 * num_halo, ny + 2 * num_halo, nz) )
        out_field = in_field

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


end program main
