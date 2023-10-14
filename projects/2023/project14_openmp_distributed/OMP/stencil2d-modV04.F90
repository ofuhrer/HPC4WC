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

    real (kind=wp), allocatable :: in_field(:, :, :)
    real (kind=wp), allocatable :: out_field(:, :, :)

    integer :: timer_work
    real (kind=8) :: runtime

    integer :: cur_setup, num_setups = 1
    integer :: nx_setups(7) = (/ 16, 32, 48, 64, 96, 128, 192 /)
    integer :: ny_setups(7) = (/ 16, 32, 48, 64, 96, 128, 192 /)
    
    !variable useful for parallelization 
    integer :: n_threads 
    integer :: thread_id, sub_nx, sub_ny

#ifdef CRAYPAT
    include "pat_apif.h"
    integer :: istat
    call PAT_record( PAT_STATE_OFF, istat )
#endif

    call init()

    if ( is_master() ) then
       !write(*, '(a)') '# ranks nx ny ny nz num_iter time'
       !write(*, '(a)') 'data = np.array( [ \'
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
            !write(*, '(a, i5, a, i5, a, i5, a, i5, a, i8, a, e15.7, a)') &
            !    '[', num_rank(), ',', nx, ',', ny, ',', nz, ',', num_iter, ',', runtime, '], \'
            write(*, '(e15.7, a)') runtime, ','

    end do

    if ( is_master() ) then
        !write(*, '(a)') '] )'
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
        real (kind=wp), allocatable :: halo(:, :, :)
        real (kind=wp), allocatable :: tmp1_field(:, :)
        real (kind=wp), allocatable :: sub_infield (:, :, :)
        real (kind=wp) :: laplap
        integer :: iter, i, j, k, a, b
        
        !$omp parallel
        !$omp master
        n_threads = omp_get_num_threads()
        call subdomains (in_field, n_threads, sub_infield, tmp1_field, sub_nx, sub_ny)
        !$omp end master
        !$omp end parallel
        
        ! this is only done the first time this subroutine is called (warmup)
        ! or when the dimensions of the fields change
        if ( allocated(tmp1_field) .and. &
            any( shape(tmp1_field) /= (/sub_nx + 2 * num_halo, sub_ny + 2 * num_halo/) ) ) then
            deallocate( tmp1_field )
        end if
        if ( .not. allocated(tmp1_field) ) then
            allocate( tmp1_field(sub_nx + 2 * num_halo, sub_ny + 2 * num_halo) )
            tmp1_field = 0.0_wp
        end if
        
        allocate( halo (nx + 2 * num_halo, ny + 2 * num_halo, nz) )
        
        call update_halo( in_field )
        halo(:, :, :) = in_field (:, :, :)
        
        !$omp parallel default(none) private(sub_infield, tmp1_field, laplap, iter, thread_id, i, j, k, a, b) shared(num_halo, nx, ny, nz, sub_nx, sub_ny, alpha, in_field, halo, out_field, num_iter, n_threads)
        
        n_threads = omp_get_num_threads()
        thread_id = omp_get_thread_num()

        i = 1
        j = 1
        call adjust_index( i, j, thread_id, sub_nx, sub_ny )
        sub_infield = in_field( i : i-1 + sub_nx + 2*num_halo, j:j-1 + sub_ny + 2*num_halo, : )        
        
        do iter = 1, num_iter
        
            do k = 1, nz
                
                do j = 1 + num_halo - 1, sub_ny + num_halo + 1
                do i = 1 + num_halo - 1, sub_nx + num_halo + 1
                    tmp1_field(i, j) = -4._wp * sub_infield(i, j, k)        &
                        + sub_infield(i - 1, j, k) + sub_infield(i + 1, j, k)  &
                        + sub_infield(i, j - 1, k) + sub_infield(i, j + 1, k)
                end do
                end do

                do j = 1 + num_halo, sub_ny + num_halo
                do i = 1 + num_halo, sub_nx + num_halo
                
                    laplap = -4._wp * tmp1_field(i, j)       &
                        + tmp1_field(i - 1, j) + tmp1_field(i + 1, j)  &
                        + tmp1_field(i, j - 1) + tmp1_field(i, j + 1)
                        
                    if ( iter == num_iter ) then
                        a = i 
                        b = j
                        call adjust_index (a, b, thread_id, sub_nx, sub_ny)
                        out_field(a, b, k) = sub_infield(i, j, k) - alpha * laplap                        
                    else
                        if (is_inner_region(i, j, sub_nx, sub_ny) ) then
                            sub_infield(i, j, k) = sub_infield(i, j, k) - alpha * laplap
                        else
                            a = i 
                            b = j
                            call adjust_index (a, b, thread_id, sub_nx, sub_ny)
                            halo(a, b, k) = halo (a, b, k) - alpha * laplap
                        end if
                    end if
                    
                end do
                end do

            end do
        
        !$omp barrier
        
        !$omp master
        call update_halo( halo )
        !$omp end master

        call update_subhalo ( halo, sub_infield, thread_id, sub_nx, sub_ny )
        
        end do
        
        !$omp end parallel
        
    end subroutine apply_diffusion
    
    
    ! IN: initial field to be decomposed, number of thread
    ! WARNING: Area-size of field must be evenly divisble by n_threads
    !          -> n_threads needs to be a square number
    ! OUT: empty subdomains in_subfield, tmp1_field
    subroutine subdomains ( field, n_threads, sub_infield, tmp1_field, sub_nx, sub_ny )
        implicit none
        integer, intent(in) :: n_threads
        real (kind=wp), intent(in) :: field(:, :, :)
        real (kind=wp), allocatable, intent(inout) :: sub_infield (:, :, :)
        real (kind=wp), allocatable, intent(inout) :: tmp1_field(:, :)
        integer, intent(inout) :: sub_nx, sub_ny

        ! local variable
        integer :: n, in_size

        n = nint( sqrt( real( n_threads ) ) )
        
        in_size = size( field, 1 ) - 2*num_halo
        sub_nx = in_size / n
        sub_ny = in_size / n

        allocate( sub_infield( sub_nx + 2*num_halo, sub_ny + 2*num_halo, nz ) )
        allocate( tmp1_field ( sub_nx + 2*num_halo, sub_ny + 2*num_halo) ) 

    end subroutine subdomains
    
    
    ! Shift the indices from sub_field to the entire shared field based on thread_id
    subroutine adjust_index (i, j, thread_id, sub_nx, sub_ny)
        implicit none

        integer, intent(inout) :: i, j
        integer, intent(in) :: thread_id, sub_nx, sub_ny
        integer :: n, row, col
        
        n = nint( sqrt( real( n_threads ) ) )

        row = floor ( real ( thread_id / n ) )
        col = mod( thread_id, n )

        i = i + row * sub_nx
        j = j + col * sub_ny
        
    end subroutine adjust_index



    ! Shift the indeces from the entire shared field to the sub_field based on thread_id
    subroutine reset_index (i, j, thread_id, sub_nx, sub_ny)
        implicit none

        integer, intent(inout) :: i, j
        integer, intent(in) :: thread_id, sub_nx, sub_ny
        integer :: n, row, col
        
        n = nint( sqrt( real( n_threads ) ) )

        row = floor ( real ( thread_id / n ) )
        col = modulo( thread_id, n )

        i = i - row * sub_nx
        j = j - col * sub_ny
        
    end subroutine reset_index
    
    
    ! Given values of i, j, return if the grid cell is in the inner region or should be saved in the shared halo
    logical function is_inner_region (i, j, sub_nx, sub_ny)
        implicit none

        integer, intent(in) :: i, j, sub_nx, sub_ny

        is_inner_region = ( (i > 2*num_halo) .AND. (i <= sub_nx) ) &
                    .AND. ( (j > 2*num_halo) .AND. (j <= sub_ny) )

    end function is_inner_region




    ! Copy the shared grid cells form the halo array to the private sub_infield
    subroutine update_subhalo ( halo, sub_infield, thread_id, sub_nx, sub_ny )
        implicit none

        ! argument
        real (kind=wp), intent(in) :: halo (:, :, :)
        real (kind=wp), intent(inout) :: sub_infield (:, :, :)
        integer, intent(in) :: thread_id, sub_nx, sub_ny

        ! local
        integer :: i, j, k, a, b

        ! bottom edge (without corners)
        do j = 1, 2*num_halo
        do i = 1 + 2*num_halo, sub_nx            
            a = i
            b = j
            call adjust_index ( a, b, thread_id, sub_nx, sub_ny )
            sub_infield(i, j, :) = halo(a, b, :)
        end do
        end do

        ! top edge (without corners)
        do j = sub_ny + 1, sub_ny + 2*num_halo
        do i = 1 + 2*num_halo, sub_nx
            a = i
            b = j
            call adjust_index ( a, b, thread_id, sub_nx, sub_ny )
            sub_infield(i, j, :) = halo(a, b, :)
        end do
        end do

        ! left edge (including corners)
        do j = 1, sub_ny + 2 * num_halo
        do i = 1, 2*num_halo
            a = i
            b = j
            call adjust_index ( a, b, thread_id, sub_nx, sub_ny )
            sub_infield(i, j, :) = halo(a, b, :)
        end do
        end do

        ! right edge (including corners)
        do j = 1, sub_ny + 2*num_halo
        do i = sub_nx + 1, sub_nx + 2*num_halo
            a = i
            b = j
            call adjust_index ( a, b, thread_id, sub_nx, sub_ny )
            sub_infield(i, j, :) = halo(a, b, :)
        end do
        end do

    end subroutine update_subhalo
    

    
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
        integer :: i, j
            
        ! bottom edge (without corners)
        do j = 1, num_halo
        do i = 1 + num_halo, nx + num_halo
            field(i, j, :) = field(i, j + ny, :)
        end do
        end do
            
        ! top edge (without corners)
        do j = ny + num_halo + 1, ny + 2 * num_halo
        do i = 1 + num_halo, nx + num_halo
            field(i, j, :) = field(i, j - ny, :)
        end do
        end do
        
        ! left edge (including corners)
        do j = 1, ny + 2 * num_halo
        do i = 1, num_halo
            field(i, j, :) = field(i + nx, j, :)
        end do
        end do
                
        ! right edge (including corners)
        do j = 1, ny + 2 * num_halo
        do i = nx + num_halo + 1, nx + 2 * num_halo
            field(i, j, :) = field(i - nx, j, :)
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
