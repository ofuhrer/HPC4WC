! ******************************************************
!     Program: stencil2d
!      Author: Oliver Fuhrer; modified by group10
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
    
    ! needed for unstructured grid
    integer :: nPoint
    real (kind=wp), allocatable :: in_field(:, :) ! (nPoint,nz)
    real (kind=wp), allocatable :: out_field(:, :)
    integer, allocatable :: xPoint(:), yPoint(:)
    integer, allocatable :: gridindex(:, :) ! obtain gridindex from (x,y)
    integer, allocatable :: neighbors(:, :) ! obtain neighbor of a particular grid
    
    ! structured grid version of in_field and out_field
    real (kind=wp), allocatable :: in_field_structured(:, :, :) 
    real (kind=wp), allocatable :: out_field_structured(:, :, :) 
    
    integer :: num_halo = 2
    real (kind=wp) :: alpha = 1.0_wp / 32.0_wp

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

    if ( is_master() ) then
        write(*, '(a)') '# ranks nx ny nz num_iter time'
        write(*, '(a)') 'data = np.array( [ \'
    end if

    if ( scan ) num_setups = size(nx_setups) * size(ny_setups)
    do cur_setup = 0, num_setups - 1

        if ( scan ) then
            nx = nx_setups( modulo(cur_setup, size(ny_setups) ) + 1 )
            ny = ny_setups( cur_setup / size(ny_setups) + 1 )
            nPoint = nx * ny ! number of points on 2D plane
        end if
        
        ! call the mapping function to map the structured grid into unstructured grid
        call mapping()
        
      
        ! Find the neighboring points of each grid and store in the array
        call find_neighbors(neighbors, gridindex)
        call setup()

        if ( .not. scan .and. is_master() ) then
            call inverse_mapping( in_field, in_field_structured )
            call write_field_to_file( in_field_structured, "in_field.dat" )
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

        ! call update_halo( out_field )
        if ( .not. scan .and. is_master() ) then
            call inverse_mapping( out_field, out_field_structured )
            call write_field_to_file( out_field_structured, "out_field.dat" )
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
    ! ------ functions specific for unstructured grids-------------------------
    subroutine swap(a, b)
        integer, intent(inout) :: a, b
        integer :: temp
        temp = a
        a = b
        b = temp
    end subroutine swap
    
    subroutine mapping()
        implicit none
        integer :: i, j, n, tmp, tmp_idx
        integer, allocatable :: idx(:)
        real :: u

        nPoint = nx * ny
        allocate(xPoint(nPoint)); allocate(yPoint(nPoint))
        allocate(gridindex(nx, ny))
        allocate(idx(nPoint))
        idx = (/(n, n = 1, nPoint, 1)/)
        
        ! shuffle the idx array using Fisher–Yates shuffle
        do n = 1, nPoint - 1
            call random_number(u)
            tmp_idx = n + FLOOR((nPoint - n + 1) * u) 
            call swap(idx(n), idx(tmp_idx))
        end do 
        
        ! loop to do mapping
        n = 1; xPoint = 0; yPoint = 0
        do j = 1, ny
            do i = 1, nx
                xPoint(idx(n)) = i
                yPoint(idx(n)) = j
                gridindex(i, j) = idx(n)
                n = n + 1
            end do
        end do
                
    end subroutine mapping
    
    ! map the unstructured grid back to structured grid
    subroutine inverse_mapping( unstructured_field, structured_field )
        implicit none

        integer :: n, k
        real (kind=wp), intent(in) :: unstructured_field(:, :)
        real (kind=wp), intent(out) :: structured_field(:, :, :)

        do k = 1, nz
            do n = 1, nPoint
                structured_field(xPoint(n), yPoint(n), k) = unstructured_field(n, k)
            end do
        end do

    end subroutine inverse_mapping
    
    ! find the grid point of the 4 neighbors of each grid index
    subroutine find_neighbors(neighbors, gridindex)
        implicit none

        integer, allocatable, intent(inout) :: neighbors(:,:)
        integer, intent(in) :: gridindex(:,:)
        integer :: xcoord, ycoord
        integer :: top, btm, left, right
        integer :: n ! loop indices

        ! (nPoint, 4); 4 dimensions of neighbors (top, btm, left, right)
        allocate(neighbors(nPoint, 4))
        neighbors = 0

        do n = 1, nPoint
            ! coordinates of the current grid
            xcoord = xPoint(n)
            ycoord = yPoint(n)
            top = MODULO(ycoord, ny) + 1
            btm = MODULO(ycoord - 2, ny) + 1
            left = MODULO(xcoord - 2, nx) + 1
            right = MODULO(xcoord, nx) + 1

            ! Ensure indices are within bounds
            if (top < 1) top = ny
            if (btm < 1) btm = ny
            if (left < 1) left = nx
            if (right < 1) right = nx

            if (xcoord < 1 .or. xcoord > nx .or. ycoord < 1 .or. ycoord > ny) then
                print *, 'Error in neighbors: xcoord, ycoord = ', xcoord, ycoord
                stop
            endif

            ! Assign neighbors
            neighbors(n, 1) = gridindex(xcoord, top)    ! top neighbor
            neighbors(n, 2) = gridindex(xcoord, btm)    ! bottom neighbor
            neighbors(n, 3) = gridindex(left, ycoord)   ! left neighbor
            neighbors(n, 4) = gridindex(right, ycoord)  ! right neighbor

            ! Debugging output
            ! print *, 'Neighbors for point ', n, ': ', neighbors(n, 1), neighbors(n, 2), neighbors(n, 3), neighbors(n, 4)
        end do
    end subroutine find_neighbors

    
    ! --------------------------------------------------------------------------

    ! Integrate 4th-order diffusion equation by a certain number of iterations.
    !
    !  in_field          -- input field (nx x ny x nz with halo in x- and y-direction)
    !  out_field         -- result (must be same size as in_field)
    !  alpha             -- diffusion coefficient (dimensionless)
    !  num_iter          -- number of iterations to execute
    !
    subroutine apply_diffusion(in_field, out_field, alpha, num_iter)
        implicit none

        ! arguments
        real (kind=wp), intent(inout) :: in_field(:, :)
        real (kind=wp), intent(inout) :: out_field(:, :)
        real (kind=wp), intent(in) :: alpha
        integer, intent(in) :: num_iter

        ! local
        real (kind=wp), save, allocatable :: tmp1_field(:)
        real (kind=wp) :: laplap
        integer :: iter, n, k

        ! this is only done the first time this subroutine is called (warmup)
        ! or when the dimensions of the fields change
        if (allocated(tmp1_field) .and. &
            any(shape(tmp1_field) /= (/nPoint/))) then
            deallocate(tmp1_field)
        end if
        if (.not. allocated(tmp1_field)) then
            allocate(tmp1_field(nPoint))
            tmp1_field = 0.0_wp
        end if

        do iter = 1, num_iter

            do k = 1, nz

                do n = 1, nPoint
                    tmp1_field(n) = -4._wp * in_field(n, k)        &
                        + in_field(neighbors(n, 1), k) + in_field(neighbors(n, 2), k)  &
                        + in_field(neighbors(n, 3), k) + in_field(neighbors(n, 4), k)
                end do

                do n = 1, nPoint
                    laplap = -4._wp * tmp1_field(n)       &
                        + tmp1_field(neighbors(n, 1)) + tmp1_field(neighbors(n, 2))  &
                        + tmp1_field(neighbors(n, 3)) + tmp1_field(neighbors(n, 4))

                    if (iter == num_iter) then
                        out_field(n, k) = in_field(n, k) - alpha * laplap
                    else
                        in_field(n, k) = in_field(n, k) - alpha * laplap
                    end if

                end do
            end do
        end do

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
        ! use m_utils, only : timer_init
        implicit none

        ! local
        integer :: i, j, k, n

        ! call timer_init()
        allocate( in_field_structured(nx, ny, nz) ); allocate( in_field(nPoint, nz) )
        in_field_structured = 0.0_wp; in_field = 0.0_wp

        do k = 1 + nz / 4, 3 * nz / 4
        do j = 1 + ny / 4, 3 * ny / 4
        do i = 1 + nx / 4, 3 * nx / 4
        
            in_field_structured(i, j, k) = 1.0_wp
            in_field(gridindex(i, j), k) = 1.0_wp ! unstructured in_field
            
        end do
        end do
        end do

        allocate( out_field_structured(nx, ny, nz) ); allocate( out_field(nPoint, nz) )
        out_field_structured = in_field_structured ; out_field = in_field

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
