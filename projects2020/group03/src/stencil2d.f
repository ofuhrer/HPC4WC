! ******************************************************
!     Program: stencil2d
!      Author: Oliver Fuhrer
!       Email: oliverf@vulcan.com
!        Date: 20.05.2020
! Description: Simple stencil example (4th-order diffusion)
! ******************************************************

! Driver for apply_diffusion() that sets up fields and does timings
program main
    use, intrinsic :: iso_fortran_env, only: REAL32, REAL64
    use mpi, only: MPI_COMM_WORLD
    use m_utils, only: timer_start, timer_end, timer_get, is_master, num_rank, write_field_to_file
    use m_partitioner, only: Partitioner
    implicit none

    ! local
    integer :: nx, ny, nz, num_iter, rank
    logical :: scan

    integer :: num_halo = 2
    real(kind = REAL32) :: alpha = 1.0_REAL32 / 32.0_REAL32

    real(kind = REAL32), allocatable :: in_field(:, :, :)
    real(kind = REAL32), allocatable :: out_field(:, :, :)
    real(kind = REAL32), allocatable :: in_field_p(:, :, :)
    real(kind = REAL32), allocatable :: out_field_p(:, :, :)

    integer :: timer_work
    real(kind = REAL64) :: runtime

    integer :: cur_setup, num_setups = 1
    integer :: nx_setups(7) = [16, 32, 48, 64, 96, 128, 192]
    integer :: ny_setups(7) = [16, 32, 48, 64, 96, 128, 192]

#ifdef CRAYPAT
    include "pat_apif.h"
    integer :: istat
    call PAT_record( PAT_STATE_OFF, istat )
#endif

    call init()

    if (is_master()) then
        write(*, '(a)') '# ranks nx ny ny nz num_iter time'
        write(*, '(a)') 'data = np.array( [ \'
    end if

    if (scan) then
        num_setups = size(nx_setups) * size(ny_setups)
    end if

    do cur_setup = 0, num_setups - 1
        if (scan) then
            nx = nx_setups( modulo(cur_setup, size(ny_setups) ) + 1 )
            ny = ny_setups( cur_setup / size(ny_setups) + 1 )
        end if

        call setup()

        if (.not. scan .and. is_master()) then
            call write_field_to_file(in_field, num_halo, "in_field.dat")
        end if

        call calc(in_field, out_field, num_halo, alpha, num_iter)

        if (.not. scan .and. is_master()) then
            call write_field_to_file(out_field, num_halo, "out_field.dat")
        end if

        call cleanup()

        runtime = timer_get( timer_work )
        if (is_master()) then
            write(*, '(a, i5, a, i5, a, i5, a, i5, a, i8, a, e15.7, a)') &
                '[', num_rank(), ',', nx, ',', ny, ',', nz, ',', num_iter, ',', runtime, '], \'
        end if
    end do

    if (is_master()) then
        write(*, '(a)') '] )'
    end if

    call finalize()
contains
    subroutine calc(in_field, out_field, num_halo, alpha, num_iter)
        use, intrinsic :: iso_fortran_env, only: REAL32
        use m_partitioner, only: Partitioner
        use m_diffusion, only: apply_diffusion

        real(kind = REAL32), intent(in) :: in_field(:, :, :)
        real(kind = REAL32), intent(out) :: out_field(:, :, :)
        integer, intent(in) :: num_halo
        real(kind = REAL32), intent(in) :: alpha
        integer, intent(in) :: num_iter

        integer :: nx
        integer :: ny
        integer :: nz
        integer :: ns(3)
        type(Partitioner) :: p
        real(kind = REAL32), allocatable :: in_field_p(:, :, :)
        real(kind = REAL32), allocatable :: out_field_p(:, :, :)

        nx = size(in_field, 1) - 2 * num_halo
        ny = size(in_field, 2) - 2 * num_halo
        nz = size(in_field, 3)
        ns = [nx, ny, nz]

        p = Partitioner(MPI_COMM_WORLD, ns, num_halo)
        in_field_p = p%scatter(in_field)
        out_field_p = p%scatter(out_field)

        ! warmup caches
        call apply_diffusion(in_field_p, out_field_p, num_halo, alpha, p, 1)

        ! time the actual work
#ifdef CRAYPAT
        call PAT_record(PAT_STATE_ON, istat)
#endif
        timer_work = -999
        call timer_start('work', timer_work)

        call apply_diffusion(in_field_p, out_field_p, num_halo, alpha, p, num_iter)

        call timer_end(timer_work)
#ifdef CRAYPAT
        call PAT_record(PAT_STATE_OFF, istat)
#endif

        out_field = p%gather(out_field_p)
    end subroutine

    ! initialize at program start
    ! (init MPI, read command line arguments)
    subroutine init()
        use mpi, only : MPI_Init_Thread, MPI_THREAD_FUNNELED, MPI_Comm_Rank, MPI_COMM_WORLD
        use m_utils, only : error
        implicit none

        ! local
        integer :: provided, ierror

        ! initialize MPI environment
        call MPI_Init_Thread(MPI_THREAD_FUNNELED, provided, ierror)
        call error(ierror /= 0, 'Problem with MPI_INIT', code=ierror)
        call MPI_Comm_Rank(MPI_COMM_WORLD, rank, ierror)
        call error(ierror /= 0, 'Problem with MPI_COMM_RANK', code=ierror)

        call read_cmd_line_arguments()
    end subroutine

    ! setup everything before work
    ! (init timers, allocate memory, initialize fields)
    subroutine setup()
        use m_utils, only : timer_init
        implicit none

        ! local
        integer :: i, j, k

        call timer_init()

        allocate(in_field(nx + 2 * num_halo, ny + 2 * num_halo, nz))
        in_field = 0.0_REAL32
        do k = 1 + nz / 4, 3 * nz / 4
        do j = 1 + num_halo + ny / 4, num_halo + 3 * ny / 4
        do i = 1 + num_halo + nx / 4, num_halo + 3 * nx / 4
            in_field(i, j, k) = 1.0_REAL32
        end do
        end do
        end do

        allocate(out_field(nx + 2 * num_halo, ny + 2 * num_halo, nz))
        out_field = in_field
    end subroutine


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
    end subroutine

    ! cleanup at end of work
    ! (report timers, free memory)
    subroutine cleanup()
        implicit none
    end subroutine

    ! finalize at end of program
    ! (finalize MPI)
    subroutine finalize()
        use mpi, only : MPI_Finalize
        use m_utils, only : error
        implicit none

        integer :: ierror

        call MPI_Finalize(ierror)
        call error(ierror /= 0, 'Problem with MPI_FINALIZE', code=ierror)
    end subroutine
end program

! vim: filetype=fortran expandtab tabstop=4 softtabstop=4 shiftwidth=4 :
