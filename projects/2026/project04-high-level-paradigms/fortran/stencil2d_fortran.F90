! ******************************************************
!     Program: stencil2d
!      Author: Oliver Fuhrer
!       Email: oliverf@vulcan.com
!        Date: 20.05.2020
! Description: Simple stencil example (4th-order diffusion) k-loop parallelization, without MPI
! ******************************************************

program main
    use omp_lib
    implicit none

    ! constants
    integer, parameter :: wp = 4

    ! local
    integer :: nx, ny, nz, num_iter
    logical :: scan

    integer :: num_halo = 2
    real (kind=wp) :: alpha = 1.0_wp / 64.0_wp

    real (kind=wp), allocatable :: in_field(:, :, :)
    real (kind=wp), allocatable :: out_field(:, :, :)

    integer (kind=8) :: c0, c1, crate
    real (kind=8) :: runtime

    integer :: cur_setup, num_setups = 1
    integer :: nx_setups(7) = (/ 16, 32, 48, 64, 96, 128, 192 /)
    integer :: ny_setups(7) = (/ 16, 32, 48, 64, 96, 128, 192 /)

    character(len=1024) :: in_field_path = ""
    character(len=1024) :: out_field_path = ""
    character(len=64) :: device = "cpu"

    call init()

    !$omp parallel
    !$omp master
    write(*,*) '#threads = ', omp_get_num_threads()
    !$omp end master
    !$omp end parallel

    write(*, '(a)') '# ranks nx ny nz num_iter time'
    write(*, '(a)') 'data = np.array( [ \'

    if ( scan ) num_setups = size(nx_setups) * size(ny_setups)
    do cur_setup = 0, num_setups - 1

        if ( scan ) then
            nx = nx_setups( modulo(cur_setup, size(ny_setups) ) + 1 )
            ny = ny_setups( cur_setup / size(ny_setups) + 1 )
        end if

        call setup()

        if ( .not. scan .and. len_trim(in_field_path) == 0 ) &
            call write_field_to_file( in_field, num_halo, "in_field.dat" )

        ! warmup caches
        call apply_diffusion( in_field, out_field, alpha, num_iter=1 )

        ! time the actual work
        call system_clock(count=c0, count_rate=crate)

        call apply_diffusion( in_field, out_field, alpha, num_iter=num_iter )

        call system_clock(count=c1)
        runtime = real(c1 - c0, 8) / real(crate, 8)

        call update_halo( out_field )

        if ( .not. scan .and. len_trim(out_field_path) == 0 ) &
            call write_field_to_file( out_field, num_halo, "out_field.dat" )

        if ( len_trim(out_field_path) > 0 ) &
            call write_raw_field( out_field_path, out_field, nx, ny, nz, num_halo )

        call cleanup()

        write(*, '(a, i5, a, i5, a, i5, a, i5, a, i8, a, e15.7, a)') &
            '[', 1, ',', nx, ',', ny, ',', nz, ',', num_iter, ',', runtime, '], \'

    end do

    write(*, '(a)') '] )'

    write(*, '(es24.15e3)') runtime

    call finalize()

contains


    ! Integrate 4th-order diffusion equation by a certain number of iterations.
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

            !$omp parallel do default(none) private(i, j, laplap, tmp1_field) shared(num_halo, ny, nx, nz, in_field, iter, num_iter, out_field, alpha)
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
                        in_field(i, j, k) = in_field(i, j, k) - alpha * laplap
                    end if

                end do
                end do
            end do

            !$omp end parallel do

        end do

    end subroutine apply_diffusion



    ! Update the halo-zone 
    subroutine update_halo( field )
        implicit none

        real (kind=wp), intent(inout) :: field(:, :, :)
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
    subroutine init()
        implicit none
        call read_cmd_line_arguments()
    end subroutine init


    ! setup everything before work
    subroutine setup()
        implicit none

        integer :: i, j, k

        allocate( in_field(nx + 2 * num_halo, ny + 2 * num_halo, nz) )
        allocate( out_field(nx + 2 * num_halo, ny + 2 * num_halo, nz) )

        if ( len_trim(in_field_path) > 0 ) then

            call read_raw_field( in_field_path, in_field, nx, ny, nz, num_halo )

        else

            ! original internally-generated field 
            in_field = 0.0_wp
            do k = 1 + nz / 4, 3 * nz / 4
            do j = 1 + num_halo + ny / 4, num_halo + 3 * ny / 4
            do i = 1 + num_halo + nx / 4, num_halo + 3 * nx / 4
                in_field(i, j, k) = 1.0_wp
            end do
            end do
            end do

        end if

        out_field = in_field

    end subroutine setup

    ! Read a raw, headerless float32 binary stream into field
    subroutine read_raw_field( path, field, nx, ny, nz, num_halo )
        implicit none

        character(len=*), intent(in) :: path
        integer, intent(in) :: nx, ny, nz, num_halo
        real (kind=wp), intent(out) :: field(nx + 2 * num_halo, ny + 2 * num_halo, nz)

        integer :: iunit, ios

        open(newunit=iunit, file=trim(path), access='stream', form='unformatted', &
            status='old', action='read', iostat=ios)
        call error(ios /= 0, 'Could not open in_field_path for reading: ' // trim(path))
        read(iunit) field
        close(iunit)

    end subroutine read_raw_field

    ! Write field as a raw, headerless float32 binary stream.
    subroutine write_raw_field( path, field, nx, ny, nz, num_halo )
        implicit none

        character(len=*), intent(in) :: path
        integer, intent(in) :: nx, ny, nz, num_halo
        real (kind=wp), intent(in) :: field(nx + 2 * num_halo, ny + 2 * num_halo, nz)

        integer :: iunit, ios

        open(newunit=iunit, file=trim(path), access='stream', form='unformatted', &
            status='replace', action='write', iostat=ios)
        call error(ios /= 0, 'Could not open out_field_path for writing: ' // trim(path))
        write(iunit) field
        close(iunit)

    end subroutine write_raw_field


    ! Local replacement for m_utils' write_3d_float32_field_to_file
    subroutine write_field_to_file( field, num_halo, filename )
        implicit none

        real (kind=wp), intent(in) :: field(:, :, :)
        integer, intent(in) :: num_halo
        character(len=*), intent(in) :: filename

        integer :: iunit, ios

        open(newunit=iunit, file=trim(filename), access="stream", form='unformatted', &
            status='replace', action='write', iostat=ios)
        call error(ios /= 0, 'Could not open file for writing: ' // trim(filename))
        write(iunit) 3, 32, num_halo
        write(iunit) shape(field)
        write(iunit) field
        close(iunit)

    end subroutine write_field_to_file


    ! Local replacement for m_utils' error()
    subroutine error(yes, msg, code)
        implicit none

        logical, intent(in) :: yes
        character(len=*), intent(in) :: msg
        integer, intent(in), optional :: code

        if (yes) then
            write(0, *) 'FATAL PROGRAM ERROR!'
            write(0, *) msg
            if (present(code)) write(0, *) code
            write(0, *) 'Execution aborted...'
            stop 1
        end if

    end subroutine error


    ! read and parse the command line arguments
    subroutine read_cmd_line_arguments()
        implicit none

        integer iarg, num_arg
        character(len=1024) :: arg, arg_val

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
            case ("--num_halo")
                call error(iarg + 1 > num_arg, "Missing value for --num_halo argument")
                call get_command_argument(iarg + 1, arg_val)
                read(arg_val, *) num_halo
                iarg = iarg + 1
            case ("--in_field_path")
                call error(iarg + 1 > num_arg, "Missing value for --in_field_path argument")
                call get_command_argument(iarg + 1, in_field_path)
                iarg = iarg + 1
            case ("--out_field_path")
                call error(iarg + 1 > num_arg, "Missing value for --out_field_path argument")
                call get_command_argument(iarg + 1, out_field_path)
                iarg = iarg + 1
            case ("--device")
                call error(iarg + 1 > num_arg, "Missing value for --device argument")
                call get_command_argument(iarg + 1, device)
                call error(trim(device) /= "cpu", "This kernel only supports --device cpu")
                iarg = iarg + 1
            case ("--scan")
                scan = .true.
            case default
                call error(.true., "Unknown command line argument encountered: " // trim(arg))
            end select
            iarg = iarg + 1
        end do

        if (.not. scan) then
            call error(nx == -1, 'You have to specify nx')
            call error(ny == -1, 'You have to specify ny')
        end if
        call error(nz == -1, 'You have to specify nz')
        call error(num_iter == -1, 'You have to specify num_iter')

        if (.not. scan) then
            call error(nx < 0 .or. nx > 1024*1024, "Please provide a reasonable value of nx")
            call error(ny < 0 .or. ny > 1024*1024, "Please provide a reasonable value of ny")
        end if
        call error(nz < 0 .or. nz > 1024, "Please provide a reasonable value of nz")
        call error(num_iter < 1 .or. num_iter > 1024*1024, "Please provide a reasonable value of num_iter")

    end subroutine read_cmd_line_arguments


    ! cleanup at end of work
    subroutine cleanup()
        implicit none
        deallocate(in_field, out_field)
    end subroutine cleanup


    ! finalize at end of program
    subroutine finalize()
        implicit none
    end subroutine finalize


end program main