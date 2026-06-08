! ******************************************************
!      Module: m_utils
!      Author: Oliver Fuhrer
!       Email: oliver.fuhrer@vulcan.com
!        Date: 09.09.2010
! Description: Collection of utility routines for
!              program flow managment and timing.
! ******************************************************

module m_utils
    use mpi, only : MPI_COMM_WORLD, MPI_Abort
    implicit none

    integer, parameter :: stderr = 0

    integer, parameter :: dp = 8
    integer, parameter :: max_timing = 100

    logical  :: ltiming = .false.
    logical  :: ltiming_list(max_timing)
    
    character (len=28) :: tag_list(max_timing)
    
    integer :: cur_timing = 0
    integer :: ncalls(max_timing)
    
    real (kind=dp) :: stiming(max_timing)
    real (kind=dp) :: rtiming(max_timing)

    interface write_field_to_file
        module procedure write_2d_float32_field_to_file, write_3d_float32_field_to_file, &
                         write_2d_float64_field_to_file, write_3d_float64_field_to_file
    end interface write_field_to_file
    
contains


    subroutine write_2d_float32_field_to_file( field, num_halo, filename )
        implicit none

        ! arguments
        real (kind=4), intent(in) :: field(:, :)
        integer, intent(in) :: num_halo
        character(len=*), intent(in) :: filename

        ! local
        integer :: iunit = 42
        integer :: i, j, k

        open(iunit, file=trim(filename), access="stream")
        write(iunit) 2, 32, num_halo
        write(iunit) shape(field)
        write(iunit) field
        close(iunit)

    end subroutine write_2d_float32_field_to_file


    subroutine write_3d_float32_field_to_file( field, num_halo, filename )
        implicit none

        ! arguments
        real (kind=4), intent(in) :: field(:, :, :)
        integer, intent(in) :: num_halo
        character(len=*), intent(in) :: filename

        ! local
        integer :: iunit = 42
        integer :: i, j, k

        open(iunit, file=trim(filename), access="stream")
        write(iunit) 3, 32, num_halo
        write(iunit) shape(field)
        write(iunit) field
        close(iunit)

    end subroutine write_3d_float32_field_to_file


    subroutine write_2d_float64_field_to_file( field, num_halo, filename )
        implicit none

        ! arguments
        real (kind=8), intent(in) :: field(:, :)
        integer, intent(in) :: num_halo
        character(len=*), intent(in) :: filename

        ! local
        integer :: iunit = 42
        integer :: i, j, k

        open(iunit, file=trim(filename), access="stream")
        write(iunit) 2, 64, num_halo
        write(iunit) shape(field)
        write(iunit) field
        close(iunit)

    end subroutine write_2d_float64_field_to_file


    subroutine write_3d_float64_field_to_file( field, num_halo, filename )
        implicit none

        ! arguments
        real (kind=8), intent(in) :: field(:, :, :)
        integer, intent(in) :: num_halo
        character(len=*), intent(in) :: filename

        ! local
        integer :: iunit = 42
        integer :: i, j, k

        open(iunit, file=trim(filename), access="stream")
        write(iunit) 3, 64, num_halo
        write(iunit) shape(field)
        write(iunit) field
        close(iunit)

    end subroutine write_3d_float64_field_to_file


    function is_master()
        implicit none

        ! function value
        logical :: is_master

        is_master = my_rank() == 0

    end function is_master


    subroutine sync()
        use mpi, only : MPI_COMM_WORLD, MPI_BARRIER
        implicit none

        ! local
        integer :: ierror

        call MPI_BARRIER(MPI_COMM_WORLD, ierror)
        call error(ierror /= 0, 'Problem with MPI_BARRIER', code=ierror)

    end subroutine


    function num_rank()
        use mpi, only : MPI_COMM_WORLD, MPI_COMM_SIZE
        implicit none

        ! function value
        integer :: num_rank

        ! local
        integer :: ierror

        call MPI_COMM_SIZE(MPI_COMM_WORLD, num_rank, ierror)
        call error(ierror /= 0, 'Problem with MPI_COMM_SIZE', code=ierror)

    end function num_rank


    function my_rank()
        use mpi, only : MPI_COMM_WORLD, MPI_COMM_RANK
        implicit none

        ! function value
        integer :: my_rank

        ! local
        integer :: ierror

        call MPI_COMM_RANK(MPI_COMM_WORLD, my_rank, ierror)
        call error(ierror /= 0, 'Problem with MPI_COMM_RANK', code=ierror)

    end function my_rank


    subroutine error(yes, msg, code)
        implicit none

        ! in
        logical, intent(in) :: yes
        character(len=*), intent(in) :: msg
        integer, intent(in), optional :: code

        ! local
        integer :: ierror

        if (yes) then
            write(stderr,*) 'FATAL PROGRAM ERROR!'
            write(stderr,*) msg
            if (present(code)) then
                write(stderr,*) code
            end if
            write(stderr,*) 'Execution aborted...'
            call MPI_Abort(MPI_COMM_WORLD, 42, ierror)
            stop
        end if

    end subroutine error


    subroutine timer_init()
        implicit none

        ltiming         = .true.
        ltiming_list(:) = .false.
        rtiming(:)      = 0.0_dp
        ncalls(:)       = 0
        stiming(:)      = -1.0_dp

    end subroutine timer_init


    subroutine timer_reset()
        implicit none

        rtiming(:) = 0.0_dp
        ncalls(:)  = 0
        stiming(:) = -1.0_dp

    end subroutine timer_reset


    subroutine timer_start(tag, inum)
        use mpi, only : MPI_WTIME
        implicit none

        ! arguments
        character (LEN=*), intent(in) :: tag
        integer, intent(inout) :: inum

        ! local
        integer :: ierror

        ! assign new index
        if (inum <= 0) then
            cur_timing = cur_timing + 1
            inum = cur_timing
        end if

        ! check inum
        call error(inum < 1 .or. inum > max_timing, &
            'ERROR: Problem in start_loc_timing (inum < 1 or inum > max_timing)')

        ! save tag if this is the first call (check tag in debug mode otherwise)
        if (stiming(inum) < 0.d0) then
            tag_list(inum) = trim(tag)
            ltiming_list(inum) = .true.
        end if

        stiming(inum) = MPI_WTIME()
        ncalls(inum) = ncalls(inum) + 1

    end subroutine timer_start


    subroutine timer_end(inum)
        use mpi, only : MPI_WTIME
        implicit none

        ! arguments
        integer, intent(in) :: inum

        ! local
        real (kind=dp) :: ztime
        integer :: ierror

        ztime = MPI_WTIME() - stiming(inum)
        rtiming(inum) = rtiming(inum) + ztime

    end subroutine timer_end


    function timer_get(inum)
        use mpi, only : MPI_WTIME, MPI_DOUBLE_PRECISION, MPI_SUM
        implicit none

        ! arguments
        integer, intent(in) :: inum
        real (kind=dp) :: timer_get

        ! local
        real (kind=dp) :: ztime_sum
        integer :: ierror

        call MPI_REDUCE(rtiming(inum), ztime_sum, 1, MPI_DOUBLE_PRECISION, MPI_SUM, 0, MPI_COMM_WORLD, ierror)
        timer_get = ztime_sum / REAL(num_rank(), dp)

    end function timer_get


    subroutine timer_print()
        use mpi, only : MPI_COMM_WORLD, MPI_DOUBLE_PRECISION, MPI_MIN, MPI_MAX, MPI_SUM
        implicit none

        ! local
        integer :: inum
        real (kind=dp) :: ztime_mean, ztime_min, ztime_max
        integer :: ierror

        if (ltiming) then

            if (is_master()) print *, ' Timer   Tag                     #calls        min[s]        max[s]       mean[s] '

            do inum = 1, max_timing

                ! NOTE: this if-statement can go terribly wrong if not all locations are
                ! called on all ranks (cf. MPI_REDUCE below)
                if (ltiming_list(inum)) then 

                    ! get run time
                    call MPI_REDUCE(rtiming(inum), ztime_min, 1, MPI_DOUBLE_PRECISION, MPI_MIN, 0, MPI_COMM_WORLD, ierror)
                    call MPI_REDUCE(rtiming(inum), ztime_max, 1, MPI_DOUBLE_PRECISION, MPI_MAX, 0, MPI_COMM_WORLD, ierror)
                    call MPI_REDUCE(rtiming(inum), ztime_mean, 1, MPI_DOUBLE_PRECISION, MPI_SUM, 0, MPI_COMM_WORLD, ierror)
                    ztime_mean = ztime_mean / REAL(num_rank(), dp)

                    if (is_master()) WRITE(*,"(I4,A28,I8,F14.4,F14.4,F14.4)")  &
                            inum, "      " // adjustl(tag_list(inum)), ncalls(inum), ztime_min, ztime_max, ztime_mean

                end if
            end do

        end if

    end subroutine timer_print


end module m_utils
