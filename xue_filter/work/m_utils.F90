! ******************************************************
!      Module: m_utils
!      Author: Oliver Fuhrer
!       Email: oliver.fuhrer@epfl.ch
!        Date: 09.09.2006
! Description: Collection of utility routines for
!              program flow managment.
!              See individual routines
!              for more information.
! ******************************************************

module m_utils
#ifdef USE_MPI
    use mpi, only : MPI_COMM_WORLD, MPI_Abort
#endif
    use m_constants, only : stderr
    implicit none

    integer, parameter :: dp = 8
    integer, parameter :: max_timing = 100

    logical  :: ltiming = .false.
    logical  :: ltiming_list(max_timing)
    
    character (len=28) :: tag_list(max_timing)
    
    integer :: cur_timing = 0
    integer :: icountrate, icountmax
    integer :: icountnew(max_timing)
    integer :: icountold(max_timing)
    integer :: ncalls(max_timing)
    
#ifdef USE_MPI
    real (kind=dp) :: stiming(max_timing)
#endif
    real (kind=dp) :: rtiming(max_timing)
    real (kind=dp) :: rsync(max_timing)
    
contains


function is_master()
    implicit none

    ! function value
    logical :: is_master
    
    is_master = my_rank() == 0

end function is_master


subroutine sync()
#ifdef USE_MPI
    use mpi, only : MPI_COMM_WORLD, MPI_BARRIER
#endif
    implicit none

    ! local
    integer :: ierror

#ifdef USE_MPI
    call MPI_BARRIER(MPI_COMM_WORLD, ierror)
    call error(ierror /= 0, 'Problem with MPI_BARRIER', code=ierror)
#endif

end subroutine


function num_rank()
#ifdef USE_MPI
    use mpi, only : MPI_COMM_WORLD, MPI_COMM_SIZE
#endif
    implicit none

    ! function value
    integer :: num_rank

    ! local
    integer :: ierror

#ifdef USE_MPI
    call MPI_COMM_SIZE(MPI_COMM_WORLD, num_rank, ierror)
    call error(ierror /= 0, 'Problem with MPI_COMM_SIZE', code=ierror)
#else
    num_rank = 1
#endif

end function num_rank


function my_rank()
#ifdef USE_MPI
    use mpi, only : MPI_COMM_WORLD, MPI_COMM_RANK
#endif
    implicit none

    ! function value
    integer :: my_rank

    ! local
    integer :: ierror

#ifdef USE_MPI
    call MPI_COMM_RANK(MPI_COMM_WORLD, my_rank, ierror)
    call error(ierror /= 0, 'Problem with MPI_COMM_RANK', code=ierror)
#else
    my_rank = 0
#endif

end function my_rank


! write error message and terminate
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
#ifdef USE_MPI
    call MPI_Abort(MPI_COMM_WORLD, 42, ierror)
#endif
    stop
end if

end subroutine error


subroutine timer_init()
    implicit none

    ! local
    integer :: idummy

    ltiming         = .true.
    ltiming_list(:) = .false.
    rtiming(:)      = 0.0_dp
    rsync(:)        = 0.0_dp
    ncalls(:)       = 0
    icountold(:)    = -1  ! set to -1 to signal that this timer has never been used
#ifndef USE_MPI
    call system_clock(count=idummy, count_rate=icountrate, count_max=icountmax)
#else
    stiming(:) = -1.0_dp
    icountrate  = 0
    icountmax   = 0
#endif

end subroutine timer_init


subroutine timer_reset()
    implicit none

    rtiming(:) = 0.0_dp
    rsync(:)   = 0.0_dp
    ncalls(:)  = 0
#ifdef USE_MPI
    stiming(:) = -1.0_dp
#endif

end subroutine timer_reset


subroutine timer_start(tag, inum)
#ifdef USE_MPI
    use mpi, only : MPI_WTIME
#endif
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

    ! make sure this is the first call or previously end_loc_timing has been called
    call error(icountold(inum) /= -1 .and. icountold(inum) /= -2, &
        'ERROR: Problem in start_loc_timing (no previous end_loc_timing)')

    ! save tag if this is the first call (check tag in debug mode otherwise)
    if (icountold(inum) == -1) then
        tag_list(inum) = trim(tag)
        ltiming_list(inum) = .true.
    end if

#ifndef USE_MPI
    call system_clock(count=icountold(inum))
#else
    stiming(inum) = MPI_WTIME()
    icountold(inum) = 1  !Flag inum in use
#endif
    ncalls(inum) = ncalls(inum) + 1

end subroutine timer_start


subroutine timer_end(inum)
#ifdef USE_MPI
    use mpi, only : MPI_WTIME
#endif
    implicit none

    ! arguments
    integer, intent(in) :: inum

    ! local
    real (kind=dp) :: etiming,ztime
    integer :: ierror

    call error(icountold(inum) == -1 .or. icountold(inum) == -2, &
        'ERROR: Problem in end_loc_timing (no matching start_loc_timing)' )

#ifndef USE_MPI
    call system_clock(count=icountnew(inum))
    if ( icountnew(inum) >= icountold(inum) ) then
        ztime = ( real(icountnew(inum) - icountold(inum), dp) )      &
                    / real(icountrate, dp)
    else
        ztime = real(icountmax - (icountold(inum)-icountnew(inum) ), dp)     &
                    / real(icountrate, dp)
    end if
#else
    etiming = MPI_WTIME()
    ztime = etiming - stiming(inum)
    icountnew(inum) = 1
#endif
    rtiming(inum) = rtiming(inum) + ztime

    ! release timer
    icountold(inum) = -2 ! set to -2 to ensure a start_timing is called before next end_timing

end subroutine timer_end


subroutine timer_print()
#ifdef USE_MPI
    use mpi, only : MPI_COMM_WORLD, MPI_DOUBLE_PRECISION, MPI_MIN, MPI_MAX, MPI_SUM
#endif
    implicit none

    ! local
    integer :: inum
    real (kind=dp) :: ztime_mean, ztime_min, ztime_max
    integer :: ierror

    if (ltiming) then

        if (is_master()) print *, '--------------------------------------------------------------------------'
        if (is_master()) print *, ' Timers:'
        if (is_master()) WRITE(*, "(A,I4)") '   number of workers = ', num_rank()
        if (is_master()) print *, '--------------------------------------------------------------------------'
        if (is_master()) print *, ' Id      Tag                     #calls        min[s]        max[s]       mean[s] '

        do inum = 1, max_timing

            ! NOTE: this if-statement can go terribly wrong if not all locations are
            ! called on all ranks (cf. MPI_REDUCE below)
            if (ltiming_list(inum)) then 

                ! get run time
#ifdef USE_MPI
                call MPI_REDUCE(rtiming(inum), ztime_min, 1, MPI_DOUBLE_PRECISION, MPI_MIN, 0, MPI_COMM_WORLD, ierror)
                call MPI_REDUCE(rtiming(inum), ztime_max, 1, MPI_DOUBLE_PRECISION, MPI_MAX, 0, MPI_COMM_WORLD, ierror)
                call MPI_REDUCE(rtiming(inum), ztime_mean, 1, MPI_DOUBLE_PRECISION, MPI_SUM, 0, MPI_COMM_WORLD, ierror)
                ztime_mean = ztime_mean / REAL(num_rank(), dp)
#else
                ztime_mean = rtiming(inum)
                ztime_min  = rtiming(inum)
                ztime_max  = rtiming(inum)
#endif

                if (is_master()) WRITE(*,"(I4,A28,I8,F14.4,F14.4,F14.4)")  &
                        inum, "      " // adjustl(tag_list(inum)), ncalls(inum), ztime_min, ztime_max, ztime_mean

            end if
        end do
        if (is_master()) print *, '--------------------------------------------------------------------------'

    end if

end subroutine timer_print

end module m_utils
