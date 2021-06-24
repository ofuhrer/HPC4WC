! Module:
! countdown
!
! Author:
! Oliver Fuhrer, Thu May 31 18:35:30 CEST 2007
! oliver.fuhrer@epfl.ch
!
! Description:
! This module provides countdown features which also enable
! the calculation of estimated time for completion

module m_countdown
  implicit none

  ! flag if we want debuggin
  logical, private :: cd_debug=.false.

  ! flag if we need initialization
  logical, private :: cd_set=.false.

  ! countdown
  real (kind=4), private :: cd_time
  integer, private :: cd_ntot
  integer, private :: cd_meslen

contains
  

  ! *************************************************************
  ! start counddown timer and store all necessary information.
  subroutine cdstart(n)
    implicit none

    ! in
    integer, intent(in) :: n

    ! local
    real (kind=4) :: dummy, r(2)
    character(len=256) :: s

    call cddebug('cdstart: entry')

    ! initialize (if necessary)
    call cderror(cd_set, &
         'cdstart: countdown already started and not stopped')

    ! set start flag
    cd_set=.true.

    ! set total number of steps
    cd_ntot=n

    ! get current time
    call dtime(r, dummy)
    cd_time=0.0

    ! print first status
    s='progress:   00%  ETA: ---'
    cd_meslen=len_trim(s)
    write(*,'(a,$)') s(1:cd_meslen)

    call cddebug('cdstart: exit')

  end subroutine cdstart


  ! *************************************************************
  ! show current progress and estimated time to completion
  subroutine cdshow(n)
    implicit none

    ! in
    integer, intent(in) :: n

    ! local
    integer :: i,oldlen,idone
    real (kind=4) :: r(2),now,done,eta
    character(len=256) :: smsg,seta,sunit
    
    call cddebug('cdshow: entry')

    ! initialize (if necessary)
    call cderror(.not.cd_set, &
         'cdshow: counddown not started')

    ! prepare new message
    oldlen=cd_meslen
    call dtime(r, now)
    cd_time=cd_time+now
    done=real(n,8)/real(cd_ntot,8)
    idone=int(done*100.0+0.5)
    if (done.gt.0) then
       eta=cd_time/done*(1.0-done)
       sunit='s'
       if (eta.gt.60.0) then
          eta=eta/60.0
          sunit='min'
          if (eta.gt.60.0) then
             eta=eta/60.0
             sunit='h'
             if (eta.gt.24.0) then
                eta=eta/24.0
                sunit='d'
             end if
          end if
       end if
       write(seta,'(f5.1,x,a)') eta,sunit(1:len_trim(sunit))
    else
       seta='   -----'
    end if
    write(smsg,'(a,i5,a,a)') &
         'progress:',idone,'%  ETA: ',seta(1:len_trim(seta))
    cd_meslen=len_trim(smsg)

    ! delete old message
    do i=1,oldlen
       write(*,'(a,$)') achar(8)
    end do
    do i=1,oldlen
       write(*,'(a,$)') ' '
    end do
    do i=1,oldlen
       write(*,'(a,$)') achar(8)
    end do

    ! print new message
    write(*,'(a,$)') smsg(1:cd_meslen)

    call cddebug('cdshow: exit')

  end subroutine cdshow


  ! *************************************************************
  ! stop current countdown, show total time and clean up
  subroutine cdstop()
    implicit none

    ! local
    integer :: i,oldlen,idone
    real (kind=4) :: r(2),now,done,eta
    character(len=256) :: smsg,seta,sunit

    call cddebug('cdstop: entry')

    ! unset start flag
    cd_set=.false.

    ! prepare new message
    oldlen=cd_meslen
    call dtime(r, now)
    cd_time=cd_time+now
    idone=100
    eta=cd_time
    sunit='s'
    if (eta.gt.60.0) then
       eta=eta/60.0
       sunit='min'
       if (eta.gt.60.0) then
          eta=eta/60.0
          sunit='h'
          if (eta.gt.24.0) then
             eta=eta/24.0
             sunit='d'
          end if
       end if
    end if
    write(seta,'(f5.1,x,a)') eta,sunit(1:len_trim(sunit))
    write(smsg,'(a,i5,a,a)') &
         'progress:',idone,'%  time: ',seta(1:len_trim(seta))
    cd_meslen=len_trim(smsg)

    ! delete old message
    do i=1,oldlen
       write(*,'(a,$)') achar(8)
    end do
    do i=1,oldlen
       write(*,'(a,$)') ' '
    end do
    do i=1,oldlen
       write(*,'(a,$)') achar(8)
    end do

    ! print new message
    write(*,'(a)') smsg(1:cd_meslen)

    call cddebug('cdstop: exit')

  end subroutine cdstop


  ! *************************************************************
  ! echo debugging information (if on)
  subroutine cddebug(str)
    implicit none
    
    ! in
    character(len=*) :: str

    if (.not.cd_debug) return
    write(0,*) 'cd>',str

  end subroutine cddebug


  !**************************************
  ! write error message and terminate
  subroutine cderror(yes,msg)
    implicit none
    
    ! in
    logical, intent(in) :: yes
    character(len=*) msg
    
    ! local
    integer, external :: lnblnk
    
    if (yes) then
       write(0,*) 'COUNTDOWN MODULE FATAL ERROR!'
       write(0,*) 'Message: ',msg
       write(0,*) 'Execution aborted...'
       stop
    end if
    
  end subroutine cderror

end module m_countdown
