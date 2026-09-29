!>
!! Message/error-reporting utility, simplified for single-worker (non-MPI) use.
!!
!! This is a trimmed-down version of ICON's mo_exception: the original dealt
!! with reporting from many MPI ranks (rank-aware printing, process-splitting,
!! collective abort) and used ICON's mo_io_units/mo_impl_constants for unit
!! numbers and string-length limits, plus a system backtrace utility on
!! finish(). None of that applies when there is only one worker, so all of
!! it has been removed: every message is simply printed by this one process,
!! and the unit numbers / string-length limit are defined locally below
!! instead of being pulled in from mo_io_units/mo_impl_constants.
!!
!! @par Revision History
!! - Initial version taken from ECHAM6;
!! - Subroutines print_status and print_value taken from mo_submodel of
!!   ECHAM6 by Hui Wan (2010-07-14).
!!
MODULE mo_exception

  USE mo_kind, ONLY: wp, i8

  IMPLICIT NONE

  PRIVATE

  PUBLIC :: message_text
  PUBLIC :: message, warning, finish, print_status, print_value
  PUBLIC :: em_none, em_info, em_warn, em_error, em_param, em_debug
  PUBLIC :: open_log, close_log
  PUBLIC :: debug_messages_on, debug_messages_off
  PUBLIC :: number_of_warnings, number_of_errors
  PUBLIC :: get_filename_noext
  PUBLIC :: msg_timestamp

  ! Output units, in place of ICON's mo_io_units: nerr = standard error,
  ! nlog = an arbitrary free unit used for the (optional) log file.
  INTEGER, PARAMETER :: nerr = 0
  INTEGER, PARAMETER :: nlog = 10

  ! Longest filename this module will hand back from get_filename_noext(),
  ! in place of ICON's mo_io_units%filename_max.
  INTEGER, PARAMETER :: filename_max = 256

  ! Longest message string, in place of ICON's mo_impl_constants%MAX_CHAR_LENGTH.
  INTEGER, PARAMETER :: max_char_length = 256

  INTEGER, PARAMETER :: em_none  = 0   !< normal message
  INTEGER, PARAMETER :: em_info  = 1   !< informational message
  INTEGER, PARAMETER :: em_warn  = 2   !< warning message: number of warnings counted
  INTEGER, PARAMETER :: em_error = 3   !< error message: number of errors counted
  INTEGER, PARAMETER :: em_param = 4   !< report parameter value
  INTEGER, PARAMETER :: em_debug = 5   !< debugging message

  CHARACTER(len=max_char_length) :: message_text = ''

  LOGICAL :: l_debug = .FALSE.
  LOGICAL :: l_log   = .FALSE.

  !> Flag. If .TRUE., precede output messages by time stamp.
  LOGICAL :: msg_timestamp = .FALSE.

  INTEGER :: number_of_warnings  = 0
  INTEGER :: number_of_errors    = 0

  INTERFACE print_value            !< report on a parameter value
    MODULE PROCEDURE print_lvalue  !< logical
    MODULE PROCEDURE print_ivalue  !< integer
    MODULE PROCEDURE print_i8value !< integer(i8)
    MODULE PROCEDURE print_rvalue  !< real
  END INTERFACE

CONTAINS

  SUBROUTINE debug_messages_on
    l_debug = .TRUE.
  END SUBROUTINE debug_messages_on

  SUBROUTINE debug_messages_off
    l_debug = .FALSE.
  END SUBROUTINE debug_messages_off

  !>
  !! Strip the extension off a filename (everything from the last '.' onward).
  FUNCTION get_filename_noext(name) result(filename)
    CHARACTER(len=*), INTENT(in) :: name
    CHARACTER(len=filename_max)  :: filename

    INTEGER :: end_name

    end_name = INDEX(name, '.', back=.true.)
    IF (end_name > 0) THEN
      filename = name(1:end_name-1)
    ELSE
      filename = TRIM(name)
    ENDIF

  END FUNCTION get_filename_noext

  !>
  !! @brief Report a fatal error and stop the program.
  SUBROUTINE finish (name, text, exit_no)

    CHARACTER(len=*), INTENT(in)           :: name
    CHARACTER(len=*), INTENT(in), OPTIONAL :: text
    INTEGER,          INTENT(in), OPTIONAL :: exit_no

    INTEGER :: iexit

    WRITE (nerr,'(/,80("="),/)')
    IF (l_log) WRITE (nlog,'(/,80("="),/)')

    IF (PRESENT(exit_no)) THEN
       iexit = exit_no
    ELSE
       iexit = 1     ! POSIX defines this as EXIT_FAILURE
    END IF

    IF (PRESENT(text)) THEN
      IF (iexit == 1) THEN
        WRITE (nerr,'(a,a,a,a)') 'FATAL ERROR in ', TRIM(name), ': ', TRIM(text)
      ELSE
        WRITE (nerr,'(1x,a,a,a)') TRIM(name), ': ', TRIM(text)
      ENDIF
      IF (l_log) WRITE (nlog,'(1x,a,a,a)') TRIM(name), ': ', TRIM(text)
    ELSE
      IF (iexit == 1) THEN
        WRITE (nerr,'(a,a)') 'FATAL ERROR in ', TRIM(name)
      ELSE
        WRITE (nerr,'(1x,a)') TRIM(name)
      ENDIF
      IF (l_log) WRITE (nlog,'(a)') TRIM(name)
    ENDIF

    WRITE (nerr,'(/,80("-"),/,/)')
    IF (l_log) WRITE (nlog,'(/,80("-"),/,/)')
    WRITE (nerr,'(/,80("="),/)')
    IF (l_log) WRITE (nlog,'(/,80("="),/)')

    STOP 1

  END SUBROUTINE finish

  !>
  SUBROUTINE warning (name, text)

    CHARACTER (len=*), INTENT(in) :: name
    CHARACTER (len=*), INTENT(in) :: text

    CALL message (name, text, level=em_warn)

  END SUBROUTINE warning

  !>
  !! @brief Print a message. With one worker, every call prints -- there is
  !!        no "am I the process that should report this" decision to make.
  SUBROUTINE message (name, text, out, level, all_print, adjust_right)

    CHARACTER (len=*), INTENT(in) :: name
    CHARACTER (len=*), INTENT(in) :: text
    !> unit to write to, defaults to standard error
    INTEGER,           INTENT(in), OPTIONAL :: out
    INTEGER,           INTENT(in), OPTIONAL :: level
    !> kept for interface compatibility; every call already prints (single worker)
    LOGICAL,           INTENT(in), OPTIONAL :: all_print
    LOGICAL,           INTENT(in), OPTIONAL :: adjust_right

    INTEGER :: iout
    INTEGER :: ilevel
    LOGICAL :: ladjust

    CHARACTER(len=8) :: prefix

    CHARACTER(len=10) :: ctime
    CHARACTER(len=8)  :: cdate

    IF (PRESENT(adjust_right)) THEN
      ladjust = adjust_right
    ELSE
      ladjust = .FALSE.
    ENDIF

    IF (PRESENT(out)) THEN
      iout = out
    ELSE
      iout = nerr
    END IF

    IF (PRESENT(level)) THEN
      ilevel = level
    ELSE
      ilevel = em_none
    END IF

    SELECT CASE (ilevel)
    CASE (em_none)  ; prefix = '        '
    CASE (em_info)  ; prefix = 'INFO   :'
    CASE (em_warn)  ; prefix = 'WARNING:' ; number_of_warnings  = number_of_warnings+1
    CASE (em_error) ; prefix = 'ERROR  :' ; number_of_errors    = number_of_errors+1
    CASE (em_param) ; prefix = '---     '
    CASE (em_debug) ; prefix = 'DEBUG  :'
    END SELECT

    IF (.NOT. ladjust) THEN
      message_text = ADJUSTL(text)
    ELSE
      message_text = text
    ENDIF
    IF (name /= '')  THEN
      message_text = TRIM(name) // ': ' // message_text
    ENDIF
    IF (ilevel > em_none) THEN
      message_text = TRIM(prefix) // ' ' // message_text
    ENDIF

    IF (msg_timestamp) THEN
      CALL DATE_AND_TIME(date=cdate,time=ctime)
      WRITE(iout,'(a)') '['//cdate//' '//ctime//'] '//TRIM(message_text)
      IF (l_log) WRITE(nlog,'(a)') '['//cdate//' '//ctime//']: '//TRIM(message_text)
    ELSE
      WRITE(iout,'(1x,a)') TRIM(message_text)
      IF (l_log) WRITE(nlog,'(1x,a)') TRIM(message_text)
    END IF

  END SUBROUTINE message

  !>
  SUBROUTINE print_status (mstring, flag)

    CHARACTER(len=*), intent(in)   :: mstring
    LOGICAL, intent(in)            :: flag

    IF ( flag ) THEN
       WRITE(message_text,'(a60,1x,":",a)') mstring,'active'
    ELSE
       WRITE(message_text,'(a60,1x,":",a)') mstring,'*not* active'
    END IF
    CALL message('', message_text, level=em_param)

  END SUBROUTINE print_status

  !>
  !! Report the value of a logical, integer or real variable
  !! Convenience routine interfaced by print_value(mstring, value)
  !!
  SUBROUTINE print_lvalue (mstring, lvalue, routine)

    CHARACTER(len=*), intent(in)   :: mstring
    LOGICAL, INTENT(in)            :: lvalue
    CHARACTER(len=*), TARGET, OPTIONAL, INTENT(in) :: routine
    CHARACTER(len=:), POINTER :: rtn
    CHARACTER(len=1), TARGET :: dummy

    IF (PRESENT(routine)) THEN
      rtn => routine
    ELSE
      dummy = ' '
      rtn => dummy(1:0)
    END IF
    WRITE(message_text,'(a60,1x,": ",a)') mstring, &
         MERGE('TRUE ', 'FALSE', lvalue)
    CALL message(rtn, message_text, level=em_param)

  END SUBROUTINE print_lvalue

  !>
  SUBROUTINE print_ivalue(mstring, ivalue, routine)

    CHARACTER(len=*), intent(in)   :: mstring
    INTEGER, intent(in)            :: ivalue
    CHARACTER(len=*), TARGET, OPTIONAL, INTENT(in) :: routine
    CHARACTER(len=:), POINTER :: rtn
    CHARACTER(len=1), TARGET :: dummy

    IF (PRESENT(routine)) THEN
      rtn => routine
    ELSE
      dummy = ' '
      rtn => dummy(1:0)
    END IF
    WRITE(message_text,'(a60,1x,":",i10)') mstring, ivalue
    CALL message(rtn, message_text, level=em_param)

  END SUBROUTINE print_ivalue

  !>
  SUBROUTINE print_i8value (mstring, i8value, routine)

    CHARACTER(len=*), intent(in)   :: mstring
    INTEGER(i8), intent(in)        :: i8value
    CHARACTER(len=*), TARGET, OPTIONAL, INTENT(in) :: routine
    CHARACTER(len=:), POINTER :: rtn
    CHARACTER(len=1), TARGET :: dummy

    IF (PRESENT(routine)) THEN
      rtn => routine
    ELSE
      dummy = ' '
      rtn => dummy(1:0)
    END IF
    WRITE(message_text,'(a60,1x,":",i10)') mstring, i8value
    CALL message(rtn, message_text, level=em_param)

  END SUBROUTINE print_i8value

  !>
  SUBROUTINE print_rvalue (mstring, rvalue, routine)

    CHARACTER(len=*), intent(in)   :: mstring
    REAL(wp), intent(in)           :: rvalue
    CHARACTER(len=*), TARGET, OPTIONAL, INTENT(in) :: routine
    CHARACTER(len=:), POINTER :: rtn
    CHARACTER(len=1), TARGET :: dummy

    IF (PRESENT(routine)) THEN
      rtn => routine
    ELSE
      dummy = ' '
      rtn => dummy(1:0)
    END IF
    WRITE(message_text,'(a60,1x,":",g12.5)') mstring, rvalue
    CALL message(rtn, message_text, level=em_param)

  END SUBROUTINE print_rvalue

  !>
  SUBROUTINE open_log (logfile_name)

    CHARACTER(len=*), INTENT(in) :: logfile_name
    LOGICAL                      :: l_opened

    INQUIRE (UNIT=nlog,OPENED=l_opened)

    IF (l_opened) THEN
      WRITE (message_text,'(a)') 'log file unit has been used already.'
      CALL message ('open_log', message_text)
      WRITE (message_text,'(a)') 'Close unit and reopen for log file.'
      CALL message ('open_log', message_text, level=em_warn)
      CLOSE (nlog)
    ENDIF

    OPEN (nlog,file=TRIM(logfile_name))

    l_log = .TRUE.

  END SUBROUTINE open_log

  !>
  SUBROUTINE close_log
    LOGICAL :: l_opened

    INQUIRE (UNIT=nlog,OPENED=l_opened)
    IF (l_opened) THEN
      CLOSE (nlog)
    ENDIF

    l_log = .FALSE.

  END SUBROUTINE close_log

END MODULE mo_exception
