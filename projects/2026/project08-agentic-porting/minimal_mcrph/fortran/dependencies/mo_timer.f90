! Minimal, single-worker timer utility, in place of ICON's mo_timer.
!
! The real ICON mo_timer integrates with a full MPI-aware statistics and
! output framework built around many named timers spanning the whole model.
! This scheme only ever needs a handful of independent stopwatches to see
! where time goes within one microphysics call: initializing the wet-growth
! lookup table, pre/post-processing, the main microphysical process calls,
! and sedimentation. Each one here is just a named CPU-time accumulator that
! can be started and stopped repeatedly (once per timestep, typically), so
! the totals can be compared against each other afterwards.
MODULE mo_timer

  USE mo_kind, ONLY: wp

  IMPLICIT NONE

  PRIVATE

  PUBLIC :: timers_level
  PUBLIC :: timer_start, timer_stop, timer_reset, timer_report

  PUBLIC :: timer_phys_2mom_dmin_init
  PUBLIC :: timer_phys_2mom_wetgrowth
  PUBLIC :: timer_phys_2mom_prepost
  PUBLIC :: timer_phys_2mom_proc
  PUBLIC :: timer_phys_2mom_sedi

  !> Call sites guard every timer_start/timer_stop with
  !> "IF (timers_level > N) CALL timer_...", so timing is opt-in: raise this
  !> above the relevant threshold (currently 10 everywhere) to enable it.
  INTEGER :: timers_level = 0

  INTEGER, PARAMETER :: timer_phys_2mom_dmin_init = 1
  INTEGER, PARAMETER :: timer_phys_2mom_wetgrowth = 2
  INTEGER, PARAMETER :: timer_phys_2mom_prepost   = 3
  INTEGER, PARAMETER :: timer_phys_2mom_proc      = 4
  INTEGER, PARAMETER :: timer_phys_2mom_sedi      = 5
  INTEGER, PARAMETER :: n_timers = 5

  CHARACTER(len=24), PARAMETER :: timer_name(n_timers) = [ CHARACTER(len=24) :: &
       'phys_2mom_dmin_init', 'phys_2mom_wetgrowth', 'phys_2mom_prepost', &
       'phys_2mom_proc',      'phys_2mom_sedi' ]

  REAL(wp) :: elapsed(n_timers)    = 0.0_wp
  REAL(wp) :: t_start(n_timers)    = 0.0_wp
  INTEGER  :: call_count(n_timers) = 0

CONTAINS

  SUBROUTINE timer_start(timer_id)
    INTEGER, INTENT(in) :: timer_id
    CALL CPU_TIME(t_start(timer_id))
  END SUBROUTINE timer_start

  SUBROUTINE timer_stop(timer_id)
    INTEGER, INTENT(in) :: timer_id
    REAL(wp) :: t_now
    CALL CPU_TIME(t_now)
    elapsed(timer_id)    = elapsed(timer_id) + (t_now - t_start(timer_id))
    call_count(timer_id) = call_count(timer_id) + 1
  END SUBROUTINE timer_stop

  SUBROUTINE timer_reset(timer_id)
    INTEGER, INTENT(in) :: timer_id
    elapsed(timer_id)    = 0.0_wp
    call_count(timer_id) = 0
  END SUBROUTINE timer_reset

  !> Print accumulated CPU time and call count per timer, so the individual
  !> stopwatches can be separated out and compared against each other.
  SUBROUTINE timer_report()
    INTEGER :: i
    WRITE(*,'(A)') 'mo_timer: accumulated CPU time per timer'
    DO i = 1, n_timers
      WRITE(*,'(2X,A,T26,A,F12.4,A,I8,A)') &
           TRIM(timer_name(i)), ': ', elapsed(i), ' s over ', call_count(i), ' call(s)'
    END DO
  END SUBROUTINE timer_report

END MODULE mo_timer
