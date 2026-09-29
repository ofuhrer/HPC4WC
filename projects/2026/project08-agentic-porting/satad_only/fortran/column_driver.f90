! Top-level single-column driver for saturation adjustment (satad) alone.
!
! Reads a 1D atmospheric column from one CSV file, calls satad_v_3D once, and
! writes the updated column back out.
!
!   example/fields.csv one row of headers, then nlev data rows, columns:
!                       rho,tk,qv,qc
!
! Saturation adjustment relaxes each level to thermodynamic equilibrium at
! constant total density: it moves water between vapour (qv) and cloud (qc)
! and adjusts the temperature (tk) by the associated latent heat. It treats
! every level independently, so -- unlike the full microphysics -- no
! half-level heights, layer thickness, pressure or vertical coupling are
! needed here; rho and tk/qv/qc are the whole story.
!
! nlev is inferred from the number of data rows in fields.csv. The iteration
! controls (maxiter, tol) are hardcoded PARAMETERs below; edit and rebuild to
! change them.

PROGRAM column_driver

  USE mo_kind,   ONLY: wp
  USE mo_satad,  ONLY: satad_v_3D

  IMPLICIT NONE

  ! -------------------------------------------------------------------
  ! File names -- defaults, overridable via command-line arguments 1 and 2.
  ! Defaults are relative to the fortran/ directory (../example/ is the shared
  ! CSV location used by fortran, gt4py and the test harness).
  ! -------------------------------------------------------------------
  CHARACTER(len=1024) :: fields_file = '../example/fields.csv'
  CHARACTER(len=1024) :: output_file = '../example/output_fields.csv'

  ! Column order of fields.csv -- read and written in this exact order.
  CHARACTER(len=*), PARAMETER :: field_header = 'rho,tk,qv,qc'

  ! -------------------------------------------------------------------
  ! Saturation-adjustment controls -- defaults, overridable via arguments 3, 4.
  ! -------------------------------------------------------------------
  INTEGER  :: maxiter = 10          ! max Newton iterations per level
  REAL(wp) :: tol     = 1.0e-3_wp   ! abs. accuracy [K] of adjusted temperature

  ! -------------------------------------------------------------------
  ! Column state
  ! -------------------------------------------------------------------
  INTEGER :: nlev

  REAL(wp), ALLOCATABLE :: rho(:), tk(:), qv(:), qc(:)

  ! Copies of the input, kept so we can report the per-level change satad made.
  REAL(wp), ALLOCATABLE :: tk_in(:), qv_in(:), qc_in(:)

  INTEGER :: errstat, k

  ! -------------------------------------------------------------------
  ! Command-line overrides:  column_driver [in_csv [out_csv [tol [maxiter]]]]
  ! -------------------------------------------------------------------
  CALL parse_cli(fields_file, output_file, tol, maxiter)

  ! -------------------------------------------------------------------
  ! Read input
  ! -------------------------------------------------------------------
  CALL count_data_rows(TRIM(fields_file), nlev)
  IF (nlev < 1) CALL die('read_fields: '//TRIM(fields_file)//' has no data rows')

  ALLOCATE(rho(nlev), tk(nlev), qv(nlev), qc(nlev))
  ALLOCATE(tk_in(nlev), qv_in(nlev), qc_in(nlev))

  CALL read_fields(TRIM(fields_file), nlev, rho, tk, qv, qc)

  tk_in = tk
  qv_in = qv
  qc_in = qc

  WRITE(*,'(A,I0)') 'column_driver: nlev = ', nlev

  ! -------------------------------------------------------------------
  ! Call saturation adjustment for the whole column (klo=1, kup=nlev)
  ! -------------------------------------------------------------------
  CALL satad_v_3D(maxiter=maxiter, tol=tol, te=tk, qve=qv, qce=qc, &
       &          rhotot=rho, kdim=nlev, klo=1, kup=nlev,          &
       &          errstat=errstat)

  IF (errstat /= 0) THEN
    WRITE(*,'(A,I0)') 'column_driver: satad returned errstat = ', errstat
    CALL die('saturation adjustment failed')
  END IF

  ! -------------------------------------------------------------------
  ! Report the per-level change and write the updated column
  ! -------------------------------------------------------------------
  WRITE(*,'(A)') 'column_driver: per-level change from saturation adjustment'
  WRITE(*,'(A)') '  lev        dT [K]        dqv [kg/kg]        dqc [kg/kg]'
  DO k = 1, nlev
    WRITE(*,'(I5,3ES18.8E3)') k, tk(k)-tk_in(k), qv(k)-qv_in(k), qc(k)-qc_in(k)
  END DO

  CALL write_fields(TRIM(output_file), nlev, rho, tk, qv, qc)

  WRITE(*,'(A)') 'column_driver: wrote '//TRIM(output_file)

CONTAINS

  ! Parse optional positional command-line arguments, overriding the defaults:
  !   arg 1: input CSV path
  !   arg 2: output CSV path
  !   arg 3: tol      (real,    e.g. 1.0e-3)
  !   arg 4: maxiter  (integer, e.g. 10)
  ! Any argument that is omitted keeps its default value.
  SUBROUTINE parse_cli(in_file, out_file, tolerance, itermax)
    CHARACTER(len=*), INTENT(inout) :: in_file, out_file
    REAL(wp),         INTENT(inout) :: tolerance
    INTEGER,          INTENT(inout) :: itermax

    INTEGER :: nargs, ios
    CHARACTER(len=1024) :: buf

    nargs = COMMAND_ARGUMENT_COUNT()
    IF (nargs >= 1) CALL GET_COMMAND_ARGUMENT(1, in_file)
    IF (nargs >= 2) CALL GET_COMMAND_ARGUMENT(2, out_file)
    IF (nargs >= 3) THEN
      CALL GET_COMMAND_ARGUMENT(3, buf)
      READ(buf, *, iostat=ios) tolerance
      IF (ios /= 0) CALL die('parse_cli: could not parse tol from "'//TRIM(buf)//'"')
    END IF
    IF (nargs >= 4) THEN
      CALL GET_COMMAND_ARGUMENT(4, buf)
      READ(buf, *, iostat=ios) itermax
      IF (ios /= 0) CALL die('parse_cli: could not parse maxiter from "'//TRIM(buf)//'"')
    END IF
  END SUBROUTINE parse_cli

  ! Count the data rows in a CSV file that has exactly one header line.
  SUBROUTINE count_data_rows(filename, n)
    CHARACTER(len=*), INTENT(in)  :: filename
    INTEGER,          INTENT(out) :: n

    INTEGER :: unit, ios
    CHARACTER(len=1) :: dummy

    OPEN(newunit=unit, file=filename, status='old', action='read')
    READ(unit,*)  ! skip header
    n = 0
    DO
      READ(unit,*,iostat=ios) dummy
      IF (ios /= 0) EXIT
      n = n + 1
    END DO
    CLOSE(unit)
  END SUBROUTINE count_data_rows

  SUBROUTINE read_fields(filename, nlev, rho, tk, qv, qc)
    CHARACTER(len=*), INTENT(in) :: filename
    INTEGER,          INTENT(in) :: nlev
    REAL(wp), DIMENSION(nlev), INTENT(out) :: rho, tk, qv, qc

    INTEGER :: unit, k

    OPEN(newunit=unit, file=filename, status='old', action='read')
    READ(unit,*)  ! skip header
    DO k = 1, nlev
      READ(unit,*) rho(k), tk(k), qv(k), qc(k)
    END DO
    CLOSE(unit)
  END SUBROUTINE read_fields

  SUBROUTINE write_fields(filename, nlev, rho, tk, qv, qc)
    CHARACTER(len=*), INTENT(in) :: filename
    INTEGER,          INTENT(in) :: nlev
    REAL(wp), DIMENSION(nlev), INTENT(in) :: rho, tk, qv, qc

    INTEGER :: unit, k

    OPEN(newunit=unit, file=filename, status='replace', action='write')
    WRITE(unit,'(A)') field_header
    DO k = 1, nlev
      ! Full float64 precision (~17 sig figs) so the harness can compare against
      ! GT4Py at rtol=1e-11 without losing precision to the CSV round-trip.
      WRITE(unit,'(*(ES24.16E3,:,","))') rho(k), tk(k), qv(k), qc(k)
    END DO
    CLOSE(unit)
  END SUBROUTINE write_fields

  SUBROUTINE die(msg)
    CHARACTER(len=*), INTENT(in) :: msg
    WRITE(*,'(A)') 'column_driver: FATAL: '//msg
    STOP 1
  END SUBROUTINE die

END PROGRAM column_driver
