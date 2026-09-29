! Opt-in, full-precision dumps of the column state at each process boundary.
!
! WHY THIS EXISTS
! ---------------
! The GT4Py port in ../ is validated by holding it against this Fortran scheme.
! Whole-column agreement alone cannot say *which* process a discrepancy came from,
! so the Python harness needs the intermediate state too. The alternative -- adding
! a temporary WRITE, running once, copying the numbers into a test file and
! reverting -- is what the first round of tests did, and it produced reference
! values that nobody can regenerate and that carry only the 9-12 digits somebody
! happened to paste. This module makes those same values a reproducible product of
! an ordinary run.
!
! CONTRACT
! --------
! Disabled unless `set_stage_dump_dir` is called with a non-empty path (the driver
! does that only when given its 5th command-line argument). When disabled every
! entry point returns immediately, so a normal `make run` executes exactly the
! instructions it did before this module existed.
!
! Values are written at ES24.16E3 -- 17 significant digits, which round-trips a
! float64 exactly, so the file never limits the precision of a comparison.
!
! This module writes output and nothing else. It must never be given a role in the
! physics: no state it holds may be read back by the scheme.

MODULE mo_stage_dump

  USE mo_kind, ONLY: wp

  IMPLICIT NONE

  PRIVATE

  PUBLIC :: set_stage_dump_dir, stage_dump_enabled, dump_stage, dump_coeffs

  INTEGER, PARAMETER :: max_path = 1024

  CHARACTER(len=max_path) :: dump_dir = ''

  ! Column set written at every stage. Deliberately narrower than the driver's 23:
  ! these are the fields in scope at *all* the call sites (ssat/ninagi/qrsflux are
  ! not, and are untouched by this configuration anyway). The Python side reads by
  ! column name via mcrph_common.csv_io.read_columns, so the set need not match the
  ! driver's -- only the names have to agree.
  CHARACTER(len=*), PARAMETER :: stage_header = &
       'rho,pres,w,tk,qv,qc,qnc,qr,qnr,qi,qni,qs,qns,'// &
       'qg,qng,qh,qnh,nccn,ninpot,ninact'

CONTAINS

  SUBROUTINE set_stage_dump_dir(path)
    CHARACTER(len=*), INTENT(in) :: path
    dump_dir = path
  END SUBROUTINE set_stage_dump_dir

  LOGICAL FUNCTION stage_dump_enabled()
    stage_dump_enabled = (LEN_TRIM(dump_dir) > 0)
  END FUNCTION stage_dump_enabled

  ! Write the column state to <dump_dir>/<name>.csv.
  !
  ! nccn/ninpot are OPTIONAL because the prognostic-aerosol arrays are not in scope
  ! at every boundary; absent ones are written as zeros rather than skipped, so the
  ! header stays identical across stages and the reader stays trivial.
  SUBROUTINE dump_stage(name, nlev, rho, pres, w, tk, qv, &
       &                cq, cn, rq, rn, iq, in_, sq, sn, gq, gn, hq, hn, &
       &                nccn, ninpot, ninact)

    CHARACTER(len=*), INTENT(in) :: name
    INTEGER,          INTENT(in) :: nlev
    REAL(wp), DIMENSION(nlev), INTENT(in) :: rho, pres, w, tk, qv
    REAL(wp), DIMENSION(nlev), INTENT(in) :: cq, cn, rq, rn, iq, in_
    REAL(wp), DIMENSION(nlev), INTENT(in) :: sq, sn, gq, gn, hq, hn
    REAL(wp), DIMENSION(nlev), INTENT(in), OPTIONAL :: nccn, ninpot, ninact

    REAL(wp), DIMENSION(nlev) :: ccn_l, inpot_l, inact_l
    INTEGER :: unit, k

    IF (.NOT. stage_dump_enabled()) RETURN

    ccn_l   = 0.0_wp
    inpot_l = 0.0_wp
    inact_l = 0.0_wp
    IF (PRESENT(nccn))   ccn_l   = nccn
    IF (PRESENT(ninpot)) inpot_l = ninpot
    IF (PRESENT(ninact)) inact_l = ninact

    OPEN(newunit=unit, file=TRIM(dump_dir)//'/'//TRIM(name)//'.csv', &
         status='replace', action='write')
    WRITE(unit,'(A)') stage_header
    DO k = 1, nlev
      WRITE(unit,'(*(ES24.16E3,:,","))') rho(k), pres(k), w(k), tk(k), qv(k), &
           cq(k), cn(k), rq(k), rn(k), iq(k), in_(k), sq(k), sn(k), &
           gq(k), gn(k), hq(k), hn(k), ccn_l(k), inpot_l(k), inact_l(k)
    END DO
    CLOSE(unit)

  END SUBROUTINE dump_stage

  ! Append one particle's derived coefficients to <dump_dir>/coeffs.csv.
  !
  ! These are pure functions of the static particle constants, computed once in
  ! init_2mom_scheme_once. The Fortran prints only init_2mom_sedi_vel's three (via
  ! the hardcoded `isprint`), and only to 7 digits, which is why the port's
  ! coefficient test used to assert at rtol=1e-6 against pasted values -- loose
  ! enough that a genuine 1e-7 error in a_f would have passed unnoticed.
  SUBROUTINE dump_coeffs(name, a_f, b_f, c_i, c_z, alfa_n, alfa_q, lambda)

    CHARACTER(len=*), INTENT(in) :: name
    REAL(wp),         INTENT(in) :: a_f, b_f, c_i, c_z, alfa_n, alfa_q, lambda

    INTEGER :: unit
    LOGICAL :: exists

    IF (.NOT. stage_dump_enabled()) RETURN

    INQUIRE(file=TRIM(dump_dir)//'/coeffs.csv', exist=exists)
    IF (exists) THEN
      OPEN(newunit=unit, file=TRIM(dump_dir)//'/coeffs.csv', &
           status='old', position='append', action='write')
    ELSE
      OPEN(newunit=unit, file=TRIM(dump_dir)//'/coeffs.csv', &
           status='new', action='write')
      WRITE(unit,'(A)') 'name,a_f,b_f,c_i,c_z,coeff_alfa_n,coeff_alfa_q,coeff_lambda'
    END IF

    WRITE(unit,'(A,7(",",ES24.16E3))') TRIM(name), a_f, b_f, c_i, c_z, &
         alfa_n, alfa_q, lambda
    CLOSE(unit)

  END SUBROUTINE dump_coeffs

END MODULE mo_stage_dump
