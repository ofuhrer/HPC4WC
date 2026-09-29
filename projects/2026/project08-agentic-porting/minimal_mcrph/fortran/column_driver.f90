! Top-level single-column driver.
!
! Reads a 1D atmospheric column from two CSV files, calls the two-moment
! microphysics scheme once, and writes the updated column back out.
!
!   example/hhl.csv    one column "hhl": nlev+1 half-level heights [m],
!                       index 1 = model top, index nlev+1 = surface
!                       (ICON convention: height decreases with index).
!   example/fields.csv one row of headers, then nlev data rows, columns:
!                       rho,pres,w,tk,qv,ssat,qc,qnc,qr,qnr,qi,qni,qs,qns,
!                       qg,qng,qh,qnh,nccn,ninpot,ninagi,ninact,qrsflux
!
! nlev is inferred from the number of data rows in fields.csv (and cross-
! checked against hhl.csv, which must have exactly one more row). The layer
! thickness dz is not read in -- it is computed from hhl: dz(k) = hhl(k) -
! hhl(k+1).
!
! The physics switches below are hardcoded for now; edit the PARAMETERs in
! the "physics switches" block to change them.

PROGRAM column_driver

  USE mo_kind,               ONLY: wp
  USE mo_2mom_mcrph_driver,  ONLY: two_moment_mcrph_init
  USE mo_nwp_gscp_interface, ONLY: nwp_microphysics
  USE mo_stage_dump,         ONLY: set_stage_dump_dir

  IMPLICIT NONE

  ! -------------------------------------------------------------------
  ! File names and time step. These are the defaults used by `make run`;
  ! all four can be overridden positionally on the command line (see
  ! parse_cli below), which is what lets the Python consistency harness
  ! drive this same binary over scenario columns of its own.
  ! -------------------------------------------------------------------
  CHARACTER(len=1024) :: fields_file = '../example/fields.csv'
  CHARACTER(len=1024) :: output_file = '../example/output_fields.csv'
  CHARACTER(len=1024) :: hhl_file    = '../example/hhl.csv'
  REAL(wp)            :: dt          = 30.0_wp  ! time step [s]
  ! Empty = no per-stage dumps, which is what `make run` does. See mo_stage_dump.f90.
  CHARACTER(len=1024) :: dump_dir    = ''

  ! Column order of fields.csv -- read and written in this exact order.
  CHARACTER(len=*), PARAMETER :: field_header = &
       'rho,pres,w,tk,qv,ssat,qc,qnc,qr,qnr,qi,qni,qs,qns,'// &
       'qg,qng,qh,qnh,nccn,ninpot,ninagi,ninact,qrsflux'

  ! -------------------------------------------------------------------
  ! Physics switches (hardcoded for now)
  ! -------------------------------------------------------------------
  INTEGER,  PARAMETER :: igscp          = 5       ! ICON scheme number (2mom w/ prognostic CCN+IN)
  INTEGER,  PARAMETER :: kstart         = 1        ! first level with active moist physics (1 = whole column)
  LOGICAL,  PARAMETER :: lsatad         = .TRUE.   ! saturation adjustment on/off
  INTEGER,  PARAMETER :: ithermo_water  = 1        ! 0: constant latent heat, /=0: temperature-dependent
  INTEGER,  PARAMETER :: ice_type       = 1        ! ice nucleation parameterization choice (>0 enables INP nucleation)
  LOGICAL,  PARAMETER :: luse_agi       = .FALSE.  ! use AgI (cloud seeding) tracer
  LOGICAL,  PARAMETER :: lexpl_supersat = .FALSE.  ! explicit supersaturation prediction on/off
  INTEGER,  PARAMETER :: msg_level      = 10       ! message/debug verbosity

  ! -------------------------------------------------------------------
  ! Column state
  ! -------------------------------------------------------------------
  INTEGER :: nlev

  REAL(wp), ALLOCATABLE :: hhl(:), dz(:)
  REAL(wp), ALLOCATABLE :: rho(:), pres(:), w(:), tk(:), qv(:), ssat(:)
  REAL(wp), ALLOCATABLE :: qc(:), qnc(:), qr(:), qnr(:), qi(:), qni(:)
  REAL(wp), ALLOCATABLE :: qs(:), qns(:), qg(:), qng(:), qh(:), qnh(:)
  REAL(wp), ALLOCATABLE :: nccn(:), ninpot(:), ninagi(:), ninact(:), qrsflux(:)

  REAL(wp), POINTER :: tke(:) => NULL()  ! no turbulence data available -> disassociated

  REAL(wp) :: prec_r, prec_i, prec_s, prec_g, prec_h, prec_gsp_rate

  ! Scratch outputs of the one-time scheme initialization; not otherwise used.
  REAL(wp) :: N_cn0, z0_nccn, z1e_nccn, N_in0, z0_nin, z1e_nin

  INTEGER :: k

  ! -------------------------------------------------------------------
  ! Read input
  ! -------------------------------------------------------------------
  CALL parse_cli(fields_file, output_file, hhl_file, dt, dump_dir)
  CALL set_stage_dump_dir(TRIM(dump_dir))

  CALL read_hhl(TRIM(hhl_file), hhl, nlev)

  ALLOCATE(dz(nlev), rho(nlev), pres(nlev), w(nlev), tk(nlev), qv(nlev), ssat(nlev))
  ALLOCATE(qc(nlev), qnc(nlev), qr(nlev), qnr(nlev), qi(nlev), qni(nlev))
  ALLOCATE(qs(nlev), qns(nlev), qg(nlev), qng(nlev), qh(nlev), qnh(nlev))
  ALLOCATE(nccn(nlev), ninpot(nlev), ninagi(nlev), ninact(nlev), qrsflux(nlev))

  CALL read_fields(TRIM(fields_file), nlev, rho, pres, w, tk, qv, ssat, qc, qnc, qr, qnr, &
       qi, qni, qs, qns, qg, qng, qh, qnh, nccn, ninpot, ninagi, ninact, qrsflux)

  ! Layer thickness from half-level heights (index increases downward, so
  ! hhl(k) > hhl(k+1), giving a positive thickness).
  DO k = 1, nlev
    dz(k) = hhl(k) - hhl(k+1)
  END DO

  WRITE(*,'(A,I0)') 'column_driver: nlev = ', nlev

  ! -------------------------------------------------------------------
  ! One-time scheme initialization (coefficient tables, CCN/IN config).
  ! Passing the N_cn0/z0_nccn/... outputs (even though unused here) is what
  ! makes two_moment_mcrph_init populate in_coeffs -- see mo_2mom_mcrph_driver.f90.
  ! This will generate/write the graupel and hail wet-growth lookup tables
  ! to disk on first run if they are not already present (one-time cost).
  ! -------------------------------------------------------------------
  CALL two_moment_mcrph_init(igscp=igscp, ice_type=ice_type,           &
       &                     N_cn0=N_cn0, z0_nccn=z0_nccn, z1e_nccn=z1e_nccn, &
       &                     N_in0=N_in0, z0_nin=z0_nin,   z1e_nin=z1e_nin,   &
       &                     msg_level=msg_level)

  prec_r = 0.0_wp
  prec_i = 0.0_wp
  prec_s = 0.0_wp
  prec_g = 0.0_wp
  prec_h = 0.0_wp

  ! -------------------------------------------------------------------
  ! Call the microphysics for one time step
  ! -------------------------------------------------------------------
  CALL nwp_microphysics(nlev=nlev, kstart=kstart, dt=dt, lsatad=lsatad,   &
       &                dz=dz, hhl=hhl, rho=rho, pres=pres, w=w, tke=tke, &
       &                tk=tk, qv=qv, ssat=ssat,                          &
       &                qc=qc, qnc=qnc, qr=qr, qnr=qnr,                   &
       &                qi=qi, qni=qni, qs=qs, qns=qns,                   &
       &                qg=qg, qng=qng, qh=qh, qnh=qnh,                   &
       &                nccn=nccn, ninpot=ninpot, ninagi=ninagi, ninact=ninact, &
       &                qrsflux=qrsflux,                                  &
       &                prec_r=prec_r, prec_i=prec_i, prec_s=prec_s,      &
       &                prec_g=prec_g, prec_h=prec_h,                     &
       &                prec_gsp_rate=prec_gsp_rate,                      &
       &                ithermo_water=ithermo_water, ice_type=ice_type,   &
       &                luse_agi=luse_agi, lexpl_supersat=lexpl_supersat, &
       &                msg_level=msg_level)

  ! -------------------------------------------------------------------
  ! Report scalar (non-column) outputs and write the updated column
  ! -------------------------------------------------------------------
  WRITE(*,'(A)')            'column_driver: surface precipitation rates [kg m-2 s-1]'
  WRITE(*,'(A,ES14.6)')     '  rain    prec_r        = ', prec_r
  WRITE(*,'(A,ES14.6)')     '  ice     prec_i        = ', prec_i
  WRITE(*,'(A,ES14.6)')     '  snow    prec_s        = ', prec_s
  WRITE(*,'(A,ES14.6)')     '  graupel prec_g        = ', prec_g
  WRITE(*,'(A,ES14.6)')     '  hail    prec_h        = ', prec_h
  WRITE(*,'(A,ES14.6)')     '  total   prec_gsp_rate = ', prec_gsp_rate

  CALL write_fields(TRIM(output_file), nlev, rho, pres, w, tk, qv, ssat, qc, qnc, qr, qnr, &
       qi, qni, qs, qns, qg, qng, qh, qnh, nccn, ninpot, ninagi, ninact, qrsflux)

  WRITE(*,'(A)') 'column_driver: wrote '//TRIM(output_file)

CONTAINS

  ! Parse optional positional command-line arguments, overriding the defaults:
  !   arg 1: input fields CSV path
  !   arg 2: output CSV path
  !   arg 3: hhl CSV path
  !   arg 4: dt (real, e.g. 30.0)
  !   arg 5: directory for per-stage state dumps (empty/omitted = no dumps)
  ! Any argument that is omitted keeps its default value, so a bare
  ! `./build/column_driver` (what `make run` does) still regenerates
  ! example/output_fields.csv exactly as before.
  SUBROUTINE parse_cli(in_file, out_file, half_level_file, timestep, stage_dir)
    CHARACTER(len=*), INTENT(inout) :: in_file, out_file, half_level_file, stage_dir
    REAL(wp),         INTENT(inout) :: timestep

    INTEGER :: nargs, ios
    CHARACTER(len=1024) :: buf

    nargs = COMMAND_ARGUMENT_COUNT()
    IF (nargs >= 1) CALL GET_COMMAND_ARGUMENT(1, in_file)
    IF (nargs >= 2) CALL GET_COMMAND_ARGUMENT(2, out_file)
    IF (nargs >= 3) CALL GET_COMMAND_ARGUMENT(3, half_level_file)
    IF (nargs >= 4) THEN
      CALL GET_COMMAND_ARGUMENT(4, buf)
      READ(buf, *, iostat=ios) timestep
      IF (ios /= 0) CALL die('parse_cli: could not parse dt from "'//TRIM(buf)//'"')
    END IF
    IF (nargs >= 5) CALL GET_COMMAND_ARGUMENT(5, stage_dir)
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

  SUBROUTINE read_hhl(filename, hhl, nlev)
    CHARACTER(len=*),       INTENT(in)    :: filename
    REAL(wp), ALLOCATABLE,  INTENT(out)   :: hhl(:)
    INTEGER,                INTENT(out)   :: nlev

    INTEGER :: unit, k, nlevp1

    CALL count_data_rows(filename, nlevp1)
    nlev = nlevp1 - 1
    IF (nlev < 1) CALL die('read_hhl: '//filename//' needs at least 2 rows (nlev+1 half levels)')

    ALLOCATE(hhl(nlevp1))
    OPEN(newunit=unit, file=filename, status='old', action='read')
    READ(unit,*)  ! skip header
    DO k = 1, nlevp1
      READ(unit,*) hhl(k)
    END DO
    CLOSE(unit)
  END SUBROUTINE read_hhl

  SUBROUTINE read_fields(filename, nlev, rho, pres, w, tk, qv, ssat, qc, qnc, qr, qnr, &
       qi, qni, qs, qns, qg, qng, qh, qnh, nccn, ninpot, ninagi, ninact, qrsflux)
    CHARACTER(len=*), INTENT(in) :: filename
    INTEGER,          INTENT(in) :: nlev
    REAL(wp), DIMENSION(nlev), INTENT(out) :: rho, pres, w, tk, qv, ssat
    REAL(wp), DIMENSION(nlev), INTENT(out) :: qc, qnc, qr, qnr, qi, qni
    REAL(wp), DIMENSION(nlev), INTENT(out) :: qs, qns, qg, qng, qh, qnh
    REAL(wp), DIMENSION(nlev), INTENT(out) :: nccn, ninpot, ninagi, ninact, qrsflux

    INTEGER :: unit, k, nrows

    CALL count_data_rows(filename, nrows)
    IF (nrows /= nlev) THEN
      WRITE(*,'(A,I0,A,I0)') 'read_fields: row count mismatch: hhl.csv implies nlev=', nlev, &
           ' but fields.csv has ', nrows
      CALL die('read_fields: '//filename//' row count does not match hhl.csv')
    END IF

    OPEN(newunit=unit, file=filename, status='old', action='read')
    READ(unit,*)  ! skip header
    DO k = 1, nlev
      READ(unit,*) rho(k), pres(k), w(k), tk(k), qv(k), ssat(k), qc(k), qnc(k), qr(k), qnr(k), &
           qi(k), qni(k), qs(k), qns(k), qg(k), qng(k), qh(k), qnh(k), &
           nccn(k), ninpot(k), ninagi(k), ninact(k), qrsflux(k)
    END DO
    CLOSE(unit)
  END SUBROUTINE read_fields

  SUBROUTINE write_fields(filename, nlev, rho, pres, w, tk, qv, ssat, qc, qnc, qr, qnr, &
       qi, qni, qs, qns, qg, qng, qh, qnh, nccn, ninpot, ninagi, ninact, qrsflux)
    CHARACTER(len=*), INTENT(in) :: filename
    INTEGER,          INTENT(in) :: nlev
    REAL(wp), DIMENSION(nlev), INTENT(in) :: rho, pres, w, tk, qv, ssat
    REAL(wp), DIMENSION(nlev), INTENT(in) :: qc, qnc, qr, qnr, qi, qni
    REAL(wp), DIMENSION(nlev), INTENT(in) :: qs, qns, qg, qng, qh, qnh
    REAL(wp), DIMENSION(nlev), INTENT(in) :: nccn, ninpot, ninagi, ninact, qrsflux

    INTEGER :: unit, k

    OPEN(newunit=unit, file=filename, status='replace', action='write')
    WRITE(unit,'(A)') field_header
    DO k = 1, nlev
      ! ES24.16E3 = 17 significant digits, which round-trips a float64 exactly.
      ! The previous ES16.8E3 (9 digits) put a ~1e-9 relative floor under every
      ! comparison against this file, well above the ~1e-13 the port actually
      ! achieves -- the reference file must not be the limiting precision.
      WRITE(unit,'(*(ES24.16E3,:,","))') rho(k), pres(k), w(k), tk(k), qv(k), ssat(k), &
           qc(k), qnc(k), qr(k), qnr(k), qi(k), qni(k), qs(k), qns(k), &
           qg(k), qng(k), qh(k), qnh(k), nccn(k), ninpot(k), ninagi(k), ninact(k), qrsflux(k)
    END DO
    CLOSE(unit)
  END SUBROUTINE write_fields

  SUBROUTINE die(msg)
    CHARACTER(len=*), INTENT(in) :: msg
    WRITE(*,'(A)') 'column_driver: FATAL: '//msg
    STOP 1
  END SUBROUTINE die

END PROGRAM column_driver
