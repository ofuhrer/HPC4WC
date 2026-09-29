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

  IMPLICIT NONE

  ! -------------------------------------------------------------------
  ! File names (edit these, or point them at your own data)
  ! -------------------------------------------------------------------
  CHARACTER(len=*), PARAMETER :: hhl_file    = '../example/hhl.csv'
  CHARACTER(len=*), PARAMETER :: fields_file = '../example/fields.csv'
  CHARACTER(len=*), PARAMETER :: output_file = '../example/output_fields.csv'

  ! Column order of fields.csv -- read and written in this exact order.
  CHARACTER(len=*), PARAMETER :: field_header = &
       'rho,pres,w,tk,qv,ssat,qc,qnc,qr,qnr,qi,qni,qs,qns,'// &
       'qg,qng,qh,qnh,nccn,ninpot,ninagi,ninact,qrsflux'

  ! -------------------------------------------------------------------
  ! Physics switches (hardcoded for now)
  ! -------------------------------------------------------------------
  INTEGER,  PARAMETER :: igscp          = 5       ! ICON scheme number (2mom w/ prognostic CCN+IN)
  INTEGER,  PARAMETER :: kstart         = 1        ! first level with active moist physics (1 = whole column)
  REAL(wp), PARAMETER :: dt             = 30.0_wp  ! time step [s]
  LOGICAL,  PARAMETER :: lsatad         = .TRUE.   ! saturation adjustment on/off
  INTEGER,  PARAMETER :: ithermo_water  = 1        ! 0: constant latent heat, /=0: temperature-dependent
  INTEGER,  PARAMETER :: ice_type       = 0        ! ice nucleation parameterization choice
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
  CALL read_hhl(hhl_file, hhl, nlev)

  ALLOCATE(dz(nlev), rho(nlev), pres(nlev), w(nlev), tk(nlev), qv(nlev), ssat(nlev))
  ALLOCATE(qc(nlev), qnc(nlev), qr(nlev), qnr(nlev), qi(nlev), qni(nlev))
  ALLOCATE(qs(nlev), qns(nlev), qg(nlev), qng(nlev), qh(nlev), qnh(nlev))
  ALLOCATE(nccn(nlev), ninpot(nlev), ninagi(nlev), ninact(nlev), qrsflux(nlev))

  CALL read_fields(fields_file, nlev, rho, pres, w, tk, qv, ssat, qc, qnc, qr, qnr, &
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

  CALL write_fields(output_file, nlev, rho, pres, w, tk, qv, ssat, qc, qnc, qr, qnr, &
       qi, qni, qs, qns, qg, qng, qh, qnh, nccn, ninpot, ninagi, ninact, qrsflux)

  WRITE(*,'(A)') 'column_driver: wrote '//output_file

CONTAINS

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
      WRITE(unit,'(*(ES16.8E3,:,","))') rho(k), pres(k), w(k), tk(k), qv(k), ssat(k), &
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
