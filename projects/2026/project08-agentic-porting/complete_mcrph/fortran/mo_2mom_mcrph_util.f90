! ICON
!
! ---------------------------------------------------------------
! Copyright (C) 2004-2024, DWD, MPI-M, DKRZ, KIT, ETH, MeteoSwiss
! Contact information: icon-model.org
!
! See AUTHORS.TXT for a list of authors
! See LICENSES/ for license information
! SPDX-License-Identifier: BSD-3-Clause
! ---------------------------------------------------------------

! Two-moment bulk microphysics after Seifert, Beheng and Blahak
!
! Description:
! Provides various subroutines and functions for the two-moment microphysics

!NEC$ options "-finline-max-depth=3 -finline-max-function-size=1000"

MODULE mo_2mom_mcrph_util

  USE mo_kind,               ONLY: wp,sp,dp
  USE mo_exception,          ONLY: finish, message, txt => message_text
  USE mo_physical_constants, ONLY: &
       & rhoh2o,           & ! density of liquid water
       & T_3   => tmelt      ! melting temperature of ice
  USE netcdf,                ONLY: nf90_open, nf90_noerr, NF90_NOWRITE, NF90_GLOBAL, &
       &                           nf90_inq_varid, nf90_inq_dimid, nf90_inquire_dimension, &
       &                           nf90_get_var, nf90_close, nf90_get_att, nf90_strerror
  USE mo_2mom_mcrph_types,   ONLY: particle, lookupt_1D, lookupt_4D
  USE mo_2mom_mcrph_dmin_wetgrowth, ONLY: dmin_wetgrowth_lookupcreate, &
       &                           write_dmin_wetgrowth_table, get_filebase_dmin_lut_file
  
  IMPLICIT NONE

  PRIVATE

  PUBLIC :: init_dmin_wetgrowth

  PUBLIC :: &
       & rat2do3,                    & ! main
       & dyn_visc_sutherland,        & ! main
       & Dv_Rasmussen,               & ! main
       & ka_Rasmussen,               & ! main
       & lh_evap_RH87,               & ! main
       & lh_melt_RH87,               & ! main
       & gamlookuptable,             & ! main
       & nlookup, nlookuphr_dummy,   & ! main
       & incgfct_lower_lookupcreate, & ! main
       & incgfct_lower_lookup,       & ! main
       & incgfct_upper_lookup,       & ! main
       & init_dmin_wg_gr_ltab_equi,  & ! driver
       & lookupt_4D,                 & !
       & dmin_wg_gr_ltab_equi,       & ! main
       & dmin_wetgrowth_fit_check,   &
       & dmin_wetgrowth_fun,         &
       & luse_dmin_wetgrowth_table,  &
       & lprintout_comp_table_fit,   &
       & set_qnc,                    &
       & set_qni,                    &
       & set_qnr,                    &
       & set_qns,                    &
       & set_qng,                    &
       & set_qnh_Dmean,              &
       & set_qnh_expPSD_N0const,     &
       & e_stick,                     &
       & init_estick_ltab_equi,      &
       & estick_ltab_equi


  CHARACTER(len=*), PARAMETER :: modname = 'mo_2mom_mcrph_util'

  ! In place of ICON's mo_math_constants: pi is the only constant this
  ! module needs from it.
  REAL(wp), PARAMETER :: pi = 3.14159265358979323846_wp

  ! Use look-up table for dmin_wetgrowth?
  ! If .true., a lookup table file (netcdf or ascii format) is read from input directory if present
  !  or produced on the fly. Both for graupel and hail.
  ! If .false., an internal 4D-fit for a specific default graupel kind is used. hail is not implemented, so
  !  this option is not usable any more. LEAVE IT .true.!
  LOGICAL, PARAMETER :: luse_dmin_wetgrowth_table  = .true.
  ! Do a test printout to stdout of the 4D-fit compared to the tabulated values?
  ! If .true., the lookup table file to compare with is read from input directory
  !  regardless of use_dmin_wetgrowth_table.
  LOGICAL, PARAMETER :: lprintout_comp_table_fit   = .false.

  ! Variables for wet growth diameter lookup tables:
  REAL(wp), DIMENSION(:,:,:,:), ALLOCATABLE :: dmin_wg_g
  REAL(wp), DIMENSION(:),       ALLOCATABLE :: pvec_wg_g, Tvec_wg_g, qwvec_wg_g, qivec_wg_g
  INTEGER                                   :: anzp_wg, anzT_wg, anzi_wg, anzw_wg

  ! Structure for holding the data of a lookup table for the incomplete gamma function:
  INTEGER, PARAMETER                     :: nlookup   = 2000    ! Internal number of bins (low res part)
  INTEGER, PARAMETER                     :: nlookuphr = 10000   ! Internal number of bins (high res part)

  ! dummy of internal number of bins (high res part) in case the high resolution part is not really needed:
  INTEGER, PARAMETER                     :: nlookuphr_dummy = 10

  ! Type to hold the lookup table for the incomplete gamma functions.
  ! The table is divided into a low resolution part, which spans the
  ! whole range of x-values up to the 99.5 % x-value, and a high resolution part for the
  ! smallest 1 % of these x-values, where the incomplete gamma function may increase
  ! very rapidly and nonlinearily, depending on paramter a.
  ! For some applications (e.g., Newtons Method in future subroutine
  ! graupel_hail_conv_wetgrowth_Dg_gamlook() ), this rapid change requires a much higher
  ! accuracy of the table lookup as compared to be achievable with the low resolution table.

  TYPE gamlookuptable
    LOGICAL :: is_initialized = .FALSE.
    ! Number of bins in the tables:
    INTEGER                         :: n        ! Internal number of bins (low res part)
    INTEGER                         :: nhr      ! Internal number of bins (high res part)
    REAL(wp)                        :: a        ! a-parameter
    REAL(wp), DIMENSION(:), POINTER :: x => NULL()   ! vector of x-parameters (limit of integration) -
                                                ! always starts at 0 and has equidistant dx (low resolution part)
    REAL(wp), DIMENSION(:), POINTER :: xhr => NULL() ! vector of x-parameters (limit of integration) -
                                                ! always starts at 0 and has equidistant dxhr (high resolution part)
    REAL(wp)                        :: dx       ! dx   (low resolution part)
    REAL(wp)                        :: dxhr     ! dxhr (high resolution part)
    REAL(wp)                        :: odx      ! one over dx
    REAL(wp)                        :: odxhr    ! one over dxhr
    REAL(wp), DIMENSION(:), POINTER :: igf => NULL()   ! value of the inc. gamma function at (a,x) (low res)
    REAL(wp), DIMENSION(:), POINTER :: igfhr => NULL() ! value of the inc. gamma function at (a,x) (high res)
  END TYPE gamlookuptable

CONTAINS

  !*******************************************************************************
  ! Special functions and utility functions like look-up tables
  !*******************************************************************************


  !*******************************************************************************
  !       Incomplete Gamma function
  !*******************************************************************************

  !*******************************************************************************
  ! 1) some helper functions:

  SUBROUTINE gamma_help_cf(gammcf,a,x,gln)

    REAL(dp), INTENT(in)  :: a, x
    REAL(dp), INTENT(out) :: gammcf, gln

    INTEGER,  PARAMETER   :: maxiter = 100
    REAL(dp), PARAMETER   :: eps = 3.d-7, fpmin = 1.d-30
    INTEGER               :: i
    REAL(dp)              :: an, b, c, d, del, h

    gln = LOG(GAMMA(a))
    b   = x + 1.0_dp - a
    c   = 1.0_dp / fpmin
    d   = 1.0_dp / b
    h   = d
    DO i = 1, maxiter
      an = -i*(i-a)
      b  = b + 2.0_dp
      d  = an*d + b
      IF (ABS(d) < fpmin) d = fpmin
      c  = b + an/c
      IF (ABS(c) < fpmin) c = fpmin
      d  = 1.0_dp / d
      del= d * c
      h  = h * del
      IF (ABS(del-1.0_dp) < EPS) EXIT
    END DO

    IF (ABS(del-1.0_dp) >= eps) THEN
      WRITE (txt,*) 'ERROR in GAMMA_HELP_CF: a too large, maxiter too small'
      CALL message(modname,TRIM(txt))
      gammcf = 0.0_dp
      CALL finish(TRIM(modname),'Error in gamma_help_cf')
    END IF

    gammcf = EXP(-x + a*LOG(x) - gln) * h

    RETURN
  END SUBROUTINE gamma_help_cf

  SUBROUTINE gamma_help_ser(gamser,a,x,gln)

    REAL(dp), INTENT(in)  :: a, x
    REAL(dp), INTENT(out) :: gamser, gln

    INTEGER,  PARAMETER   :: maxiter = 100
    REAL(dp), PARAMETER   :: eps = 3.d-7
    INTEGER               :: n
    REAL(dp)              :: ap,del,sum

    gln = LOG(GAMMA(a))
    IF (x <= 0.0_dp) THEN
      IF (x < 0.0_dp) THEN
        WRITE (txt,*) 'ERROR in GAMMA_HELP_SER: x < 0'
        CALL message(modname,TRIM(txt))
        CALL finish(TRIM(modname),'Error in gamma_help_ser')
      END IF
      
      gamser = 0.0_dp
    
    ELSE

      ap  = a
      sum = 1.0_dp/a
      del = sum
      DO n = 1, maxiter
        ap  = ap + 1.0_dp
        del = del*x/ap
        sum = sum + del
        IF (ABS(del) < ABS(sum)*eps) EXIT
      END DO

      IF (ABS(del) >= ABS(sum)*eps) THEN
        WRITE (txt,*) 'ERROR in GAMMA_HELP_SER: a too large, maxiter too small' ;
        CALL message(modname,TRIM(txt))
        gamser = 0.0_dp
        CALL finish(TRIM(modname),'Error in gamma_help_ser')
      END IF

      gamser = sum * EXP(-x + a*LOG(x) - gln)

    END IF

    RETURN
  END SUBROUTINE gamma_help_ser

  REAL(dp) FUNCTION gamma_p(a,x,gln)

    REAL(dp), INTENT(in)  :: a, x
    REAL(dp), INTENT(out) :: gln
    REAL(dp)              :: gammcf, gamser

    IF (x < 0.0_dp .OR. a <= 0.0_dp) THEN
      WRITE (txt,*) 'ERROR in GAMMA_P: bad arguments'
      CALL message(modname,TRIM(txt))
      gamma_p = 0.0d0
      CALL finish(TRIM(modname),'Error in gamma_p')
    END IF
    
    IF (x < a+1.0_dp) THEN
      CALL gamma_help_ser(gamser,a,x,gln)
      gamma_p = gamser
    ELSE
      CALL gamma_help_cf(gammcf,a,x,gln)
      gamma_p = 1.0_dp - gammcf
    ENDIF
    
    RETURN
  END FUNCTION gamma_p

  REAL(dp) FUNCTION gamma_q(a,x,gln)

    REAL(dp), INTENT(in)  :: a, x
    REAL(dp), INTENT(out) :: gln
    REAL(dp)              :: gammcf, gamser

    IF (x < 0.0_dp .OR. a <= 0.0_dp) THEN
      WRITE (txt,*) 'ERROR in GAMMA_Q: bad arguments'
      CALL message(modname,TRIM(txt))
      gamma_q = 0.0_dp
      CALL finish(TRIM(modname),'Error in gamma_q')
    END IF

    IF (x < a+1.0_dp) THEN
      CALL gamma_help_ser(gamser,a,x,gln)
      gamma_q = 1.0_dp - gamser
    ELSE
      CALL gamma_help_cf(gammcf,a,x,gln)
      gamma_q = gammcf
    ENDIF
    
    RETURN
  END FUNCTION gamma_q

  ! End helper functions
  !*******************************************************************************

  !*******************************************************************************
  ! Upper incomplete gamma function
  !              int(x)(oo) exp(-t) t^(a-1) dt
  !*******************************************************************************

  REAL(dp) FUNCTION incgfct_upper(a,x)

    REAL(dp), INTENT(in) :: a, x
    REAL(dp) :: gam, gln

    gam = gamma_q(a,x,gln)
    incgfct_upper = EXP(gln) * gam

  END FUNCTION incgfct_upper

  !*******************************************************************************
  ! Lower incomplete gamma function
  !              int(0)(x) exp(-t) t^(a-1) dt
  !*******************************************************************************

  REAL(dp) FUNCTION incgfct_lower(a,x)
    
    REAL(dp), INTENT(in) :: a, x
    REAL(dp) :: gam, gln

    gam = gamma_p(a,x,gln)
    incgfct_lower = EXP(gln) * gam

  END FUNCTION incgfct_lower

  !*******************************************************************************
  ! Incomplete gamma function
  !              int(x1)(x2) exp(-t) t^(a-1) dt
  !*******************************************************************************

  REAL(dp) FUNCTION incgfct(a,x1,x2)

    REAL(dp), INTENT(in) :: a, x1, x2

    incgfct = incgfct_lower(a,x2) - incgfct_lower(a,x1)

  END FUNCTION incgfct

  !*******************************************************************************
  ! Create Lookup-table vectors for the lower incomplete gamma function,
  !              int(0)(x) exp(-t) t^(a-1) dt
  ! as function of x at constant a.
  ! The table runs from x=0 to the 99.5 % - value of the normalized
  ! incomplete gamma function. This 99.5 % - value has been fitted
  ! with high accuracy as function of a in the range a in [0;20], but can
  ! safely be applied also to higher values of a. (Fit created with the
  ! matlab-program "gamma_unvoll_lower_lookup.m" by Ulrich Blahak, 2008/11/13).
  !
  ! The last value in the table corresponds to x = infinity, so that
  ! during the reconstruction of incgfct-values from the table,
  ! the x-value can safely be truncated at the maximum table x-value.
  !
  !*******************************************************************************

  SUBROUTINE incgfct_lower_lookupcreate(a,ltable,nl,nlhr)
    REAL(dp), INTENT(in) :: a  ! value of a
    TYPE(gamlookuptable), INTENT(inout) :: ltable
    INTEGER, INTENT(in) :: nl, nlhr
    INTEGER :: i, err
    REAL(dp), PARAMETER ::   &
         c1 =  36.629433904824623d0, &
         c2 = -0.119475603955226d0,  &
         c3 =  0.339332937820052d0,  &
         c4 =  1.156369000458310d0

    IF (.NOT. ltable%is_initialized) THEN

      ! Store parameters in the structure ltable:
      ltable%a = a
      ltable%n = nl
      ltable%nhr = nlhr

      ! Allocate Memory for the table vectors:
      NULLIFY(ltable%x)
      NULLIFY(ltable%xhr)
      NULLIFY(ltable%igf)
      NULLIFY(ltable%igfhr)

      ALLOCATE(ltable%x(nl), STAT=err)
      IF (err /= 0) THEN
        WRITE (txt,*) 'INCGFCT_LOWER_LOOKUPCREATE: Allocation error x' ; CALL message(modname,TRIM(txt))
        CALL finish(TRIM(modname),'Error in incgfct_lower_lookupcreate!')
      END IF
      ALLOCATE(ltable%xhr(nlhr), STAT=err)
      IF (err /= 0) THEN
        WRITE (txt,*) 'INCGFCT_LOWER_LOOKUPCREATE: Allocation error xhr' ; CALL message(modname,TRIM(txt))
        CALL finish(TRIM(modname),'Error in incgfct_lower_lookupcreate!')
      END IF
      ALLOCATE(ltable%igf(nl), STAT=err)
      IF (err /= 0) THEN
        WRITE (txt,*) 'INCGFCT_LOWER_LOOKUPCREATE: Allocation error igf' ; CALL message(modname,TRIM(txt))
        CALL finish(TRIM(modname),'Error in incgfct_lower_lookupcreate!')
      END IF
      ALLOCATE(ltable%igfhr(nlhr), STAT=err)
      IF (err /= 0) THEN
        WRITE (txt,*) 'INCGFCT_LOWER_LOOKUPCREATE: Allocation error igfhr' ; CALL message(modname,TRIM(txt))
        CALL finish(TRIM(modname),'Error in incgfct_lower_lookupcreate!')
      END IF

      !==================================================================
      ! low resolution part of the table:
      !==================================================================

      ! maximum x-value of the lookup table (99.5-%-value):
      ltable%x(ltable%n-1) = c1 * ( 1.0d0 - EXP(c2*a**c3) ) + c4*a

      ! create lookup table vectors:
      ltable%dx = ltable%x(ltable%n-1) / (ltable%n-2.0d0)
      ltable%odx = 1.0d0 / ltable%dx
      ! Diese Schleife vektorisiert nicht wg. incgfct_lower():
      DO i = 1, ltable%n - 1
        ltable%x(i) = (i-1) * ltable%dx
        ltable%igf(i) = incgfct_lower(a,ltable%x(i))
      END DO

      ! The last value is for x = infinity:
      ltable%x(ltable%n) = (ltable%n-1) * ltable%dx
!      ltable%igf(ltable%n) = gfct_lanc(a)
      ltable%igf(ltable%n) = GAMMA(a)

      !==================================================================
      ! high resolution part of the table (lowest 2 % of the X-values):
      !==================================================================

      ! create lookup table vectors:
      ltable%dxhr = ltable%x(NINT(0.01*(ltable%n-1))) / (ltable%nhr-1.0d0)
      ltable%odxhr = 1.0d0 / ltable%dxhr
      ! Diese Schleife vektorisiert nicht wg. incgfct_lower():
      DO i = 1, ltable%nhr
        ltable%xhr(i) = (i-1) * ltable%dxhr
        ltable%igfhr(i) = incgfct_lower(a,ltable%xhr(i))
      END DO

      ltable%is_initialized = .TRUE.

    END IF

    !$ACC ENTER DATA COPYIN(ltable, ltable%x, ltable%xhr, ltable%igf, ltable%igfhr)

    RETURN
  END SUBROUTINE incgfct_lower_lookupcreate

  !*******************************************************************************
  ! Retrieve values from a lookup table of the lower incomplete gamma function,
  ! as function of x at a constant a, for which the lookup table has been
  ! created.
  !
  ! The last value in the table has to correspond to x = infinity, so that
  ! during the reconstruction of incgfct-values from the table,
  ! the x-value can safely be truncated at the maximum table x-value:
  !
  ! ltable%igf( ltable%x(ltable%n),...) = gfct(a)
  !
  ! Profiling with ifort on a Linux-PC shows, that table lookup for the
  ! incompl. gamma-Funktion is faster by a factor of about 15 compared
  ! to the original function without optimization (-O0). Using optimization
  ! could change this ratio (we encoutered up to 300 depending on function inlining).
  !
  ! Concerning the accuracy, comparisons show that the results of table lookup
  ! are accurate to within better than 0.1 % or even much less, except for
  ! very small values of X, for which the absolute values are however very
  ! close to 0. For X -> infinity (X > 99.5 % - value), accuracy may be
  ! somewhat reduced up to about 0.5 % ,
  ! because the table is truncated at the 99.5 % value (second-last value)
  ! and the last value is set to the ordinary gamma function.
  !
  ! This function only uses the low resolution part of the table!
  !*******************************************************************************

  REAL(dp) FUNCTION incgfct_lower_lookup(x, ltable)

    !$ACC ROUTINE SEQ

    REAL(dp), INTENT(in) :: x  ! value of x for table lookup
    TYPE(gamlookuptable), INTENT(in) :: ltable
    INTEGER :: iu, io
    REAL(dp) :: xt

    ! Trunkcate x to the range of the table:
    xt = MAX(MIN(x, ltable%x(ltable%n)), 0.0d0)

    ! calculate indices of the neighbouring regular x-values
    ! in the table:
    iu = MIN(FLOOR(xt * ltable%odx) + 1, ltable%n-1)
    io = iu + 1

    ! interpolate linearily and subtract from the ordinary gamma function to get the upper
    ! incomplete gamma function:
    incgfct_lower_lookup = ltable%igf(iu) + &
         (ltable%igf(io) - ltable%igf(iu)) * ltable%odx * (xt-ltable%x(iu))

  END FUNCTION incgfct_lower_lookup

  ! Statt linearer Interpolation wird eine quadratische Interpolation gemacht
  ! und hierfuer am "kleinen" Ende der lookup table der 50-fach hoeher aufgeloeste
  ! high-resolution Ast benutzt.
  ! (an benachbarte 3 Punkte eine Parabel interpolieren und interpolierten Wert von der Parabel nehmen --
  !  weil es jeweils 2 moegliche 3-Punkte-Nachbarschaften gibt, wird aus Stetigkeitsgruenden der Mittelwert
  ! von beiden genommen):
  REAL(dp) FUNCTION incgfct_lower_lookup_parabolic(x, ltable)
    REAL(dp), INTENT(in)     :: x       ! value of x for table lookup
    TYPE(gamlookuptable), INTENT(in) :: ltable

    INTEGER :: iu, im, io
    REAL(dp) :: xt, f12, f23, f123, yn1, yn2

    ! If x is within the high-resolution part of the table:
    IF (x <= ltable%xhr(ltable%nhr)) THEN

      ! Truncate x to the range of the table:
      xt = MAX(x, 0.0d0)

      ! Neighbourhood 1:
      ! calculate indices of the neighbouring regular x-values
      ! in the table:
      im = MIN(MAX(FLOOR(xt * ltable%odxhr) + 1, 2),  ltable%nhr-1)
      iu = im - 1
      io = im + 1

      ! interpolate parabolicly:
      ! coefficients of the interpolating parabola wrsp. to Newton's polynome form
      ! with Newton's tableau ("divided differences"):
      f12  = (ltable%igfhr(im) - ltable%igfhr(iu)) * ltable%odxhr
      f23  = (ltable%igfhr(io) - ltable%igfhr(im)) * ltable%odxhr
      f123 = 0.5d0 * (f23 - f12) * ltable%odxhr
      ! Horner's scheme to calculate the interpolated value from the interpolating parabola:
      yn1 = f123 * (xt - ltable%xhr(im)) + f12
      yn1 = yn1  * (xt - ltable%xhr(iu)) + ltable%igfhr(iu)

      IF (im < ltable%nhr - 1) THEN
        ! Neighbourhood 2:
        ! calculate indices of the neighbouring regular x-values
        ! in the table:

        im = MIN(MAX(CEILING(xt * ltable%odxhr) + 1, 2),  ltable%nhr-1)
        iu = im - 1
        io = im + 1

        ! interpolate parabolicly:
        ! coefficients of the interpolating parabola wrsp. to Newton's polynome form
        ! with Newton's tableau ("divided differences"):
        f12  = (ltable%igfhr(im) - ltable%igfhr(iu)) * ltable%odxhr
        f23  = (ltable%igfhr(io) - ltable%igfhr(im)) * ltable%odxhr
        f123 = 0.5d0 * (f23 - f12) * ltable%odxhr
        ! Horner's scheme to calculate the interpolated value from the interpolating parabola:
        yn2 = f123 * (xt - ltable%xhr(im)) + f12
        yn2 = yn2  * (xt - ltable%xhr(iu)) + ltable%igfhr(iu)

        incgfct_lower_lookup_parabolic = 0.5d0 * ( yn1 + yn2 )

      ELSE
        incgfct_lower_lookup_parabolic = yn1
      END IF

    ELSE

      ! Truncate x to the range of the table:
      xt = MAX(MIN(x, ltable%x(ltable%n)), 0.0d0)

      ! Neighbourhood 1:
      ! calculate indices of the neighbouring regular x-values
      ! in the table:
      im = MIN(MAX(FLOOR(xt * ltable%odx) + 1, 2),  ltable%n-1)
      iu = im - 1
      io = im + 1

      ! interpolate parabolicly:
      ! coefficients of the interpolating parabola wrsp. to Newton's polynome form
      ! with Newton's tableau ("divided differences"):
      f12  = (ltable%igf(im) - ltable%igf(iu)) * ltable%odx
      f23  = (ltable%igf(io) - ltable%igf(im)) * ltable%odx
      f123 = 0.5d0 * (f23 - f12) * ltable%odx
      ! Horner's scheme to calculate the interpolated value from the interpolating parabola:
      yn1 = f123 * (xt - ltable%x(im)) + f12
      yn1 = yn1  * (xt - ltable%x(iu)) + ltable%igf(iu)

      IF (im < ltable%n - 2) THEN
        ! Neighbourhood 2:
        ! calculate indices of the neighbouring regular x-values
        ! in the table:

        im = MIN(MAX(CEILING(xt * ltable%odx) + 1, 2),  ltable%n-2)
        iu = im - 1
        io = im + 1

        ! interpolate parabolicly:
        ! coefficients of the interpolating parabola wrsp. to Newton's polynome form
        ! with Newton's tableau ("divided differences"):
        f12  = (ltable%igf(im) - ltable%igf(iu)) * ltable%odx
        f23  = (ltable%igf(io) - ltable%igf(im)) * ltable%odx
        f123 = 0.5d0 * (f23 - f12) * ltable%odx
        ! Horner's scheme to calculate the interpolated value from the interpolating parabola:
        yn2 = f123 * (xt - ltable%x(im)) + f12
        yn2 = yn2  * (xt - ltable%x(iu)) + ltable%igf(iu)

        incgfct_lower_lookup_parabolic = 0.5d0 * ( yn1 + yn2 )

      ELSE
        incgfct_lower_lookup_parabolic = yn1
      END IF

    END IF

    incgfct_lower_lookup_parabolic = MAX(incgfct_lower_lookup_parabolic, 0.0d0)

  END FUNCTION incgfct_lower_lookup_parabolic

  !*******************************************************************************
  !
  ! Retrieve values of the upper incomplete gamma function
  ! from a lookup table of the lower incomplete gamma function,
  ! as function of x at a constant a, for which the lookup table has been
  ! created.
  !
  ! The last value in the table has to correspond to x = infinity
  ! (the ordinary gamma function of a), so that
  ! during the reconstruction of incgfct-values from the table,
  ! the x-value can safely be truncated at the maximum table x-value:
  !
  ! ltable%igf( ltable%x(ltable%n),...) = gfct(a)
  !
  ! This function only uses the low resolution part of the table!
  !
  !*******************************************************************************

  REAL(dp) FUNCTION incgfct_upper_lookup(x, ltable)

    !$ACC ROUTINE SEQ

    REAL(dp), INTENT(in)     :: x    ! value of x for table lookup
    TYPE(gamlookuptable), INTENT(in) :: ltable

    INTEGER :: iu, io
    REAL(dp) :: xt

    ! Trunkcate x to the range of the table:
    xt = MAX(MIN(x, ltable%x(ltable%n)), 0.0d0)

    ! calculate indices of the neighbouring regular x-values
    ! in the table:
    iu = MIN(FLOOR(xt * ltable%odx) + 1, ltable%n-1)
    io = iu + 1

    ! interpolate lower inc. gamma function linearily and subtract from
    ! the ordinary gamma function to get the upper
    ! incomplete gamma function:
    incgfct_upper_lookup = ltable%igf(ltable%n) - ltable%igf(iu) -  &
         (ltable%igf(io) - ltable%igf(iu)) * ltable%odx * (xt-ltable%x(iu))

    ! Aufgrund von Rundungsfehlern (Differenz von 2 fast gleichen Zahlen) kann es beim table lookup passieren,
    ! dass incgfct_upper_lookup(x, ltable) kleiner 0 wird, wenn eigentlich nahezu 0.0 herauskommen muesste.
    ! Dies kommt vor allem dann vor, wenn x sehr gross ist.
    ! Deswegen Begrenzung:

    incgfct_upper_lookup = MAX(incgfct_upper_lookup, 0.0d0)

    RETURN
  END FUNCTION incgfct_upper_lookup

  ! Statt linearer Interpolation wird eine quadratische Interpolation gemacht
  ! und hierfuer am "kleinen" Ende der lookup table der 50-fach hoeher aufgeloeste
  ! high-resolution Ast benutzt.
  ! (an benachbarte 3 Punkte eine Parabel interpolieren und interpolierten Wert von der Parabel nehmen --
  !  weil es jeweils 2 moegliche 3-Punkte-Nachbarschaften gibt, wird aus Stetigkeitsgruenden der Mittelwert
  ! von beiden genommen):
  REAL(dp) FUNCTION incgfct_upper_lookup_parabolic(x, ltable)
    REAL(dp), INTENT(in)     :: x  ! value of x for table lookup
    TYPE(gamlookuptable), INTENT(in) :: ltable

    INTEGER :: iu, im, io
    REAL(dp) :: xt, f12, f23, f123, yn1, yn2

    ! If x is within the high-resolution part of the table:
    IF (x <= ltable%xhr(ltable%nhr)) THEN

      ! Truncate x to the range of the table:
      xt = MAX(x, 0.0d0)

      ! Neighbourhood 1:
      ! calculate indices of the neighbouring regular x-values
      ! in the table:
      im = MIN(MAX(FLOOR(xt * ltable%odxhr) + 1, 2),  ltable%nhr-1)
      iu = im - 1
      io = im + 1

      ! interpolate parabolicly (lower incomplete gamma function --
      ! will be converted to upper function later):
      ! coefficients of the interpolating parabola wrsp. to Newton's polynome form
      ! with Newton's tableau ("divided differences"):
      f12  = (ltable%igfhr(im) - ltable%igfhr(iu)) * ltable%odxhr
      f23  = (ltable%igfhr(io) - ltable%igfhr(im)) * ltable%odxhr
      f123 = 0.5d0 * (f23 - f12) * ltable%odxhr
      ! Horner's scheme to calculate the interpolated value from the interpolating parabola:
      yn1 = f123 * (xt - ltable%xhr(im)) + f12
      yn1 = yn1  * (xt - ltable%xhr(iu)) + ltable%igfhr(iu)

      IF (im < ltable%nhr - 1) THEN
        ! Neighbourhood 2:
        ! calculate indices of the neighbouring regular x-values
        ! in the table:

        im = MIN(MAX(CEILING(xt * ltable%odxhr) + 1, 2),  ltable%nhr-1)
        iu = im - 1
        io = im + 1

        ! interpolate parabolicly:
        ! coefficients of the interpolating parabola wrsp. to Newton's polynome form
        ! with Newton's tableau ("divided differences"):
        f12  = (ltable%igfhr(im) - ltable%igfhr(iu)) * ltable%odxhr
        f23  = (ltable%igfhr(io) - ltable%igfhr(im)) * ltable%odxhr
        f123 = 0.5d0 * (f23 - f12) * ltable%odxhr
        ! Horner's scheme to calculate the interpolated value from the interpolating parabola:
        yn2 = f123 * (xt - ltable%xhr(im)) + f12
        yn2 = yn2  * (xt - ltable%xhr(iu)) + ltable%igfhr(iu)

        incgfct_upper_lookup_parabolic = 0.5d0 * ( yn1 + yn2 )

      ELSE

        incgfct_upper_lookup_parabolic = yn1

      END IF

    ELSE

      ! Truncate x to the range of the table:
      xt = MAX(MIN(x, ltable%x(ltable%n)), 0.0d0)

      ! Neighbourhood 1:
      ! calculate indices of the neighbouring regular x-values
      ! in the table:
      im = MIN(MAX(FLOOR(xt * ltable%odx) + 1, 2),  ltable%n-1)
      iu = im - 1
      io = im + 1

      ! interpolate parabolicly (lower incomplete gamma function --
      ! will be converted to upper FUNCTION later):
      ! coefficients of the interpolating parabola wrsp. to Newton's polynome form
      ! with Newton's tableau ("divided differences"):
      f12  = (ltable%igf(im) - ltable%igf(iu)) * ltable%odx
      f23  = (ltable%igf(io) - ltable%igf(im)) * ltable%odx
      f123 = 0.5d0 * (f23 - f12) * ltable%odx
      ! Horner's scheme to calculate the interpolated value from the interpolating parabola:
      yn1 = f123 * (xt - ltable%x(im)) + f12
      yn1 = yn1  * (xt - ltable%x(iu)) + ltable%igf(iu)

      IF (im < ltable%n - 2) THEN
        ! Neighbourhood 2:
        ! calculate indices of the neighbouring regular x-values
        ! in the table:

        im = MIN(MAX(CEILING(xt * ltable%odx) + 1, 2),  ltable%n-2)
        iu = im - 1
        io = im + 1

        ! interpolate parabolicly:
        ! coefficients of the interpolating parabola wrsp. to Newton's polynome form
        ! with Newton's tableau ("divided differences"):
        f12  = (ltable%igf(im) - ltable%igf(iu)) * ltable%odx
        f23  = (ltable%igf(io) - ltable%igf(im)) * ltable%odx
        f123 = 0.5d0 * (f23 - f12) * ltable%odx
        ! Horner's scheme to calculate the interpolated value from the interpolating parabola:
        yn2 = f123 * (xt - ltable%x(im)) + f12
        yn2 = yn2  * (xt - ltable%x(iu)) + ltable%igf(iu)

        incgfct_upper_lookup_parabolic = 0.5d0 * ( yn1 + yn2 )

      ELSE

        incgfct_upper_lookup_parabolic = yn1

      END IF
    END IF

    ! Convert to upper incomplete gamma function:
    incgfct_upper_lookup_parabolic = ltable%igf(ltable%n) - incgfct_upper_lookup_parabolic

    incgfct_upper_lookup_parabolic = MAX(incgfct_upper_lookup_parabolic, 0.0d0)

    RETURN
  END FUNCTION incgfct_upper_lookup_parabolic

  !===========================================================================
  ! OBSOLETE:
  !===========================================================================
  ! Subroutinen fuer die Wet Growth Parametrisierung:
  ! Initialisierung: Einlesen der Lookup-table aus einer Textdatei.
  ! Diese Subroutine muss von der Interface-Routine des 2-M-Schemas
  ! aufgerufen werden.
  ! Eventuelle Verteilung der Table auf alle Knoten bei Parallelbetrieb
  ! muss ebenfalls von der Interface-Routine besorgt werden.
  !===========================================================================

  SUBROUTINE init_dmin_wetgrowth(dateiname, unitnr)
    CHARACTER(len=*), INTENT(in) :: dateiname
    INTEGER, INTENT(in) :: unitnr
    INTEGER :: error, in_aux(4)
    CHARACTER(len=*), PARAMETER :: routine = 'init_dmin_wetgrowth'

    CALL message (TRIM(routine), " Trying to read "//TRIM(dateiname))
    OPEN(unitnr, file=TRIM(dateiname), status='old', form='formatted', iostat=error)
    IF (error /= 0) THEN
      WRITE (txt,*) 'init_dmin_wetgrowth: lookup-table ' // TRIM(dateiname) // ' not found'
      CALL message(modname,TRIM(txt))
      CALL finish(TRIM(modname),'Error in init_dmin_wetgrowth')
    END IF

    READ (unitnr,*) in_aux(1:4) ! anzp_wg, anzT_wg, anzi_wg, anzw_wg

    anzp_wg = in_aux(1)
    anzT_wg = in_aux(2)
    anzi_wg = in_aux(3)
    anzw_wg = in_aux(4)

    IF (ALLOCATED(pvec_wg_g)) THEN

      IF (anzp_wg /= SIZE(pvec_wg_g)) THEN
        WRITE (txt,*) 'init_dmin_wetgrowth: Error re-reading pvec from ' // TRIM(dateiname) // ': wrong size anzp'
        CALL message(modname,TRIM(txt))
        CALL finish(TRIM(modname),'Error in init_dmin_wetgrowth')
      END IF
      IF (anzT_wg /= SIZE(Tvec_wg_g)) THEN
        WRITE (txt,*) 'init_dmin_wetgrowth: Error re-reading Tvec from ' // TRIM(dateiname) // ': wrong size anzT'
        CALL message(modname,TRIM(txt))
        CALL finish(TRIM(modname),'Error in init_dmin_wetgrowth')
      END IF
      IF (anzi_wg /= SIZE(qivec_wg_g)) THEN
        WRITE (txt,*) 'init_dmin_wetgrowth: Error re-reading qivec from ' // TRIM(dateiname) // ': wrong size anzi'
        CALL message(modname,TRIM(txt))
        CALL finish(TRIM(modname),'Error in init_dmin_wetgrowth')
      END IF
      IF (anzw_wg /= SIZE(qwvec_wg_g)) THEN
        WRITE (txt,*) 'init_dmin_wetgrowth: Error re-reading qwvec from ' // TRIM(dateiname) // ': wrong size anzw'
        CALL message(modname,TRIM(txt))
        CALL finish(TRIM(modname),'Error in init_dmin_wetgrowth')
      END IF

    ELSE

      CALL message(modname,'init_dmin_wetgrowth: Initializing non-equidistant lookup table for graupel wet growth diameter (2mom)')

      ALLOCATE(pvec_wg_g(anzp_wg))
      ALLOCATE(Tvec_wg_g(anzT_wg))
      ALLOCATE(qwvec_wg_g(anzw_wg))
      ALLOCATE(qivec_wg_g(anzi_wg))
      ALLOCATE(dmin_wg_g(anzp_wg,anzT_wg,anzw_wg,anzi_wg))

      READ (unitnr,*,iostat=error) pvec_wg_g(1:anzp_wg)
      IF (error /= 0) THEN
        WRITE (txt,*) 'init_dmin_wetgrowth: Error reading pvec from ' // TRIM(dateiname)
        CALL message(modname,TRIM(txt))
        CALL finish(TRIM(modname),'Error in init_dmin_wetgrowth')
      END IF
      READ (unitnr,*,iostat=error) Tvec_wg_g(1:anzT_wg)
      IF (error /= 0) THEN
        WRITE (txt,*) 'init_dmin_wetgrowth: Error reading Tvec from ' // TRIM(dateiname)
        CALL message(modname,TRIM(txt))
        CALL finish(TRIM(modname),'Error in init_dmin_wetgrowth')
      END IF
      READ (unitnr,*,iostat=error) qwvec_wg_g(1:anzw_wg)
      IF (error /= 0) THEN
        WRITE (txt,*) 'init_dmin_wetgrowth: Error reading qwvec from ' // TRIM(dateiname)
        CALL message(modname,TRIM(txt))
        CALL finish(TRIM(modname),'Error in init_dmin_wetgrowth')
      END IF
      READ (unitnr,*,iostat=error) qivec_wg_g(1:anzi_wg)
      IF (error /= 0) THEN
        WRITE (txt,*) 'init_dmin_wetgrowth: Error reading qivec from ' // TRIM(dateiname)
        CALL message(modname,TRIM(txt))
        CALL finish(TRIM(modname),'Error in init_dmin_wetgrowth')
      END IF
      READ (unitnr,*,iostat=error) dmin_wg_g(1:anzp_wg,1:anzT_wg,1:anzw_wg,1:anzi_wg)
      IF (error /= 0) THEN
        WRITE (txt,*) 'init_dmin_wetgrowth: Error reading dmin from ' // TRIM(dateiname)
        CALL message(modname,TRIM(txt))
        CALL finish(TRIM(modname),'Error in init_dmin_wetgrowth')
      END IF
      CLOSE(unitnr)

    END IF

    RETURN
  END SUBROUTINE init_dmin_wetgrowth

  !===========================================================================
  ! OBSOLETE: wet growth Grenzdurchmesser fuer graupelhail2test in m:
  !===========================================================================
  FUNCTION dmin_wetgrowth_graupel(p_a,T_a,qw_a,qi_a)

    REAL(wp) :: dmin_wetgrowth_graupel
    REAL(wp), INTENT(in) :: p_a,T_a,qw_a,qi_a
    REAL(wp) :: p_lok,T_lok,qw_lok,qi_lok

    INTEGER :: i
    INTEGER :: iu, io, ju, jo, ku, ko, lu, lo

    REAL(wp) :: hilf1(2,2,2,2), hilf2(2,2,2), hilf3(2,2), hilf4(2)

    LOGICAL :: found_p, found_T, found_w, found_i

    found_p = .FALSE.
    found_T = .FALSE.
    found_w = .FALSE.
    found_i = .FALSE.
    dmin_wetgrowth_graupel = 999.99

    p_lok = MIN(MAX(p_a,pvec_wg_g(1)),pvec_wg_g(anzp_wg))
    IF (p_a <= pvec_wg_g(1)) THEN
      found_p = .TRUE.
      iu = 1
      io = 2
    ELSE IF (p_a >= pvec_wg_g(anzp_wg)) THEN
      found_p = .TRUE.
      iu = anzp_wg - 1
      io = anzp_wg
    ELSE
      iu = 1
      DO i=1, anzp_wg-1
        IF (p_a >= pvec_wg_g(i) .AND. p_a < pvec_wg_g(i+1)) THEN
          iu = i
          found_p = .TRUE.
          EXIT
        END IF
      END DO
      io = iu + 1
    END IF

    T_lok = MIN(MAX(T_a,Tvec_wg_g(1)),Tvec_wg_g(anzT_wg))
    IF (T_a <= Tvec_wg_g(1)) THEN
      found_T = .TRUE.
      ju = 1
      jo = 2
    ELSE IF (T_a >= Tvec_wg_g(anzT_wg)) THEN
      found_T = .TRUE.
      dmin_wetgrowth_graupel = 0.0
      RETURN
    ELSE
      ju = 1
      DO i=1, anzT_wg-1
        IF (T_a >= Tvec_wg_g(i) .AND. T_a < Tvec_wg_g(i+1)) THEN
          ju = i
          found_T = .TRUE.
          EXIT
        END IF
      END DO
      jo = ju + 1
    END IF

    qw_lok = MIN(MAX(qw_a,qwvec_wg_g(1)),qwvec_wg_g(anzw_wg))
    IF (qw_a <= qwvec_wg_g(1)) THEN
      found_w = .TRUE.
      dmin_wetgrowth_graupel = 999.99
      RETURN
    ELSE IF (qw_a >= qwvec_wg_g(anzw_wg)) THEN
      found_w = .TRUE.
      ku = anzw_wg - 1
      ko = anzw_wg
    ELSE
      ku = 1
      DO i=1, anzw_wg-1
        IF (qw_a >= qwvec_wg_g(i) .AND. qw_a < qwvec_wg_g(i+1)) THEN
          ku = i
          found_w = .TRUE.
          EXIT
        END IF
      END DO
      ko = ku + 1
    END IF

    qi_lok = MIN(MAX(qi_a,qivec_wg_g(1)),qivec_wg_g(anzi_wg))
    IF (qi_a <= qivec_wg_g(1)) THEN
      found_i = .TRUE.
      lu = 1
      lo = 2
    ELSE IF (qi_a >= qivec_wg_g(anzi_wg)) THEN
      found_i = .TRUE.
      lu = anzi_wg - 1
      lo = anzi_wg
    ELSE
      lu = 1
      DO i=1, anzi_wg-1
        IF (qi_a >= qivec_wg_g(i) .AND. qi_a < qivec_wg_g(i+1)) THEN
          lu = i
          found_i = .TRUE.
          EXIT
        END IF
      END DO
      lo = lu + 1
    END IF

    IF (.NOT.found_p .OR. .NOT.found_T .OR. .NOT.found_w .OR. .NOT. found_i) THEN
       WRITE (txt,*) 'dmin_wetgrowth_graupel: interpolation point not found in lookup table'
       CALL message(modname,TRIM(txt))
       dmin_wetgrowth_graupel = 999.99
    ELSE

      ! Tetra-lineare Interpolation von Dmin:
      hilf1 = dmin_wg_g(iu:io,ju:jo,ku:ko,lu:lo)
      hilf2 = hilf1(1,:,:,:) + &
           (hilf1(2,:,:,:)-hilf1(1,:,:,:)) / &
           (pvec_wg_g(io)-pvec_wg_g(iu)) * (p_lok-pvec_wg_g(iu))
      hilf3 = hilf2(1,:,:) + &
           (hilf2(2,:,:)-hilf2(1,:,:)) / (Tvec_wg_g(jo)-Tvec_wg_g(ju)) * (T_lok-Tvec_wg_g(ju))

      hilf4 = hilf3(1,:) + &
           (hilf3(2,:)-hilf3(1,:)) / (qwvec_wg_g(ko)-qwvec_wg_g(ku)) * (qw_lok-qwvec_wg_g(ku))

      dmin_wetgrowth_graupel = hilf4(1) + &
           (hilf4(2)-hilf4(1)) / (qivec_wg_g(lo)-qivec_wg_g(lu)) * (qi_lok-qivec_wg_g(lu))

    END IF

    RETURN
  END FUNCTION dmin_wetgrowth_graupel

  !===========================================================================
  !
  ! Subroutine for setting up the wet growth diameter of a frozen hydrometeor
  ! type as function of supercooled LWC qw, frozen content qi, pressure p and temperature T.
  ! Is needed for the Parameterization of conversion from graupel to hail
  ! via wet growth of graupel.
  ! A corresponding 4D lookup table is read from an external file and is made
  ! equidistant along all table dimensions for better vectorization of table lookup
  ! (quadro-linear interpolation). Here, 3 of the dimensions (qw, qi, p) are already assumed
  ! to be equidistant in the table file. Only T can be non-equidistant and is
  ! made equidistant by linear interpolation.
  !
  ! The file format can be either a NETCDF table file (.nc) or a classic ASCII table (.dat).
  ! In case of NETCDF, the "hydrotypename" argument cross-checked with the
  ! correspondig global attribute in the NETCDF file,
  ! if it matches the hydrometeor type for which the table has been constructed.
  !
  ! The subroutine reads the table on one node only (NEC: a dedicated VH node)
  ! and distributes the table and the table vectors to all workers.
  !
  !===========================================================================

  SUBROUTINE init_dmin_wg_gr_ltab_equi(filebasename, parti, unitnr, ndT, ltab, msg_level)

    CHARACTER(len=*),       INTENT(in)    :: filebasename
    ! To cross-check that the table matches the desired hydrometeor type:
    CLASS(particle),        INTENT(in)    :: parti
    ! File unit number for ASCII file reading:
    INTEGER,                INTENT(in)    :: unitnr
    ! Desired number of elements for the fine equidistant grid vector for T:
    INTEGER,                INTENT(in)    :: ndT
    TYPE(lookupt_4D),       INTENT(inout) :: ltab
    INTEGER,                INTENT (IN)   :: msg_level

    CHARACTER(len=300)  :: alt_filebasename
    
    ! grid spacings of the desired fine grid vectors:
    REAL(wp)            :: minT, maxT
    INTEGER             :: i, j, k, l, error, ii

    REAL(wp), DIMENSION(:), ALLOCATABLE :: pvec_wg_g_loc, qwvec_wg_g_loc, qivec_wg_g_loc
    REAL(wp), ALLOCATABLE :: Tvec_wg_g_loc(:), dmin_wg_g_loc(:,:,:,:)
    INTEGER             :: anzT_wg_loc
    INTEGER             :: ju, jo

    CHARACTER(len=*), PARAMETER :: routine = TRIM(modname)//'::init_dmin_wg_gr_ltab_equi'

    error = 0

    !------------------------------------------------------------------------------------
    ! 1) Read the original lookup table from a file. This table may be made of a
    !    nonequidistant grid vector for T.
    !    The grid vectors for p, qw and qi have to be equidistant.
    !
    !    If reading fails for the standar filename, try a more specific filename
    !    having the relevant parameters in the filename. If this also fails,
    !    generate the table on the fly and write it to a netcdf file whose
    !    name has the specific parameters in its filename.

    IF (.NOT. ltab%is_initialized) THEN 

      CALL message(TRIM(routine),'Initializing equidistant lookup table for graupel wet growth diameter (2mom)')

      CALL message(TRIM(routine),'Reading Dmin table from file '//TRIM(filebasename)//'.nc')

        CALL read_dmin_wetgrowth_table(filebasename, parti, unitnr, ltab, &
             Tvec_wg_g_loc, anzT_wg_loc, dmin_wg_g_loc, error, msg_level)

        ! .. table vectors ltab%x1, ltab%x2, and ltab%x3, Tvec_wg_g_loc and dmin_wg_g_loc are in SI at this point

        IF (error /= 0) THEN

          alt_filebasename = get_filebase_dmin_lut_file(TRIM(filebasename), parti)
          CALL message(TRIM(routine),'Failed reading Dmin table, trying alternative file '//TRIM(alt_filebasename)//'.nc')

          ! .. The file was the wrong file or was not present. We try to read a file whose name has the parameters as part
          !    of the filename. If that does not work, we generate the table on the fly and write
          !    it as netcdf to disk.

          error = 0
          CALL read_dmin_wetgrowth_table(alt_filebasename, parti, unitnr, ltab, &
               Tvec_wg_g_loc, anzT_wg_loc, dmin_wg_g_loc, error, msg_level)

          ! .. table vectors ltab%x1, ltab%x2, and ltab%x3, Tvec_wg_g_loc and dmin_wg_g_loc are in SI at this point

          IF (error /= 0) THEN

            CALL message(TRIM(routine),'Failed reading Dmin table, generating table on the fly')
            
            ! .. If the other file is also not present, we create the LUT on the fly:
            CALL dmin_wetgrowth_lookupcreate (parti, dmin_wg_g_loc, ltab%n1, anzT_wg_loc, ltab%n3, ltab%n4, &
                 pvec_wg_g_loc, Tvec_wg_g_loc, qwvec_wg_g_loc, qivec_wg_g_loc)

            ! .. dmin_wg_g_loc in m, pvec_wg_g_loc in Pa, Tvec_wg_g_loc in degC, qwvec_wg_g_loc in kg/m3, qivec_wg_g_loc in kg/m3

            CALL message(TRIM(routine),'Table generated, now writing it to file '//TRIM(alt_filebasename)//'.nc')

            ! .. and save it to a netcdf file:
            CALL write_dmin_wetgrowth_table(TRIM(filebasename), parti, dmin_wg_g_loc, &
                 pvec_wg_g_loc, Tvec_wg_g_loc, qwvec_wg_g_loc, qivec_wg_g_loc, error, filename=TRIM(alt_filebasename)//'.nc')
            IF (error /= 0) THEN
              CALL finish(TRIM(routine),'Error writing Dmin table to file! Stop!')
            END IF
            
            ! .. re-read the newly produced file in order to avoid nonreproducible results because of rounding error
            !    differences of table values which have been observed between the netcdf-file and the on-the-fly LUT:
            CALL message(TRIM(routine),'Re-reading table from file'//TRIM(alt_filebasename)//'.nc to achieve reproducible results')
            CALL read_dmin_wetgrowth_table(alt_filebasename, parti, unitnr, ltab, &
                 Tvec_wg_g_loc, anzT_wg_loc, dmin_wg_g_loc, error, msg_level)
            IF (error /= 0) THEN
              CALL finish(TRIM(routine),'Error re-reading Dmin table from file! Stop!')
            END IF

            ! .. table vectors ltab%x1, ltab%x2, and ltab%x3, Tvec_wg_g_loc and dmin_wg_g_loc are in SI at this point

          ELSE
            CALL message(TRIM(routine),'Reading of alternative file '//TRIM(alt_filebasename)//'.nc was successful')
          END IF
        ELSE
          CALL message(TRIM(routine),'Reading of file '//TRIM(filebasename)//'.nc was successful')
        END IF

      !------------------------------------------------------------------------------------
      ! 2) Generate equidistant table vector T and construct the
      !    equidistant Dmin-lookuptable by linear oversampling:
      !    (all in SI units!)
      
      ltab%n2 = ndT

      NULLIFY ( ltab%x2 )
      NULLIFY ( ltab%ltable )

      ALLOCATE( ltab%x2(ltab%n2) )
      ALLOCATE( ltab%ltable(ltab%n1,ltab%n2,ltab%n3,ltab%n4) )

      minT  = Tvec_wg_g_loc (1)
      maxT  = Tvec_wg_g_loc (anzT_wg_loc)

      ltab%dx1      = ltab%x1(2) - ltab%x1(1)
      ltab%odx1     = 1.0d0 / ltab%dx1
      ltab%dx2      = (maxT - minT) / (ndT - 1.0d0)
      ltab%odx2     = 1.0d0 / ltab%dx2
      ltab%dx3      = ltab%x3(2) - ltab%x3(1)
      ltab%odx3     = 1.0d0 / ltab%dx3
      ltab%dx4      = ltab%x4(2) - ltab%x4(1)
      ltab%odx4     = 1.0d0 / ltab%dx4

      ! Equidistant grid vectors for T:
      DO j=1, ltab%n2
        ltab%x2(j) = minT + (j-1) * ltab%dx2
      END DO

      ! Linear interpolation w.r.t. T of the equidistant Dmin-lookuptable from
      ! the original table in the datafile, which may be non-equidistant
      ! w.r.t. T:

!NEC$ unroll_completely
      DO j=1, ndT
        ju = 1
        DO ii=1, anzT_wg_loc-1
          IF (ltab%x2(j) >= Tvec_wg_g_loc(ii) .AND. ltab%x2(j) <= Tvec_wg_g_loc(ii+1)) THEN
            ju = ii
            EXIT
          END IF
        END DO
        jo = ju + 1

        ! Linear interplation of Dmin with respect to T:
        ltab%ltable(:,j,:,:) = dmin_wg_g_loc(:,ju,:,:) + &
             (dmin_wg_g_loc(:,jo,:,:) - dmin_wg_g_loc(:,ju,:,:)) / &
             (Tvec_wg_g_loc(jo)-Tvec_wg_g_loc(ju))  * (ltab%x2(j)-Tvec_wg_g_loc(ju))

      END DO

      !$ACC ENTER DATA COPYIN(ltab, ltab%x1, ltab%x2, ltab%x3, ltab%x4, ltab%ltable)

      ! clean up memory:
      DEALLOCATE(Tvec_wg_g_loc,dmin_wg_g_loc)

      ltab%is_initialized = .TRUE.

    END IF

    IF (lprintout_comp_table_fit) THEN
      CALL dmin_wetgrowth_fun_check(parti, ltab)
    END IF

  END SUBROUTINE init_dmin_wg_gr_ltab_equi

  SUBROUTINE read_dmin_wetgrowth_table(filebasename, parti, unitnr, ltab, &
       Tvec_wg_g_loc, anzT_wg_loc, dmin_wg_g_loc, error, msg_level)

    CHARACTER(len=*), INTENT(in)       :: filebasename
    ! To cross-check that the table matches the desired hydrometeor type:
    CLASS(particle),  INTENT(in)       :: parti
    ! File unit number for ASCII file reading:
    INTEGER,          INTENT(in)       :: unitnr
    TYPE(lookupt_4D), INTENT(inout)    :: ltab
    REAL(wp), ALLOCATABLE, INTENT(out) :: Tvec_wg_g_loc(:), dmin_wg_g_loc(:,:,:,:)
    INTEGER,          INTENT(out)      :: anzT_wg_loc
    INTEGER,          INTENT(out)      :: error
    INTEGER,          INTENT (IN)      :: msg_level

    CHARACTER(len=*), PARAMETER :: routine = TRIM(modname)//'::read_dmin_wetgrowth_table'

    INTEGER             :: i,j,k,l,in_aux(4)
    CHARACTER(len=300)  :: filename                  ! Full file name with .nc or .dat extension
    INTEGER             :: status_netcdf             ! Retrun from netcdf functions
    INTEGER             :: id_netcdf                 ! ID from file
    INTEGER             :: dims_id(4)                ! Dimensions from netcf
    INTEGER             :: p_id, T_id, qw_id, qi_id  ! Ids of variables in netcdf
    INTEGER             :: dmin_id                   ! Id of table
    LOGICAL             :: l_netcdf_format           ! True for netcdf, false for ascii

    CHARACTER(len=100)  :: nc_hydrotype
    REAL(wp)            :: nc_ageo, nc_bgeo, nc_avel, nc_bvel, dmin_fillval

    error = 0

    filename = TRIM(filebasename)//'.nc'
    status_netcdf = check_nc( nf90_open(TRIM(ADJUSTL(filename)), NF90_NOWRITE, id_netcdf), &
         'opening '//TRIM(filebasename)//'.nc')

    IF (status_netcdf == nf90_noerr) THEN 

      ! Check hydrometeor type:
      nc_hydrotype(:) = ' '
      status_netcdf = check_nc( nf90_get_att(id_netcdf, NF90_GLOBAL, 'hydrometeorType', nc_hydrotype), 'hydrometeorType')
      IF (TRIM(nc_hydrotype) /= TRIM(parti%name)) THEN
        CALL message(routine,'INFO: need table file for hydrometeor type '// &
             TRIM(parti%name)//' but file '//TRIM(filename) //' is for '//TRIM(nc_hydrotype))
      END IF
      status_netcdf = check_nc( nf90_get_att(id_netcdf, NF90_GLOBAL, 'a_geo', nc_ageo), 'a_geo')
      IF ( ABS(nc_ageo - parti%a_geo) > 1e-3_wp ) THEN
        txt(:) = ' '
        WRITE(txt,'(a,es12.5,a,es12.5)') 'Error: wrong a_geo in table file: need a_geo=', parti%a_geo, &
             ' but file '//TRIM(filename) //' is for ', nc_ageo
        IF (msg_level > 6) CALL message(routine, TRIM(txt))
        error = 1
      END IF
      status_netcdf = check_nc( nf90_get_att(id_netcdf, NF90_GLOBAL, 'b_geo', nc_bgeo), 'b_geo')
      IF ( ABS(nc_bgeo - parti%b_geo) > 1e-3_wp ) THEN
        txt(:) = ' '
        WRITE(txt,'(a,es12.5,a,es12.5)') 'Error: wrong b_geo in table file: need b_geo=', parti%b_geo, &
             ' but file '//TRIM(filename) //' is for ', nc_bgeo
        IF (msg_level > 6) CALL message(routine, TRIM(txt))
        error = 2
      END IF
      status_netcdf = check_nc( nf90_get_att(id_netcdf, NF90_GLOBAL, 'a_vel', nc_avel), 'a_vel')
      IF ( ABS(nc_avel - parti%a_vel) > 1e-3_wp ) THEN
        txt(:) = ' '
        WRITE(txt,'(a,es12.5,a,es12.5)') 'Error: wrong a_vel in table file: need a_vel=', parti%a_vel, &
             ' but file '//TRIM(filename) //' is for ', nc_avel
        IF (msg_level > 6) CALL message(routine, TRIM(txt))
        error = 3
      END IF
      status_netcdf = check_nc( nf90_get_att(id_netcdf, NF90_GLOBAL, 'b_vel', nc_bvel), 'b_vel')
      IF ( ABS(nc_bvel - parti%b_vel) > 1e-3_wp ) THEN
        txt(:) = ' '
        WRITE(txt,'(a,es12.5,a,es12.5)') 'Error: wrong b_vel in table file: need b_vel=', parti%b_vel, &
             ' but file '//TRIM(filename) //' is for ', nc_bvel
        IF (msg_level > 6) CALL message(routine, TRIM(txt))
        error = 4
      END IF

      status_netcdf = check_nc( nf90_inq_dimid(id_netcdf, 'npres', dims_id(1)), 'npres')
      status_netcdf = check_nc( nf90_inq_dimid(id_netcdf, 'ntemp', dims_id(2)), 'ntemp')
      status_netcdf = check_nc( nf90_inq_dimid(id_netcdf, 'nqw'  , dims_id(3)), 'nqw'  )
      status_netcdf = check_nc( nf90_inq_dimid(id_netcdf, 'nqi'  , dims_id(4)), 'nqi'  )
      DO j=1,4
        status_netcdf = check_nc( nf90_inquire_dimension(id_netcdf, dims_id(j), len=in_aux(j)), 'dimension')
      END DO
      l_netcdf_format = .true.

      ! Try ascii format if no netcdf is available  
    ELSE

      CALL message(routine,'Table file '//TRIM(ADJUSTL(filename))// ' not found. Trying ascii file.')   
      OPEN(unitnr, file=TRIM(filebasename)//'.dat', status='old', form='formatted', iostat=error)
      IF (error /= 0) THEN
        WRITE (txt,*) 'Error: table file ' // TRIM(filebasename)//'.dat' // ' not found.'
        error = 5
        l_netcdf_format = .TRUE.
      ELSE
        READ (unitnr,*) in_aux(1:4) !  ltab%n1, anzT_wg_loc, ltab%n3, ltab%n4
        l_netcdf_format = .FALSE.
      END IF

    END IF

    IF (error == 0) THEN

      ltab%n1 = in_aux(1)
      anzT_wg_loc = in_aux(2)
      ltab%n3 = in_aux(3)
      ltab%n4 = in_aux(4)

      IF (ASSOCIATED(ltab%x1)) DEALLOCATE(ltab%x1)
      IF (ALLOCATED(Tvec_wg_g_loc)) DEALLOCATE(Tvec_wg_g_loc)
      IF (ASSOCIATED(ltab%x3)) DEALLOCATE(ltab%x3)
      IF (ASSOCIATED(ltab%x4)) DEALLOCATE(ltab%x4)
      IF (ALLOCATED(dmin_wg_g_loc)) DEALLOCATE(dmin_wg_g_loc)

      NULLIFY (ltab%x1, ltab%x3, ltab%x4)
      
      ALLOCATE( Tvec_wg_g_loc(anzT_wg_loc) )
      ALLOCATE( ltab%x1(ltab%n1) )
      ALLOCATE( ltab%x3(ltab%n3) )
      ALLOCATE( ltab%x4(ltab%n4) )
      ALLOCATE( dmin_wg_g_loc(ltab%n1,anzT_wg_loc,ltab%n3,ltab%n4) )

      IF ( l_netcdf_format ) THEN

        status_netcdf = check_nc( nf90_inq_varid(id_netcdf, 'p'                    , p_id   ), 'p' )
        status_netcdf = check_nc( nf90_inq_varid(id_netcdf, 'T'                    , T_id   ), 'T' )
        status_netcdf = check_nc( nf90_inq_varid(id_netcdf, 'qw'                   , qw_id  ), 'qw')
        status_netcdf = check_nc( nf90_inq_varid(id_netcdf, 'qi'                   , qi_id  ), 'qi')
        status_netcdf = check_nc( nf90_inq_varid(id_netcdf, 'Dmin_wetgrowth_table' , dmin_id), 'Dmin_wetgrowth_table')

        ! The precision in the nc table file is dp, but we can safely feed wp precision
        ! to the 3rd argument to nf90_get_var(), because type conversion is done
        ! internally in netcdf-lib as needed:
        status_netcdf = check_nc( nf90_get_var(id_netcdf, p_id   , ltab%x1      , start=(/1/))      , 'p' )
        status_netcdf = check_nc( nf90_get_var(id_netcdf, T_id   , Tvec_wg_g_loc, start=(/1/))      , 'T' )
        status_netcdf = check_nc( nf90_get_var(id_netcdf, qw_id  , ltab%x3      , start=(/1/))      , 'qw')
        status_netcdf = check_nc( nf90_get_var(id_netcdf, qi_id  , ltab%x4      , start=(/1/))      , 'qi')
        status_netcdf = check_nc( nf90_get_var(id_netcdf, dmin_id, dmin_wg_g_loc, start=(/1,1,1,1/)), 'Dmin_wetgrowth_table')
        status_netcdf = check_nc( nf90_get_att(id_netcdf, dmin_id, '_FillValue',  dmin_fillval)     , 'Dmin:_FillValue')
        status_netcdf = check_nc( nf90_close(id_netcdf), 'closing file')

        WHERE( ABS(dmin_wg_g_loc-dmin_fillval) < 1e-6_wp)
          dmin_wg_g_loc = 999.99
        END WHERE

        ! Unit conversion to SI from netcdf file:
        ltab%x1 = ltab%x1 * 100.0_wp           ! Conversion from hPa to Pa
        Tvec_wg_g_loc = Tvec_wg_g_loc + T_3    ! Conversion from deg C to K
        ltab%x3 = ltab%x3 * 0.001_wp           ! Conversion from g/m^3 to kg/m^3
        ltab%x4 = ltab%x4 * 0.001_wp           ! Conversion from g/m^3 to kg/m^3
        dmin_wg_g_loc = dmin_wg_g_loc*0.001_wp ! Conversion from mm to m

      ELSE

        READ (unitnr,*,iostat=error) ltab%x1(1:ltab%n1)
        IF (error /= 0) THEN
          txt(:) = ' '
          WRITE (txt,*) 'Error reading pvec from ' // TRIM(filename)
          IF (msg_level > 6) CALL message(TRIM(routine),TRIM(txt))
          error = 7
        END IF
        READ (unitnr,*,iostat=error) Tvec_wg_g_loc(1:anzT_wg_loc)
        IF (error /= 0) THEN
          txt(:) = ' '
          WRITE (txt,*) 'Error reading Tvec from ' // TRIM(filename)
          IF (msg_level > 6) CALL message(TRIM(routine),TRIM(txt))
          error = 7
        END IF
        READ (unitnr,*,iostat=error) ltab%x3(1:ltab%n3)
        IF (error /= 0) THEN
          txt(:) = ' '
          WRITE (txt,*) 'Error reading qwvec from ' // TRIM(filename)
          IF (msg_level > 6) CALL message(TRIM(routine),TRIM(txt))
          error = 7
        END IF
        READ (unitnr,*,iostat=error) ltab%x4(1:ltab%n4)
        IF (error /= 0) THEN
          txt(:) = ' '
          WRITE (txt,*) 'Error reading qivec from ' // TRIM(filename)
          IF (msg_level > 6) CALL message(TRIM(routine),TRIM(txt))
          error = 7
        END IF

        DO l=1, ltab%n4
          DO k=1, ltab%n3
            DO j=1, anzT_wg_loc
              DO i=1,ltab%n1
                READ (unitnr,*,iostat=error) dmin_wg_g_loc(i,j,k,l)
                IF (error /= 0) THEN
                  WRITE (txt,'(a,4(1x,i4))') 'Error reading dmin from '//TRIM(filename)//' at position',i,j,k,l
                  IF (msg_level > 6) CALL message(TRIM(routine),TRIM(txt))
                  error = 7
                END IF
              END DO
            END DO
          END DO
        END DO

        CLOSE(unitnr)
      ENDIF

    ENDIF

  CONTAINS

    FUNCTION check_nc ( istat, rinfo ) RESULT (ostat)
      INTEGER, INTENT(in)          :: istat
      CHARACTER(len=*), INTENT(in) :: rinfo
      INTEGER                      :: ostat

      ostat = istat
      IF (istat /= NF90_NOERR) THEN
        txt(:) = ' '
        WRITE (txt,'(a)') 'reading '//TRIM(filename)//': '// &
             & TRIM(rinfo)//': '//TRIM(NF90_strerror(istat))
        CALL message('ERROR '//TRIM(routine),TRIM(txt))
        error = 6
      END IF

    END FUNCTION check_nc

  END SUBROUTINE read_dmin_wetgrowth_table

  ! wet growth Grenzdurchmesser in m
  FUNCTION dmin_wg_gr_ltab_equi(p_a,T_a,qw_a,qi_a,ltab) RESULT (dmin_loc)

    !$ACC ROUTINE SEQ

    REAL(wp) :: dmin_loc
    REAL(wp), INTENT(in) :: p_a,T_a,qw_a,qi_a
    TYPE(lookupt_4D), INTENT(in) :: ltab
    REAL(wp) :: p_lok,T_lok,qw_lok,qi_lok

    INTEGER :: iu, io, ju, jo, ku, ko, lu, lo
    REAL(wp) :: hilf1_1111, hilf1_1112, hilf1_1121, hilf1_1122, &
         &      hilf1_1211, hilf1_1212, hilf1_1221, hilf1_1222, &
         &      hilf1_2111, hilf1_2112, hilf1_2121, hilf1_2122, &
         &      hilf1_2211, hilf1_2212, hilf1_2221, hilf1_2222
    REAL(wp) :: hilf2_111, hilf2_112, hilf2_121, hilf2_122, &
         &      hilf2_211, hilf2_212, hilf2_221, hilf2_222
    REAL(wp) :: hilf3_11, hilf3_12, hilf3_21, hilf3_22
    REAL(wp) :: hilf4_1, hilf4_2

    IF (T_a >= ltab%x2(ltab%n2)) THEN
      dmin_loc = 0.0d0
    ELSE IF (T_a < ltab%x2(1)) THEN
      dmin_loc = 999.99d0
    ELSE

      p_lok = MIN(MAX(p_a,ltab%x1(1)),ltab%x1(ltab%n1))
      iu = MIN(FLOOR((p_lok - ltab%x1(1)) * ltab%odx1 ) + 1, ltab%n1-1)
      io = iu + 1
      T_lok = MIN(MAX(T_a,ltab%x2(1)),ltab%x2(ltab%n2))
      ju = MIN(FLOOR((T_lok - ltab%x2(1)) * ltab%odx2 ) + 1, ltab%n2-1)
      jo = ju + 1
      qw_lok = MIN(MAX(qw_a,ltab%x3(1)),ltab%x3(ltab%n3))
      ku = MIN(FLOOR((qw_lok - ltab%x3(1)) * ltab%odx3 ) + 1, ltab%n3-1)
      ko = ku + 1
      qi_lok = MIN(MAX(qi_a,ltab%x4(1)),ltab%x4(ltab%n4))
      lu = MIN(FLOOR((qi_lok - ltab%x4(1)) * ltab%odx4 ) + 1, ltab%n4-1)
      lo = lu + 1

      ! Tetra-linear interpolation of Dmin:
      ! The following is an explicit manual expansion of this old code for better vectorization:
      !    REAL(wp) :: hilf1(2,2,2,2), hilf2(2,2,2), hilf3(2,2), hilf4(2)
      !      hilf1 = ltab%ltable(iu:io,ju:jo,ku:ko,lu:lo)
      !      hilf2 = hilf1(1,:,:,:) + (hilf1(2,:,:,:) - hilf1(1,:,:,:)) * ltab%odx1 * (p_lok-ltab%x1(iu) )
      !      hilf3 = hilf2(1,:,:)   + (hilf2(2,:,:)   - hilf2(1,:,:)  ) * ltab%odx2 * (T_lok-ltab%x2(ju) )
      !      hilf4 = hilf3(1,:)     + (hilf3(2,:)     - hilf3(1,:)    ) * ltab%odx3 * (qw_lok-ltab%x3(ku))

      hilf1_1111 = ltab%ltable(iu,ju,ku,lu)
      hilf1_1112 = ltab%ltable(iu,ju,ku,lo)
      hilf1_1121 = ltab%ltable(iu,ju,ko,lu)
      hilf1_1122 = ltab%ltable(iu,ju,ko,lo)
      hilf1_1211 = ltab%ltable(iu,jo,ku,lu)
      hilf1_1212 = ltab%ltable(iu,jo,ku,lo)
      hilf1_1221 = ltab%ltable(iu,jo,ko,lu)
      hilf1_1222 = ltab%ltable(iu,jo,ko,lo)
      hilf1_2111 = ltab%ltable(io,ju,ku,lu)
      hilf1_2112 = ltab%ltable(io,ju,ku,lo)
      hilf1_2121 = ltab%ltable(io,ju,ko,lu)
      hilf1_2122 = ltab%ltable(io,ju,ko,lo)
      hilf1_2211 = ltab%ltable(io,jo,ku,lu)
      hilf1_2212 = ltab%ltable(io,jo,ku,lo)
      hilf1_2221 = ltab%ltable(io,jo,ko,lu)
      hilf1_2222 = ltab%ltable(io,jo,ko,lo)

      hilf2_111 = hilf1_1111 + (hilf1_2111 - hilf1_1111) * ltab%odx1 * (p_lok-ltab%x1(iu) )
      hilf2_112 = hilf1_1112 + (hilf1_2112 - hilf1_1112) * ltab%odx1 * (p_lok-ltab%x1(iu) )
      hilf2_121 = hilf1_1121 + (hilf1_2121 - hilf1_1121) * ltab%odx1 * (p_lok-ltab%x1(iu) )
      hilf2_122 = hilf1_1122 + (hilf1_2122 - hilf1_1122) * ltab%odx1 * (p_lok-ltab%x1(iu) )
      hilf2_211 = hilf1_1211 + (hilf1_2211 - hilf1_1211) * ltab%odx1 * (p_lok-ltab%x1(iu) )
      hilf2_212 = hilf1_1212 + (hilf1_2212 - hilf1_1212) * ltab%odx1 * (p_lok-ltab%x1(iu) )
      hilf2_221 = hilf1_1221 + (hilf1_2221 - hilf1_1221) * ltab%odx1 * (p_lok-ltab%x1(iu) )
      hilf2_222 = hilf1_1222 + (hilf1_2222 - hilf1_1222) * ltab%odx1 * (p_lok-ltab%x1(iu) )

      hilf3_11  = hilf2_111  + (hilf2_211  - hilf2_111 ) * ltab%odx2 * (T_lok-ltab%x2(ju) )
      hilf3_12  = hilf2_112  + (hilf2_212  - hilf2_112 ) * ltab%odx2 * (T_lok-ltab%x2(ju) )
      hilf3_21  = hilf2_121  + (hilf2_221  - hilf2_121 ) * ltab%odx2 * (T_lok-ltab%x2(ju) )
      hilf3_22  = hilf2_122  + (hilf2_222  - hilf2_122 ) * ltab%odx2 * (T_lok-ltab%x2(ju) )
      
      hilf4_1   = hilf3_11   + (hilf3_21   - hilf3_11  ) * ltab%odx3 * (qw_lok-ltab%x3(ku))
      hilf4_2   = hilf3_12   + (hilf3_22   - hilf3_12  ) * ltab%odx3 * (qw_lok-ltab%x3(ku))

      dmin_loc = hilf4_1 + (hilf4_2 - hilf4_1)  * ltab%odx4 * (qi_lok-ltab%x4(lu))
    END IF

    RETURN
  END FUNCTION dmin_wg_gr_ltab_equi

  !*******************************************************************************
  ! 4D rational function to approximate the dmin_wetgrowth_table                 *
  ! for dmin_graupelhail2test4_wetgrowth_lookup.dat                              *
  !*******************************************************************************

  REAL(wp) ELEMENTAL FUNCTION dmin_wetgrowth_fun(pres,Tk,lwc,iwc) result(dmin)
    IMPLICIT NONE
    
    REAL(wp), INTENT(IN)               :: lwc,iwc,Tk,pres

    REAL(wp), PARAMETER, DIMENSION(0:12) :: &
         & a = (/ -2.33480700e+01, -4.27065283e+01, -8.24313122e-01, -1.13320431e+03, &
         &         1.69568678e+01, -1.15100315e+00, -1.89611461e-01, -3.42732844e+00, &
         &        -2.16286180e+01,  1.45545961e+01, -4.55024573e-01, -1.40540857e-02, &
         &        -4.15266452e-01 /)
    REAL(wp), PARAMETER, DIMENSION(0:8)  :: &
         & b = (/ 4.20962118e+02,  5.41644773e-02,  2.52002794e+00,                   &
         &       -1.20674620e+00,  8.50123638e-01, -6.93148367e-02,  2.50286359e+00,  &
         &        6.79791825e-02,  8.44801073e+00 /)
    REAL(wp), PARAMETER, DIMENSION(0:5)  :: &
         & c = (/ -2.24270084e-03, -6.80963043e-05, -1.14260514e-07, -2.43983379e-03, &
         &        -5.24851050e-05,  4.76633146e-07 /)

    REAL(wp) :: qw,qi,T,p,p1,p2,q1,q2,pp

    p  = pres * 1e-2  ! pressure in hPa
    T  = Tk - T_3     ! Celsius temperature
    qw = lwc * 1e3    ! liquid water in g/m3
    qi = iwc * 1e3    ! ice water in g/m3
    
    p1 = a(0)+a(1)*qw+a(2)*qi+a(3)*T+a(4)*qw*qw+a(5)*qw*qi+a(6)*qi*qi+a(7)*qi*T+a(8)*T*T+a(9)*qw*T &
         &   +a(10)*qw*qw*qw+a(11)*qi*qi*qi+a(12)*T*T*T
    q1 = 1.00+b(0)*qw+b(1)*qi+b(2)*T+b(3)*qw*qw+b(4)*qw*qi+b(5)*qi*qi+b(6)*qi*T+b(7)*T*T+b(8)*qw*T

    pp  = p - 700.0  ! change to pressure deviation from reference pressure of 700 hPa

    p2 = 1.0 + c(0)*pp + c(1)*pp*T + c(2)*pp*pp 
    q2 = 1.0 + c(3)*pp + c(4)*pp*T + c(5)*pp*pp  

    dmin = p1/q1 * p2/q2 * 1e-3    ! Dmin in m

!   some limit values (Dmin in meters)
    if (dmin.gt.0.9) then
       dmin = 0.9
    elseif (dmin.lt.0.and.qi/qw.gt.1) then
       dmin = 1.
    elseif (dmin.lt.0) then
       dmin = -999.
    end if

    return
  end function dmin_wetgrowth_fun

  !*******************************************************************************
  ! initial check of the 4D rational functions of the dmin_wetgrowth_table       *
  !*******************************************************************************

  SUBROUTINE dmin_wetgrowth_fun_check(parti, ltab)
    implicit none
    CLASS(particle) :: parti
    TYPE(lookupt_4D), INTENT(in) :: ltab

    integer  :: i,j,k,m
    real(wp) :: qw,qi,T,p
    real(wp) :: qliq(6) = (/0.2e-3,0.5e-3,1e-3,2e-3,5e-3,10e-3/)
    real(wp) :: qice(6) = (/0.2e-3,0.5e-3,1e-3,2e-3,5e-3,10e-3/)
    real(wp) :: pp(3) = (/300e2,700e2,1000e2/)
    real(wp) :: tt(3) = (/-30.,-20.,-10./)

    WRITE(*,*)    
    WRITE(*,*) 'Dmin_wetgrowth comparison of table and 4d-fit:'
    WRITE(*,'(A,L6)') ' dmin_wetgrowth_fit_check = ',dmin_wetgrowth_fit_check(parti)
    WRITE(*,'(9a20)') 'Tc [Celsius]','Tk [K]','p [hPa]','qw','qi','dmin_table','dmin_fit'

    DO m=1,size(pp)
      p = pp(m)
      DO k=1,size(tt)
        T = T_3 + tt(k)
        DO j=1,size(qice)
          qi = qice(j)
          DO i=1,size(qliq)
            qw = qliq(i)
            WRITE (*,'(5f20.3,2f20.3)') T-T_3, T, p*1e-2, qw*1e3, qi*1e3,         &
                 & dmin_wg_gr_ltab_equi(p,T,qw,qi,ltab)*1e3, &
                 & dmin_wetgrowth_fun(p,T,qw,qi)*1e3
          END DO
        END DO
      END DO
    END DO
    
  END SUBROUTINE dmin_wetgrowth_fun_check

  LOGICAL FUNCTION dmin_wetgrowth_fit_check(p)
    CLASS(particle) :: p
    REAL(wp), PARAMETER :: dmin_fit_a_geo = 1.42d-01
    REAL(wp), PARAMETER :: dmin_fit_b_geo = 0.314
    REAL(wp), PARAMETER :: dmin_fit_a_vel = 86.89371
    REAL(wp), PARAMETER :: dmin_fit_b_vel = 0.268325
    
    IF (p%a_geo.NE.dmin_fit_a_geo .OR. p%b_geo.NE.dmin_fit_b_geo &
         & .OR. p%a_vel.NE.dmin_fit_a_vel .OR. p%b_vel.NE.dmin_fit_b_vel) THEN
      dmin_wetgrowth_fit_check = .FALSE.
    ELSE
      dmin_wetgrowth_fit_check = .TRUE.
    END IF

  END FUNCTION dmin_wetgrowth_fit_check

  !*******************************************************************************
  ! Sticking efficiency of ice and snow as function of temperature
  !*******************************************************************************

  FUNCTION e_stick(T_a, istick) RESULT(e_i)

    REAL(wp), INTENT(IN) :: T_a     ! ambient T [K]
    INTEGER, INTENT(in)  :: istick  ! Flag to choose a specific parameterization

    REAL(wp) :: e_i, T_c

    T_c = T_a - T_3

    SELECT CASE (istick)
    CASE (1)
      ! temperature dependent sticking efficiency following Lin et al. (1983) which they
      ! use for rimed particles.
      ! (this relation is criticized by Ackerman et al. 2015, ACP, as being much too large at cold conditions)
      e_i = MIN(EXP(0.09_wp*T_c),1.0_wp)
    CASE (2)
      ! even higher sticking efficiency also suggested by Lin et al. (1983) and there used for unrimed particles.
      ! Also used by Spichtinger and Gierens, e.g., doi:10.5194/acp-13-9021-2013
      ! (similar to the values given in Pruppacher and Klett, Ch. 16.2, page 601, as used by Mitchell 1988, JAS)
      e_i = MIN(EXP(0.025_wp*T_c),1.0_wp) 
    CASE (3)
      ! as previous setting, but reduced sticking eff. below -40 C 
      IF ( (T_a-T_3) > -40_wp) THEN
        e_i = MIN(EXP(0.025_wp*T_c),1.0_wp) 
      ELSE
        e_i = 0.01_wp
      END IF
    CASE (4)
      ! piecewise defined sticking efficiency with maximum at -15 C,
      ! as given in Pruppacher and Klett, Ch. 16.2, page 601, as used by Mitchell 1988, JAS
      ! here extended below -20 C with values similar to Lin et al.
      IF ( T_c >= -4._wp ) THEN
        e_i = 0.1_wp
      ELSEIF ( T_c >= -6._wp ) THEN
        e_i = 0.6_wp
      ELSEIF ( T_c >= -9._wp ) THEN
        e_i = 0.1_wp
      ELSEIF ( T_c >= -12.5_wp ) THEN
        e_i = 0.4_wp
      ELSEIF ( T_c >= -17_wp ) THEN
        e_i = 1.0_wp
      ELSEIF ( T_c >= -20_wp ) THEN
        e_i = 0.40_wp
      ELSEIF ( T_c >= -30_wp ) THEN
        e_i = 0.25_wp
      ELSEIF ( T_c >= -40_wp ) THEN
        e_i = 0.10_wp
      ELSE
        e_i = 0.02_wp
      END IF
    CASE (5)
      ! piecewise linear sticking efficiency with maximum at -15 C,
      ! inspired by Figure 14 of Connolly et al. ACP 2012, doi:10.5194/acp-12-2055-2012
      ! Value at -40 C is based on Kajikawa and Heymsfield as cited by Philips et al. (2015, JAS)
      IF ( T_c >= 0_wp ) THEN
        e_i = 0.14_wp
      ELSEIF ( T_c >= -10_wp ) THEN
        e_i = -0.01_wp*(T_c+10_wp)+0.24_wp
      ELSEIF ( T_c >= -15_wp ) THEN
        e_i = -0.08_wp*(T_c+15_wp)+0.64_wp
      ELSEIF ( T_c >= -20_wp ) THEN
        e_i =  0.10_wp*(T_c+20_wp)+0.14_wp
      ELSEIF ( T_c >= -40_wp ) THEN
        e_i = 0.005_wp*(T_c+40_wp)+0.04_wp
      ELSE
        e_i =  0.04_wp
      END IF
    CASE (6)
      ! as option 5, but with factor 0.5, i.e., the lower range of Figure 14
      IF ( T_c >= 0_wp ) THEN
        e_i = 0.14_wp
      ELSEIF ( T_c >= -10_wp ) THEN
        e_i = -0.01_wp*(T_c+10_wp)+0.24_wp
      ELSEIF ( T_c >= -15_wp ) THEN
        e_i = -0.08_wp*(T_c+15_wp)+0.64_wp
      ELSEIF ( T_c >= -20_wp ) THEN
        e_i =  0.10_wp*(T_c+20_wp)+0.14_wp
      ELSEIF ( T_c >= -40_wp ) THEN
        e_i = 0.005_wp*(T_c+40_wp)+0.04_wp
      ELSE
        e_i =  0.04_wp
      END IF
      e_i = 0.5_wp * e_i
    CASE (7)
      e_i = 0.1_wp
    CASE (8)
      ! inspired by PK (option 4) and Connolly (option 5)
      IF ( T_c >= -1_wp ) THEN
        e_i = 1.0_wp
      ELSEIF (  T_c >= -6_wp ) THEN
        e_i = 0.19_wp*(T_c+6_wp)+0.24_wp
      ELSEIF (  T_c >= -10_wp ) THEN
        e_i = 0.24_wp
      ELSEIF (  T_c >= -12.5_wp ) THEN
        e_i = -0.304*(T_c+12.5_wp)+1.0_wp
      ELSEIF (  T_c >= -17_wp ) THEN
        e_i = 1.0_wp
      ELSEIF (  T_c >= -20_wp ) THEN
        e_i =  0.286666_wp*(T_c+20_wp)+0.14_wp
      ELSEIF (  T_c >= -40_wp ) THEN
        e_i = 0.005_wp*(T_c+40_wp)+0.04_wp
      ELSE
        e_i =  0.04_wp
      END IF
    CASE (9)
      ! as option 5, but with factor 0.75, i.e., the lower range of Figure 14
      IF ( T_c >= 0_wp ) THEN
        e_i = 0.14_wp
      ELSEIF ( T_c >= -10_wp ) THEN
        e_i = -0.01_wp*(T_c+10_wp)+0.24_wp
      ELSEIF ( T_c >= -15_wp ) THEN
        e_i = -0.08_wp*(T_c+15_wp)+0.64_wp
      ELSEIF ( T_c >= -20_wp ) THEN
        e_i =  0.10_wp*(T_c+20_wp)+0.14_wp
      ELSEIF ( T_c >= -40_wp ) THEN
        e_i = 0.005_wp*(T_c+40_wp)+0.04_wp
      ELSE
        e_i =  0.04_wp
      END IF
      e_i = 0.75_wp * e_i
    CASE (10)
      !.. Temperaturabhaengige Efficiency nach Cotton et al. (1986)
      !   (siehe auch Straka, 1989; S. 53)
      e_i = MIN(10**(0.035_wp*T_c-0.7_wp),0.2_wp)
    CASE default
      e_i = 1.0_wp
    END SELECT

    RETURN
  END FUNCTION e_stick

  SUBROUTINE init_estick_ltab_equi (ltab, istick, name)

    TYPE(lookupt_1D), INTENT(inout) :: ltab
    INTEGER, INTENT(in)             :: istick
    CHARACTER(len=*), INTENT(in)    :: name

    INTEGER  :: i, n
    REAL(wp) :: T_a
    CHARACTER(len=*), PARAMETER :: routine = TRIM(modname)//'::init_estick_ltab_equi'

    INTEGER, PARAMETER  :: ndT  = 801      ! Number of table nodes
    REAL(wp), PARAMETER :: Tmin = -70.0_wp ! Start T of table [deg C]
    REAL(wp), PARAMETER :: Tmax = 10.0_wp  ! End T of table [deg C]
    
    IF (.NOT. ltab%is_initialized) THEN

      ltab%name(:) = ' '
      ltab%name    = TRIM(name)
      ltab%iflag   = istick

      ltab%n1 = ndT
      ltab%dx1 = (Tmax - Tmin) / (ndT - 1.0_wp)
      ltab%odx1 = 1.0_wp / ltab%dx1

      NULLIFY (ltab%x1, ltab%ltable)
      
      ALLOCATE (ltab%x1(ltab%n1))
      ALLOCATE (ltab%ltable(ltab%n1))

      DO i = 1, ltab%n1
        T_a = Tmin + (i-1)*ltab%dx1 + T_3    ! [K]
        ltab%x1(i) = T_a                     ! equidistant table vector [K]
        ltab%ltable(i) = e_stick(T_a, istick)
      END DO
      
      ltab%is_initialized = .TRUE.

    ELSE

      IF (ltab%iflag /= istick) THEN
        ! conflicting initialization, don't know what to do:
        txt(:) = ' '
        WRITE(txt, '(a,i0,a,i0)') 'Conflicting re-initialization of LUT for '//TRIM(ltab%name)// &
             ', old istick=',ltab%iflag, ' / new istick=', istick
        CALL finish(TRIM(routine),TRIM(txt))
      END IF
    END IF

  END SUBROUTINE init_estick_ltab_equi

  FUNCTION estick_ltab_equi(T_a, ltab) RESULT (e_stick)

    !$ACC ROUTINE SEQ

    REAL(wp) :: e_stick
    REAL(wp), INTENT(in) :: T_a     ! ambient T [K]
    TYPE(lookupt_1D), INTENT(in) :: ltab

    INTEGER  :: iu, io
    REAL(wp) :: T_loc
    
    T_loc = MIN( MAX( T_a, ltab%x1(1)), ltab%x1(ltab%n1) )
    iu = MIN(FLOOR((T_loc - ltab%x1(1)) * ltab%odx1 ) + 1, ltab%n1-1)
    io = iu + 1

    e_stick = ltab%ltable(iu) + (ltab%ltable(io)-ltab%ltable(iu)) * ltab%odx1 * (T_loc-ltab%x1(iu))

  END FUNCTION estick_ltab_equi
  
  !*******************************************************************************
  ! 2D rational functions to evaluate bulk approximations                        *
  ! following Frick et al. (2013; cf. Eq. (31)), for n=2 and n=3                 *
  !*******************************************************************************

  REAL(wp) FUNCTION rat2do3(x,y,a,b)
    implicit none

    REAL(wp), INTENT(IN)                :: x,y
    REAL(wp), INTENT(IN), DIMENSION(10) :: a
    REAL(wp), INTENT(IN), DIMENSION(9)  :: b
    REAL(wp), PARAMETER :: eins = 1.0_wp
    REAL(wp)            :: p1,p2

    p1 = a(1)+a(2)*x+a(3)*y+a(4)*x*x+a(5)*x*y+a(6)*y*y &
         &   +a(7)*x*x*x+a(8)*x*x*y+a(9)*x*y*y+a(10)*y*y*y 
    p2 = eins+b(1)*x+b(2)*y+b(3)*x*x+b(4)*x*y+b(5)*y*y &
         &  + b(6)*x*x*x+b(7)*x*x*y+b(8)*x*y*y+b(9)*y*y*y 

    rat2do3 = p1/p2

    RETURN
  END FUNCTION rat2do3

  ELEMENTAL REAL(wp) FUNCTION dyn_visc_sutherland(Ta)
    !
    ! Calculate dynamic viscosity of air [kg m-1 s-1]
    ! following Sutherland's formula of an ideal
    ! gas with reference temp. T = 291.15 K
    !
    ! There is another alternative in P&K97 on
    ! page 417
    !
    IMPLICIT NONE
    REAL(wp), INTENT(in) :: Ta   ! ambient temp. [K]
    REAL(wp), PARAMETER :: &
         C = 120.d0      , &     ! Sutherland's constant (for air) [K]
         T0 = 291.15d0   , &     ! Reference temp. [K]
         eta0 = 1.827d-5         ! Reference dyn. visc. [kg m-1 s-1]
    REAL(wp) :: a, b

    a = T0 + C
    b = Ta + C
    dyn_visc_sutherland = eta0 * a/b * (Ta/T0)**(3.d0/2.d0)

    RETURN
  END FUNCTION dyn_visc_sutherland
  ! ---------------------------------------------------------------------
  ELEMENTAL REAL(wp) FUNCTION Dv_Rasmussen(Ta,pa)
    !
    ! Calculating the diffusivity of water vapor in air
    ! following Rasmussen et al. 1987, App. A, Tab. A1
    ! Changed: Units of D_v in m2 s-1
    !
    IMPLICIT NONE
    REAL(wp), INTENT(in) :: Ta, pa  ! Temp. and pressure in [K] and [Pa]
    REAL(wp), PARAMETER  :: p_0 = 1013.25e2_wp

    Dv_Rasmussen = 0.211d-4*(p_0/pa)*(Ta/T_3)**1.94
    RETURN
  END FUNCTION Dv_Rasmussen
  ! ---------------------------------------------------------------------
  ELEMENTAL REAL(wp) FUNCTION ka_Rasmussen(Ta)
    !
    ! Calculating the thermal conductivity of air
    ! following Rasmussen et al. 1987, App. A, Tab. A1
    !
    IMPLICIT NONE
    REAL(wp), INTENT(in)  :: Ta  ! ambient temp. [K]
    REAL(wp), PARAMETER :: &
         c_unit = 4.1840d2      ! for transforming units

    ! transform [cal cm-1 s-1 C-1] into [W m-1 K-1]
    ka_rasmussen = c_unit * (5.69 + 0.017*(Ta-T_3))*1.d-5
    RETURN
  END FUNCTION ka_Rasmussen
  ! ---------------------------------------------------------------------
  ELEMENTAL REAL(wp) FUNCTION lh_evap_RH87(T)
    !
    ! Calculating the latent heat of evaporation
    ! following the formulation of RH87a
    !
    IMPLICIT NONE
    REAL(wp), INTENT(in) :: T    ! ambient temp.
    REAL(wp) :: lh_e0, gam

    !.latent heat of evap. at T_3
    lh_e0 = 2.5008d6
    !.exponent for calculation
    gam = 0.167d0 + 3.67d-4 * T
    !.latent heat of evap. as a fct. of temp.
    lh_evap_RH87 = lh_e0 * (T_3 / T)**gam
    RETURN
  END FUNCTION lh_evap_RH87
  ! ---------------------------------------------------------------------
  ELEMENTAL REAL(wp) FUNCTION lh_melt_RH87(T)
    !
    ! Calculating the latent heat of melting
    ! following the formulation of RH87a
    !
    IMPLICIT NONE
    REAL(wp), INTENT(in) :: T    ! ambient temp.
    REAL(wp), PARAMETER :: &
         c_unit = 4.1840d3       ! constant to transform [cal g-1] to [J kg-1]

    !.latent heat of melt. as a fct. of temp.
    lh_melt_RH87 = c_unit * ( 79.7d0 + 0.485d0*(T-T_3) - 2.5d-3*(T-T_3)**2)
    RETURN
  END FUNCTION lh_melt_RH87

!==============================================================================

  REAL(wp) Function set_qnc(qc)

    !$ACC ROUTINE SEQ

    REAL(wp), INTENT(in)  :: qc  ! either [kg/kg] or [kg/m^3]
    REAL(wp), PARAMETER   :: Dmean = 10e-6_wp    ! Diameter of mean particle mass:

!    set_qnc = qc * 6.0_wp / (pi * rhoh2o * Dmean**3.0_wp)
    set_qnc = qc * 6.0_wp / (pi * rhoh2o * EXP(LOG(Dmean)*3.0_wp) )

  END FUNCTION set_qnc

  REAL(wp) Function set_qni(qi)

    !$ACC ROUTINE SEQ

    REAL(wp), INTENT(in)  :: qi  ! either [kg/kg] or [kg/m^3]

!    set_qni  = qi / 1e-10   !  qiin / ( ( Dmean / ageo) ** (1.0_wp / bgeo) )
    set_qni  = qi / 1e-10   !  qiin / ( exp(log(( Dmean / ageo)) * (1.0_wp / bgeo)) )
!     set_qni =  5.0E+0_wp * EXP(0.304_wp *  (T_3 - T))   ! FR: Cooper (1986) used by Greg Thompson(2008)
      
  END FUNCTION set_qni

  REAL(wp) Function set_qnr(qr)

    !$ACC ROUTINE SEQ

    REAL(wp), INTENT(in)  :: qr  ! has to be [kg/m^3]
    REAL(wp), PARAMETER   :: N0r = 8000.0e3_wp ! intercept of MP distribution

    !    set_qnr = N0r * ( qr * 6.0_wp / (pi * rhoh2o * N0r * gamma(4.0_wp)))**(0.25_wp)
    IF (qr >= 1e-20_wp) THEN
      set_qnr = N0r * EXP( LOG( qr * 6.0_wp / (pi * rhoh2o * N0r * GAMMA(4.0_wp))) * (0.25_wp) )
    ELSE
      set_qnr = 0.0_wp
    END IF

  END FUNCTION set_qnr

  REAL(wp) Function set_qns(qs)

    !$ACC ROUTINE SEQ

    REAL(wp), INTENT(in)  :: qs  ! has to be [kg/m^3]
    REAL(wp), PARAMETER   :: N0s = 800.0e3_wp
    REAL(wp), PARAMETER   :: ams = 0.038_wp  ! needs to be connected to snow-type
    REAL(wp), PARAMETER   :: bms = 2.0_wp

!    set_qns = N0s * ( qs / ( ams * N0s * gamma(bms+1.0_wp)))**( 1.0_wp/(1.0_wp+bms) )
    IF (qs >= 1e-20_wp) THEN
      set_qns = N0s * EXP( LOG( qs / ( ams * N0s * GAMMA(bms+1.0_wp))) * ( 1.0_wp/(1.0_wp+bms) ) )
    ELSE
      set_qns = 0.0_wp
    END IF
    
  END FUNCTION set_qns

  REAL(wp) Function set_qng(qg)

    !$ACC ROUTINE SEQ

    REAL(wp), INTENT(in)  :: qg  ! has to be [kg/m^3]
    REAL(wp), PARAMETER   :: N0g = 4000.0e3_wp
    REAL(wp), PARAMETER   :: amg = 169.6_wp     ! needs to be connected to graupel-type
    REAL(wp), PARAMETER   :: bmg = 3.1_wp

!    set_qng = N0g * ( qg / ( amg * N0g * gamma(bmg+1.0_wp)))**( 1.0_wp/(1.0_wp+bmg) )
    IF (qg >= 1e-20_wp) THEN
      set_qng = N0g * EXP( LOG ( qg / ( amg * N0g * GAMMA(bmg+1.0_wp))) * ( 1.0_wp/(1.0_wp+bmg) ) )
    ELSE
      set_qng = 0.0_wp
    END IF

  END FUNCTION set_qng

  FUNCTION set_qnh_Dmean(qh, rhobulk_hail, Dmean) RESULT (qnh)

    !$ACC ROUTINE SEQ

    REAL(wp), INTENT(in) :: qh           ! either [kg/kg] or [kg/m^3]
    REAL(wp), INTENT(in) :: rhobulk_hail ! assumed bulk density of hail [kg/m^3]
    REAL(wp), INTENT(in) :: Dmean        ! assumed mean mass diameter [m]

    REAL(wp) :: qnh

!    qnh = qh * 6.0_wp / (pi * rhobulk_hail * Dmean**3.0_wp)
    qnh = qh * 6.0_wp / (pi * rhobulk_hail * EXP(LOG(Dmean)*3.0_wp) )
    
  END FUNCTION set_qnh_Dmean

  FUNCTION set_qnh_expPSD_N0const(qh, rhobulk_hail, N0_h) RESULT (qnh)

    !$ACC ROUTINE SEQ

    ! .. Sets qnh based on assumption of an exponential PSD w.r.t. diameter D
    
    REAL(wp), INTENT(in) :: qh           ! has to be [kg/m^3] because of N0 held constant
    REAL(wp), INTENT(in) :: rhobulk_hail ! assumed bulk density of hail [kg/m^3]
    REAL(wp), INTENT(in) :: N0_h         ! assumed constant N0-parameter of expon. size distrib. [1/m^4]

    REAL(wp) :: qnh
 
!    set_qnh = N0_h * ( qh / ( pi * rhobulk_hail * N0_h) )**(0.25)
    IF (qh >= 1e-20_wp) THEN
      qnh = N0_h * EXP( LOG ( qh / ( pi * rhobulk_hail * N0_h) ) * ( 0.25_wp ) )
    ELSE
      qnh = 0.0_wp
    END IF
    
  END FUNCTION set_qnh_expPSD_N0const

 
END MODULE mo_2mom_mcrph_util
