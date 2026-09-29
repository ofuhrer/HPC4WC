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

! Module determines kinds for different precisions
! Number model from which the SELECTED_*\\_KIND are requested: <br>
!
! @f{tabular}{{r@{\hspace*{3em}}c@{\hspace*{3em}}c}
!                     &4 byte REAL     &8 byte REAL        \\\
!        CRAY:        &-               &precision =   13   \\\
!                     &                &exponent  = 2465   \\\
!        IEEE:        &precision = 6   &precision =   15   \\\
!                     &exponent  = 37  &exponent  =  307
! @f}
! \\medskip
!
!  Most likely this are the only possible models.

MODULE mo_kind

  IMPLICIT NONE

  PRIVATE

  !--------------------------------------------------------------------
  !
  ! Floating point section
  ! ----------------------
  !
  INTEGER, PARAMETER :: ps =   6
  INTEGER, PARAMETER :: rs =  37
  !
  INTEGER, PARAMETER :: pd =  12
  INTEGER, PARAMETER :: rd = 307
  !
  INTEGER, PARAMETER :: sp = SELECTED_REAL_KIND(ps,rs) !< single precision
  INTEGER, PARAMETER :: dp = SELECTED_REAL_KIND(pd,rd) !< double precision
  !
  INTEGER, PARAMETER :: qp = SELECTED_REAL_KIND(32)
  INTEGER, PARAMETER :: wp = dp                        !< selected working precision
  !
#ifdef __MIXED_PRECISION
  INTEGER, PARAMETER :: vp = sp
#else
  INTEGER, PARAMETER :: vp = wp
#endif


  !
  ! Integer section
  ! ---------------
  !
  INTEGER, PARAMETER :: pi1 =  2
  INTEGER, PARAMETER :: pi2 =  4
  INTEGER, PARAMETER :: pi4 =  9
  INTEGER, PARAMETER :: pi8 = 14  ! could be larger, but SX cannot do some operations otherwise
  !
  INTEGER, PARAMETER :: i1 = SELECTED_INT_KIND(pi1)   !< at least 1 byte integer
  INTEGER, PARAMETER :: i2 = SELECTED_INT_KIND(pi2)   !< at least 2 byte integer
  INTEGER, PARAMETER :: i4 = SELECTED_INT_KIND(pi4)   !< at least 4 byte integer
  INTEGER, PARAMETER :: i8 = SELECTED_INT_KIND(pi8)   !< at least 8 byte integer
  !
  !
  ! The following variable is made available internally only. configure needs to detect
  ! the addressing mode and according to this mo_kind has to be updated by an preprocessor
  ! directive and #include <config.h>. This needs some changes.
  !
  INTEGER, PARAMETER :: wi = i4                       !< selected working precission
  !
  PUBLIC :: sp, dp, wp, vp, i1, i2, i4, i8
  !
  !--------------------------------------------------------------------

END MODULE mo_kind

