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

! Constants used for the computation of lookup tables of the saturation
! mixing ratio over liquid water (*c_les*) or ice(*c_ies*)

MODULE mo_lookup_tables_constants

  USE, INTRINSIC :: iso_fortran_env, ONLY: wp => real64
  USE mo_physical_constants, ONLY: alv, als, cpd, rd, rv, tmelt


  IMPLICIT NONE

  PRIVATE

  PUBLIC :: c1es, c2es, c3les, c3ies, c4les, c4ies, c5les, c5ies, &
    &       c5alvcp, c5alscp, alvdcp, alsdcp

  REAL (wp), PARAMETER :: c1es  = 610.78_wp              !
  REAL (wp), PARAMETER :: c2es  = c1es*rd/rv             !
  REAL (wp), PARAMETER :: c3les = 17.269_wp              !
  REAL (wp), PARAMETER :: c3ies = 21.875_wp              !
  REAL (wp), PARAMETER :: c4les = 35.86_wp               !
  REAL (wp), PARAMETER :: c4ies = 7.66_wp                !
  REAL (wp), PARAMETER :: c5les = c3les*(tmelt-c4les)    !
  REAL (wp), PARAMETER :: c5ies = c3ies*(tmelt-c4ies)    !
  REAL (wp), PARAMETER :: c5alvcp = c5les*alv/cpd        !
  REAL (wp), PARAMETER :: c5alscp = c5ies*als/cpd        !
  REAL (wp), PARAMETER :: alvdcp  = alv/cpd              !
  REAL (wp), PARAMETER :: alsdcp  = als/cpd              !
  !$ACC DECLARE COPYIN(c1es, c2es, c3les, c3ies, c4les, c4ies, c5les, c5ies) &
  !$ACC   COPYIN(c5alvcp, c5alscp, alvdcp, alsdcp)

END MODULE mo_lookup_tables_constants
