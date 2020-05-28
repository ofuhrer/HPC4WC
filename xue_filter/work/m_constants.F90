! ******************************************************
!      Module: m_constants
!      Author: Oliver Fuhrer
!       Email: oliverf@vulcan.com
!        Date: 09.09.2006
! Description: Definitions of numerical, fortran,
!              mathematical and physical constants. A
!              set of basic physical constants is read
!              from a namelist. Others are derived from
!              this basic set and computed here.
! ******************************************************

module m_constants
    implicit none

    ! version
    character(len=4) :: version='v1_0'
  
    ! numerical precision constants
    integer, parameter :: wp = SELECTED_REAL_KIND (12,200)
    real (kind=wp), parameter :: epsilon = 1.0d-12
  
    ! fortran constants
    integer, parameter :: stderr = 0
    integer, parameter :: stdin = 5
    integer, parameter :: stdout = 6
  
    ! mathematical constants
    real (kind=wp), parameter :: pi = 3.14159265358979323846264338_wp
    real (kind=wp), parameter :: zero = 0.0_wp
    real (kind=wp), parameter :: one = 1.0_wp
    real (kind=wp), parameter :: two = 2.0_wp
    real (kind=wp), parameter :: half = 0.5_wp
    real (kind=wp), parameter :: third = 1.0_wp/3.0_wp
    real (kind=wp), parameter :: fourth = 0.25_wp
  
end module m_constants
