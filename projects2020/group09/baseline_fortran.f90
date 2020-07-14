! ******************************************************
!     Program: baseline_fortran
!      Author: Ulrike Proske
!        Date: 14.07.2020
! Description: generic_filter-np.nanmean example in fortran
! ******************************************************

program main
    
    implicit none

    integer, parameter :: wp = 4

    integer :: nvar, nx, ny, nz

    real(kind=wp), allocatable :: in_field(:,:,:,:)
    real(kind=wp), allocatable :: out_field(:,:,:,:)

    CALL setup()

contains

    ! setup everything before work
    ! (init timers, allocate memory, initialize fields)
    subroutine setup()
        !use m_utils, only : timer_init
        implicit none

        ! local
        integer :: h, i, j, k

        !call timer_init()

        allocate( in_field(nvar, nx, ny, nz) )
        in_field = 1.0_wp
        do k = 1, nz
        do j = 1, ny
        do i = 1, nx
        do h = 1, nvar
            in_field(h, i, j, k) = (h+k)/(i+j+1) ! random filling
        end do
        end do
        end do
        end do

        allocate( out_field(nvar, nx, ny, nz) )
        out_field = in_field

        write(*,*) 'done'

    end subroutine setup

end program main
