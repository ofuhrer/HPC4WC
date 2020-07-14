! ******************************************************
!     Program: baseline_fortran
!      Author: Ulrike Proske
!        Date: 14.07.2020
! Description: generic_filter-np.nanmean example in fortran
! ******************************************************

program main
    
    implicit none

    integer, parameter :: wp = 4
    integer, parameter :: nhalo = 2

    integer :: nvar, nx, ny, nz
    integer :: ivar

    real(kind=wp), allocatable :: in_field(:,:,:,:)
    real(kind=wp), allocatable :: out_field(:,:,:,:)
    real(kind=wp), allocatable :: weights(:,:,:,:)
    real(kind=wp), allocatable :: mask(:,:,:,:)

    nvar = 3
    nx = 30
    ny = 30
    nz = 30

    CALL setup()

    ! TODO: start timer 
    do ivar = 1, nvar
        CALL weightsum(mask(ivar,:,:,:), weights(ivar,:,:,:))
        CALL nanmean(in_field(ivar,:,:,:), weights(ivar,:,:,:), mask(ivar,:,:,:), out_field(ivar,:,:,:))
    end do
    ! TODO: stop timer

contains

    subroutine weightsum(mask, weights)
        implicit none

        ! argument
        real(kind=wp), intent(in) :: mask(:,:,:)
        real(kind=wp), intent(inout) :: weights(:,:,:)

        ! local
        integer :: i, j, k

        do k = 1+nhalo, nz-nhalo
        do j = 1+nhalo, ny-nhalo
        do i = 1+nhalo, nx-nhalo
            CALL weights_stencil(mask(i-2:i+2,j-2:j+2,k-2:k+2), weights(i,j,k))
        end do
        end do
        end do

    end subroutine weightsum

    subroutine nanmean(in_field, weights, mask, out_field)
        implicit none

        ! argument
        real(kind=wp), intent(in) :: in_field(:,:,:)
        real(kind=wp), intent(in) :: mask(:,:,:)
        real(kind=wp), intent(in) :: weights(:,:,:)
        real(kind=wp), intent(inout) :: out_field(:,:,:)

        ! local
        integer :: i, j, k

        do k = 1+nhalo, nz-nhalo
        do j = 1+nhalo, ny-nhalo
        do i = 1+nhalo, nx-nhalo
            CALL nanmean_stencil(in_field(i-2:i+2,j-2:j+2,k-2:k+2), weights(i-2:i+2,j-2:j+2,k-2:k+2), out_field(i,j,k))
        end do
        end do
        end do


    end subroutine nanmean


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
        allocate( mask(nvar, nx, ny, nz))
        mask = 0.0_wp
        do k = 1, nz
        do j = 1, ny
        do i = 1, nx
        do h = 1, nvar
            in_field(h, i, j, k) = (h+k)/(i+j+1) ! random filling
            if (in_field(h,i,j,k) < 0.5) mask(h,i,j,k) = 1.0_wp
        end do
        end do
        end do
        end do

        allocate( out_field(nvar, nx, ny, nz) )
        out_field = in_field

        allocate( weights(nvar, nx, ny, nz) )
        weights = 0.0_wp


        write(*,*) 'done'

    end subroutine setup

    subroutine nanmean_stencil(v, w, outvalue)

            implicit none

            real (kind=wp), intent(in) :: v(5,5,5)
            real (kind=wp), intent(in) :: w(5,5,5)
            real (kind=wp), intent(inout) :: outvalue

            outvalue = &

            (v[2,3,3] * w[2,3,3] + v[4,3,3] * w[4,3,3] + ! middle 2d slice 1 point each dir
            v[3,2,3] * w[3,2,3] + v[3,4,3] * w[3,4,3] +
            v[2,2,3] * w[2,2,3] + v[4,4,3] * w[4,4,3] +
            v[2,4,3] * w[2,4,3] + v[4,2,3] * w[4,2,3] +

            v[2,3,4] * w[2,3,4] + v[4,3,4] * w[4,3,4] + ! 2d slice before incl middle
            v[3,2,4] * w[3,2,4] + v[3,4,4] * w[3,4,4] +
            v[2,2,4] * w[2,2,4] + v[4,4,4] * w[4,4,4] +
            v[2,4,4] * w[2,4,4] + v[4,2,4] * w[4,2,4] +
            v[3,3,4] * w[3,3,4] +

            v[2,3,2] * w[2,3,2] + v[4,3,2] * w[4,3,2] + ! 2d slice after incl middle
            v[3,2,2] * w[3,2,2] + v[3,4,2] * w[3,4,2] +
            v[2,2,2] * w[2,2,2] + v[4,4,2] * w[4,4,2] +
            v[2,4,2] * w[2,4,2] + v[4,2,2] * w[4,2,2] +
            v[3,3,2] * w[3,3,2] +

            v[1,3,3] * w[1,3,3] + v[5,3,3] * w[5,3,3] + ! middle 2d slice 2 points l&r
            v[1,2,3] * w[1,2,3] + v[5,2,3] * w[5,2,3] +
            v[1,1,3] * w[1,1,3] + v[5,1,3] * w[5,1,3] +
            v[1,4,3] * w[1,4,3] + v[5,4,3] * w[5,4,3] +
            v[1,5,3] * w[1,5,3] + v[5,5,3] * w[5,5,3] +

            ! TODO: continue modifying for fortran syntax here (also echange [] for ())
            v[0,-2,0] * w[0,-2,0] + v[0,+2,0] * w[0,+2,0] + ! middle 2d slice 2 points u&d
            v[-1,-2,0] * w[-1,-2,0] + v[-1,+2,0] * w[-1,+2,0] +
            v[1,-2,0] * w[1,-2,0] + v[1,+2,0] * w[1,+2,0] +
        ! TODO: maybe there's something missing here?

            v[-2,0,1] * w[-2,0,1] + v[+2,0,1] * w[+2,0,1] + ! 2d slice before 2 points l&r
            v[-2,-1,1] * w[-2,-1,1] + v[+2,-1,1] * w[+2,-1,1] +
            v[-2,-2,1] * w[-2,-2,1] + v[+2,-2,1] * w[+2,-2,1] +
            v[-2,1,1] * w[-2,1,1] + v[+2,1,1] * w[+2,1,1] +
            v[-2,2,1] * w[-2,2,1] + v[+2,2,1] * w[+2,2,1] +

            v[0,-2,1] * w[0,-2,1] + v[0,+2,1] * w[0,+2,1] + ! 2d slice before 2 points u&d
            v[-1,-2,1] * w[-1,-2,1] + v[-1,+2,1] * w[-1,+2,1] +
            v[1,-2,1] * w[1,-2,1] + v[1,+2,1] * w[1,+2,1] +

            v[-2,0,-1] * w[-2,0,-1] + v[+2,0,-1] * w[+2,0,-1] + ! 2d slice after 2 points l&r
            v[-2,-1,-1] * w[-2,-1,-1] + v[+2,-1,-1] * w[+2,-1,-1] +
            v[-2,-2,-1] * w[-2,-2,-1] + v[+2,-2,-1] * w[+2,-2,-1] +
            v[-2,1,-1] * w[-2,1,-1] + v[+2,1,-1] * w[+2,1,-1] +
            v[-2,2,-1] * w[-2,2,-1] + v[+2,2,-1] * w[+2,2,-1] +

            v[0,-2,-1] * w[0,-2,-1] + v[0,+2,-1] * w[0,+2,-1] + ! 2d slice after 2 points u&d
            v[-1,-2,-1] * w[-1,-2,-1] + v[-1,+2,-1] * w[-1,+2,-1] +
            v[1,-2,-1] * w[1,-2,-1] + v[1,+2,-1] * w[1,+2,-1] +

            v[-2,-2,-2] * w[-2,-2,-2] +
            v[-2,-1,-2] * w[-2,-1,-2] +
            v[-2,0,-2] * w[-2,0,-2] +
            v[-2,+1,-2] * w[-2,+1,-2] +
            v[-2,+2,-2] * w[-2,+2,-2] +

            v[-1,-2,-2] * w[-1,-2,-2] +
            v[-1,-1,-2] * w[-1,-1,-2] +
            v[-1,0,-2] * w[-1,0,-2] +
            v[-1,+1,-2] * w[-1,+1,-2] +
            v[-1,+2,-2] * w[-1,+2,-2] +

            v[0,-2,-2] * w[0,-2,-2] +
            v[0,-1,-2] * w[0,-1,-2] +
            v[0,0,-2] * w[0,0,-2] +
            v[0,+1,-2] * w[0,+1,-2] +
            v[0,+2,-2] * w[0,+2,-2] +

            v[+1,-2,-2] * w[+1,-2,-2] +
            v[+1,-1,-2] * w[+1,-1,-2] +
            v[+1,0,-2] * w[+1,0,-2] +
            v[+1,+1,-2] * w[+1,+1,-2] +
            v[+1,+2,-2] * w[+1,+2,-2] +

            v[+2,-2,-2] * w[+2,-2,-2] +
            v[+2,-1,-2] * w[+2,-1,-2] +
            v[+2,0,-2] * w[+2,0,-2] +
            v[+2,+1,-2] * w[+2,+1,-2] +
            v[+2,+2,-2] * w[+2,+2,-2] +

            v[-2,-2,2] * w[-2,-2,2] +
            v[-2,-1,2] * w[-2,-1,2] +
            v[-2,0,2] * w[-2,0,2] +
            v[-2,+1,2] * w[-2,+1,2] +
            v[-2,+2,2] * w[-2,+2,2] +

            v[-1,-2,2] * w[-1,-2,2] +
            v[-1,-1,2] * w[-1,-1,2] +
            v[-1,0,2] * w[-1,0,2] +
            v[-1,+1,2] * w[-1,+1,2] +
            v[-1,+2,2] * w[-1,+2,2] +

            v[0,-2,2] * w[0,-2,2] +
            v[0,-1,2] * w[0,-1,2] +
            v[0,0,2] * w[0,0,2] +
            v[0,+1,2] * w[0,+1,2] +
            v[0,+2,2] * w[0,+2,2] +

            v[+1,-2,2] * w[+1,-2,2] +
            v[+1,-1,2] * w[+1,-1,2] +
            v[+1,0,2] * w[+1,0,2] +
            v[+1,+1,2] * w[+1,+1,2] +
            v[+1,+2,2] * w[+1,+2,2] +

            v[+2,-2,2] * w[+2,-2,2] +
            v[+2,-1,2] * w[+2,-1,2] +
            v[+2,0,2] * w[+2,0,2] +
            v[+2,+1,2] * w[+2,+1,2] +
            v[+2,+2,2] * w[+2,+2,2]) / max(w(3,3,3), 1)
    end subroutine

    subroutine weights_stencil(w, outvalue)

            implicit none

            real (kind=wp), intent(in) :: w(5,5,5)
            real (kind=wp), intent(inout) :: outvalue

            outvalue = &
            w(1,1,5) + w(1,2,5) + w(1,3,5) + w(1,4,5) + w(1,5,5) + &
            w(2,1,5) + w(2,2,5) + w(2,3,5) + w(2,4,5) + w(2,5,5) + &
            w(3,1,5) + w(3,2,5) + w(3,3,5) + w(3,4,5) + w(3,5,5) + &
            w(4,1,5) + w(4,2,5) + w(4,3,5) + w(4,4,5) + w(4,5,5) + &
            w(5,1,5) + w(5,2,5) + w(5,3,5) + w(5,4,5) + w(5,5,5) + &

            w(1,1,4) + w(1,2,4) + w(1,3,4) + w(1,4,4) + w(1,5,4) + &
            w(2,1,4) + w(2,2,4) + w(2,3,4) + w(2,4,4) + w(2,5,4) + &
            w(3,1,4) + w(3,2,4) + w(3,3,4) + w(3,4,4) + w(3,5,4) + &
            w(4,1,4) + w(4,2,4) + w(4,3,4) + w(4,4,4) + w(4,5,4) + &
            w(5,1,4) + w(5,2,4) + w(5,3,4) + w(5,4,4) + w(5,5,4) + &

            w(1,1,3) + w(1,2,3) + w(1,3,3) + w(1,4,3) + w(1,5,3) + &
            w(2,1,3) + w(2,2,3) + w(2,3,3) + w(2,4,3) + w(2,5,3) + &
            w(3,1,3) + w(3,2,3) + w(3,4,3) + w(3,5,3) + & ! w/o point itself!
            w(4,1,3) + w(4,2,3) + w(4,3,3) + w(4,4,3) + w(4,5,3) + &
            w(5,1,3) + w(5,2,3) + w(5,3,3) + w(5,4,3) + w(5,5,3) + &

            w(1,1,2) + w(1,2,2) + w(1,3,2) + w(1,4,2) + w(1,5,2) + &
            w(2,1,2) + w(2,2,2) + w(2,3,2) + w(2,4,2) + w(2,5,2) + &
            w(3,1,2) + w(3,2,2) + w(3,3,2) + w(3,4,2) + w(3,5,2) + &
            w(4,1,2) + w(4,2,2) + w(4,3,2) + w(4,4,2) + w(4,5,2) + &
            w(5,1,2) + w(5,2,2) + w(5,3,2) + w(5,4,2) + w(5,5,2) + &

            w(1,1,1) + w(1,2,1) + w(1,3,1) + w(1,4,1) + w(1,5,1) + &
            w(2,1,1) + w(2,2,1) + w(2,3,1) + w(2,4,1) + w(2,5,1) + &
            w(3,1,1) + w(3,2,1) + w(3,3,1) + w(3,4,1) + w(3,5,1) + &
            w(4,1,1) + w(4,2,1) + w(4,3,1) + w(4,4,1) + w(4,5,1) + &
            w(5,1,1) + w(5,2,1) + w(5,3,1) + w(5,4,1) + w(5,5,1)

    end subroutine

end program main
