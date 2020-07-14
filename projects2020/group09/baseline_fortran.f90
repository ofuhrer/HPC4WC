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

    integer :: start_time, stop_time
    real :: count_rate

    real(kind=wp), allocatable :: in_field(:,:,:,:)
    real(kind=wp), allocatable :: out_field(:,:,:,:)
    real(kind=wp), allocatable :: weights(:,:,:,:)
    real(kind=wp), allocatable :: mask(:,:,:,:)

    nvar = 3
    nx = 30
    ny = 720
    nz = 1440

    CALL setup()

    ! start timer 
    call system_clock(start_time, count_rate)

    do ivar = 1, nvar
        CALL weightsum(mask(ivar,:,:,:), weights(ivar,:,:,:))
        CALL nanmean(in_field(ivar,:,:,:), weights(ivar,:,:,:), mask(ivar,:,:,:), out_field(ivar,:,:,:))
        write(*,*) 'ivar = ', ivar
    end do
    
    ! stop timer
    call system_clock(stop_time)

    write(*,*) 'This function took ', (stop_time - start_time)/count_rate, "seconds"

contains

    subroutine weightsum(mask, weights)
        ! sum surrounding mask points into the weights
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
        ! loop over all points of a certain var-field and compute generic_filter for each
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
            CALL nanmean_stencil(in_field(i-2:i+2,j-2:j+2,k-2:k+2), mask(i-2:i+2,j-2:j+2,k-2:k+2), weights(i,j,k), out_field(i,j,k))
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

    subroutine nanmean_stencil(v, w, wvalue, outvalue)

            implicit none

            real (kind=wp), intent(in) :: v(5,5,5)
            real (kind=wp), intent(in) :: w(5,5,5) ! mask/weights surrounding that point
            real (kind=wp), intent(in) :: wvalue ! sum of weights at this point (masks surrounding this point)
            real (kind=wp), intent(inout) :: outvalue

            outvalue = &

            (v(2,3,3) * w(2,3,3) + v(4,3,3) * w(4,3,3) + & ! middle 2d slice 1 point each dir
            v(3,2,3) * w(3,2,3) + v(3,4,3) * w(3,4,3) + &
            v(2,2,3) * w(2,2,3) + v(4,4,3) * w(4,4,3) + &
            v(2,4,3) * w(2,4,3) + v(4,2,3) * w(4,2,3) + &

            v(2,3,4) * w(2,3,4) + v(4,3,4) * w(4,3,4) +  &! 2d slice before incl middle
            v(3,2,4) * w(3,2,4) + v(3,4,4) * w(3,4,4) + &
            v(2,2,4) * w(2,2,4) + v(4,4,4) * w(4,4,4) + &
            v(2,4,4) * w(2,4,4) + v(4,2,4) * w(4,2,4) + &
            v(3,3,4) * w(3,3,4) + &

            v(2,3,2) * w(2,3,2) + v(4,3,2) * w(4,3,2) +  &! 2d slice after incl middle
            v(3,2,2) * w(3,2,2) + v(3,4,2) * w(3,4,2) + &
            v(2,2,2) * w(2,2,2) + v(4,4,2) * w(4,4,2) + &
            v(2,4,2) * w(2,4,2) + v(4,2,2) * w(4,2,2) + &
            v(3,3,2) * w(3,3,2) + &

            v(1,3,3) * w(1,3,3) + v(5,3,3) * w(5,3,3) + & ! middle 2d slice 2 points l&r
            v(1,2,3) * w(1,2,3) + v(5,2,3) * w(5,2,3) + &
            v(1,1,3) * w(1,1,3) + v(5,1,3) * w(5,1,3) + &
            v(1,4,3) * w(1,4,3) + v(5,4,3) * w(5,4,3) + &
            v(1,5,3) * w(1,5,3) + v(5,5,3) * w(5,5,3) + &

            v(3,1,3) * w(3,1,3) + v(3,5,3) * w(3,5,3) +  &! middle 2d slice 2 points u&d
            v(2,1,3) * w(2,1,3) + v(2,5,3) * w(2,5,3) + &
            v(4,1,3) * w(4,1,3) + v(4,5,3) * w(4,5,3) + &
        ! TODO: maybe there's something missing here? But also missing in Verena's version

            v(1,3,4) * w(1,3,4) + v(5,3,4) * w(5,3,4) +  &! 2d slice before 2 points l&r
            v(1,2,4) * w(1,2,4) + v(5,2,4) * w(5,2,4) + &
            v(1,1,4) * w(1,1,4) + v(5,1,4) * w(5,1,4) + &
            v(1,4,4) * w(1,4,4) + v(5,4,4) * w(5,4,4) + &
            v(1,5,4) * w(1,5,4) + v(5,5,4) * w(5,5,4) + &

            v(3,1,4) * w(3,1,4) + v(3,5,4) * w(3,5,4) +  &! 2d slice before 2 points u&d
            v(2,1,4) * w(2,1,4) + v(2,5,4) * w(2,4,4) + &
            v(4,1,4) * w(4,1,4) + v(4,5,4) * w(4,5,4) + &

            v(1,3,2) * w(1,3,2) + v(5,3,2) * w(5,3,2) +  &! 2d slice after 2 points l&r
            v(1,2,2) * w(1,2,2) + v(5,2,2) * w(5,2,2) + &
            v(1,1,2) * w(1,1,2) + v(5,1,2) * w(5,1,2) + &
            v(1,4,2) * w(1,4,2) + v(5,4,2) * w(5,4,2) + &
            v(1,5,2) * w(1,5,2) + v(5,5,2) * w(5,5,2) + &

            v(3,1,2) * w(3,1,2) + v(3,5,2) * w(3,5,2) +  &! 2d slice after 2 points u&d
            v(2,1,2) * w(2,1,2) + v(2,5,2) * w(2,5,2) + &
            v(4,1,2) * w(4,1,2) + v(4,5,2) * w(4,5,2) + &

            v(1,1,1) * w(1,1,1) +  &
            v(1,2,1) * w(1,2,1) + &
            v(1,3,1) * w(1,3,1) + &
            v(1,4,1) * w(1,4,1) + &
            v(1,5,1) * w(1,5,1) + &

            v(2,1,1) * w(2,1,1) + &
            v(2,2,1) * w(2,2,1) + &
            v(2,3,1) * w(2,3,1) + &
            v(2,4,1) * w(2,4,1) + &
            v(2,5,1) * w(2,5,1) + &

            v(3,1,1) * w(3,1,1) + &
            v(3,2,1) * w(3,2,1) + &
            v(3,3,1) * w(3,3,1) + &
            v(3,4,1) * w(3,4,1) + &
            v(3,5,1) * w(3,5,1) + &

            v(4,1,1) * w(4,1,1) + &
            v(4,2,1) * w(4,2,1) + &
            v(4,3,1) * w(4,3,1) + &
            v(4,4,1) * w(4,4,1) + &
            v(4,5,1) * w(4,5,1) + &

            v(5,1,1) * w(5,1,1) + &
            v(5,2,1) * w(5,2,1) + &
            v(5,3,1) * w(5,3,1) + &
            v(5,4,1) * w(5,4,1) + &
            v(5,5,1) * w(5,5,1) + &

            v(1,1,2) * w(1,1,2) + &
            v(1,2,2) * w(1,2,2) + &
            v(1,3,2) * w(1,3,2) + &
            v(1,4,2) * w(1,4,2) + &
            v(1,5,2) * w(1,5,2) + &

            v(2,1,2) * w(2,1,2) + &
            v(2,2,2) * w(2,2,2) + &
            v(2,3,2) * w(2,3,2) + &
            v(2,4,2) * w(2,4,2) + &
            v(2,5,2) * w(2,5,2) + &

            v(3,1,2) * w(3,1,2) + &
            v(3,2,2) * w(3,2,2) + &
            v(3,3,2) * w(3,3,2) + &
            v(3,4,2) * w(3,4,2) + &
            v(3,5,2) * w(3,5,2) + &

            v(4,1,2) * w(4,1,2) + &
            v(4,2,2) * w(4,2,2) + &
            v(4,3,2) * w(4,3,2) + &
            v(4,4,2) * w(4,4,2) + &
            v(4,5,2) * w(4,5,2) + &

            v(5,1,2) * w(5,1,2) + &
            v(5,2,2) * w(5,2,2) + &
            v(5,3,2) * w(5,3,2) + &
            v(5,4,2) * w(5,4,2) + &
            v(5,5,2) * w(5,5,2) ) / max(wvalue, 1.0_wp)
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
