! ******************************************************
!     Program: baseline_fortran
!      Author: Ulrike Proske
!        Date: 14.07.2020
! Description: generic_filter-np.nanmean example in fortran
! ******************************************************

! To compile with netcdf:
! module load netcdf
! gfortran -I /usr/local/netcdf-4.6.1-4.4.4-gnu-7.4.1/include/ -L /usr/local/netcdf-4.6.1-4.4.4-gnu-7.4.1/lib/ -lnetcdf -lnetcdff baseline_fortran.f90

program main
    
    USE netcdf 

    implicit none

    integer, parameter :: wp = 4
    integer, parameter :: nhalo = 2

    integer :: nvar, nx, ny, nz, blockx, blocky, blockz
    integer :: ivar

    integer :: start_time, stop_time
    integer :: count_rate

    real(kind=wp), allocatable :: in_field(:,:,:,:)
    real(kind=wp), allocatable :: out_field(:,:,:,:)
    real(kind=wp), allocatable :: weights(:,:,:,:)
    real(kind=wp), allocatable :: mask(:,:,:,:)

    character(len=7) :: outfile

    integer, parameter :: lblocking = 1 ! 1; blocking with blocks as specified below, 0: no blocking
    integer, parameter :: lwriteout = 0 ! 1: write out the fields (careful! do that only when they are not too large)
    ! 0: don't write any netcdf files

    nvar = 3
    nx = 3653
    ny = 1440
    nz = 720
    blockx = 100
    blocky = 100
    blockz = 100

    CALL setup()

    !WRITE(*,*) in_field

    if (lwriteout == 1) then
            outfile = 'foin.nc'
            CALL writegrid(outfile,in_field)

            outfile = 'fmas.nc'
            CALL writegrid(outfile,mask)
    end if

    ! start timer 
    write(*,*) 'Start timer'
    call system_clock(start_time, count_rate)

    do ivar = 1, nvar
        if (lblocking == 1) then
            CALL weightsum_blocking(mask(ivar,:,:,:), weights(ivar,:,:,:))
            CALL nanmean_blocking(in_field(ivar,:,:,:), weights(ivar,:,:,:), mask(ivar,:,:,:), out_field(ivar,:,:,:))
        else
            CALL weightsum(mask(ivar,:,:,:), weights(ivar,:,:,:))
            CALL nanmean(in_field(ivar,:,:,:), weights(ivar,:,:,:), mask(ivar,:,:,:), out_field(ivar,:,:,:))
        end if
        write(*,*) 'ivar = ', ivar
    end do

    ! stop timer
    write(*,*) 'Stop timer'
    call system_clock(stop_time)
    
    if (lwriteout == 1) then
            outfile = 'fwei.nc'
            CALL writegrid(outfile,weights)

            outfile = 'fout.nc'
            CALL writegrid(outfile,out_field)
    end if

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

    subroutine weightsum_blocking(mask, weights)
        ! sum surrounding mask points into the weights
        implicit none

        ! argument
        real(kind=wp), intent(in) :: mask(:,:,:)
        real(kind=wp), intent(inout) :: weights(:,:,:)

        ! local
        integer :: i, j, k, block_i, block_j, block_k, k_local, j_local, i_local

        do block_k = 0, nz/blockz+1
        do block_j = 0, ny/blocky+1
        do block_i = 0, nx/blockx+1
        do k_local = 1, blockz
        do j_local = 1, blocky
        do i_local = 1, blockx
            k = blockz*block_k + k_local
            j = blocky*block_j + j_local
            i = blockx*block_i + i_local
            if (k < 1+nhalo .or. k > nz-nhalo .or. j < 1+nhalo .or. j > ny-nhalo .or. i < 1+nhalo .or. i > nx-nhalo) then
               cycle
            end if
            CALL weights_stencil(mask(i-2:i+2,j-2:j+2,k-2:k+2), weights(i,j,k))
            !write(*,*) weights(i,j,k)
        end do
        end do
        end do
        end do
        end do
        end do

    end subroutine weightsum_blocking

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

    subroutine nanmean_blocking(in_field, weights, mask, out_field)
        ! loop over all points of a certain var-field and compute generic_filter for each
        implicit none

        ! argument
        real(kind=wp), intent(in) :: in_field(:,:,:)
        real(kind=wp), intent(in) :: mask(:,:,:)
        real(kind=wp), intent(in) :: weights(:,:,:)
        real(kind=wp), intent(inout) :: out_field(:,:,:)

        ! local
        integer :: i, j, k, block_i, block_j, block_k, k_local, j_local, i_local

        do block_k = 0, nz/blockz+1 ! for decimals that cut off a block +1
        do block_j = 0, ny/blocky+1 ! TODO: this could probably be done in a smarter way
        do block_i = 0, nx/blockx+1 ! but looping over too many should not be very expensive
        do k_local = 1, blockz
        do j_local = 1, blocky
        do i_local = 1, blockx
            k = blockz*block_k + k_local
            j = blocky*block_j + j_local
            i = blockx*block_i + i_local
            !write(*,*) i,j,k,1+nhalo,nz-nhalo,ny-nhalo,nx-nhalo
            if (k < 1+nhalo .or. k > nz-nhalo .or. j < 1+nhalo .or. j > ny-nhalo .or. i < 1+nhalo .or. i > nx-nhalo) then
               cycle
            end if
            !write(*,*) 'b'
            !write(*,*) weights(i,j,k)
            CALL nanmean_stencil(in_field(i-2:i+2,j-2:j+2,k-2:k+2), mask(i-2:i+2,j-2:j+2,k-2:k+2), weights(i,j,k), out_field(i,j,k))
        end do
        end do
        end do
        end do
        end do
        end do
    end subroutine nanmean_blocking


    ! setup everything before work
    ! (init timers, allocate memory, initialize fields)
    subroutine setup()
        !use m_utils, only : timer_init
        implicit none

        ! local
        integer :: h, i, j, k

        !call timer_init()

        allocate( in_field(nvar, nx, ny, nz) )
        in_field(:,:,:,:) = 1.0_wp
        allocate( mask(nvar, nx, ny, nz))
        mask = 1.0_wp
        CALL RANDOM_NUMBER(in_field)
        do k = 1, nz
        do j = 1, ny
        do i = 1, nx
        do h = 1, nvar
                ! Since fortran does not have nans, we set all values below 0.5 to 0 and treat them as nans here
                if (in_field(h,i,j,k) < 0.5) then
                        mask(h,i,j,k) = 0.0_wp
                        in_field(h,i,j,k) = 0.0_wp
                end if 
        end do
        end do
        end do
        end do

        allocate( out_field(nvar, nx, ny, nz) )
        out_field = in_field

        allocate( weights(nvar, nx, ny, nz) )
        weights = 0.0_wp


        write(*,*) 'done with setup'

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
            v(3,3,3) * w(3,3,3) + & ! including middle point itself

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
            ! This is shorter, because the corners are already included in the block above

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

            ! Up to here is all points from (:,:,(2,3,4))

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

            v(1,1,5) * w(1,1,5) + &
            v(1,2,5) * w(1,2,5) + &
            v(1,3,5) * w(1,3,5) + &
            v(1,4,5) * w(1,4,5) + &
            v(1,5,5) * w(1,5,5) + &

            v(2,1,5) * w(2,1,5) + &
            v(2,2,5) * w(2,2,5) + &
            v(2,3,5) * w(2,3,5) + &
            v(2,4,5) * w(2,4,5) + &
            v(2,5,5) * w(2,5,5) + &

            v(3,1,5) * w(3,1,5) + &
            v(3,2,5) * w(3,2,5) + &
            v(3,3,5) * w(3,3,5) + &
            v(3,4,5) * w(3,4,5) + &
            v(3,5,5) * w(3,5,5) + &

            v(4,1,5) * w(4,1,5) + &
            v(4,2,5) * w(4,2,5) + &
            v(4,3,5) * w(4,3,5) + &
            v(4,4,5) * w(4,4,5) + &
            v(4,5,5) * w(4,5,5) + &

            v(5,1,5) * w(5,1,5) + &
            v(5,2,5) * w(5,2,5) + &
            v(5,3,5) * w(5,3,5) + &
            v(5,4,5) * w(5,4,5) + &
            v(5,5,5) * w(5,5,5) ) / max(wvalue, 1.0_wp)
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
            w(3,1,3) + w(3,2,3) + w(3,4,3) + w(3,5,3) + w(3,3,3) + & ! w point itself!
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

            ! Seems correct: WRITE(*,*) outvalue
    end subroutine

    SUBROUTINE check(iret)
            IMPLICIT NONE
            INTEGER(KIND=wp) iret
            IF (iret .NE. NF90_NOERR) THEN
                  WRITE(*,*) nf90_strerror(iret)    
            ENDIF       
    END SUBROUTINE check

    subroutine writegrid(outfile,idata)
        ! http://home.chpc.utah.edu/~thorne/computing/Examples_netCDF.pdf
        IMPLICIT NONE
        real (kind=wp), intent(in) :: idata(:,:,:,:)
        integer(kind=wp), dimension(4) :: dimids
        character (len=7), intent(in):: outfile

        integer(kind=wp) :: ncid, x_dimid, y_dimid, t_dimid, var_dimid
        integer(kind=wp) :: x_varid, y_varid, var_varid, t_varid, varid

        ! Create the netcdf file
        CALL check(nf90_create(outfile, NF90_CLOBBER, ncid))

        ! Define the dimensions
        CALL check(nf90_def_dim(ncid, "lon", nx, x_dimid))
        CALL check(nf90_def_dim(ncid, "lat", ny, y_dimid))
        CALL check(nf90_def_dim(ncid, "time", nz, t_dimid))
        CALL check(nf90_def_dim(ncid, "var", nvar, var_dimid))

        ! Define coordinate variables
        !CALL check(nf90_def_var(ncid, "lon", NF90_REAL, x_dimid, x_varid))
        !CALL check(nf90_def_var(ncid, "lat", NF90_REAL, y_dimid, y_varid))
        !CALL check(nf90_def_var(ncid, "time", NF90_REAL, t_dimid, t_varid))
        !CALL check(nf90_def_var(ncid, "var", NF90_REAL, var_dimid, var_varid))
        dimids = (/ var_dimid, x_dimid, y_dimid, t_dimid /) !TODO: is this the right order?

        ! Define variable
        CALL check(nf90_def_var(ncid, "Data", NF90_FLOAT, dimids, varid))
        CALL check(nf90_enddef(ncid)) !End Definitions

        ! Write Data
        !CALL check(nf90_put_var(ncid, x_varid, xpos))
        !CALL check(nf90_put_var(ncid, y_varid, ypos))
        !CALL check(nf90_put_var(ncid, t_varid, tpos))
        CALL check(nf90_put_var(ncid, varid, idata))
        CALL check(nf90_close(ncid))
!
    end subroutine

end program main
