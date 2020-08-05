! ******************************************************
!     Program: fortran_nocopy
!      Author: Ulrike Proske
!        Date: 14.07.2020
! Description: generic_filter-np.nanmean example in fortran
!              faster version with minimized copies and calls to subroutines
! ******************************************************

! To compile with netcdf:
! module load netcdf
! gfortran -I /usr/local/netcdf-4.6.1-4.4.4-gnu-7.4.1/include/ -L /usr/local/netcdf-4.6.1-4.4.4-gnu-7.4.1/lib/ -lnetcdf -lnetcdff fortran_nocopy.f90
! To compile fast:
! gfortran -I /usr/local/netcdf-4.6.1-4.4.4-gnu-7.4.1/include/ -L /usr/local/netcdf-4.6.1-4.4.4-gnu-7.4.1/lib/ -lnetcdf -lnetcdff
! -O3 -ftree-vectorize -funroll-loops fortran_nocopy.f90

! Profiling:
! Doesn't seem to be installed at IAC:
!module load perftools-lite
! Instead use gprof:
! gfortran -I /usr/local/netcdf-4.6.1-4.4.4-gnu-7.4.1/include/ -L /usr/local/netcdf-4.6.1-4.4.4-gnu-7.4.1/lib/ -lnetcdf -lnetcdff -g
! -gp baseline_fortran.f90
! ./a.out
! gprof -l ./a.out
! Source: https://ftp.gnu.org/old-gnu/Manuals/gprof-2.9.1/html_chapter/gprof_5.html

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

    integer, parameter :: lblocking = 0 ! 1; blocking with blocks as specified below, 0: no blocking
    ! Choose to write out the results for a validation with baseline_fortran_check.py:
    integer, parameter :: lwriteout = 0 ! 1: write out the fields (careful! do that only when they are not too large)
    ! 0: don't write any netcdf files

    nvar = 3
    nx = 3653 !3653
    ny = 1440 !1440
    nz = 720 !720
    blockx = 100
    blocky = 100
    blockz = 100

    CALL setup()

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
        if (lblocking == 0) then
            CALL weightsumnanmean_combined(in_field(:,:,:,:), mask(:,:,:,:), out_field(:,:,:,:), ivar)
        else if (lblocking == 1) then
            CALL weightsumnanmean_combined_blocking(in_field(:,:,:,:), mask(:,:,:,:), out_field(:,:,:,:), ivar)
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
    write(*,*) 'lblocking: ', lblocking
    write(*,*) 'nx, ny, nz: ', nx, ny, nz
    write(*,*) 'blockx, blocky, blockz: ', blockx, blocky, blockz

contains

    subroutine weightsumnanmean_combined(in_field, mask, out_field, ivar)
        ! loop over all points of a certain var-field and compute generic_filter for each
        implicit none

        ! argument
        real(kind=wp), intent(in) :: in_field(:,:,:,:)
        real(kind=wp), intent(in) :: mask(:,:,:,:)
        integer, intent(in) :: ivar
        real(kind=wp), intent(inout) :: out_field(:,:,:,:)

        ! local
        integer :: i, j, k
        integer :: il, jl, kl
        real(kind=wp) :: tmp, tmpm

        ! loop over all data points
        do k = 1+nhalo, nz-nhalo
        do j = 1+nhalo, ny-nhalo
        do i = 1+nhalo, nx-nhalo
            tmp = 0
            tmpm = 0
            ! loop over surrounding box
            do il = i-2,i+2
            do jl = j-2,j+2
            do kl = k-2,k+2
                tmp = tmp + in_field(ivar,il,jl,kl) * mask(ivar,il,jl,kl)
                tmpm = tmpm + mask(ivar,il,jl,kl)
            end do
            end do
            end do
            out_field(ivar,i,j,k) = tmp / max(tmpm, 1.0_wp)
        end do
        end do
        end do
    end subroutine weightsumnanmean_combined

    subroutine weightsumnanmean_combined_blocking(in_field, mask, out_field, ivar)
        ! loop over all points of a certain var-field and compute generic_filter for each
        implicit none

        ! argument
        real(kind=wp), intent(in) :: in_field(:,:,:,:)
        real(kind=wp), intent(in) :: mask(:,:,:,:)
        integer, intent(in) :: ivar
        real(kind=wp), intent(inout) :: out_field(:,:,:,:)

        ! local
        integer :: i, j, k, block_i, block_j, block_k, k_local, j_local, i_local
        integer :: il, jl, kl
        real(kind=wp) :: tmp, tmpm

        ! loop over blocks
        do block_k = 0, nz/blockz+1 ! +1 to include all points even if the block size does not fit the dimension
        do block_j = 0, ny/blocky+1
        do block_i = 0, nx/blockx+1
        ! loop inside blocks
        do k_local = 1, blockz
        do j_local = 1, blocky
        do i_local = 1, blockx
            ! construct absolute indices
            k = blockz*block_k + k_local
            j = blocky*block_j + j_local
            i = blockx*block_i + i_local
            ! ignore if we're inside the halo zone
            if (k < 1+nhalo .or. k > nz-nhalo .or. j < 1+nhalo .or. j > ny-nhalo .or. i < 1+nhalo .or. i > nx-nhalo) then
               cycle
            end if
            tmp = 0
            tmpm = 0
            ! loop over surrounding box
            do il = i-2,i+2
            do jl = j-2,j+2
            do kl = k-2,k+2
                tmp = tmp + in_field(ivar,il,jl,kl) * mask(ivar,il,jl,kl)
                tmpm = tmpm + mask(ivar,il,jl,kl)
            end do
            end do
            end do
            ! result = sum of all points in the box / number of values in the box
            out_field(ivar,i,j,k) = tmp / max(tmpm, 1.0_wp)
        end do
        end do
        end do
        end do
        end do
        end do
    end subroutine weightsumnanmean_combined_blocking

    ! setup everything before work
    ! (allocate memory, initialize fields)
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
                ! Since fortran does not have nans, we set all values below 0.5 to 0 and treat them as nans here,
                ! using the mask construct
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
        dimids = (/ var_dimid, x_dimid, y_dimid, t_dimid /) ! not the right order

        ! Define variable
        CALL check(nf90_def_var(ncid, "Data", NF90_FLOAT, dimids, varid))
        CALL check(nf90_enddef(ncid)) !End Definitions

        ! Write Data
        CALL check(nf90_put_var(ncid, varid, idata))
        CALL check(nf90_close(ncid))
!
    end subroutine

end program main
