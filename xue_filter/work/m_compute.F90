! ******************************************************
!      Module: m_compute
!      Author: Oliver Fuhrer
!       Email: oliverf@vulcan.com
!        Date: 18.05.2020
! Description: Contains the main computation of the
!              n-th order diffusion operator
! ******************************************************

module m_compute
    use m_constants, only : wp
    implicit none

    integer :: itim_warmup = -999, itim_core = -999

contains
  
subroutine compute_filter(in_field, out_field, num_halo, num_iter, order, alpha)
    use m_utils, only : error, sync, timer_start, timer_end
    implicit none

    ! arguments
    real (kind=wp), intent(in) :: in_field(:, :, :)
    real (kind=wp), intent(out) :: out_field(:, :, :)
    integer, intent(in) :: num_halo, num_iter, order
    real (kind=wp), intent(in) :: alpha

    ! local
    real (kind=wp), allocatable, save :: tmp1_field(:, :, :)
    real (kind=wp), allocatable, save :: tmp2_field(:, :, :)
    integer :: iter, iord, n_indent
    integer :: i_start, i_end, j_start, j_end, k_start, k_end
    real (kind=wp) :: alpha_eff

    call timer_start('compute_warmup', itim_warmup)

    call error( any(lbound(in_field) /= lbound(out_field)) .or.   &
                any(ubound(in_field) /= ubound(out_field)),       &
                "The shape of in and lap must match in compute_laplacian" )

    call error( order / 2 > num_halo, "Number of halo points must be at least order / 2")

    ! allocate temporary fields
    if ( .not. allocated(tmp1_field) ) then
        allocate(tmp1_field, source=in_field)
        tmp1_field = -999.0_wp
    end if
    if ( .not. allocated(tmp2_field) ) then
        allocate(tmp2_field, source=in_field)
        tmp2_field = -999.0_wp
    end if

    ! get bounds of compute domain (without halos)
    i_start = lbound(in_field, 1) + num_halo
    i_end = ubound(in_field, 1) - num_halo
    j_start = lbound(in_field, 2) + num_halo
    j_end = ubound(in_field, 2) - num_halo
    k_start = lbound(in_field, 3)
    k_end = ubound(in_field, 3)

    ! setup coefficient
    alpha_eff = ( -1.0_wp )**(order / 2 + 1) * ( 0.5_wp )**order * alpha

    ! do some warmup (should initialize caches)
    do iter = 1, 3
        call compute_laplacian( in_field=out_field, out_field=tmp2_field, n_indent=num_halo - 1)
        call compute_laplacian( in_field=tmp2_field, out_field=tmp1_field, n_indent=num_halo - 1)
    end do

    call timer_end(itim_warmup)

    call sync()
    
    call timer_start('compute_core', itim_core)

    ! fill halo zone from in_field
    out_field(1:i_start-1, :, :) = in_field(1:i_start-1, :, :)
    out_field(i_end+1:i_end+num_halo, :, :) = in_field(i_end+1:i_end+num_halo, :, :)
    out_field(i_start:i_end, 1:j_start-1, :) = in_field(i_start:i_end, 1:j_start-1, :)
    out_field(i_start:i_end, j_end+1:j_end+num_halo, :) = in_field(i_start:i_end, j_end+1:j_end+num_halo, :)

    do iter = 1, num_iter

        select case ( order )
        case (2)
            call compute_laplacian( in_field=out_field, out_field=tmp1_field, n_indent=num_halo)
        case (4)
            call compute_laplacian( in_field=out_field, out_field=tmp2_field, n_indent=num_halo - 1)
            call compute_laplacian( in_field=tmp2_field, out_field=tmp1_field, n_indent=num_halo)
        case (6)
            call compute_laplacian( in_field=out_field, out_field=tmp1_field, n_indent=num_halo - 2)
            call compute_laplacian( in_field=tmp1_field, out_field=tmp2_field, n_indent=num_halo - 1)
            call compute_laplacian( in_field=tmp2_field, out_field=tmp1_field, n_indent=num_halo)
        case (8)
            call compute_laplacian( in_field=out_field, out_field=tmp2_field, n_indent=num_halo - 3)
            call compute_laplacian( in_field=tmp2_field, out_field=tmp1_field, n_indent=num_halo - 2)
            call compute_laplacian( in_field=tmp1_field, out_field=tmp2_field, n_indent=num_halo - 1)
            call compute_laplacian( in_field=tmp2_field, out_field=tmp1_field, n_indent=num_halo)
        case default
            call error(.true., "Only orders 2, 4, 6, and 8 are implemented")
        end select

        ! fill compute domain with result
        out_field(i_start:i_end, j_start:j_end, k_start:k_end) =        &
            out_field(i_start:i_end, j_start:j_end, k_start:k_end)      &
            + alpha_eff * tmp1_field(i_start:i_end, j_start:j_end, k_start:k_end)
        
    end do

    call timer_end(itim_core)

    call sync()
    
end subroutine compute_filter


subroutine compute_laplacian( in_field, out_field, n_indent )
    implicit none

    ! arguments
    real (kind=wp), intent(in) :: in_field(:, :, :)
    real (kind=wp), intent(out) :: out_field(:, :, :)
    integer, intent(in) :: n_indent

    ! local
    integer :: i, j, k
    integer :: i_start, j_start, k_start
    integer :: i_end, j_end, k_end

    i_start = lbound(in_field, 1) + n_indent
    i_end = ubound(in_field, 1) - n_indent
    j_start = lbound(in_field, 2) + n_indent
    j_end = ubound(in_field, 2) - n_indent
    k_start = lbound(in_field, 3)
    k_end = ubound(in_field, 3)

    do k = k_start, k_end
        do j = j_start, j_end
            do i = i_start, i_end
                out_field(i,j,k) = -3.0_wp * in_field(i,j,k)   &
                     + 0.5_wp * in_field(i-1,j,k) + 0.5_wp * in_field(i+1,j,k)   &
                     + 0.5_wp * in_field(i,j-1,k) + 0.5_wp * in_field(i,j+1,k)   &
                     + 0.25_wp * in_field(i-1,j-1,k) + 0.25_wp * in_field(i+1,j-1,k)   &
                     + 0.25_wp * in_field(i-1,j+1,k) + 0.25_wp * in_field(i+1,j+1,k)
            end do
        end do
    end do

end subroutine compute_laplacian

end module m_compute
