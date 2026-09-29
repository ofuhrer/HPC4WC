! Program: stencil2d-filter-zalesak
!  Authors: Stefanie Boersig <stefanie.boersig@env.ethz.ch>
!           Boaz Ko <boazko@student.ethz.ch>
!           Ben Bullinger <ben.bullinger@inf.ethz.ch>
!  Based on: the course day-3 MPI baseline by Oliver Fuhrer
!
! Zalesak-type flux-corrected variant of stencil2d-filter (Xue 2000,
! Eqs. 5-9). Needs halo n/2+1. --cone and --czero force C == 1 and
! C == 0 for validation.

! Driver for apply_diffusion() that sets up fields and does timings
program main
    use mpi, only: MPI_COMM_WORLD
    use m_utils, only: timer_init, timer_start, timer_end, timer_get, is_master, num_rank, write_field_to_file
    use m_partitioner, only: Partitioner
    implicit none

    ! constants
    integer, parameter :: wp = 4

    ! local
    integer :: global_nx, global_ny, global_nz, num_iter
    logical :: scan

    integer :: num_halo = -1     ! runtime option --num_halo (default: order/2)
    real (kind=wp) :: alpha = -1.0_wp  ! runtime option --alpha (default: 1/8^(order/2))
    logical :: force_cone = .false.    ! --cone: C == 1 (validation)
    logical :: force_czero = .false.   ! --czero: C == 0 (validation)
    real (kind=wp) :: alpha2 = 1.0_wp / 8.0_wp   ! low-order coefficient
    integer :: order = 4               ! runtime option --order (2,4,6,8)
    real (kind=wp) :: sgn              ! (-1)^(order/2+1), fixed by order
    character(len=16) :: ic = 'box'    ! runtime option --ic (box|mode)
    integer :: kx_mode = 4, ky_mode = 3  ! runtime options --kx/--ky (ic=mode)

    real (kind=8) :: c_sum = 0.0d0, c_cnt = 0.0d0
    real (kind=8) :: global_c_sum, global_c_cnt

    real (kind=wp), allocatable :: in_field(:, :, :), f_in(:, :, :)
    real (kind=wp), allocatable :: out_field(:, :, :), f_out(:, :, :)
#ifdef GNU_MPI_WORKAROUND
    real (kind=wp), allocatable :: dummy_field(:, :, :)
#endif

    integer :: timer_work
    real (kind=8) :: runtime

    integer :: cur_setup, num_setups = 1
    integer :: nx_setups(7) = (/ 16, 32, 48, 64, 96, 128, 192 /)
    integer :: ny_setups(7) = (/ 16, 32, 48, 64, 96, 128, 192 /)

    type(Partitioner) :: p

#ifdef CRAYPAT
    include "pat_apif.h"
    integer :: istat
    call PAT_record( PAT_STATE_OFF, istat )
#endif

    call init()

    if ( is_master() ) then
        write(*, '(a)') '# ranks nx ny nz num_iter time'
        write(*, '(a, i2, a, e15.7, a, a)') &
            '# order = ', order, ', alpha = ', alpha, ', scheme = zalesak, ic = ', trim(ic)
        write(*, '(a)') 'data = np.array( [ \'
    end if

    if ( scan ) num_setups = size(nx_setups) * size(ny_setups)
    do cur_setup = 0, num_setups - 1

        call timer_init()

        if ( scan ) then
            global_nx = nx_setups( modulo(cur_setup, size(ny_setups) ) + 1 )
            global_ny = ny_setups( cur_setup / size(ny_setups) + 1 )
        end if

        if ( is_master() ) then
            call setup()
#ifdef GNU_MPI_WORKAROUND
        else
            if ( .not. allocated(dummy_field) ) &
                allocate( dummy_field(1, 1, 1) )
#endif
        end if

        if ( .not. scan .and. is_master() ) &
            call write_field_to_file( in_field, num_halo, "in_field.dat" )

        p = Partitioner(MPI_COMM_WORLD, (/global_nx, global_ny, global_nz/), num_halo, periodic=(/.true., .true./))

#ifdef GNU_MPI_WORKAROUND
        if ( is_master() ) then
            f_in = p%scatter(in_field, root=0)
        else
            f_in = p%scatter(dummy_field, root=0)
        end if
#else
        f_in = p%scatter(in_field, root=0)
#endif
        allocate(f_out, source=f_in)

        ! warmup caches
        call apply_diffusion( f_in, f_out, alpha, num_iter=1, p=p )

        ! time the actual work
#ifdef CRAYPAT
        call PAT_record( PAT_STATE_ON, istat )
#endif
        timer_work = -999
        call timer_start('work', timer_work)

        call apply_diffusion( f_in, f_out, alpha, num_iter=num_iter, p=p )

        call timer_end( timer_work )
#ifdef CRAYPAT
        call PAT_record( PAT_STATE_OFF, istat )
#endif

        call update_halo( f_out, p )

#ifdef DEBUG_DUMP_RANKS
        block
            character(len=32) :: dbg_fn
            write(dbg_fn, '(a,i1,a)') 'rank', p%rank(), '_out.dat'
            call write_field_to_file( f_out, num_halo, trim(dbg_fn) )
        end block
#endif
        out_field = p%gather(f_out, root=0)

        if ( .not. scan .and. is_master() ) &
            call write_field_to_file( out_field, num_halo, "out_field.dat" )

        if ( is_master() ) &
            call cleanup()

        runtime = timer_get( timer_work )
        if ( is_master() ) &
            write(*, '(a, i5, a, i5, a, i5, a, i5, a, i8, a, e15.7, a)') &
                '[', num_rank(), ',', global_nx, ',', global_ny, ',', global_nz, &
                ',', num_iter, ',', runtime, '], \'

        global_c_sum = get_global_rcounter( c_sum )
        global_c_cnt = get_global_rcounter( c_cnt )
        if ( is_master() ) &
            write(*, '(a, e15.7)') '# zalesak_mean_C = ', &
                global_c_sum / max( global_c_cnt, 1.0d0 )

    end do

    if ( is_master() ) then
        write(*, '(a)') '] )'
    end if

    call finalize()

contains


    ! Integrate 4th-order diffusion equation by a certain number of iterations.
    !
    !  in_field          -- input field (nx x ny x nz with halo in x- and y-direction)
    !  out_field         -- result (must be same size as in_field)
    !  alpha             -- diffusion coefficient (dimensionless)
    !  num_iter          -- number of iterations to execute
    !
    subroutine apply_diffusion( in_field, out_field, alpha, num_iter, p )
        implicit none

        ! arguments
        real (kind=wp), intent(inout), target :: in_field(:, :, :)
        real (kind=wp), intent(inout) :: out_field(:, :, :)
        real (kind=wp), intent(in) :: alpha
        integer, intent(in) :: num_iter
        type(Partitioner), intent(in) :: p

        ! local (naive: full-size 3D temporaries)
        real (kind=wp), save, allocatable, target :: tmp1_field(:, :, :)
        real (kind=wp), save, allocatable, target :: tmp2_field(:, :, :)
        real (kind=wp), save, allocatable :: fxhl(:, :, :), fyhl(:, :, :)
        real (kind=wp), save, allocatable :: dxl(:, :, :), dyl(:, :, :)
        real (kind=wp), save, allocatable :: phis(:, :, :)
        real (kind=wp), save, allocatable :: rp(:, :, :), rm(:, :, :)
        real (kind=wp), pointer :: src(:, :, :), cur(:, :, :), nxt(:, :, :), swp(:, :, :)
        real (kind=wp), parameter :: eps = 1.0e-15_wp
        real (kind=wp) :: alpha_eff, pmax, pmin, pin, pout, c
        integer :: iter, i, j, k, ss, m
        integer :: dims(3), nx, ny, nz
        integer :: i0, i1, j0, j1

        m = order / 2
        alpha_eff = sgn * alpha

        dims = p%shape()
        nx = dims(1) - 2 * p%num_halo()
        ny = dims(2) - 2 * p%num_halo()
        nz = dims(3)

        if ( allocated(tmp1_field) .and. &
            any( shape(tmp1_field) /= (/nx + 2 * num_halo, ny + 2 * num_halo, nz /) ) ) then
            deallocate( tmp1_field, tmp2_field, fxhl, fyhl, dxl, dyl, phis, rp, rm )
        end if
        if ( .not. allocated(tmp1_field) ) then
            allocate( tmp1_field(nx + 2 * num_halo, ny + 2 * num_halo, nz) )
            allocate( tmp2_field, fxhl, fyhl, dxl, dyl, phis, rp, rm, mold=tmp1_field )
            tmp1_field = 0.0_wp; tmp2_field = 0.0_wp
            fxhl = 0.0_wp; fyhl = 0.0_wp; dxl = 0.0_wp; dyl = 0.0_wp
            phis = 0.0_wp; rp = 0.0_wp; rm = 0.0_wp
        end if

        i0 = 1 + num_halo; i1 = nx + num_halo
        j0 = 1 + num_halo; j1 = ny + num_halo

        do iter = 1, num_iter

            call update_halo( in_field, p )

            ! pre-sweeps with one extra ring (src needed on interior+2)
            if ( m == 1 ) then
                src => in_field
            else
                cur => tmp1_field
                nxt => tmp2_field
                call laplacian( in_field, cur, num_halo, extend=m )
                do ss = 2, m - 1
                    call laplacian( cur, nxt, num_halo, extend=m+1-ss )
                    swp => cur; cur => nxt; nxt => swp
                end do
                src => cur
            end if

            ! corrective fluxes A^HL = A^H - A^L and raw low-order gradients,
            ! one ring beyond the interior interfaces
            do k = 1, nz
            do j = j0 - 1, j1 + 1
            do i = i0 - 1, i1 + 2
                dxl(i, j, k) = in_field(i, j, k) - in_field(i - 1, j, k)
                fxhl(i, j, k) = -alpha_eff * (src(i, j, k) - src(i - 1, j, k)) &
                                + alpha2 * dxl(i, j, k)
            end do
            end do
            end do
            do k = 1, nz
            do j = j0 - 1, j1 + 2
            do i = i0 - 1, i1 + 1
                dyl(i, j, k) = in_field(i, j, k) - in_field(i, j - 1, k)
                fyhl(i, j, k) = -alpha_eff * (src(i, j, k) - src(i, j - 1, k)) &
                                + alpha2 * dyl(i, j, k)
            end do
            end do
            end do

            ! provisional low-order (monotone) solution phi* = in - div(A^L),
            ! written to match the order-2 filter update bit-exactly
            do k = 1, nz
            do j = j0 - 1, j1 + 1
            do i = i0 - 1, i1 + 1
                phis(i, j, k) = in_field(i, j, k) + alpha2 * (      &
                      dxl(i + 1, j, k) - dxl(i, j, k)               &
                    + dyl(i, j + 1, k) - dyl(i, j, k) )
            end do
            end do
            end do

            ! Zalesak ratio factors from the phi^n extrema headroom
            do k = 1, nz
            do j = j0 - 1, j1 + 1
            do i = i0 - 1, i1 + 1
                pmax = max( in_field(i, j, k), in_field(i - 1, j, k), in_field(i + 1, j, k), &
                            in_field(i, j - 1, k), in_field(i, j + 1, k) )
                pmin = min( in_field(i, j, k), in_field(i - 1, j, k), in_field(i + 1, j, k), &
                            in_field(i, j - 1, k), in_field(i, j + 1, k) )
                pin  = max( 0.0_wp,  fxhl(i, j, k) ) - min( 0.0_wp, fxhl(i + 1, j, k) ) &
                     + max( 0.0_wp,  fyhl(i, j, k) ) - min( 0.0_wp, fyhl(i, j + 1, k) )
                pout = max( 0.0_wp,  fxhl(i + 1, j, k) ) - min( 0.0_wp, fxhl(i, j, k) ) &
                     + max( 0.0_wp,  fyhl(i, j + 1, k) ) - min( 0.0_wp, fyhl(i, j, k) )
                rp(i, j, k) = max( 0.0_wp, min( 1.0_wp, (pmax - phis(i, j, k)) / (pin + eps) ) )
                rm(i, j, k) = max( 0.0_wp, min( 1.0_wp, (phis(i, j, k) - pmin) / (pout + eps) ) )
            end do
            end do
            end do

            ! limit corrective fluxes: C per interface from the R factors
            ! (A >= 0 transports into cell i, out of cell i-1)
            do k = 1, nz
            do j = j0, j1
            do i = i0, i1 + 1
                if ( fxhl(i, j, k) >= 0.0_wp ) then
                    c = min( rp(i, j, k), rm(i - 1, j, k) )
                else
                    c = min( rp(i - 1, j, k), rm(i, j, k) )
                end if
                if ( force_cone ) c = 1.0_wp
                if ( force_czero ) c = 0.0_wp
                c_sum = c_sum + c; c_cnt = c_cnt + 1.0d0
                fxhl(i, j, k) = c * fxhl(i, j, k)
            end do
            end do
            end do
            do k = 1, nz
            do j = j0, j1 + 1
            do i = i0, i1
                if ( fyhl(i, j, k) >= 0.0_wp ) then
                    c = min( rp(i, j, k), rm(i, j - 1, k) )
                else
                    c = min( rp(i, j - 1, k), rm(i, j, k) )
                end if
                if ( force_cone ) c = 1.0_wp
                if ( force_czero ) c = 0.0_wp
                c_sum = c_sum + c; c_cnt = c_cnt + 1.0d0
                fyhl(i, j, k) = c * fyhl(i, j, k)
            end do
            end do
            end do

            ! antidiffusive update: out = phi* - div(C * A^HL)
            do k = 1, nz
            do j = j0, j1
            do i = i0, i1
                out_field(i, j, k) = phis(i, j, k) - (               &
                      fxhl(i + 1, j, k) - fxhl(i, j, k)              &
                    + fyhl(i, j + 1, k) - fyhl(i, j, k) )
            end do
            end do
            end do

            if ( iter /= num_iter ) then
                do k = 1, nz
                do j = j0, j1
                do i = i0, i1
                    in_field(i, j, k) = out_field(i, j, k)
                end do
                end do
                end do
            end if

        end do

    end subroutine apply_diffusion


    ! Compute Laplacian using 2nd-order centered differences.
    !
    !  in_field          -- input field (nx x ny x nz with halo in x- and y-direction)
    !  lap_field         -- result (must be same size as in_field)
    !  num_halo          -- number of halo points
    !  extend            -- extend computation into halo-zone by this number of points
    !
    subroutine laplacian( field, lap, num_halo, extend )
        implicit none

        ! argument
        real (kind=wp), intent(in) :: field(:, :, :)
        real (kind=wp), intent(inout) :: lap(:, :, :)
        integer, intent(in) :: num_halo, extend

        ! local
        integer :: i, j, k

        do k = lbound(field,3), ubound(field,3)
        do j = lbound(field,2) + num_halo - extend, ubound(field,2) - num_halo + extend
        do i = lbound(field,1) + num_halo - extend, ubound(field,1) - num_halo + extend
            lap(i, j, k) = -4._wp * field(i, j, k)      &
                + field(i - 1, j, k) + field(i + 1, j, k)  &
                + field(i, j - 1, k) + field(i, j + 1, k)
        end do
        end do
        end do

    end subroutine laplacian


    ! Update the halo-zone using an up/down and left/right strategy.
    !
    !  field             -- input/output field (nz x ny x nx with halo in x- and y-direction)
    !
    !  Note: corners are updated in the left/right phase of the halo-update
    !
    subroutine update_halo( field, p )
        use mpi, only : MPI_FLOAT, MPI_DOUBLE, MPI_SUCCESS, MPI_STATUS_SIZE
        use m_utils, only : error
        implicit none

        ! argument
        real (kind=wp), intent(inout) :: field(:, :, :)
        type(Partitioner), intent(in) :: p

        ! local
        integer :: i, j, k
        integer :: dims(3), nx, ny, nz
        integer :: lr_size, tb_size, dtype
        integer :: tb_req(4), lr_req(4)
        integer :: ierror, status(MPI_STATUS_SIZE, 4), icount
        real (kind=wp), save, allocatable :: sndbuf_l(:), sndbuf_r(:), sndbuf_t(:), sndbuf_b(:)
        real (kind=wp), save, allocatable :: rcvbuf_l(:), rcvbuf_r(:), rcvbuf_t(:), rcvbuf_b(:)

        ! set datatype
        if (wp == 4) then
            dtype = MPI_FLOAT
        else
            dtype = MPI_DOUBLE
        end if

        ! get dimensions
        dims = p%shape()
        nx = dims(1) - 2 * p%num_halo()
        ny = dims(2) - 2 * p%num_halo()
        nz = dims(3)

        ! compute sizes of buffers
        tb_size = nz * num_halo * nx
        lr_size = nz * num_halo * (ny + 2 * num_halo)

        ! this is only done the first time this subroutine is called (warmup)
        ! or when the dimensions of the fields change
        if ( allocated(sndbuf_l) .and. &
            ( ( size(sndbuf_l) /= lr_size ) .or. ( size(sndbuf_t) /= tb_size ) ) ) then
            deallocate( sndbuf_l, sndbuf_r, sndbuf_t, sndbuf_b )
            deallocate( rcvbuf_l, rcvbuf_r, rcvbuf_t, rcvbuf_b )
        end if
        if ( .not. allocated(sndbuf_l) ) then
            allocate( sndbuf_l(lr_size), sndbuf_r(lr_size), sndbuf_t(tb_size), sndbuf_b(tb_size) )
            allocate( rcvbuf_l(lr_size), rcvbuf_r(lr_size), rcvbuf_t(tb_size), rcvbuf_b(tb_size) )
            sndbuf_l = 0.0_wp; sndbuf_r = 0.0_wp; sndbuf_t = 0.0_wp; sndbuf_b = 0.0_wp
            rcvbuf_l = 0.0_wp; rcvbuf_r = 0.0_wp; rcvbuf_t = 0.0_wp; rcvbuf_b = 0.0_wp
        end if

        ! pre-post the receives
        call MPI_Irecv(rcvbuf_b, tb_size, dtype, p%bottom(), 1000, p%comm(), tb_req(1), ierror)
        call error(ierror /= MPI_SUCCESS, 'Problem with MPI_Irecv(bottom)', code=ierror)
        call MPI_Irecv(rcvbuf_t, tb_size, dtype, p%top(), 1001, p%comm(), tb_req(2), ierror)
        call error(ierror /= MPI_SUCCESS, 'Problem with MPI_Irecv(top)', code=ierror)
        call MPI_Irecv(rcvbuf_l, lr_size, dtype, p%left(), 1002, p%comm(), lr_req(1), ierror)
        call error(ierror /= MPI_SUCCESS, 'Problem with MPI_Irecv(left)', code=ierror)
        call MPI_Irecv(rcvbuf_r, lr_size, dtype, p%right(), 1003, p%comm(), lr_req(2), ierror)
        call error(ierror /= MPI_SUCCESS, 'Problem with MPI_Irecv(right)', code=ierror)

        ! pack the tb-buffers (without corners)
        icount = 0
        do k = 1, nz
        do j = 1, num_halo
        do i = 1 + num_halo, nx + num_halo
            icount = icount + 1
            sndbuf_t(icount) = field(i, j + ny, k)
            sndbuf_b(icount) = field(i, j + num_halo, k)
        end do
        end do
        end do

        ! send tb-buffers
        call MPI_Isend(sndbuf_t, tb_size, dtype, p%top(), 1000, p%comm(), tb_req(3), ierror)
        call error(ierror /= MPI_SUCCESS, 'Problem with MPI_Isend(top)', code=ierror)
        call MPI_Isend(sndbuf_b, tb_size, dtype, p%bottom(), 1001, p%comm(), tb_req(4), ierror)
        call error(ierror /= MPI_SUCCESS, 'Problem with MPI_Isend(bottom)', code=ierror)

        ! wait for tb-comm to finish
        call MPI_Waitall(4, tb_req, status, ierror)
        call error(ierror /= MPI_SUCCESS, 'Problem with MPI_Waitall(tb)', code=ierror)

        ! unpack the tb-buffers (without corners); must precede the
        ! lr-pack, or the corners forwarded left/right are stale
        icount = 0
        do k = 1, nz
        do j = 1, num_halo
        do i = 1 + num_halo, nx + num_halo
            icount = icount + 1
            field(i, j, k) = rcvbuf_b(icount)
            field(i, j + num_halo + ny, k) = rcvbuf_t(icount)
        end do
        end do
        end do

        ! pack the lr-buffers (including corners)
        icount = 0
        do k = 1, nz
        do j = 1, ny + 2 * num_halo
        do i = 1, num_halo
            icount = icount + 1
            sndbuf_r(icount) = field(i + nx, j, k)
            sndbuf_l(icount) = field(i + num_halo, j, k)
        end do
        end do
        end do

        call MPI_Isend(sndbuf_r, lr_size, dtype, p%right(), 1002, p%comm(), lr_req(3), ierror)
        call error(ierror /= MPI_SUCCESS, 'Problem with MPI_Isend(right)', code=ierror)
        call MPI_Isend(sndbuf_l, lr_size, dtype, p%left(), 1003, p%comm(), lr_req(4), ierror)
        call error(ierror /= MPI_SUCCESS, 'Problem with MPI_Isend(left)', code=ierror)

        ! wait for lr-comm to finish
        call MPI_Waitall(4, lr_req, status, ierror)
        call error(ierror /= MPI_SUCCESS, 'Problem with MPI_Waitall(lr)', code=ierror)

        ! unpack the lr-buffers (with corners)
        icount = 0
        do k = 1, nz
        do j = 1, ny + 2 * num_halo
        do i = 1, num_halo
            icount = icount + 1
            field(i, j, k) = rcvbuf_l(icount)
            field(i + num_halo + nx, j, k) = rcvbuf_r(icount)
        end do
        end do
        end do

    end subroutine update_halo


    ! initialize at program start
    ! (init MPI, read command line arguments)
    subroutine init()
        use mpi, only : MPI_INIT
        use m_utils, only : error
        implicit none

        ! local
        integer :: ierror

        ! initialize MPI environment
        call MPI_INIT(ierror)
        call error(ierror /= 0, 'Problem with MPI_INIT', code=ierror)

        call read_cmd_line_arguments()

    end subroutine init


    ! setup everything before work
    ! (init timers, allocate memory, initialize fields)
    subroutine setup()
        use m_utils, only : error
        implicit none

        ! local
        integer :: i, j, k
        real (kind=8) :: pi

        allocate( in_field(global_nx + 2 * num_halo, global_ny + 2 * num_halo, global_nz) )
        in_field = 0.0_wp

        select case (trim(ic))
        case ('box')
            do k = 1 + global_nz / 4, 3 * global_nz / 4
            do j = 1 + num_halo + global_ny / 4, num_halo + 3 * global_ny / 4
            do i = 1 + num_halo + global_nx / 4, num_halo + 3 * global_nx / 4
                in_field(i, j, k) = 1.0_wp
            end do
            end do
            end do
        case ('mode')
            ! discrete eigenmode of the periodic Laplacian (damping: tools/check_amplification.py)
            pi = 4.0d0 * atan(1.0d0)
            do k = 1, global_nz
            do j = 1 + num_halo, global_ny + num_halo
            do i = 1 + num_halo, global_nx + num_halo
                in_field(i, j, k) = real( &
                    cos( 2.0d0 * pi * kx_mode * dble(i - num_halo - 1) / global_nx ) &
                  * cos( 2.0d0 * pi * ky_mode * dble(j - num_halo - 1) / global_ny ), wp )
            end do
            end do
            end do
        case default
            call error(.true., 'Unknown IC: ' // trim(ic))
        end select

        allocate( out_field(global_nx + 2 * num_halo, global_ny + 2 * num_halo, global_nz) )
        out_field = in_field

    end subroutine setup


    ! read and parse the command line arguments
    ! (read values, convert type, ensure all required arguments are present,
    !  ensure values are reasonable)
    subroutine read_cmd_line_arguments()
        use m_utils, only : error
        implicit none

        ! local
        integer iarg, num_arg
        character(len=256) :: arg, arg_val

        ! setup defaults
        global_nx = -1
        global_ny = -1
        global_nz = -1
        num_iter = -1
        scan = .false.

        num_arg = command_argument_count()
        iarg = 1
        do while ( iarg <= num_arg )
            call get_command_argument(iarg, arg)
            select case (arg)
            case ("--nx")
                call error(iarg + 1 > num_arg, "Missing value for --nx argument")
                call get_command_argument(iarg + 1, arg_val)
                call error(arg_val(1:1) == "-", "Missing value for --nx argument")
                read(arg_val, *) global_nx
                iarg = iarg + 1
            case ("--ny")
                call error(iarg + 1 > num_arg, "Missing value for --ny argument")
                call get_command_argument(iarg + 1, arg_val)
                call error(arg_val(1:1) == "-", "Missing value for --ny argument")
                read(arg_val, *) global_ny
                iarg = iarg + 1
            case ("--nz")
                call error(iarg + 1 > num_arg, "Missing value for --nz argument")
                call get_command_argument(iarg + 1, arg_val)
                call error(arg_val(1:1) == "-", "Missing value for --nz argument")
                read(arg_val, *) global_nz
                iarg = iarg + 1
            case ("--num_iter")
                call error(iarg + 1 > num_arg, "Missing value for --num_iter argument")
                call get_command_argument(iarg + 1, arg_val)
                call error(arg_val(1:1) == "-", "Missing value for --num_iter argument")
                read(arg_val, *) num_iter
                iarg = iarg + 1
            case ("--num_halo")
                call error(iarg + 1 > num_arg, "Missing value for --num_halo argument")
                call get_command_argument(iarg + 1, arg_val)
                call error(arg_val(1:1) == "-", "Missing value for --num_halo argument")
                read(arg_val, *) num_halo
                iarg = iarg + 1
            case ("--cone")
                force_cone = .true.
            case ("--czero")
                force_czero = .true.
            case ("--order")
                call error(iarg + 1 > num_arg, "Missing value for --order argument")
                call get_command_argument(iarg + 1, arg_val)
                read(arg_val, *) order
                iarg = iarg + 1
            case ("--alpha")
                call error(iarg + 1 > num_arg, "Missing value for --alpha argument")
                call get_command_argument(iarg + 1, arg_val)
                read(arg_val, *) alpha
                iarg = iarg + 1
            case ("--ic")
                call error(iarg + 1 > num_arg, "Missing value for --ic argument")
                call get_command_argument(iarg + 1, ic)
                iarg = iarg + 1
            case ("--kx")
                call error(iarg + 1 > num_arg, "Missing value for --kx argument")
                call get_command_argument(iarg + 1, arg_val)
                read(arg_val, *) kx_mode
                iarg = iarg + 1
            case ("--ky")
                call error(iarg + 1 > num_arg, "Missing value for --ky argument")
                call get_command_argument(iarg + 1, arg_val)
                read(arg_val, *) ky_mode
                iarg = iarg + 1
            case ("--scan")
                scan = .true.
            case default
                call error(.true., "Unknown command line argument encountered: " // trim(arg))
            end select
            iarg = iarg + 1
        end do

        ! make sure everything is set
        if (.not. scan) then
            call error(global_nx == -1, 'You have to specify nx')
            call error(global_ny == -1, 'You have to specify ny')
        end if
        call error(global_nz == -1, 'You have to specify nz')
        call error(num_iter == -1, 'You have to specify num_iter')

        ! check consistency of values
        if (.not. scan) then
            call error(global_nx < 0 .or. global_nx > 1024*1024, "Please provide a reasonable value of nx")
            call error(global_ny < 0 .or. global_ny > 1024*1024, "Please provide a reasonable value of ny")
        end if
        call error(global_nz < 0 .or. global_nz > 1024, "Please provide a reasonable value of nz")
        call error(num_iter < 1 .or. num_iter > 1024*1024, "Please provide a reasonable value of num_iter")

        ! order-dependent defaults and checks
        call error(.not. (order == 2 .or. order == 4 .or. order == 6 .or. order == 8), &
                   "--order must be one of 2, 4, 6, 8")
        if ( num_halo == -1 ) num_halo = order / 2 + 1
        call error(num_halo < order / 2 + 1 .or. num_halo > 8, &
                   "num_halo must be in [order/2 + 1 .. 8] (Zalesak needs one extra ring)")
        if ( alpha < 0.0_wp ) alpha = 1.0_wp / real( 8 ** (order / 2), wp )
        sgn = merge( 1.0_wp, -1.0_wp, mod( order / 2 + 1, 2 ) == 0 )

    end subroutine read_cmd_line_arguments


    ! sum a real accumulator over all ranks (returns the total on rank 0)
    function get_global_rcounter( counter )
        use mpi, only : MPI_COMM_WORLD, MPI_DOUBLE_PRECISION, MPI_SUM
        implicit none

        real (kind=8) :: get_global_rcounter
        real (kind=8) :: counter

        real (kind=8) :: global_counter
        integer :: ierror

        call MPI_REDUCE( counter, global_counter, 1, MPI_DOUBLE_PRECISION, &
                         MPI_SUM, 0, MPI_COMM_WORLD, ierror )
        get_global_rcounter = global_counter

    end function get_global_rcounter


    ! sum a counter over all ranks (returns the total on rank 0)
    function get_global_counter( counter )
        use mpi, only : MPI_COMM_WORLD, MPI_DOUBLE_PRECISION, MPI_SUM
        implicit none

        real (kind=8) :: get_global_counter
        integer (kind=8) :: counter

        real (kind=8) :: global_counter
        integer :: ierror

        call MPI_REDUCE( dble( counter ), global_counter, 1, MPI_DOUBLE_PRECISION, &
                         MPI_SUM, 0, MPI_COMM_WORLD, ierror )
        get_global_counter = global_counter

    end function get_global_counter


    ! cleanup at end of work
    ! (report timers, free memory)
    subroutine cleanup()
        implicit none

        deallocate(in_field, out_field)

    end subroutine cleanup


    ! finalize at end of program
    ! (finalize MPI)
    subroutine finalize()
        use mpi, only : MPI_FINALIZE
        use m_utils, only : error
        implicit none

        integer :: ierror

        call MPI_FINALIZE(ierror)
        call error(ierror /= 0, 'Problem with MPI_FINALIZE', code=ierror)

    end subroutine finalize


end program main
