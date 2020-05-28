! ******************************************************
!      Module: main
!      Author: Oliver Fuhrer
!       Email: oliverf@vulcan.com
!        Date: 20.05.2020
! Description: Program that implements n-th order 2-dimensional monotonic diffusion operator
!              following the publication of Xue 2000 (Monthly Weather Review)
!              URL: http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.73.9013&rep=rep1&type=pdf
! ******************************************************


program main
    use m_constants, only : wp
    use m_compute, only : compute_filter
#ifdef USE_IO
    use m_cdf, only : cdfvarput
#endif
    use m_utils, only : timer_init, timer_print, timer_start, timer_end
    implicit none

    ! local
    logical :: io_ready = .false.
    integer :: nx, ny, nz, num_halo, num_iter, order
    integer :: itim_init = -999, itim_io = -999, itim_compute = -999, itim_clean = -999
    real (kind=wp) :: alpha
    real (kind=wp), allocatable :: in_field(:, :, :), out_field(:, :, :)

    call init()

#ifdef USE_IO
    call timer_start('i/o', itim_io)
    call cdfvarput('f', in_field, (/'x', 'y', 'z'/), units='-', time=real(0, wp))
    call timer_end(itim_io)
#endif

    call timer_start('compute', itim_compute)
    call compute_filter(in_field, out_field, num_halo, num_iter, order, alpha)
    call timer_end(itim_compute)

#ifdef USE_IO
    call timer_start('i/o', itim_io)
    call cdfvarput('f', out_field, (/'x', 'y', 'z'/), units='-', time=real(num_iter, wp))
    call timer_end(itim_io)
#endif

    call cleanup()

contains


subroutine init()
#ifdef USE_MPI
    use mpi, only : MPI_INIT
#endif
#ifdef USE_IO
    use m_cdf, only : cdfbegin, cdftime, cdfdimput, cdfattput
#endif
    use m_utils, only : error, is_master
    implicit none

    ! local
    integer :: ierror
    integer :: i, j, k

#ifdef USE_MPI
    ! initialize MPI environment
    call MPI_INIT(ierror)
    call error(ierror /= 0, 'Problem with MPI_INIT', code=ierror)
#endif

    call timer_init()
    call timer_start('init', itim_init)

    call read_cmd_line_arguments()

    allocate( in_field(nx + 2*num_halo, ny + 2*num_halo, nz) )
    in_field = 0.0_wp

    ! set inner 1/3 to 1's
    do k = 1, nz
        do j = 1 + num_halo + 2*ny/6, 1 + num_halo + 4*ny/6
            do i = 1 + num_halo + 2*nx/6, 1 + num_halo + 4*nx/6
                in_field(i, j, k) = 1.0_wp
            end do
        end do
    end do
 
    allocate( out_field(nx + 2*num_halo, ny + 2*num_halo, nz) )
    out_field = -999.0_wp

#ifdef USE_IO
    ! initialize output file
    if ( is_master() ) then
        call cdfbegin(fname='out.nc', caller='main.x', create=.true., overwrite=.true.)
        call cdftime('steps','-')
        call cdfdimput('x',nx + 2 * num_halo)
        call cdfdimput('y',ny + 2 * num_halo)
        call cdfdimput('z',nz)
        call cdfattput('', 'Conventions', 'COARDS')
        call cdfattput('', 'Source', 'main.x')
        io_ready = .true.
    end if
#endif

    call timer_end(itim_init)

end subroutine init


subroutine read_cmd_line_arguments()
    use m_utils, only : error
    implicit none

    ! local
    integer iarg, num_arg
    character(len=256) :: arg, arg_val

    ! setup defaults
    nx = -1
    ny = -1
    nz = -1
    order = -1
    num_halo = -1
    num_iter = -1
    alpha = -1.0_wp

    num_arg = command_argument_count()
    iarg = 1
    do while ( iarg <= num_arg )
        call get_command_argument(iarg, arg)
        select case (arg)
        case ("--help", "-h")
            call print_help()
        case ("--nx")
            call error(iarg + 1 > num_arg, "Missing value for --nx argument")
            call get_command_argument(iarg + 1, arg_val)
            read(arg_val, *) nx
            iarg = iarg + 1
        case ("--ny")
            call error(iarg + 1 > num_arg, "Missing value for --ny argument")
            call get_command_argument(iarg + 1, arg_val)
            read(arg_val, *) ny
            iarg = iarg + 1
        case ("--nz")
            call error(iarg + 1 > num_arg, "Missing value for --nz argument")
            call get_command_argument(iarg + 1, arg_val)
            read(arg_val, *) nz
            iarg = iarg + 1
        case ("--order", "-o")
            call error(iarg + 1 > num_arg, "Missing value for --order argument")
            call get_command_argument(iarg + 1, arg_val)
            read(arg_val, *) order
            iarg = iarg + 1
        case ("--num_halo")
            call error(iarg + 1 > num_arg, "Missing value for --num_halo argument")
            call get_command_argument(iarg + 1, arg_val)
            read(arg_val, *) num_halo
            iarg = iarg + 1
        case ("--num_iter")
            call error(iarg + 1 > num_arg, "Missing value for --num_iter argument")
            call get_command_argument(iarg + 1, arg_val)
            read(arg_val, *) num_iter
            iarg = iarg + 1
        case ("--alpha")
            call error(iarg + 1 > num_arg, "Missing value for --alpha argument")
            call get_command_argument(iarg + 1, arg_val)
            read(arg_val, *) alpha
            iarg = iarg + 1
        case default
            call error(.true., "Unknown command line argument encountered: " // trim(arg))
        end select
        iarg = iarg + 1
    end do

    ! make sure everything is set
    call error(nx == -1, 'You have to specify nx')
    call error(ny == -1, 'You have to specify ny')
    call error(nz == -1, 'You have to specify nz')
    call error(order == -1, 'You have to specify order')
    call error(num_halo == -1, 'You have to specify num_halo')
    call error(num_iter == -1, 'You have to specify num_iter')
    call error(alpha == -1.0_wp, 'You have to specify alpha')

    ! check consistency of values
    call error(nx < 0 .or. nx > 1024*1024, "Please provide a reasonable value of nx")
    call error(ny < 0 .or. ny > 1024*1024, "Please provide a reasonable value of ny")
    call error(nz < 0 .or. nz > 1024, "Please provide a reasonable value of nz")
    call error(.not. any(order == (/2, 4, 6, 8/)), "Order must be one of 2, 4, 6, or 8")
    call error(num_halo < 1 .or. num_halo > 1024, "Please provide a reasonable value of num_halo")
    call error(num_iter < 1 .or. num_iter > 1024*1024, "Please provide a reasonable value of num_iter")
    call error(alpha < 0.0_wp .or. alpha > 1.0_wp ** order, "Please provide a reasonable value of alpha")
    
end subroutine read_cmd_line_arguments


subroutine print_help()

end subroutine print_help

subroutine cleanup()
#ifdef USE_MPI
    use mpi, only : MPI_FINALIZE
#endif
#ifdef USE_IO
    use m_cdf, only : cdfend
#endif
    use m_utils, only : error
    implicit none
    
    integer :: ierror
    
    call timer_start('clean', itim_clean)

#ifdef USE_IO
    if (io_ready) then
        call cdfend()
    end if
#endif

    call timer_end(itim_clean)
    call timer_print()
    
#ifdef USE_MPI
    call MPI_FINALIZE(ierror)
    call error(ierror /= 0, 'Problem with MPI_FINALIZE', code=ierror)
#endif
    
end subroutine cleanup

end program main
