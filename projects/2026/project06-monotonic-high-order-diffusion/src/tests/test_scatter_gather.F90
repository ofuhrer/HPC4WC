! Program: test_scatter_gather
!  Authors: Stefanie Boersig <stefanie.boersig@env.ethz.ch>
!           Boaz Ko <boazko@student.ethz.ch>
!           Ben Bullinger <ben.bullinger@inf.ethz.ch>
!
! Scatters a global field and gathers it back; the result must equal the
! input on the interior. The field carries a globally unique pattern.
program test_scatter_gather
    use mpi, only: MPI_COMM_WORLD, MPI_INIT, MPI_FINALIZE
    use m_partitioner, only: Partitioner
    implicit none

    integer, parameter :: wp = 4
    integer :: nx = 64, ny = 64, nz = 2, num_halo = 2
    real (kind=wp), allocatable :: g(:,:,:), f(:,:,:), back(:,:,:)
#ifdef GNU_MPI_WORKAROUND
    real (kind=wp), allocatable :: dummy(:,:,:)
#endif
    type(Partitioner) :: p
    integer :: i, j, k, ierror, nerr
    logical :: master

    call MPI_INIT(ierror)
    p = Partitioner(MPI_COMM_WORLD, (/nx, ny, nz/), num_halo, periodic=(/.true., .true./))
    master = ( p%rank() == 0 )

    if ( master ) then
        allocate( g(nx + 2*num_halo, ny + 2*num_halo, nz) )
        g = -1.0_wp
        do k = 1, nz
        do j = 1, ny + 2*num_halo
        do i = 1, nx + 2*num_halo
            g(i, j, k) = real(i + 1000*j + 1000000*k, wp)
        end do
        end do
        end do
#ifdef GNU_MPI_WORKAROUND
    else
        allocate( dummy(1,1,1) )
#endif
    end if

#ifdef GNU_MPI_WORKAROUND
    if ( master ) then
        f = p%scatter(g, root=0)
    else
        f = p%scatter(dummy, root=0)
    end if
#else
    f = p%scatter(g, root=0)
#endif

    back = p%gather(f, root=0)

    if ( master ) then
        nerr = 0
        do k = 1, nz
        do j = 1 + num_halo, ny + num_halo
        do i = 1 + num_halo, nx + num_halo
            if ( back(i, j, k) /= g(i, j, k) ) then
                nerr = nerr + 1
                if ( nerr <= 10 ) &
                    write(*,'(a,3i5,a,f12.1,a,f12.1)') 'MISMATCH at', i, j, k, &
                        ': got', back(i,j,k), ' want', g(i,j,k)
            end if
        end do
        end do
        end do
        write(*,'(a,i8,a)') 'scatter/gather roundtrip: ', nerr, ' interior mismatches'
    end if

    call MPI_FINALIZE(ierror)
end program test_scatter_gather
