program main
    use omp_lib
    implicit none

    integer :: i, size, rank

    !$omp parallel num_threads(3) private(size, rank, i)
    size = omp_get_num_threads()
    rank = omp_get_thread_num()

    !$omp do
    do i = 0, 5
        write(*,*) 'loop 1, iteration ', i
    end do
    !$omp end do nowait

    !$omp do
    do i = 0, 5
        write(*,*) 'loop 2, iteration ', i
    end do
    !$omp end do nowait

    !$omp end parallel

end program main
