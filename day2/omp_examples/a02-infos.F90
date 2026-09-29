program main
    use omp_lib
    implicit none

    integer :: size, rank

    !$omp parallel private(rank, size)
    size = omp_get_num_threads()
    rank = omp_get_thread_num()

    !$omp critical(output)
    write(*,*) 'I am thread ', rank, ' of a total of ', size, ' threads'
    !$omp end critical(output)

    !$omp end parallel

end program main
