program main
    use omp_lib
    implicit none

    integer :: size, rank

    !$omp parallel private(size, rank)
    size = omp_get_num_threads()
    rank = omp_get_thread_num()

    !$omp single
    write(*,*) 'thread ', rank, ' is present in single'
    write(*,*) 'and the size here is : ', omp_get_num_threads()
    !$omp end single

    !$omp master
    write(*,*) 'thread ', rank, ' is present in master'
    !$omp end master

    !$omp critical(somethingHard)
    write(*,*) 'thread ', rank, ' is present in critical'
    !$omp end critical(somethingHard)

    !$omp end parallel

end program main
