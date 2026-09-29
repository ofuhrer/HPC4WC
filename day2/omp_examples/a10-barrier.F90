program main
    use omp_lib
    implicit none

    integer :: size, rank

    !$omp parallel num_threads(5) private(size, rank)
    size = omp_get_num_threads()
    rank = omp_get_thread_num()

    !$omp critical(somethingHard)
    write(*,*) 'thread ', rank, ' is present in critical1'
    !$omp end critical(somethingHard)

    !$omp critical(somethingEasy)
    write(*,*) 'thread ', rank, ' is present in critical2'
    !$omp end critical(somethingEasy)

    !$omp end parallel

end program main

