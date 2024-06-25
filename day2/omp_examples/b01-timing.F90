program main
    use omp_lib
    implicit none

    integer :: nThreads, steps, t
    double precision :: tic, toc, sum
    character(len=20) :: arg

    call get_command_argument(1, arg)
    read(arg,*) nThreads

    call omp_set_num_threads(nThreads)

    tic = omp_get_wtime()

    steps = 10000000
    sum = 0.0

    !$omp parallel do reduction(+:sum)
    do t = 0, steps - 1
        sum = sum + (1.0d0 - 2.0d0 * mod(t, 2)) / (2 * t + 1)
    end do
    !$omp end parallel do

    toc = omp_get_wtime()

    !$omp parallel
    if (omp_get_thread_num() == 0) then
        write(*,*) omp_get_num_threads(), toc - tic
    end if
    !$omp end parallel

end program main
