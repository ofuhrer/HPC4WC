program main
    use omp_lib
    implicit none

    integer, parameter :: n = 10000000
    integer :: nThreads, i
    double precision :: tic, toc
    double precision, dimension(n) :: input, output
    character(len=20) :: arg

    input = 1.0d0
    output = 0.0d0

    call get_command_argument(1, arg)
    read(arg,*) nThreads

    call omp_set_num_threads(nThreads)

    tic = omp_get_wtime()

    !$omp parallel do schedule(static, 10000)
    do i = 1, n
        output(i) = 2.0d0 * input(i)
        input(i) = 0.0d0
    end do
    !$omp end parallel do

    toc = omp_get_wtime()

    !$omp parallel
    if (omp_get_thread_num() == 0) then
        write(*,*) omp_get_num_threads(), toc - tic
    end if
    !$omp end parallel

end program main
