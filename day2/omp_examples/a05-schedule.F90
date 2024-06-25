program main
    use omp_lib
    implicit none

    integer :: i

    write(*,*) 'schedule(static, 2)'

    !$omp parallel do schedule(static, 2)
    do i = 0, 9
        !$omp critical(output)
        write(*,*) 'This is iteration ', i, ' executed from thread ', omp_get_thread_num()
        !$omp end critical(output)
    end do
    !$omp end parallel do

    write(*,*) 'schedule(static, 1)'

    !$omp parallel do schedule(static, 1)
    do i = 0, 9
        !$omp critical(output)
        write(*,*) 'This is iteration ', i, ' executed from thread ', omp_get_thread_num()
        !$omp end critical(output)
    end do
    !$omp end parallel do

end program main
