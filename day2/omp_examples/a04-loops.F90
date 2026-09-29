program main
    use omp_lib
    implicit none

    integer :: i

    !$omp parallel do
    do i = 0, 9
        !$omp critical(output)
        write(*,*) 'This is iteration ', i, ' executed from thread ', omp_get_thread_num()
        !$omp end critical(output)
    end do
    !$omp end parallel do

end program main
