program main
    use omp_lib
    implicit none

    integer :: i, myvar
    myvar = -1

    ! Parallel loop with 3 threads
    !$omp parallel do num_threads(3)
    do i = 0, 9
        myvar = i
        !$omp critical(output)
        write(*,*) 'i is ', i, ' and myvar is ', myvar
        !$omp end critical(output)
    end do
    !$omp end parallel do

    ! Print final value of myvar
    write(*,*) 'myvar: ', myvar

end program main
