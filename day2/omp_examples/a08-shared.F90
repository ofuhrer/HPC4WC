program main
    use omp_lib
    implicit none

    integer :: i, myvar
    myvar = -1

    !$omp parallel do shared(myvar)
    do i = 0, 9
        !$omp critical(output)
        write(*,*) 'before writing: i is ', i, ' and myvar is ', myvar
        !$omp end critical(output)

        myvar = i

        !$omp critical(output)
        write(*,*) 'after writing: i is ', i, ' and myvar is ', myvar
        !$omp end critical(output)
    end do
    !$omp end parallel do

    write(*,*) 'myvar: ', myvar

end program main

