program main
    use omp_lib
    implicit none

    integer :: t
    double precision :: sum

    sum = 0.0

    !$omp parallel do reduction(+:sum)
    do t = 0, 9
        sum = sum + t
    end do
    !$omp end parallel do

    write(*,*) sum

end program main

