program main
    use omp_lib
    implicit none

    integer :: t
    double precision :: sum

    sum = 0.0

    ! Parallel loop with reduction on sum
    !$omp parallel do reduction(+:sum)
    do t = 0, 9
        sum = sum + t
    end do
    !$omp end parallel do

    ! Print the result
    write(*,*) sum

end program main

