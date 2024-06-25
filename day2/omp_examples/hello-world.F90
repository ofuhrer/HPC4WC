program hello_world
    use omp_lib
    implicit none

    !$omp parallel
    print *, 'Hello, world.'
    !$omp end parallel

end program hello_world
