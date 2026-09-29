program main
    use omp_lib
    implicit none

    integer :: i, seed_size
    real :: a, b
    integer, allocatable :: seed(:)

    call random_seed(size=seed_size)
    allocate(seed(seed_size))
    seed = 712
    call random_seed(put=seed)

    call random_number(a)
    a = int(a * 100)
    write(*,*) a
    write(*,*) ''

    !$omp parallel private(b)
    call random_number(b)
    b = int(b * 100)
    write(*,*) b
    !$omp end parallel

    write(*,*) ''
    write(*,*) 'this is a sequential region again'

end program main
