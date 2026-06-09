! hpc4wc:student-begin
! hpc4wc:student | program omp_example
! hpc4wc:student |   use omp_lib
! hpc4wc:student |   implicit none
! hpc4wc:student |
! hpc4wc:student |   integer :: N, i, rank, iteration
! hpc4wc:student |   character(len=100) :: output
! hpc4wc:student |   character(len=256) :: arg
! hpc4wc:student |   integer, allocatable :: values(:)
! hpc4wc:student |
! hpc4wc:student |   call get_command_argument(1, arg)
! hpc4wc:student |   read(arg, *) N
! hpc4wc:student |   allocate(values(N))
! hpc4wc:student |   values = -1
! hpc4wc:student |
! hpc4wc:student |   ! Pragmas here?
! hpc4wc:student |       do i = 1, N
! hpc4wc:student |         ! Pragmas here?
! hpc4wc:student |           rank = 1 ! YOUR IMPLEMENTATION HERE
! hpc4wc:student |           iteration = 1 ! YOUR IMPLEMENTATION HERE
! hpc4wc:student |           values(iteration) = rank
! hpc4wc:student |           write(output, '(A, I0, A, I0)') "Thread ", rank, " executed loop iteration ", iteration
! hpc4wc:student |           write(*, '(A)') trim(output)
! hpc4wc:student |         ! Pragmas here?
! hpc4wc:student |       end do
! hpc4wc:student |   ! Pragmas here?
! hpc4wc:student |
! hpc4wc:student |   deallocate(values)
! hpc4wc:student |
! hpc4wc:student | end program omp_example
! hpc4wc:student-end
! hpc4wc:solution-begin
program omp_example
  use omp_lib
  implicit none

  integer :: N, i, rank, iteration
  character(len=100) :: output
  character(len=256) :: arg
  integer, allocatable :: values(:)

  call get_command_argument(1, arg)
  read(arg, *) N
  allocate(values(N))
  values = -1

  !$omp parallel num_threads(10)
    !$omp single
      do i = 1, N
        !$omp task default(none) firstprivate(i) private(rank,iteration, output) shared(values)
          rank = omp_get_thread_num()
          iteration = i
          values(iteration) = rank
          write(output, '(A, I0, A, I0)') "Thread ", rank, " executed loop iteration ", iteration
          write(*, '(A)') trim(output)
        !$omp end task
      end do
    !$omp end single
  !$omp end parallel

  deallocate(values)

end program omp_example
! hpc4wc:solution-end
