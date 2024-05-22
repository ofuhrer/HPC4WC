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