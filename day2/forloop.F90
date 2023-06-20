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
  
  ! Pragmas here?
      do i = 1, N
        ! Pragmas here?
          rank = 1 ! YOUR IMPLEMENTATION HERE
          iteration = 1 ! YOUR IMPLEMENTATION HERE
          values(iteration) = rank
          write(output, '(A, I0, A, I0)') "Thread ", rank, " executed loop iteration ", iteration
          write(*, '(A)') trim(output)
        ! Pragmas here?
      end do
  ! Pragmas here?
  
  deallocate(values)
  
end program omp_example