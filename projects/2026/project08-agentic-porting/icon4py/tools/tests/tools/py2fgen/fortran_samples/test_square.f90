! ICON4Py - ICON inspired code in Python and GT4Py
!
! Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
! All rights reserved.
!
! Please, refer to the LICENSE file in the root directory.
! SPDX-License-Identifier: BSD-3-Clause

program call_square_wrapper_cffi_plugin
   use, intrinsic :: iso_c_binding
   use square_plugin
   implicit none
   character(len=100) :: str_buffer
   integer(c_int) :: cdim, kdim, i, j, rc, n
   logical :: computation_correct
   real(c_double), dimension(:, :), allocatable :: input, result

   ! array dimensions
   cdim = 18
   kdim = 10

   !$ACC enter data create(input, result)

   ! allocate arrays (allocate in column-major order)
   allocate (input(cdim, kdim))
   allocate (result(cdim, kdim))

   ! initialise arrays
   input = 5.0d0
   result = 0.0d0

   !$ACC data copyin(input, result)

   ! print array shapes and values before computation
   print *, "Fortran Arrays before calling Python:"
   write (str_buffer, '("Shape of input = ", I2, ",", I2)') size(input, 1), size(input, 2)
   print *, trim(str_buffer)
   write (str_buffer, '("Shape of result = ", I2, ",", I2)') size(result, 1), size(result, 2)
   print *, trim(str_buffer)
   print *
   print *, "input = ", input
   print *
   print *, "result = ", result
   print *

#ifdef USE_SQUARE_FROM_FUNCTION
   call square_from_function(input, result, rc)
#elif USE_SQUARE_ERROR
   call square_error(input, result, rc)
#elif PROFILE_SQUARE_FROM_FUNCTION
    call square_from_function(input, result, rc)
    do n = 1, 100
    call square_from_function(input, result, rc)
    end do
#else
   call square(input, result, rc)
#endif
   if (rc /= 1) then
       print *, "Python failed with exit code = ", rc
       call exit(1)
   end if

   !$ACC update host(input, result)

   ! print array shapes and values before computation
   print *, "Fortran arrays after calling Python:"
   write (str_buffer, '("Shape of input = ", I2, ",", I2)') size(input, 1), size(input, 2)
   print *, trim(str_buffer)
   write (str_buffer, '("Shape of result = ", I2, ",", I2)') size(result, 1), size(result, 2)
   print *, trim(str_buffer)
   print *
   print *, "input = ", input
   print *
   print *, "result = ", result
   print *

   ! Assert each element of result is the square of the corresponding element in input
   computation_correct = .true.
   do i = 1, cdim
      do j = 1, kdim
         if (result(i, j) /= input(i, j)**2) then
            print *, "Error: result(", i, ",", j, ") =", result(i, j), &
               "is not the square of input(", i, ",", j, ") =", input(i, j)
            computation_correct = .false.
            exit
         end if
      end do
      if (.not. computation_correct) exit
   end do

   !$ACC end data
   !$ACC exit data delete(input, result)

   ! deallocate arrays
   deallocate (input, result)

   ! Check and print the result of the assertion
   if (computation_correct) then
      print *, "passed: result has expected values."
   else
      print *, "failed: result does not have the expected values."
   end if
end program call_square_wrapper_cffi_plugin
