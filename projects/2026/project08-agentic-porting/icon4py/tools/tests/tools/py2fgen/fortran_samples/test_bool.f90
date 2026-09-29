! ICON4Py - ICON inspired code in Python and GT4Py
!
! Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
! All rights reserved.
!
! Please, refer to the LICENSE file in the root directory.
! SPDX-License-Identifier: BSD-3-Clause

program call_bool_wrapper_cffi_plugin
   use, intrinsic :: iso_c_binding
   use bool_plugin
   implicit none
   integer(c_int) :: rc, i, n
   logical(kind=c_bool) :: flag
   logical(kind=c_bool), dimension(:), allocatable :: mask
   logical :: computation_correct

   n = 10
   allocate (mask(n))
   flag = .true.
   mask = .false.

   call fill_mask(flag, mask, rc)
   if (rc /= 1) then
      print *, "Python failed with exit code = ", rc
      stop 1
   end if

   ! fill_mask writes `flag` (.true.) into every element of `mask`; this checks
   ! both that the scalar arrived as true and that the array write propagated.
   computation_correct = .true.
   do i = 1, n
      if (.not. mask(i)) then
         computation_correct = .false.
         exit
      end if
   end do

   deallocate (mask)

   if (computation_correct) then
      print *, "passed: result has expected values."
   else
      print *, "failed: result does not have the expected values."
   end if
end program call_bool_wrapper_cffi_plugin
