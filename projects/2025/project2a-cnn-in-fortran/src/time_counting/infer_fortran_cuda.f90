program inference

   use, intrinsic :: iso_fortran_env, only : sp => real32, int64
   use ftorch, only : torch_model, torch_tensor, torch_kCUDA, torch_kCPU, torch_delete, &
                      torch_tensor_from_array, torch_model_load, torch_model_forward
   use ftorch_test_utils, only : assert_isclose

   implicit none
   integer, parameter :: wp = sp

   ! C interface for sync
   interface
       subroutine cuda_sync() bind(C, name="cuda_sync_")
       end subroutine
   end interface

   call main()

contains

   subroutine main()

      integer :: num_args, ix, num_runs, batch_size, i
      character(len=128), dimension(:), allocatable :: args
      type(torch_model) :: model
      type(torch_tensor), dimension(1) :: in_tensors, out_tensors
      real(wp), dimension(:,:,:,:), allocatable, target :: in_data, out_data
      integer, parameter :: in_dims = 4, out_dims = 4
      integer, parameter :: in_shape(in_dims) = [1,5,320,320]
      integer, parameter :: out_shape(out_dims) = [1,1,320,320]
      character(len=128) :: data_dir, filename, out_filename
      integer, parameter :: tensor_length = 512000
      real(wp), dimension(:), allocatable :: inference_times
      integer(int64) :: start_count, end_count, count_rate
      real(wp) :: data_prep_time, model_load_time, start_time, end_time
      real(wp) :: avg_inference, std_inference, throughput

      ! Get arguments from command lines
      num_args = command_argument_count()
      allocate(args(num_args))
      do ix = 1, num_args
         call get_command_argument(ix,args(ix))
      end do

      if (num_args > 1) then
         data_dir = args(2)
      else
         data_dir = "../data"
      end if
      filename = trim(data_dir)//"/input_tensor.dat"
      out_filename = trim(data_dir)//"/output_fortran.dat"

      ! Date Preparation
      call cuda_sync()  ! sync
      call system_clock(start_count, count_rate)
      allocate(in_data(in_shape(1), in_shape(2), in_shape(3), in_shape(4)))
      allocate(out_data(out_shape(1), out_shape(2), out_shape(3), out_shape(4)))
      call load_data(filename, tensor_length, in_data)
      call torch_tensor_from_array(in_tensors(1), in_data, torch_kCUDA)
      call torch_tensor_from_array(out_tensors(1), out_data, torch_kCUDA)
      call cuda_sync()  ! sync
      call system_clock(end_count)
      data_prep_time = real(end_count - start_count, wp) / real(count_rate, wp)

      ! Model load
      call cuda_sync()  ! sync
      call system_clock(start_count)
      call torch_model_load(model, args(1), torch_kCUDA)
      call cuda_sync()  ! sync
      call system_clock(end_count)
      model_load_time = real(end_count - start_count, wp) / real(count_rate, wp)

      ! Warm up
      do i = 1, 10
         call torch_model_forward(model, in_tensors, out_tensors)
      end do

      ! Inference
      num_runs = 100
      allocate(inference_times(num_runs))
      do ix = 1, num_runs
         call cuda_sync()  ! sync
         call system_clock(start_count)
         call torch_model_forward(model, in_tensors, out_tensors)
         call cuda_sync()  ! sync
         call system_clock(end_count)
         inference_times(ix) = real(end_count - start_count, wp) / real(count_rate, wp)
      end do

      print *, inference_times
      avg_inference = sum(inference_times)/num_runs
      std_inference = sqrt(sum((inference_times - avg_inference)**2)/num_runs)
      batch_size = in_shape(1)
      throughput = batch_size / avg_inference

      ! Print Result
      print *, "=== Fortran (ftorch) Performance Results ==="
      print *, "Device: CUDA"
      print *, " Data preparation time: ", data_prep_time, " sec"
      print *, " Model load time:       ", model_load_time, " sec"
      print *, " Average inference (", num_runs, " runs): ", avg_inference, " sec"
      print *, " Throughput:           ", throughput, " samples/sec"

      ! Clean
      call torch_delete(model)
      call torch_delete(in_tensors)
      call torch_delete(out_tensors)
      deallocate(in_data)
      deallocate(out_data)
      deallocate(args)
      deallocate(inference_times)

      print *, "UNet inference ran successfully"
   end subroutine main

   ! -------------------- load_data --------------------
   subroutine load_data(filename, tensor_length, in_data)
      character(len=*), intent(in) :: filename
      integer, intent(in) :: tensor_length
      real(wp), dimension(:,:,:,:), intent(out) :: in_data
      real(wp) :: flat_data(tensor_length)
      integer :: ios
      character(len=100) :: ioerrmsg

      open(unit=10, file=filename, status='old', access='stream', form='unformatted', &
           action="read", iostat=ios, iomsg=ioerrmsg)
      if (ios /= 0) then
         print *, ioerrmsg
         stop 1
      end if

      read(10, iostat=ios, iomsg=ioerrmsg) flat_data
      if (ios /= 0) then
         print *, ioerrmsg
         stop 1
      end if

      close(10)
      in_data = reshape(flat_data, shape(in_data))
   end subroutine load_data

end program inference