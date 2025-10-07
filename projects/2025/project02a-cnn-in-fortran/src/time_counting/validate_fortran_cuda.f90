program inference

   use, intrinsic :: iso_fortran_env, only : sp => real32

   ! Import our library for interfacing with PyTorch
   use ftorch, only : torch_model, torch_tensor, torch_kCUDA, torch_kCPU, torch_delete, &
                      torch_tensor_from_array, torch_model_load, torch_model_forward

   ! Import our tools module for testing utils
   use ftorch_test_utils, only : assert_isclose

   implicit none

   integer, parameter :: wp = sp

   call main()

contains

   subroutine main()

      integer :: num_args, ix
      character(len=128), dimension(:), allocatable :: args

      ! Set up types of input and output data
      type(torch_model) :: model
      type(torch_tensor), dimension(1) :: in_tensors
      type(torch_tensor), dimension(1) :: out_tensors

      real(wp), dimension(:,:,:,:), allocatable, target :: in_data
      real(wp), dimension(:,:,:,:), allocatable, target :: out_data

      integer, parameter :: in_dims = 4
      integer, parameter :: in_shape(in_dims) = [1, 5, 320, 320]
      integer, parameter :: out_dims = 4
      integer, parameter :: out_shape(out_dims) = [1, 1, 320, 320]

      ! Path to input data
      character(len=128) :: data_dir
      ! Binary file containing input and output tensor
      character(len=128) :: filename, out_filename

      ! Length of tensor
      integer, parameter :: tensor_length = 512000

      ! Flag for testing
      logical :: test_pass

      ! Get TorchScript model file as a command line argument
      num_args = command_argument_count()
      allocate(args(num_args))
      do ix = 1, num_args
         call get_command_argument(ix,args(ix))
      end do

      ! Process data directory argument, if provided
      if (num_args > 1) then
        data_dir = args(2)
      else
        data_dir = "./data"
      end if
      filename = trim(data_dir)//"/input_tensor.dat"
      out_filename = trim(data_dir)//"/output_fortran.dat"

      ! Allocate one-dimensional input/output arrays, based on multiplication of all input/output dimension sizes
      allocate(in_data(in_shape(1), in_shape(2), in_shape(3), in_shape(4)))
      allocate(out_data(out_shape(1), out_shape(2), out_shape(3), out_shape(4)))

      call load_data(filename, tensor_length, in_data)

      ! Create input/output tensors from the above arrays
      call torch_tensor_from_array(in_tensors(1), in_data, torch_kCUDA)

      call torch_tensor_from_array(out_tensors(1), out_data, torch_kCPU)

      ! Load ML model (edit this line to use different models)
      call torch_model_load(model, args(1), torch_kCUDA)

      ! Infer
      call torch_model_forward(model, in_tensors, out_tensors)

      ! Save results
      call write_output_to_dat(out_data, out_shape(2), out_shape(3), out_shape(4), out_filename)

      ! Cleanup
      call torch_delete(model)
      call torch_delete(in_tensors)
      call torch_delete(out_tensors)
      deallocate(in_data)
      deallocate(out_data)
      deallocate(args)

      write (*,*) "UNet inference ran successfully"

   end subroutine main

   subroutine load_data(filename, tensor_length, in_data)

      character(len=*), intent(in) :: filename
      integer, intent(in) :: tensor_length
      real(wp), dimension(:,:,:,:), intent(out) :: in_data

      real(wp) :: flat_data(tensor_length)
      integer :: ios
      character(len=100) :: ioerrmsg

      ! Read input tensor from Python script
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

      ! Reshape data to tensor input shape
      ! This assumes the data from Python was transposed before saving
      in_data = reshape(flat_data, shape(in_data))

   end subroutine load_data

   subroutine write_output_to_dat(arr, in_channels, nx, ny, filename)
        ! Write a 3D real array to a binary .dat file using stream I/O
        integer, intent(in) :: in_channels, nx, ny
        real, intent(in) :: arr(in_channels, nx, ny)
        character(len=*), intent(in) :: filename
        
        integer :: unit, ios
        
        open(unit=10, file=filename, form='unformatted', access='stream', status='replace', iostat=ios)
        if (ios /= 0) then
           print *, "Error opening file: ", filename
           stop
        end if
        
        ! Write the array in Fortran column-major order
        write(10) arr
        close(10)
        
        print *, "Array written to", trim(filename)
   end subroutine write_output_to_dat

end program inference