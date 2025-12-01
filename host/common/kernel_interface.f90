module kernel_interface_mod
  use iso_c_binding
  implicit none

  ! Define the operation
  integer(c_int), parameter :: OP_VECTOR_ADD = 0
  integer(c_int), parameter :: OP_DOT_PRODUCT = 1
  integer(c_int), parameter :: OP_TRANSPOSE = 2
  integer(c_int), parameter :: OP_MATRIX_MUL = 3

  ! Define the data type used by each argument
  integer(c_int), parameter :: DTYPE_F32 = 0

  ! Define single argument, can be the input argument or the output argument 
  type, bind(C) :: TensorDesc
    ! data of the argument
    type(c_ptr) :: data         
    ! Shape of the argument, for a vector, this is a one element array record the size;
    ! For a 2D matrix, this is a two element array record the size in different dimension.
    type(c_ptr) :: shape       
    ! How many dimension does the argument has
    integer(c_int) :: rank     
    integer(c_int) :: dtype
  end type TensorDesc

  ! One pointer pointing to all the input arguments and output arguments
  type, bind(C) :: KernelArgs
    ! Operations on the kernel
    integer(c_int) :: op_type

    ! Number of the input arguments     
    integer(c_int) :: num_inputs
    ! Pointing to the array of tensorDesc
    type(c_ptr) :: inputs      
     
    ! Number of output arguments
    integer(c_int) :: num_outputs
    ! Pointing to first element of the array of output tensorDesc
    type(c_ptr) :: outputs     
  end type KernelArgs

  interface
    subroutine launch_kernel(args) bind(C, name="launch_kernel")
      use iso_c_binding
      import KernelArgs
      type(c_ptr), value :: args
    end subroutine
  end interface

end module kernel_interface_mod