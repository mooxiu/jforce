module host_mod
  use iso_c_binding
  use kernel_interface_mod
  implicit none

  integer(c_long), parameter :: n = 33
  real(c_float), allocatable, target :: a(:), b(:), out(:)
  integer :: i

  allocate(a(n), b(n), out(n))

  ! Initialization
  do i = 1, n
    a(i) = real(i, c_float)
    b(i) = 100.0_c_float
  end do



  ! Instead of Running OpenMP, call Cpp function using ABI:
  type(tensorDesc), target, dimension(2) :: input_descs
  type(tensorDesc), target, dimension(1) :: output_descs
  type(KernelArgs), target :: args

  input_descs(1)%data = c_loc(a)
  integer(c_long), target:: shape_A(1) = n
  input_descs(1)%shape = c_loc(shape_A)
  input_descs(1)%rank = 1
  input_descs(1)%dtype = 0

  input_descs(2)%data = c_loc(b)
  integer(c_long), target:: shape_B(1) = n
  input_descs(2)%shape = c_loc(shape_B)
  input_descs(2)%rank = 1
  input_descs(2)%dtype = 0

  output_descs(1)%data = c_loc(out)
  integer(c_long), target:: shape_Out(1) = n
  output_descs(1)%shape = c_loc(shape_Out)
  output_descs(1)%rank = 1
  output_descs(1)%dtype = 0

  args%op_type = OP_VECTOR_ADD
  args%num_inputs = 2
  args%inputs = c_loc(input_descs(1))
  args%num_outputs = 1
  args%outputs = c_loc(output_descs(1))

  call launch_kernel(c_loc(args))
  ! End of offloading


  ! Verification and deallocate
  print *, "Result out = ", out

  deallocate(a, b, out)

end program
