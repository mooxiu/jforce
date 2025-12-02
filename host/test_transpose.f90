program transpose_demo
  use iso_c_binding
  use kernel_interface_mod
  implicit none

  integer(c_long), parameter :: M = 7
  integer(c_long), parameter :: N = 9
  real(c_float), allocatable, target :: a(:, :), out(:, :)
  integer :: i, j

  allocate(a(M, N), out(N, M))

  ! Initialization
  do i = 1, M
    do j = 1, N
      a(i, j) = real(i * M + j, c_float)
    end do
  end do


  ! Instead of Running OpenMP, call Cpp function using ABI:
  block
    type(tensorDesc), target, dimension(1) :: input_descs
    type(tensorDesc), target, dimension(1) :: output_descs
    type(KernelArgs), target :: args

    integer(c_long), target:: shape_A(M,N)
    integer(c_long), target:: shape_Out(N,M)


    input_descs(1)%data = c_loc(a)
    input_descs(1)%shape = c_loc(shape_A)
    input_descs(1)%rank = 2
    input_descs(1)%dtype = 0

    output_descs(1)%data = c_loc(out)
    output_descs(1)%shape = c_loc(shape_Out)
    output_descs(1)%rank = 2
    output_descs(1)%dtype = 0

    args%op_type = OP_TRANSPOSE  
    args%num_inputs = 1
    args%inputs = c_loc(input_descs(1))
    args%num_outputs = 1
    args%outputs = c_loc(output_descs(1))

    call launch_kernel(c_loc(args))
  end block
  ! End of offloading


  ! Verification and deallocate
  print *, "Result out = ", out

  deallocate(a, out)

end program transpose_demo 
