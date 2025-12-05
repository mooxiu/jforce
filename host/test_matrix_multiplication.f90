program test_matrix_multiplication
  use iso_c_binding
  use kernel_interface_mod
  implicit none

  integer(c_long), parameter :: X = 20
  integer(c_long), parameter :: Y = 30
  integer(c_long), parameter :: Z = 40
  real(c_float), allocatable, target :: a(:, :), b(:, :), out(:, :)
  integer :: i, j

  allocate(a(X, Y), b(Y, Z), out(X, Z))

  ! Initialization
  do i = 1, X
    do j = 1, Y
      a(i, j) = real((i-1) * Y + j, c_float)
    end do
  end do

  do i = 1, Y
    do j = 1, Z
      b(i, j) = 1
    end do
  end do

  print *, "Print the input matrix a: "
  do i = 1, size(a, 1)  
      print *, a(i, :)  
  end do

  print *, "Print the input matrix b: "
  do i = 1, size(b, 1)  
      print *, b(i, :)  
  end do

  ! Instead of Running OpenMP, call Cpp function using ABI:
  block
    type(tensorDesc), target, dimension(2) :: input_descs
    type(tensorDesc), target, dimension(1) :: output_descs
    type(KernelArgs), target :: args

    integer(c_long), target :: shape_A(2)
    integer(c_long), target :: shape_B(2)
    integer(c_long), target :: shape_Out(2)

    ! Notice that in fortran, the matrix is stored by column!!!!!!!!
    ! That's why we need to pass by column, row
    shape_A(1) = Y
    shape_A(2) = X

    shape_B(1) = Z
    shape_B(2) = Y

    shape_Out(1) = X
    shape_Out(2) = Z


    input_descs(1)%data = c_loc(a)
    input_descs(1)%shape = c_loc(shape_A)
    input_descs(1)%rank = 2
    input_descs(1)%dtype = 0

    input_descs(2)%data = c_loc(b)
    input_descs(2)%shape = c_loc(shape_B)
    input_descs(2)%rank = 2
    input_descs(2)%dtype = 0


    output_descs(1)%data = c_loc(out)
    output_descs(1)%shape = c_loc(shape_Out)
    output_descs(1)%rank = 2
    output_descs(1)%dtype = 0

    args%op_type = OP_MATRIX_MUL
    args%num_inputs = 2
    args%inputs = c_loc(input_descs(1))
    args%num_outputs = 1
    args%outputs = c_loc(output_descs(1))

    call launch_kernel(c_loc(args))
  end block
  ! End of offloading


  ! Verification and deallocate
  print *, "Print the output matrix: "
  do i = 1, size(out, 1)  
      print *, out(i, :)  
  end do

  deallocate(a, b, out)

end program test_matrix_multiplication 
