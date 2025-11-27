module host_mod
  use iso_c_binding
  implicit none

  interface
     subroutine launch_kernel(a, b, out, n) bind(C, name="launch_kernel")
       use iso_c_binding
       type(c_ptr), value :: a
       type(c_ptr), value :: b
       type(c_ptr), value :: out
       integer(c_long), value :: n
     end subroutine
  end interface

contains

  subroutine run_host()
    use iso_c_binding
    implicit none

    integer(c_long), parameter :: n = 8
    real(c_float), allocatable, target :: a(:), b(:), out(:)
    integer :: i

    allocate(a(n), b(n), out(n))

    ! Initialization
    do i = 1, n
      a(i) = real(i, c_float)
      b(i) = 100.0_c_float
    end do

    ! Instead of Running OpenMP, call Cpp function using ABI:
    call launch_kernel(c_loc(a), c_loc(b), c_loc(out), n)

    print *, "Result out = ", out

    deallocate(a, b, out)
  end subroutine

end module


program main
  use host_mod
  call run_host()
end program
