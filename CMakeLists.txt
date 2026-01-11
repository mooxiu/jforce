cmake_minimum_required(VERSION 3.24)

# I have to put here before "project" so it knows the place of Fortran Compiler
set(InstallPath "/home/haruka/opt/llvm-project/release/install")
# set(LLVM_PREFIX "/home/muyao/projects/llvm-project/build")
set(LLVM_PREFIX ${InstallPath})
set(MLIR_PREFIX ${InstallPath})
# StableHLO does not have *Config.cmake, so we cannot do in that way

# Adding following to make CXX and Fortran compiler to be the one we assigned, if not it will still try to find the default location
set(CMAKE_CXX_COMPILER "${LLVM_PREFIX}/bin/clang++" CACHE FILEPATH "LLVM C++ Compiler" FORCE)
set(CMAKE_Fortran_COMPILER "${LLVM_PREFIX}/bin/flang-new" CACHE FILEPATH "LLVM Fortran Compiler" FORCE)

project(xla_glue_project LANGUAGES CXX Fortran)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type" FORCE)
set(CMAKE_CXX_STANDARD 17)


# -------------------- Compile Fortran Kernel Interface as A library --------------------  
add_library(fortran_kernel_interface 
  host/common/kernel_interface.f90
)


# -------------------- Compile CPP Runtime as A library --------------------  

include(FetchContent)
FetchContent_Declare(
  protobuf
  GIT_REPOSITORY https://github.com/protocolbuffers/protobuf.git
  GIT_TAG v32.1
)
FetchContent_MakeAvailable(protobuf)


add_subdirectory(third_party/protos)

add_library(xla_glue_runtime 
  src/runtime.cpp
  src/operations.h
  src/operations.cpp
  src/kernel_pointer_interface.h
  src/kernel_pointer_interface.cpp
)

target_link_libraries(xla_glue_runtime PRIVATE 
  stdc++
  xla_protos
  ${CMAKE_DL_LIBS} # For dlopen, dlclose, dlsym, etc.
)

set(THIRD_PARTY "${CMAKE_CURRENT_SOURCE_DIR}/third_party")

target_compile_definitions(xla_glue_runtime
    PRIVATE
    DEFAULT_PJRT_PLUGIN_PATH="${THIRD_PARTY}/pjrt_plugins/pjrt_c_api_gpu_plugin.so"
)


# -------------------- Macro for compiling Each Test Case --------------------  
# macro(add_xla_test TEST_NAME HOST_FILE)
#   add_executable("${TEST_NAME}" ${HOST_FILE})
#   # We need to link using fortran compiler
#   set_property(TARGET "${TEST_NAME}" PROPERTY LINKER_LANGUAGE Fortran)
#   target_link_libraries("${TEST_NAME}" PRIVATE
#     fortran_kernel_interface
#     xla_glue_runtime
#   )
# endmacro()


# -------------------- Adding Target for Each Test Case --------------------  
# add_xla_test(test_vector_add host/test_vector_addition.f90)
# add_xla_test(test_transpose host/test_transpose.f90)
# add_xla_test(test_dot_product host/test_dot_product.f90)
# add_xla_test(test_matrix_multiplication host/test_matrix_multiplication.f90)
