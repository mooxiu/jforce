cmake_minimum_required(VERSION 3.20)
project(xla_glue LANGUAGES CXX Fortran)

set(LLVM_PREFIX "/home/haruka/Documents/workspace/projects/llvm-project/build")

# 你的 flang-new 和 clang++
set(CMAKE_CXX_COMPILER "${LLVM_PREFIX}/bin/clang++" CACHE FILEPATH "LLVM C++ Compiler" FORCE)
set(CMAKE_Fortran_COMPILER "${LLVM_PREFIX}/bin/flang-new" CACHE FILEPATH "LLVM Fortran Compiler" FORCE)
message(CXX_COMPILER="${CMAKE_CXX_COMPILER}")

set(CMAKE_CXX_STANDARD 14)

# Fortran & C++ sources
add_executable(demo
    host/host.f90
    src/runtime.cpp
)

# We need to link using fortran compiler
set_property(TARGET demo PROPERTY LINKER_LANGUAGE Fortran)

target_link_libraries(demo PRIVATE stdc++)
