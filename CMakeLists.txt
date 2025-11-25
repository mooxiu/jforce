cmake_minimum_required(VERSION 3.20)
project(xla_glue LANGUAGES CXX Fortran)

set(LLVM_PREFIX "/home/haruka/Documents/workspace/projects/llvm-project/build")

# 你的 flang-new 和 clang++
set(CMAKE_CXX_COMPILER "${LLVM_PREFIX}/bin/clang++" CACHE FILEPATH "LLVM C++ Compiler" FORCE)
set(CMAKE_Fortran_COMPILER "${LLVM_PREFIX}/bin/flang-new" CACHE FILEPATH "LLVM Fortran Compiler" FORCE)
message(CXX_COMPILER="${CMAKE_CXX_COMPILER}")

set(CMAKE_CXX_STANDARD 14)

set(DEFAULT_PLUGIN_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/pjrt_plugins")

# Fortran & C++ sources
add_executable(demo
    host/host.f90
    src/runtime.cpp
)

target_compile_definitions(demo
    PRIVATE
    # 注意：宏定义需要双引号来保持路径中的空格或特殊字符
    DEFAULT_PJRT_PLUGIN_PATH="${DEFAULT_PLUGIN_DIR}/pjrt_c_api_cpu_plugin.so"
)


# We need to link using fortran compiler
set_property(TARGET demo PROPERTY LINKER_LANGUAGE Fortran)

target_link_libraries(demo PRIVATE stdc++)
