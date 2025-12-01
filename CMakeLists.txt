cmake_minimum_required(VERSION 3.20)

# I have to put here before "project" so it knows the place of Fortran Compiler
set(LLVM_PREFIX "/home/haruka/Documents/workspace/projects/llvm-project/build")
set(CMAKE_CXX_COMPILER "${LLVM_PREFIX}/bin/clang++" CACHE FILEPATH "LLVM C++ Compiler" FORCE)
set(CMAKE_Fortran_COMPILER "${LLVM_PREFIX}/bin/flang-new" CACHE FILEPATH "LLVM Fortran Compiler" FORCE)
project(xla_glue LANGUAGES CXX Fortran)
# set(TARGET_NAME demo)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type" FORCE)
set(CMAKE_CXX_STANDARD 14)

#-----------------------
find_package(Protobuf REQUIRED)
add_subdirectory(third_party/protos)

set(COMMON_SOURCES
    host/common/kernel_interface.f90
    src/runtime.cpp
    src/operations.h
    src/operations.cpp
    src/kernel_pointer_interface.h
    src/kernel_pointer_interface.cpp
)


# ----------------- LINK

set(THIRD_PARTY "${CMAKE_CURRENT_SOURCE_DIR}/third_party")
set(DEFAULT_PLUGIN_DIR "${THIRD_PARTY}/pjrt_plugins")


macro(add_xla_test TEST_NAME HOST_FILE)
    add_executable("${TEST_NAME}"
        ${HOST_FILE}
        ${COMMON_SOURCES}
    )

    target_compile_definitions("${TEST_NAME}"
        PRIVATE
        DEFAULT_PJRT_PLUGIN_PATH="${DEFAULT_PLUGIN_DIR}/pjrt_c_api_cpu_plugin.so"
    )

    # We need to link using fortran compiler
    set_property(TARGET "${TEST_NAME}" PROPERTY LINKER_LANGUAGE Fortran)

    target_link_libraries("${TEST_NAME}" PRIVATE 
        stdc++
        xla_protos
        protobuf::libprotobuf
    )
endmacro()


add_xla_test(test_vector_add host/host.f90)