cmake_minimum_required(VERSION 3.24)

# I have to put here before "project" so it knows the place of Fortran Compiler
set(InstallPath "/home/haruka/opt/llvm-project/release/install")
# set(LLVM_PREFIX "/home/muyao/projects/llvm-project/build")
set(LLVM_PREFIX ${InstallPath})
# StableHLO does not have *Config.cmake, so we cannot do in that way

# Adding following to make CXX and Fortran compiler to be the one we assigned, if not it will still try to find the default location
set(CMAKE_CXX_COMPILER "${LLVM_PREFIX}/bin/clang++" CACHE FILEPATH "LLVM C++ Compiler" FORCE)

project(xla_glue_project LANGUAGES CXX Fortran)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type" FORCE)
set(CMAKE_CXX_STANDARD 17)


list(APPEND CMAKE_PREFIX_PATH ${InstallPath})
find_package(MLIR REQUIRED CONFIG)


# -------------------- Compile CPP Runtime as A library --------------------  
include(FetchContent)

set(protobuf_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(protobuf_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(protobuf_INSTALL OFF CACHE BOOL "" FORCE)


FetchContent_Declare(
  protobuf
  GIT_REPOSITORY https://github.com/protocolbuffers/protobuf.git
  GIT_TAG v29.5
)
FetchContent_MakeAvailable(protobuf)

if(EXISTS "${protobuf_SOURCE_DIR}/cmake/protobuf-generate.cmake")
    include("${protobuf_SOURCE_DIR}/cmake/protobuf-generate.cmake")
else()
    message(WARNING "Could not find protobuf-generate.cmake via direct path")
endif()


add_subdirectory(third_party/protos)

add_library(jit-code-executor 
  src/jit-code-executor.cpp
  src/workdistribute-to-stablehlo.cpp
)

get_property(MLIR_LIBS GLOBAL PROPERTY MLIR_ALL_LIBS)

target_include_directories(jit-code-executor PRIVATE
  ${MLIR_INCLUDE_DIRS}
)

target_link_libraries(jit-code-executor PRIVATE 
  stdc++
  xla_protos
  libprotobuf
  ${CMAKE_DL_LIBS} # For dlopen, dlclose, dlsym, etc.
  ${MLIR_ALL_LIBS}
)

set(THIRD_PARTY "${CMAKE_CURRENT_SOURCE_DIR}/third_party")

target_compile_definitions(jit-code-executor
    PRIVATE
    DEFAULT_PJRT_PLUGIN_PATH="${THIRD_PARTY}/pjrt_plugins/pjrt_c_api_cpu_plugin.so"
)

