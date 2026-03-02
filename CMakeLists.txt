cmake_minimum_required(VERSION 3.24)

# I have to put here before "project" so it knows the place of Fortran Compiler
set(HOME $ENV{HOME})
set(CURR_ENV "LOCAL") # <- Change this for different environment
if (CURR_ENV STREQUAL "LOCAL")
  set(DEFAULT_PJRT_PLUGIN_PATH "${CMAKE_CURRENT_SOURCE_DIR}/third_party/pjrt_plugins/pjrt_c_api_cpu_plugin.so")
  set(TARGET_DEVICE_NAME "CPU")
  set(LLVM_INSTALL_PATH "${HOME}/Documents/workspace/projects/llvm-project/build")
  set(STABLEHLO_PROJECT "${HOME}/Documents/workspace/projects/stablehlo")
elseif (CURR_ENV STREQUAL "REMOTE")
  set(DEFAULT_PJRT_PLUGIN_PATH "${CMAKE_CURRENT_SOURCE_DIR}/third_party/pjrt_plugins/pjrt_c_api_gpu_plugin.so")
  set(TARGET_DEVICE_NAME "CUDA")
  set(LLVM_INSTALL_PATH "${HOME}/projects/llvm-project/build")
  set(STABLEHLO_PROJECT "${HOME}/projects/stablehlo")
endif()

# Adding following to make CXX and Fortran compiler to be the one we assigned, if not it will still try to find the default location
set(CMAKE_CXX_COMPILER "${LLVM_INSTALL_PATH}/bin/clang++" CACHE FILEPATH "LLVM C++ Compiler" FORCE)

project(xla_glue_project)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
if (no CMAKE_BUILD_TYPE) 
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
set(CMAKE_CXX_STANDARD 17)

# -------------------- Find MLIR, LLVM, Clang, Flang --------------------  
list(APPEND CMAKE_PREFIX_PATH ${LLVM_INSTALL_PATH})
find_package(MLIR REQUIRED CONFIG)
find_package(LLVM REQUIRED CONFIG)
find_package(Clang REQUIRED CONFIG)
find_package(Flang REQUIRED CONFIG)
# StableHLO does not have *Config.cmake, so we cannot do in that way

# -------------------- Find StableHLO --------------------  
# set(STABLEHLO_SRC_PATH "${STABLEHLO_PROJECT}/stablehlo")
set(STABLEHLO_BUILD_PATH "${STABLEHLO_PROJECT}/build")
set(STABLEHLO_STATIC_LIBS "${STABLEHLO_BUILD_PATH}/lib")
message(STATUS "[DEBUG] THE stablehlo project: ${STABLEHLO_PROJECT}")
message(STATUS "[DEBUG] THE stablehlo build: ${STABLEHLO_BUILD_PATH}")
file(GLOB stablehlo_all_libs ${STABLEHLO_STATIC_LIBS}/lib*.a)
include_directories(
  ${STABLEHLO_PROJECT}
  ${STABLEHLO_BUILD_PATH} # Important for generated .td files
)


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

add_library(jit-code-executor SHARED 
  src/jit-code-executor.cpp
  src/workdistribute-to-stablehlo.cpp
  src/workdistribute-shape-inference.cpp
  src/workdistribute-shape-transform.cpp
  src/kernel_launcher.cpp
  src/kernel_pointer_interface.cpp
  src/utilities.cpp
  src/jit-manager.cpp
)

get_property(MLIR_LIBS GLOBAL PROPERTY MLIR_ALL_LIBS)
get_property(LLVM_LIBS GLOBAL PROPERTY LLVM_ALL_LIBS)
get_property(FLANG_LIBS GLOBAL PROPERTY FLANG_ALL_LIBS)

target_include_directories(jit-code-executor PRIVATE
  ${MLIR_INCLUDE_DIRS}
  ${LLVM_INCLUDE_DIRS}
  ${FLANG_INCLUDE_DIRS}
  ${STABLEHLO_PROJECT}
)

target_link_libraries(jit-code-executor PRIVATE 
  stdc++
  xla_protos
  libprotobuf
  ${CMAKE_DL_LIBS} # For dlopen, dlclose, dlsym, etc.

  #${stablehlo_all_libs}
  -Wl,--start-group ${stablehlo_all_libs} -Wl,--end-group
  FIRDialect
  HLFIRDialect
  ${MLIR_LIBS} # TODO: Trim linked MLIR libs to save compile time, same for LLVM and Flang
  ${LLVM_LIBS}
)

# Pass flags
target_compile_definitions(jit-code-executor PRIVATE
  DEFAULT_PJRT_PLUGIN_PATH="${DEFAULT_PJRT_PLUGIN_PATH}"
)

target_compile_definitions(jit-code-executor PRIVATE
  TARGET_DEVICE="${TARGET_DEVICE_NAME}"
)
