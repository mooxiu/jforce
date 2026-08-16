cmake_minimum_required(VERSION 3.24)

# I have to put here before "project" so it knows the place of Fortran Compiler
set(HOME "$ENV{HOME}")

set(LLVM_INSTALL_PATH
    "${HOME}/projects/llvm-project/build"
    CACHE PATH "LLVM build or installation directory")

set(STABLEHLO_PROJECT
    "${HOME}/projects/stablehlo"
    CACHE PATH "StableHLO source directory")

# When commenting this line, need also to assign spack env on remote machines! 
# Adding following to make CXX and Fortran compiler to be the one we assigned, if not it will still try to find the default location
# set(CMAKE_CXX_COMPILER "${LLVM_INSTALL_PATH}/bin/clang++" CACHE FILEPATH "LLVM C++ Compiler" FORCE)
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fuse-ld=lld")
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -fuse-ld=lld")

project(xla_glue_project)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
if (NOT CMAKE_BUILD_TYPE) 
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
option(ENABLE_PROFILING "Profiling the execution or not, default off" OFF)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-rtti -fno-exceptions")

# -------------------- Find MLIR, LLVM, Clang, Flang --------------------  
list(APPEND CMAKE_PREFIX_PATH ${LLVM_INSTALL_PATH})
find_package(MLIR REQUIRED CONFIG)
find_package(LLVM REQUIRED CONFIG)
find_package(Clang REQUIRED CONFIG)
find_package(Flang REQUIRED CONFIG)

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
# include(TableGen)
include(AddLLVM)
include(AddMLIR)

# StableHLO does not have *Config.cmake, so we cannot do in that way

set(LLVM_LIT_ARGS "-sv" CACHE STRING "Default options for lit")

set(JFORCE_SOURCE_DIR "${PROJECT_SOURCE_DIR}")
set(JFORCE_BINARY_DIR "${PROJECT_BINARY_DIR}")
set(JFORCE_TOOLS_DIR "${PROJECT_BINARY_DIR}/src/tool")

option(JFORCE_INCLUDE_TESTS "Enable JForce tests" ON)


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

get_property(LLVM_LIBS GLOBAL PROPERTY LLVM_ALL_LIBS)
get_property(MLIR_LIBS GLOBAL PROPERTY MLIR_ALL_LIBS)

add_subdirectory(third_party/protos)
add_subdirectory(src/support)
add_subdirectory(src/transform)
add_subdirectory(src/tool)


add_library(jit-code-executor SHARED 
  src/runtime/jit-code-executor.cpp
  src/runtime/kernel_launcher.cpp
  src/runtime/jit-manager.cpp
)

# already public in transforms
target_include_directories(jit-code-executor PRIVATE
)

target_link_libraries(jit-code-executor PRIVATE 
  xla_protos
  libprotobuf
  ${CMAKE_DL_LIBS} # For dlopen, dlclose, dlsym, etc.
  JitSupport
  JitTransforms
)

# -DENABLE_PROFILING
if (ENABLE_PROFILING)
  target_compile_definitions(jit-code-executor PRIVATE ENABLE_PROFILING)
endif()

# -DCMAKE_BUILD_TYPE=Debug
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  target_compile_definitions(jit-code-executor PRIVATE ENABLE_XLA_DEBUG)
endif()

if(JFORCE_INCLUDE_TESTS)
  enable_testing()
  add_subdirectory(test)
endif()
