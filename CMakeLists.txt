cmake_minimum_required(VERSION 3.24)

# I have to put here before "project" so it knows the place of Fortran Compiler
set(InstallPath "/home/haruka/opt/llvm-project/release/install")
# set(LLVM_PREFIX "/home/muyao/projects/llvm-project/build")
set(LLVM_PREFIX ${InstallPath})
# StableHLO does not have *Config.cmake, so we cannot do in that way

# Adding following to make CXX and Fortran compiler to be the one we assigned, if not it will still try to find the default location
set(CMAKE_CXX_COMPILER "${LLVM_PREFIX}/bin/clang++" CACHE FILEPATH "LLVM C++ Compiler" FORCE)

project(xla_glue_project LANGUAGES CXX Fortran)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type" FORCE)
set(CMAKE_CXX_STANDARD 17)

# -------------------- Find MLIR, LLVM, Clang, Flang --------------------  
list(APPEND CMAKE_PREFIX_PATH ${InstallPath})
find_package(MLIR REQUIRED CONFIG)
find_package(LLVM REQUIRED CONFIG)
find_package(Clang REQUIRED CONFIG)
find_package(Flang REQUIRED CONFIG)

# -------------------- Find StableHLO --------------------  
set(STABLEHLO_PROJECT "/home/haruka/Documents/workspace/projects/stablehlo")
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
  src/kernel_launcher.cpp
  src/kernel_pointer_interface.cpp
  src/utilities.cpp
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

set(THIRD_PARTY "${CMAKE_CURRENT_SOURCE_DIR}/third_party")

target_compile_definitions(jit-code-executor
    PRIVATE
    DEFAULT_PJRT_PLUGIN_PATH="${THIRD_PARTY}/pjrt_plugins/pjrt_c_api_cpu_plugin.so"
)

# -------------------- Macro for compiling tests --------------------  
macro(add_test TEST_NAME SRC_FILE)
  add_executable(${TEST_NAME} ${SRC_FILE})
  target_link_libraries(${TEST_NAME} PRIVATE jit-code-executor)
endmacro()


add_test(test_1_dummy tests/test1.cpp)
