add_library(JitSupport STATIC
  utilities.cpp
  profiler.cpp
  kernel_pointer_interface.cpp
)

target_include_directories(JitSupport
  PUBLIC
    ${LLVM_INCLUDE_DIRS}
    ${MLIR_INCLUDE_DIRS}
    ${FLANG_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)

target_link_libraries(JitSupport
  PUBLIC
    MLIRIR
)
