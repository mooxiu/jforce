add_library(JitTransforms STATIC 
  aliasingPass.cpp
  shapeInferPass.cpp
  trimArgsPass.cpp
  workdistributeToStableHLOPass.cpp
  passes.cpp
)

target_include_directories(JitTransforms
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)

target_link_libraries(JitTransforms
  PRIVATE
    FIRDialect
    HLFIRDialect
  PUBLIC
    JitSupport
    MLIRPass
)
