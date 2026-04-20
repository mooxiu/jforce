add_library(transforms STATIC 
  aliasingPass.cpp
  shapeInferPass.cpp
  trimArgsPass.cpp
  workdistributeToStableHLOPass.cpp
  passes.cpp
)

target_include_directories(transforms 
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)

target_link_libraries(transforms 
  PRIVATE
    FIRDialect
    HLFIRDialect
    MLIRFuncDialect
  PUBLIC
    JitSupport
    MLIRPass
)
