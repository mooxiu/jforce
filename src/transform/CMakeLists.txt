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
  PUBLIC
    JitSupport
    MLIRPass
  PRIVATE
    -Wl,--start-group ${stablehlo_all_libs}  -Wl,--end-group
    FIRDialect
    HLFIRDialect
    MLIRShapeDialect
    MLIRQuantDialect
    MLIRMathDialect
    MLIRArithDialect
    MLIRIR
)
