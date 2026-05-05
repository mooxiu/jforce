add_library(JitTransforms STATIC 
  translate.cpp
  AnnotatePass.cpp
  AliasingPass.cpp
  ShapeInferPass.cpp
  TrimArgsPass.cpp
  WorkdistributeToStableHLOPass.cpp
  Passes.cpp
  AffineCFG.cpp
  CleanFIRLoopPass.cpp
  SimplifyDoLoopPass.cpp
  AffineToStableHLO.cpp
  FirLoadToAffineLoad.cpp
  FirLICMPass.cpp
  ArithRaising.cpp
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
    MLIRGPUDialect
    MLIRIR
    FIRTransforms
    HLFIRTransforms
)

# -DCMAKE_BUILD_TYPE=Debug
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  target_compile_definitions(JitTransforms PRIVATE ENABLE_XLA_DEBUG)
endif()
