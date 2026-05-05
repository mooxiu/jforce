add_library(JitTransforms STATIC 
  TranslatePass.cpp
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
)

target_include_directories(JitTransforms
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)

target_link_libraries(JitTransforms
  PUBLIC
    JitSupport
    MLIRPass
    -Wl,--start-group ${stablehlo_all_libs}  -Wl,--end-group
  PRIVATE
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
