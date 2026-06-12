find_path(ISL_INCLUDE_DIR NAMES isl/aff.h)
find_library(ISL_LIBRARY NAMES isl)

if(NOT ISL_INCLUDE_DIR OR NOT ISL_LIBRARY)
  message(FATAL_ERROR "ISL library or headers not found. Please specify -DCMAKE_PREFIX_PATH.")
else()
  message(STATUS "Found ISL include: ${ISL_INCLUDE_DIR}")
  message(STATUS "Found ISL library: ${ISL_LIBRARY}")
endif()

add_library(JitTransforms STATIC 
  MemOpsFoldingPass.cpp
  CleanTempsPass.cpp
  AnnotatePass.cpp
  AliasingPass.cpp
  ShapeInferPass.cpp
  TrimArgsPass.cpp
  WorkdistributeToStableHLOPass.cpp
  Passes.cpp
  CleanFIRLoopPass.cpp
  OptimizingMemOpsPass.cpp
  CleanFIROpsPass.cpp
  SimplifyDoLoopPass.cpp
  OutlineAffinePass.cpp
  PropagateConstantsPass.cpp
  LoopSinkingPass.cpp
  SwitchOutlineFuncPass.cpp
  IfConversionPass.cpp
  FoldRepeatConversionPass.cpp

  # INFO: some passes copied and adapted from Enzyme-JAX project
  EnzymeJAX/PolygeistMem2RegPass.cpp
  EnzymeJAX/AffineCFG.cpp
  EnzymeJAX/AffineToStableHLO.cpp
  EnzymeJAX/ArithRaising.cpp
  EnzymeJAX/FoldSCFIfPass.cpp
  EnzymeJAX/EnzymeUtils.cpp
  EnzymeJAX/Utils.cpp
)

target_include_directories(JitTransforms
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/..
  PRIVATE
    ${ISL_INCLUDE_DIR}
)

target_link_libraries(JitTransforms
  PUBLIC
    JitSupport
    MLIRPass
  PRIVATE
    ${ISL_LIBRARY}
    -Wl,--start-group ${stablehlo_all_libs}  -Wl,--end-group
    FIRDialect
    HLFIRDialect
    MLIRShapeDialect
    MLIRQuantDialect
    MLIRMathDialect
    MLIRArithDialect
    MLIRGPUDialect
    MLIRIR
    MLIRMemRefDialect
    FIRTransforms
    HLFIRTransforms
    MLIRBufferizationDialect
)

# -DCMAKE_BUILD_TYPE=Debug
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  target_compile_definitions(JitTransforms PRIVATE ENABLE_XLA_DEBUG)
endif()
