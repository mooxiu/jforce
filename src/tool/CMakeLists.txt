add_executable(jforce-opt 
  jforce-opt.cpp
)

target_link_libraries(jforce-opt 
  PRIVATE
    -Wl,--start-group ${stablehlo_all_libs} -Wl,--end-group
    MLIROptLib
    MLIRPass
    MLIRTransforms
    MLIRSupport
    JitTransforms
    MLIRShapeDialect
    MLIRQuantDialect
    MLIRMathDialect
    MLIRArithDialect
    MLIRRegisterAllPasses
    MLIRIR
    ${MLIR_LIBS}
    ${LLVM_LIBS}
)
