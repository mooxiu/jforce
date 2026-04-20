add_executable(jforce-opt 
  jforce-opt.cpp
)

target_link_libraries(jforce-opt 
  PRIVATE
    JitTransforms
    MLIROptLib
    MLIRTransforms
    MLIRRegisterAllPasses
)
