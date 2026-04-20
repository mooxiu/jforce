add_executable(jforce-opt 
  jforce-opt.cpp
)

target_link_libraries(jforce-opt 
  PRIVATE
    MLIRIR
    MLIROptLib
    MLIRTransforms
    MLIRSupport
    -Wl,--start-group JitTransforms -Wl,--end-group
    -Wl,--start-group ${stablehlo_all_libs} -Wl,--end-group
)
