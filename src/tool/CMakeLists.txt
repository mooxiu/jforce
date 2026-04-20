add_executable(jforce-opt 
  jforce-opt.cpp
)

target_include_directories(jforce-opt PRIVATE
  transform
)

target_link_libraries(jforce-opt PRIVATE
  PRIVATE
    transforms 
    MLIRPass
    MLIRIR
    -Wl,--start-group ${stablehlo_all_libs} -Wl,--end-group
)

# target_include_directories(jforce-opt PRIVATE
#   ${MLIR_INCLUDE_DIRS}
#   ${LLVM_INCLUDE_DIRS}
#   ${FLANG_INCLUDE_DIRS}
#   ${STABLEHLO_PROJECT}
# )

# target_link_libraries(jforce-opt PRIVATE 
#   stdc++
#   -Wl,--start-group ${stablehlo_all_libs} -Wl,--end-group
#   FIRDialect
#   HLFIRDialect
# )

