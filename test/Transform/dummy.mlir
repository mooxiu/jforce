// RUN: %jforce-opt %s --jforce-translate | FileCheck %s --check-prefix=IR
func.func @kernel(%arg0: !fir.ref<!fir.array<99xf64>>, %arg1: !fir.ref<!fir.array<99xf64>>, %arg2: !fir.ref<i32>, %arg3: !fir.ref<i32>) {
  %c99 = arith.constant 99 : index
  %0 = fir.shape %c99 : (index) -> !fir.shape<1>
  %1:2 = hlfir.declare %arg0(%0) {uniq_name = "_QFFrun_benchmarkEy"} : (!fir.ref<!fir.array<99xf64>>, !fir.shape<1>) -> (!fir.ref<!fir.array<99xf64>>, !fir.ref<!fir.array<99xf64>>)
  %2:2 = hlfir.declare %arg1(%0) {uniq_name = "_QFFrun_benchmarkEx"} : (!fir.ref<!fir.array<99xf64>>, !fir.shape<1>) -> (!fir.ref<!fir.array<99xf64>>, !fir.ref<!fir.array<99xf64>>)
  omp.teams {
    omp.workdistribute {
      hlfir.assign %2#0 to %1#0 : !fir.ref<!fir.array<99xf64>>, !fir.ref<!fir.array<99xf64>>
      omp.terminator
    }
    omp.terminator
  }
  omp.terminator
}

// IR-LABEL: func.func @main(%arg0: tensor<99xf64>, %arg1: tensor<99xf64>, %arg2: tensor<i32>, %arg3: tensor<i32> 
// IR-SAME: tensor<99xf64>, tensor<99xf64>, tensor<i32>, tensor<i32>
// IR-NEXT: return %arg1, %arg1, %arg2, %arg3 : tensor<99xf64>, tensor<99xf64>, tensor<i32>, tensor<i32>

