// RUN: split-file %s %t
// RUN: %jforce-opt %t/dummy.mlir --jforce-translatev2 | FileCheck %s --check-prefix=DUMMY
// RUN: %jforce-opt %t/mem-to-mem.mlir --jforce-translatev2 | FileCheck %s --check-prefix=MEM2MEM
// RUN: %jforce-opt %t/constant-to-mem.mlir --jforce-translatev2 | FileCheck %s --check-prefix=CONST
// RUN: %jforce-opt %t/sequential-assign.mlir --jforce-translatev2 | FileCheck %s --check-prefix=SEQ
// RUN: %jforce-opt %t/load-snapshot.mlir --jforce-translatev2 | FileCheck %s --check-prefix=SNAPSHOT
// RUN: %jforce-opt %t/declare-chain.mlir --jforce-translatev2 | FileCheck %s --check-prefix=DECLARE
// RUN: %jforce-opt %t/read-after-write.mlir --jforce-translatev2 | FileCheck %s --check-prefix=RAW
// RUN: %jforce-opt %t/self-assign.mlir --jforce-translatev2 | FileCheck %s --check-prefix=SELF
// XRUN : %jforce-opt %t/attention.mlir --jforce-translatev2 | FileCheck %s --check-prefix=ATTENTION

//--- dummy.mlir
// Dummy Workdistribute Example
func.func @kernel(%arg0: !fir.ref<!fir.array<99xf64>>, %arg1: !fir.ref<!fir.array<99xf64>>, %arg2: !fir.ref<i32>, %arg3: !fir.ref<i32>) {
  // DUMMY-LABEL: func.func @main(%arg0: tensor<99xf64>, %arg1: tensor<99xf64>, %arg2: tensor<i32>, %arg3: tensor<i32> 
  // DUMMY-SAME: tensor<99xf64>, tensor<99xf64>, tensor<i32>, tensor<i32>
  %c99 = arith.constant 99 : index
  %0 = fir.shape %c99 : (index) -> !fir.shape<1>
  %1:2 = hlfir.declare %arg0(%0) {uniq_name = ""} : (!fir.ref<!fir.array<99xf64>>, !fir.shape<1>) -> (!fir.ref<!fir.array<99xf64>>, !fir.ref<!fir.array<99xf64>>)
  %2:2 = hlfir.declare %arg1(%0) {uniq_name = ""} : (!fir.ref<!fir.array<99xf64>>, !fir.shape<1>) -> (!fir.ref<!fir.array<99xf64>>, !fir.ref<!fir.array<99xf64>>)
  omp.teams {
    omp.workdistribute {
      hlfir.assign %2#0 to %1#0 : !fir.ref<!fir.array<99xf64>>, !fir.ref<!fir.array<99xf64>>
      // DUMMY: return %arg1, %arg1, %arg2, %arg3 : tensor<99xf64>, tensor<99xf64>, tensor<i32>, tensor<i32>
      omp.terminator
    }
    omp.terminator
  }
  omp.terminator
}

//--- mem-to-mem.mlir
func.func @kernel(%arg0: !fir.ref<!fir.array<4xf64>>, %arg1: !fir.ref<!fir.array<4xf64>>) {
  // MEM2MEM-LABEL: func.func @main
  %c4 = arith.constant 4 : index
  %shape = fir.shape %c4 : (index) -> !fir.shape<1>
  %a:2 = hlfir.declare %arg0(%shape) {uniq_name = "a"}: (!fir.ref<!fir.array<4xf64>>, !fir.shape<1>) -> (!fir.ref<!fir.array<4xf64>>, !fir.ref<!fir.array<4xf64>>)
  %b:2 = hlfir.declare %arg1(%shape) {uniq_name = "b"}: (!fir.ref<!fir.array<4xf64>>, !fir.shape<1>) -> (!fir.ref<!fir.array<4xf64>>, !fir.ref<!fir.array<4xf64>>)
  hlfir.assign %b#0 to %a#0: !fir.ref<!fir.array<4xf64>>, !fir.ref<!fir.array<4xf64>>
  // MEM2MEM: return %arg1, %arg1 : tensor<4xf64>, tensor<4xf64>
  return
}

//--- constant-to-mem.mlir
func.func @kernel(%arg0: !fir.ref<f64>) {
  // CONST-LABEL: func.func @main
  %a:2 = hlfir.declare %arg0 {uniq_name = "a"}: (!fir.ref<f64>) -> (!fir.ref<f64>, !fir.ref<f64>)
  %cst = arith.constant 3.000000e+00 : f64
  // CONST: %[[C:.*]] = stablehlo.constant
  hlfir.assign %cst to %a#0 : f64, !fir.ref<f64>
  // CONST: return %[[C]] : tensor<f64>
  return
}

//--- sequential-assign.mlir
func.func @kernel(%arg0: !fir.ref<f64>, %arg1: !fir.ref<f64>, %arg2: !fir.ref<f64>) {
  // SEQ-LABEL: func.func @main
  %a:2 = hlfir.declare %arg0 {uniq_name = "a"} : (!fir.ref<f64>) -> (!fir.ref<f64>, !fir.ref<f64>)
  %b:2 = hlfir.declare %arg1 {uniq_name = "b"} : (!fir.ref<f64>) -> (!fir.ref<f64>, !fir.ref<f64>)
  %c:2 = hlfir.declare %arg2 {uniq_name = "c"} : (!fir.ref<f64>) -> (!fir.ref<f64>, !fir.ref<f64>)
  hlfir.assign %b#0 to %a#0 : !fir.ref<f64>, !fir.ref<f64>
  hlfir.assign %c#0 to %b#0 : !fir.ref<f64>, !fir.ref<f64>
  // SEQ: return %arg1, %arg2, %arg2 : tensor<f64>, tensor<f64>, tensor<f64>
  return
}

//--- load-snapshot.mlir
func.func @kernel(%arg0: !fir.ref<f64>, %arg1: !fir.ref<f64>, %arg2: !fir.ref<f64>) {
  // SNAPSHOT-LABEL: func.func @main
  %a:2 = hlfir.declare %arg0 {uniq_name = "a"} : (!fir.ref<f64>) -> (!fir.ref<f64>, !fir.ref<f64>)
  %b:2 = hlfir.declare %arg1 {uniq_name = "b"} : (!fir.ref<f64>) -> (!fir.ref<f64>, !fir.ref<f64>)
  %c:2 = hlfir.declare %arg2 {uniq_name = "c"} : (!fir.ref<f64>) -> (!fir.ref<f64>, !fir.ref<f64>)
  %old_a = fir.load %a#0 : !fir.ref<f64>
  hlfir.assign %b#0 to %a#0 : !fir.ref<f64>, !fir.ref<f64>
  hlfir.assign %old_a to %c#0 : f64, !fir.ref<f64>
  // SNAPSHOT: return %arg1, %arg1, %arg0 : tensor<f64>, tensor<f64>, tensor<f64>
  return
}

//--- declare-chain.mlir
func.func @kernel(%arg0: !fir.ref<f64>, %arg1: !fir.ref<f64>) {
  // DECLARE-LABEL: func.func @main
  %a0:2 = hlfir.declare %arg0 {uniq_name = "a0"} : (!fir.ref<f64>) -> (!fir.ref<f64>, !fir.ref<f64>)
  %a1:2 = hlfir.declare %a0#0 {uniq_name = "a1"} : (!fir.ref<f64>) -> (!fir.ref<f64>, !fir.ref<f64>)
  %b:2 = hlfir.declare %arg1 {uniq_name = "b"} : (!fir.ref<f64>) -> (!fir.ref<f64>, !fir.ref<f64>)
  hlfir.assign %b#0 to %a1#0 : !fir.ref<f64>, !fir.ref<f64>
  // DECLARE: return %arg1, %arg1 : tensor<f64>, tensor<f64>
  return
}

//--- read-after-write.mlir
func.func @kernel(
    %arg0: !fir.ref<f64>,
    %arg1: !fir.ref<f64>,
    %arg2: !fir.ref<f64>) {
  // RAW-LABEL: func.func @main
  %a:2 = hlfir.declare %arg0 {uniq_name = "a"}
      : (!fir.ref<f64>) -> (!fir.ref<f64>, !fir.ref<f64>)
  %b:2 = hlfir.declare %arg1 {uniq_name = "b"}
      : (!fir.ref<f64>) -> (!fir.ref<f64>, !fir.ref<f64>)
  %c:2 = hlfir.declare %arg2 {uniq_name = "c"}
      : (!fir.ref<f64>) -> (!fir.ref<f64>, !fir.ref<f64>)

  hlfir.assign %b#0 to %a#0 : !fir.ref<f64>, !fir.ref<f64>
  hlfir.assign %a#0 to %c#0 : !fir.ref<f64>, !fir.ref<f64>
  // RAW: return %arg1, %arg1, %arg1 : tensor<f64>, tensor<f64>, tensor<f64>
  return
}

//--- self-assign.mlir
func.func @kernel(%arg0: !fir.ref<f64>) {
  // SELF-LABEL: func.func @main
  %a:2 = hlfir.declare %arg0 {uniq_name = "a"}
      : (!fir.ref<f64>) -> (!fir.ref<f64>, !fir.ref<f64>)
  hlfir.assign %a#0 to %a#0 : !fir.ref<f64>, !fir.ref<f64>
  // SELF: return %arg0 : tensor<f64>
  return
}



//--- attention.mlir
// Complicated Workdistribute Example
func.func @kernel(%arg0: !fir.ref<!fir.array<1024x1024xf64>>, %arg1: !fir.ref<!fir.array<1024x1024xf64>>, %arg2: !fir.ref<f64>, %arg3: !fir.ref<!fir.array<1024x1024xf64>>, %arg4: !fir.ref<!fir.array<1024x1024xf64>>, %arg5: !fir.ref<i32>, %arg6: !fir.ref<i32>, %arg7: !fir.ref<i32>, %arg8: !fir.ref<i32>, %arg9: !fir.ref<i32>, %arg10: !fir.ref<i32>, %arg11: !fir.ref<i32>, %arg12: !fir.ref<i32>, %arg13: !fir.ref<i32>, %arg14: !fir.ref<i32>, %arg15: !fir.ref<i32>, %arg16: !fir.ref<i32>) {
// ATTENTION-LABEL: func.func @main(%arg0: tensor<1024x1024xf64>, %arg1: tensor<1024x1024xf64>, %arg2: tensor<f64>, %arg3: tensor<1024x1024xf64>, %arg4: tensor<1024x1024xf64>, %arg5: tensor<i32>, %arg6: tensor<i32>, %arg7: tensor<i32>, %arg8: tensor<i32>, %arg9: tensor<i32>, %arg10: tensor<i32>, %arg11: tensor<i32>, %arg12: tensor<i32>, %arg13: tensor<i32>, %arg14: tensor<i32>, %arg15: tensor<i32>, %arg16: tensor<i32>)
// ATTENTION-SAME: tensor<1024x1024xf64>, tensor<1024x1024xf64>, tensor<f64>, tensor<1024x1024xf64>, tensor<1024x1024xf64>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>

  %c1024 = arith.constant 1024 : index
  %cst = arith.constant 0.000000e+00 : f64
  // ATTENTION: %[[CST:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f64>
  %0 = fir.shape %c1024, %c1024 : (index, index) -> !fir.shape<2>
  %1:2 = hlfir.declare %arg0(%0) {uniq_name = ""} : (!fir.ref<!fir.array<1024x1024xf64>>, !fir.shape<2>) -> (!fir.ref<!fir.array<1024x1024xf64>>, !fir.ref<!fir.array<1024x1024xf64>>)
  %2:2 = hlfir.declare %arg1(%0) {uniq_name = ""} : (!fir.ref<!fir.array<1024x1024xf64>>, !fir.shape<2>) -> (!fir.ref<!fir.array<1024x1024xf64>>, !fir.ref<!fir.array<1024x1024xf64>>)
  %3:2 = hlfir.declare %arg2 {uniq_name = ""} : (!fir.ref<f64>) -> (!fir.ref<f64>, !fir.ref<f64>)
  %4:2 = hlfir.declare %arg3(%0) {uniq_name = ""} : (!fir.ref<!fir.array<1024x1024xf64>>, !fir.shape<2>) -> (!fir.ref<!fir.array<1024x1024xf64>>, !fir.ref<!fir.array<1024x1024xf64>>)
  %5:2 = hlfir.declare %arg4(%0) {uniq_name = ""} : (!fir.ref<!fir.array<1024x1024xf64>>, !fir.shape<2>) -> (!fir.ref<!fir.array<1024x1024xf64>>, !fir.ref<!fir.array<1024x1024xf64>>)
  omp.teams {
    %6 = fir.alloca !fir.array<1024x1024xf64> 
    %7:2 = hlfir.declare %6(%0) {uniq_name = ""} : (!fir.ref<!fir.array<1024x1024xf64>>, !fir.shape<2>) -> (!fir.ref<!fir.array<1024x1024xf64>>, !fir.ref<!fir.array<1024x1024xf64>>)
    %8 = fir.alloca !fir.array<1024x1024xf64> 
    %9:2 = hlfir.declare %8(%0) {uniq_name = ""} : (!fir.ref<!fir.array<1024x1024xf64>>, !fir.shape<2>) -> (!fir.ref<!fir.array<1024x1024xf64>>, !fir.ref<!fir.array<1024x1024xf64>>)
    omp.workdistribute {
      %10 = hlfir.transpose %2#0 : (!fir.ref<!fir.array<1024x1024xf64>>) -> !hlfir.expr<?x?xf64>
      // ATTENTION: %0 = stablehlo.transpose %arg1, dims = [1, 0] : (tensor<1024x1024xf64>) -> tensor<1024x1024xf64>
      %11 = hlfir.matmul %1#0 %10 : (!fir.ref<!fir.array<1024x1024xf64>>, !hlfir.expr<?x?xf64>) -> !hlfir.expr<?x?xf64>
      // ATTENTION-NEXT: %1 = stablehlo.dot_general %0, %arg0, contracting_dims = [1] x [0] : (tensor<1024x1024xf64>, tensor<1024x1024xf64>) -> tensor<1024x1024xf64>
      hlfir.assign %11 to %7#0 : !hlfir.expr<?x?xf64>, !fir.ref<!fir.array<1024x1024xf64>>
      hlfir.destroy %11 : !hlfir.expr<?x?xf64>
      hlfir.destroy %10 : !hlfir.expr<?x?xf64>
      %12 = fir.load %3#0 : !fir.ref<f64>
      %13 = hlfir.elemental %0 : (!fir.shape<2>) -> !hlfir.expr<1024x1024xf64> {
      ^bb0(%arg17: index, %arg18: index):
        %16 = hlfir.designate %7#0 (%arg17, %arg18)  : (!fir.ref<!fir.array<1024x1024xf64>>, index, index) -> !fir.ref<f64>
        %17 = fir.load %16 : !fir.ref<f64>
        %18 = arith.mulf %17, %12 fastmath<contract> : f64
        hlfir.yield_element %18 : f64
      }
      // ATTENTION-NEXT: %2 = stablehlo.broadcast_in_dim %arg2, dims = [] : (tensor<f64>) -> tensor<1024x1024xf64>
      // ATTENTION-NEXT: %3 = stablehlo.multiply %1, %2 : tensor<1024x1024xf64>

      %14 = hlfir.elemental %0 : (!fir.shape<2>) -> !hlfir.expr<1024x1024xf64> {
      ^bb0(%arg17: index, %arg18: index):
        %16 = hlfir.apply %13, %arg17, %arg18 : (!hlfir.expr<1024x1024xf64>, index, index) -> f64
        %17 = arith.cmpf ogt, %16, %cst fastmath<contract> : f64
        %18 = arith.select %17, %16, %cst : f64
        hlfir.yield_element %18 : f64
      }
      // ATTENTION-NEXT: %4 = stablehlo.broadcast_in_dim %[[CST:.*]], dims = [] : (tensor<f64>) -> tensor<1024x1024xf64>
      // ATTENTION-NEXT: %5 = stablehlo.compare GT, %3, %4, FLOAT : (tensor<1024x1024xf64>, tensor<1024x1024xf64>) -> tensor<1024x1024xi1>
      // ATTENTION-NEXT: %6 = stablehlo.broadcast_in_dim %[[CST:.*]], dims = [] : (tensor<f64>) -> tensor<1024x1024xf64>
      // ATTENTION-NEXT: %7 = stablehlo.select %5, %3, %6 : tensor<1024x1024xi1>, tensor<1024x1024xf64>

      hlfir.assign %14 to %9#0 : !hlfir.expr<1024x1024xf64>, !fir.ref<!fir.array<1024x1024xf64>>
      hlfir.destroy %14 : !hlfir.expr<1024x1024xf64>
      hlfir.destroy %13 : !hlfir.expr<1024x1024xf64>
      %15 = hlfir.matmul %9#0 %5#0 {fastmath = #arith.fastmath<contract>} : (!fir.ref<!fir.array<1024x1024xf64>>, !fir.ref<!fir.array<1024x1024xf64>>) -> !hlfir.expr<?x?xf64>
      // ATTENTION-NEXT: %8 = stablehlo.dot_general %arg4, %7, contracting_dims = [1] x [0] : (tensor<1024x1024xf64>, tensor<1024x1024xf64>) -> tensor<1024x1024xf64>

      hlfir.assign %15 to %4#0 : !hlfir.expr<?x?xf64>, !fir.ref<!fir.array<1024x1024xf64>>
      hlfir.destroy %15 : !hlfir.expr<?x?xf64>
      omp.terminator
    }
    omp.terminator
  }
  omp.terminator
}
// ATTENTION-NEXT: return %arg0, %arg1, %arg2, %8, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15, %arg16 : tensor<1024x1024xf64>, tensor<1024x1024xf64>, tensor<f64>, tensor<1024x1024xf64>, tensor<1024x1024xf64>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>
