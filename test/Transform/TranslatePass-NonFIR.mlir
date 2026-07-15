// INFO: memref, affine, bufferization, etc. are mainly for general fortran code rasing, they have nothing to do with workdistribute case.

// RUN: split-file %s %t
// RUN: %jforce-opt %t/fir-zero.mlir --jforce-translatev2 | FileCheck %s --check-prefix=FIR-ZERO
// RUN: %jforce-opt %t/fir-convert.mlir --jforce-translatev2 | FileCheck %s --check-prefix=FIR-CONVERT
// RUN: %jforce-opt %t/affine-load.mlir --jforce-translatev2 | FileCheck %s --check-prefix=AFFINE-LOAD
// RUN: %jforce-opt %t/affine-store.mlir --jforce-translatev2 | FileCheck %s --check-prefix=AFFINE-STORE
// RUN: %jforce-opt %t/memref-store.mlir --jforce-translatev2 | FileCheck %s --check-prefix=MEMREF-STORE

//--- fir-zero.mlir

func.func @kernel(%result: !fir.ref<f64>) {
  %result_decl:2 = hlfir.declare %result {uniq_name = "result"}
      : (!fir.ref<f64>) -> (!fir.ref<f64>, !fir.ref<f64>)

  %zero = fir.zero_bits f64

  hlfir.assign %zero to %result_decl#0
      : f64, !fir.ref<f64>

  return
}

// FIR-ZERO-LABEL: func.func @main(%arg0: tensor<f64>) -> tensor<f64>
// FIR-ZERO: %[[ZERO:.*]] = stablehlo.constant
// FIR-ZERO-SAME: tensor<f64>
// FIR-ZERO: return %[[ZERO]] : tensor<f64>


//--- fir-convert.mlir

func.func @kernel(
    %input: !fir.ref<i32>,
    %bias: !fir.ref<f64>,
    %result: !fir.ref<f64>) {
  %input_decl:2 = hlfir.declare %input {uniq_name = "input"}
      : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)

  %bias_decl:2 = hlfir.declare %bias {uniq_name = "bias"}
      : (!fir.ref<f64>) -> (!fir.ref<f64>, !fir.ref<f64>)

  %result_decl:2 = hlfir.declare %result {uniq_name = "result"}
      : (!fir.ref<f64>) -> (!fir.ref<f64>, !fir.ref<f64>)

  // Value conversion: i32 -> f64.
  %input_value = fir.load %input_decl#0 : !fir.ref<i32>
  %input_f64 = fir.convert %input_value : (i32) -> f64

  // Reference conversion: !fir.ref<f64> -> memref<f64>.
  %bias_memref = fir.convert %bias_decl#0 : (!fir.ref<f64>) -> memref<f64>

  %bias_value = affine.load %bias_memref[] : memref<f64>

  %sum = arith.addf %input_f64, %bias_value : f64

  hlfir.assign %sum to %result_decl#0
      : f64, !fir.ref<f64>

  return
}

// FIR-CONVERT-LABEL: func.func @main(
// FIR-CONVERT-SAME: %arg0: tensor<i32>
// FIR-CONVERT-SAME: %arg1: tensor<f64>
// FIR-CONVERT-SAME: %arg2: tensor<f64>

// The integer value conversion must become stablehlo.convert.
// FIR-CONVERT: %[[INPUT_F64:.*]] = stablehlo.convert %arg0
// FIR-CONVERT-SAME: tensor<i32>
// FIR-CONVERT-SAME: tensor<f64>

// The reference conversion emits no StableHLO operation.
// Reading through the converted alias must therefore produce %arg1.
// FIR-CONVERT: %[[SUM:.*]] = stablehlo.add %[[INPUT_F64]], %arg1
// FIR-CONVERT-SAME: tensor<f64>

// FIR-CONVERT: return %arg0, %arg1, %[[SUM]]


//--- affine-load.mlir

func.func @kernel(
    %input: memref<f64>,
    %scale: !fir.ref<f64>,
    %bias: !fir.ref<f64>,
    %result: !fir.ref<f64>) {
  %scale_decl:2 = hlfir.declare %scale {uniq_name = "scale"}
      : (!fir.ref<f64>) -> (!fir.ref<f64>, !fir.ref<f64>)

  %bias_decl:2 = hlfir.declare %bias {uniq_name = "bias"}
      : (!fir.ref<f64>) -> (!fir.ref<f64>, !fir.ref<f64>)

  %result_decl:2 = hlfir.declare %result {uniq_name = "result"}
      : (!fir.ref<f64>) -> (!fir.ref<f64>, !fir.ref<f64>)

  %input_value = affine.load %input[] : memref<f64>
  %scale_value = fir.load %scale_decl#0 : !fir.ref<f64>
  %bias_value = fir.load %bias_decl#0 : !fir.ref<f64>

  %scaled = arith.mulf %input_value, %scale_value : f64
  %sum = arith.addf %scaled, %bias_value : f64

  hlfir.assign %sum to %result_decl#0
      : f64, !fir.ref<f64>

  return
}

// AFFINE-LOAD-LABEL: func.func @main(
// AFFINE-LOAD-SAME: %arg0: tensor<f64>
// AFFINE-LOAD-SAME: %arg1: tensor<f64>
// AFFINE-LOAD-SAME: %arg2: tensor<f64>
// AFFINE-LOAD-SAME: %arg3: tensor<f64>

// affine.load itself should emit no StableHLO operation.
// Its result should map directly to %arg0.
// AFFINE-LOAD: %[[SCALED:.*]] = stablehlo.multiply %arg0, %arg1
// AFFINE-LOAD-SAME: tensor<f64>
// AFFINE-LOAD: %[[SUM:.*]] = stablehlo.add %[[SCALED]], %arg2
// AFFINE-LOAD-SAME: tensor<f64>

// AFFINE-LOAD: return %arg0, %arg1, %arg2, %[[SUM]]


//--- affine-store.mlir

func.func @kernel(
    %source: !fir.ref<f64>,
    %destination: !fir.ref<f64>,
    %result: !fir.ref<f64>) {
  %source_decl:2 = hlfir.declare %source {uniq_name = "source"}
      : (!fir.ref<f64>) -> (!fir.ref<f64>, !fir.ref<f64>)

  %destination_decl:2 =
      hlfir.declare %destination {uniq_name = "destination"}
      : (!fir.ref<f64>) -> (!fir.ref<f64>, !fir.ref<f64>)

  %result_decl:2 = hlfir.declare %result {uniq_name = "result"}
      : (!fir.ref<f64>) -> (!fir.ref<f64>, !fir.ref<f64>)

  %source_value = fir.load %source_decl#0 : !fir.ref<f64>

  %two = arith.constant 2.000000e+00 : f64
  %scaled = arith.mulf %source_value, %two : f64

  %destination_memref = fir.convert %destination_decl#0
      : (!fir.ref<f64>) -> memref<f64>

  affine.store %scaled, %destination_memref[] : memref<f64>

  // Read through the original FIR reference. This must observe the store.
  %reloaded = fir.load %destination_decl#0 : !fir.ref<f64>

  %one = arith.constant 1.000000e+00 : f64
  %final = arith.addf %reloaded, %one : f64

  hlfir.assign %final to %result_decl#0
      : f64, !fir.ref<f64>

  return
}

// AFFINE-STORE-LABEL: func.func @main(
// AFFINE-STORE: %[[TWO:.*]] = stablehlo.constant
// AFFINE-STORE: %[[SCALED:.*]] = stablehlo.multiply %arg0, %[[TWO]]
// AFFINE-STORE-SAME: tensor<f64>

// Reading destination after affine.store must obtain %[[SCALED]].
// AFFINE-STORE: %[[ONE:.*]] = stablehlo.constant
// AFFINE-STORE: %[[FINAL:.*]] = stablehlo.add %[[SCALED]], %[[ONE]]
// AFFINE-STORE-SAME: tensor<f64>

// destination must also return the stored value.
// AFFINE-STORE: return %arg0, %[[SCALED]], %[[FINAL]]


//--- memref-store.mlir

func.func @kernel(
    %lhs: !fir.ref<f64>,
    %rhs: !fir.ref<f64>,
    %destination: memref<f64>) {
  %lhs_decl:2 = hlfir.declare %lhs {uniq_name = "lhs"}
      : (!fir.ref<f64>) -> (!fir.ref<f64>, !fir.ref<f64>)

  %rhs_decl:2 = hlfir.declare %rhs {uniq_name = "rhs"}
      : (!fir.ref<f64>) -> (!fir.ref<f64>, !fir.ref<f64>)

  %lhs_value = fir.load %lhs_decl#0 : !fir.ref<f64>
  %rhs_value = fir.load %rhs_decl#0 : !fir.ref<f64>

  %difference = arith.subf %lhs_value, %rhs_value : f64

  memref.store %difference, %destination[] : memref<f64>

  return
}

// MEMREF-STORE-LABEL: func.func @main(
// MEMREF-STORE-SAME: %arg0: tensor<f64>
// MEMREF-STORE-SAME: %arg1: tensor<f64>
// MEMREF-STORE-SAME: %arg2: tensor<f64>

// Operand order must remain lhs - rhs.
// MEMREF-STORE: %[[DIFFERENCE:.*]] = stablehlo.subtract %arg0, %arg1
// MEMREF-STORE-SAME: tensor<f64>

// destination is updated to the subtraction result.
// MEMREF-STORE: return %arg0, %arg1, %[[DIFFERENCE]]


