// RUN: split-file %s %t
// RUN: %jforce-opt %t/slice-read.mlir --jforce-translatev2 | FileCheck %s --check-prefix=SLICE-READ
// RUN: %jforce-opt %t/slice-write.mlir --jforce-translatev2 | FileCheck %s --check-prefix=SLICE-WRITE
// RUN: %jforce-opt %t/slice-to-slice.mlir --jforce-translatev2 | FileCheck %s --check-prefix=SLICE2SLICE
// RUN: %jforce-opt %t/overlap.mlir --jforce-translatev2 | FileCheck %s --check-prefix=OVERLAP
// RUN: %jforce-opt %t/read-after-slice-write.mlir --jforce-translatev2 | FileCheck %s --check-prefix=READ-AFTER-WRITE
// RUN: %jforce-opt %t/repeated-read.mlir --jforce-translatev2 --cse | FileCheck %s --check-prefix=REPEATED
// RUN: %jforce-opt %t/slice-coordinate.mlir --jforce-translatev2 | FileCheck %s --check-prefix=SLICE-COORD
// RUN: %jforce-opt %t/negative-lower-bound-read.mlir --jforce-translatev2 | FileCheck %s --check-prefix=NEG-LB-READ
// RUN: %jforce-opt %t/negative-lower-bound-write.mlir --jforce-translatev2 | FileCheck %s --check-prefix=NEG-LB-WRITE
// RUN: %jforce-opt %t/strided-read.mlir --jforce-translatev2 | FileCheck %s --check-prefix=STRIDED-READ
// RUN: %jforce-opt %t/slice-2d.mlir --jforce-translatev2 | FileCheck %s --check-prefix=SLICE-2D

//--- slice-read.mlir
// Fortran A(2:4) becomes zero-based StableHLO [1:4).
func.func @kernel(%a: !fir.box<!fir.array<8xf64>>, %b: !fir.box<!fir.array<3xf64>>) {
  // SLICE-READ-LABEL: func.func @main(
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %shape3 = fir.shape %c3 : (index) -> !fir.shape<1>
  %a_slice = hlfir.designate %a (%c2:%c4:%c1) shape %shape3 : (!fir.box<!fir.array<8xf64>>, index, index, index, !fir.shape<1>) -> !fir.box<!fir.array<3xf64>>
  hlfir.assign %a_slice to %b : !fir.box<!fir.array<3xf64>>, !fir.box<!fir.array<3xf64>>
  // SLICE-READ: %[[SLICE:.*]] = stablehlo.slice %arg0
  // SLICE-READ: return %arg0, %[[SLICE]]
  return
}


//--- slice-write.mlir
func.func @kernel(%b: !fir.box<!fir.array<3xf64>>, %a: !fir.box<!fir.array<8xf64>>) {
  // SLICE-WRITE-LABEL: func.func @main(
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %shape3 = fir.shape %c3 : (index) -> !fir.shape<1>
  %a_slice = hlfir.designate %a (%c2:%c4:%c1) shape %shape3 : (!fir.box<!fir.array<8xf64>>, index, index, index, !fir.shape<1>) -> !fir.box<!fir.array<3xf64>>
  hlfir.assign %b to %a_slice : !fir.box<!fir.array<3xf64>>, !fir.box<!fir.array<3xf64>>
  // SLICE-WRITE: %[[UPDATED:.*]] = stablehlo.dynamic_update_slice %arg1, %arg0
  // SLICE-WRITE: return %arg0, %[[UPDATED]]
  return
}



//--- slice-to-slice.mlir

func.func @kernel(
    %a: !fir.box<!fir.array<8xf64>>,
    %b: !fir.box<!fir.array<8xf64>>) {
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %c5 = arith.constant 5 : index
  %c7 = arith.constant 7 : index

  %shape3 = fir.shape %c3 : (index) -> !fir.shape<1>

  %a_slice = hlfir.designate %a (%c2:%c4:%c1) shape %shape3
      : (!fir.box<!fir.array<8xf64>>,
         index, index, index, !fir.shape<1>)
      -> !fir.box<!fir.array<3xf64>>

  %b_slice = hlfir.designate %b (%c5:%c7:%c1) shape %shape3
      : (!fir.box<!fir.array<8xf64>>,
         index, index, index, !fir.shape<1>)
      -> !fir.box<!fir.array<3xf64>>

  hlfir.assign %a_slice to %b_slice
      : !fir.box<!fir.array<3xf64>>,
        !fir.box<!fir.array<3xf64>>

  return
}

// SLICE2SLICE-LABEL: func.func @main(
// SLICE2SLICE: %[[RHS:.*]] = stablehlo.slice %arg0
// SLICE2SLICE: %[[UPDATED:.*]] = stablehlo.dynamic_update_slice %arg1, %[[RHS]]
// SLICE2SLICE: return %arg0, %[[UPDATED]]


//--- overlap.mlir

func.func @kernel(%a: !fir.box<!fir.array<8xf64>>) {
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index

  %shape3 = fir.shape %c3 : (index) -> !fir.shape<1>

  %rhs = hlfir.designate %a (%c1:%c3:%c1) shape %shape3
      : (!fir.box<!fir.array<8xf64>>,
         index, index, index, !fir.shape<1>)
      -> !fir.box<!fir.array<3xf64>>

  %lhs = hlfir.designate %a (%c2:%c4:%c1) shape %shape3
      : (!fir.box<!fir.array<8xf64>>,
         index, index, index, !fir.shape<1>)
      -> !fir.box<!fir.array<3xf64>>

  hlfir.assign %rhs to %lhs
      : !fir.box<!fir.array<3xf64>>,
        !fir.box<!fir.array<3xf64>>

  return
}

// RHS must be materialized from the old root tensor.
//
// OVERLAP-LABEL: func.func @main(
// OVERLAP: %[[RHS:.*]] = stablehlo.slice %arg0
// OVERLAP: %[[UPDATED:.*]] = stablehlo.dynamic_update_slice %arg0, %[[RHS]]
// OVERLAP: return %[[UPDATED]]

//--- read-after-slice-write.mlir

func.func @kernel(
    %a: !fir.box<!fir.array<8xf64>>,
    %b: !fir.box<!fir.array<3xf64>>,
    %c: !fir.box<!fir.array<3xf64>>) {
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index

  %shape3 = fir.shape %c3 : (index) -> !fir.shape<1>

  %a_slice = hlfir.designate %a (%c2:%c4:%c1) shape %shape3
      : (!fir.box<!fir.array<8xf64>>,
         index, index, index, !fir.shape<1>)
      -> !fir.box<!fir.array<3xf64>>

  hlfir.assign %b to %a_slice
      : !fir.box<!fir.array<3xf64>>,
        !fir.box<!fir.array<3xf64>>

  hlfir.assign %a_slice to %c
      : !fir.box<!fir.array<3xf64>>,
        !fir.box<!fir.array<3xf64>>

  return
}

// The second slice must read from UPDATED_A, not from %arg0.
//
// READ-AFTER-WRITE-LABEL: func.func @main(
// READ-AFTER-WRITE: %[[UPDATED_A:.*]] = stablehlo.dynamic_update_slice %arg0, %arg1
// READ-AFTER-WRITE: %[[CURRENT_SLICE:.*]] = stablehlo.slice %[[UPDATED_A]]
// READ-AFTER-WRITE: return %[[UPDATED_A]], %arg1, %[[CURRENT_SLICE]]



//--- repeated-read.mlir

func.func @kernel(
    %a: !fir.box<!fir.array<8xf64>>,
    %b: !fir.box<!fir.array<3xf64>>,
    %c: !fir.box<!fir.array<3xf64>>) {
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index

  %shape3 = fir.shape %c3 : (index) -> !fir.shape<1>

  %a_slice = hlfir.designate %a (%c2:%c4:%c1) shape %shape3
      : (!fir.box<!fir.array<8xf64>>,
         index, index, index, !fir.shape<1>)
      -> !fir.box<!fir.array<3xf64>>

  hlfir.assign %a_slice to %b
      : !fir.box<!fir.array<3xf64>>,
        !fir.box<!fir.array<3xf64>>

  hlfir.assign %a_slice to %c
      : !fir.box<!fir.array<3xf64>>,
        !fir.box<!fir.array<3xf64>>

  return
}

// After CSE, the identical reads should share one StableHLO slice.
//
// REPEATED-LABEL: func.func @main(
// REPEATED-COUNT-1: stablehlo.slice
// REPEATED: return %arg0, %[[SLICE:.*]], %[[SLICE]]


//--- slice-coordinate.mlir

// Verify the exact zero-based, half-open StableHLO coordinates.
//
// Fortran:
//   A(2:4)
//
// Root lower bound:
//   1
//
// StableHLO:
//   start  = 2 - 1     = 1
//   limit  = 4 - 1 + 1 = 4
//   stride = 1
func.func @kernel(
    %a: !fir.box<!fir.array<8xf64>>,
    %b: !fir.box<!fir.array<3xf64>>) {
  // SLICE-COORD-LABEL: func.func @main(
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index

  %shape3 = fir.shape %c3 : (index) -> !fir.shape<1>

  %slice = hlfir.designate %a (%c2:%c4:%c1) shape %shape3
      : (!fir.box<!fir.array<8xf64>>,
         index, index, index, !fir.shape<1>)
      -> !fir.box<!fir.array<3xf64>>

  hlfir.assign %slice to %b
      : !fir.box<!fir.array<3xf64>>,
        !fir.box<!fir.array<3xf64>>

  // SLICE-COORD: %[[SLICE:.*]] = stablehlo.slice %arg0 [1:4]
  // SLICE-COORD: return %arg0, %[[SLICE]]
  return
}

//--- negative-lower-bound-read.mlir

// The preceding inference pass records the static lower bound on
// fir.shape_shift using:
//
//   jit.slice_shift = array<i64: -5>
//
// Fortran declaration:
//   A(-5:5)
//
// Fortran section:
//   A(-3:1)
//
// StableHLO:
//   start  = -3 - (-5)     = 2
//   limit  =  1 - (-5) + 1 = 7
//   stride = 1
func.func @kernel(
    %a: !fir.box<!fir.array<11xf64>>,
    %b: !fir.box<!fir.array<5xf64>>) {
  %cm5 = arith.constant -5 : index
  %cm3 = arith.constant -3 : index
  %c1 = arith.constant 1 : index
  %c5 = arith.constant 5 : index
  %c11 = arith.constant 11 : index

  %shape_shift = fir.shape_shift %cm5, %c11
      {jit.slice_shift = array<i64: -5>}
      : (index, index) -> !fir.shapeshift<1>

  %a_decl:2 = hlfir.declare %a(%shape_shift) {uniq_name = "a"}
      : (!fir.box<!fir.array<11xf64>>, !fir.shapeshift<1>)
      -> (!fir.box<!fir.array<11xf64>>,
          !fir.box<!fir.array<11xf64>>)

  %shape5 = fir.shape %c5 : (index) -> !fir.shape<1>

  %slice = hlfir.designate %a_decl#0 (%cm3:%c1:%c1) shape %shape5
      : (!fir.box<!fir.array<11xf64>>,
         index, index, index, !fir.shape<1>)
      -> !fir.box<!fir.array<5xf64>>

  hlfir.assign %slice to %b
      : !fir.box<!fir.array<5xf64>>,
        !fir.box<!fir.array<5xf64>>

  return
}

// NEG-LB-READ-LABEL: func.func @main(
// NEG-LB-READ: %[[SLICE:.*]] = stablehlo.slice %arg0 [2:7]
// NEG-LB-READ: return %arg0, %[[SLICE]]

//--- negative-lower-bound-write.mlir

func.func @kernel(
    %b: !fir.box<!fir.array<5xf64>>,
    %a: !fir.box<!fir.array<11xf64>>) {
  %cm5 = arith.constant -5 : index
  %cm3 = arith.constant -3 : index
  %c1 = arith.constant 1 : index
  %c5 = arith.constant 5 : index
  %c11 = arith.constant 11 : index

  %shape_shift = fir.shape_shift %cm5, %c11
      {jit.slice_shift = array<i64: -5>}
      : (index, index) -> !fir.shapeshift<1>

  %a_decl:2 = hlfir.declare %a(%shape_shift) {uniq_name = "a"}
      : (!fir.box<!fir.array<11xf64>>, !fir.shapeshift<1>)
      -> (!fir.box<!fir.array<11xf64>>,
          !fir.box<!fir.array<11xf64>>)

  %shape5 = fir.shape %c5 : (index) -> !fir.shape<1>

  %slice = hlfir.designate %a_decl#0 (%cm3:%c1:%c1) shape %shape5
      : (!fir.box<!fir.array<11xf64>>,
         index, index, index, !fir.shape<1>)
      -> !fir.box<!fir.array<5xf64>>

  hlfir.assign %b to %slice
      : !fir.box<!fir.array<5xf64>>,
        !fir.box<!fir.array<5xf64>>

  return
}

// NEG-LB-WRITE-LABEL: func.func @main(
// NEG-LB-WRITE: %[[START:.*]] = stablehlo.constant dense<2> : tensor<i64>
// NEG-LB-WRITE: %[[UPDATED:.*]] = stablehlo.dynamic_update_slice %arg1, %arg0, %[[START]]
// NEG-LB-WRITE: return %arg0, %[[UPDATED]]


//--- strided-read.mlir

// Verify a positive, non-unit static stride.
//
// Fortran:
//   A(2:8:2)
//
// Root lower bound:
//   1
//
// StableHLO:
//   start  = 2 - 1     = 1
//   limit  = 8 - 1 + 1 = 8
//   stride = 2
//
// Selected Fortran elements:
//   A(2), A(4), A(6), A(8)
func.func @kernel(
    %a: !fir.box<!fir.array<8xf64>>,
    %b: !fir.box<!fir.array<4xf64>>) {
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index

  %shape4 = fir.shape %c4 : (index) -> !fir.shape<1>

  %slice = hlfir.designate %a (%c2:%c8:%c2) shape %shape4
      : (!fir.box<!fir.array<8xf64>>,
         index, index, index, !fir.shape<1>)
      -> !fir.box<!fir.array<4xf64>>

  hlfir.assign %slice to %b
      : !fir.box<!fir.array<4xf64>>,
        !fir.box<!fir.array<4xf64>>

  return
}

// STRIDED-READ-LABEL: func.func @main(
// STRIDED-READ: %[[SLICE:.*]] = stablehlo.slice %arg0 [1:8:2]
// STRIDED-READ: return %arg0, %[[SLICE]]

//--- slice-2d.mlir

// Verify FIR/Fortran dimension order is reversed exactly once.
//
// Fortran root shape:
//   A(1:6, 1:8)
//
// Fortran section:
//   A(2:4, 3:6)
//
// Coordinates in Fortran dimension order:
//   dim 0: [1:4:1]
//   dim 1: [2:6:1]
//
// Coordinates in StableHLO tensor order:
//   [2:6:1, 1:4:1]
//
// Fortran result shape:
//   3 x 4
//
// StableHLO result shape:
//   4 x 3
func.func @kernel(
    %a: !fir.box<!fir.array<6x8xf64>>,
    %b: !fir.box<!fir.array<3x4xf64>>) {
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %c6 = arith.constant 6 : index

  %shape_result = fir.shape %c3, %c4
      : (index, index) -> !fir.shape<2>

  %slice = hlfir.designate %a
      (%c2:%c4:%c1, %c3:%c6:%c1)
      shape %shape_result
      : (!fir.box<!fir.array<6x8xf64>>,
         index, index, index,
         index, index, index,
         !fir.shape<2>)
      -> !fir.box<!fir.array<3x4xf64>>

  hlfir.assign %slice to %b
      : !fir.box<!fir.array<3x4xf64>>,
        !fir.box<!fir.array<3x4xf64>>

  return
}

// SLICE-2D-LABEL: func.func @main(
// SLICE-2D: %[[SLICE:.*]] = stablehlo.slice %arg0 [2:6, 1:4]
// SLICE-2D: return %arg0, %[[SLICE]]


