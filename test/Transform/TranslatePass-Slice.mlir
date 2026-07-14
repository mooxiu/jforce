// RUN: split-file %s %t
// RUN: %jforce-opt %t/slice-read.mlir --jforce-translatev2 | FileCheck %s --check-prefix=SLICE-READ
// RUN : %jforce-opt %t/slice-write.mlir --jforce-translatev2 | FileCheck %s --check-prefix=SLICE-WRITE
// RUN : %jforce-opt %t/slice-to-slice.mlir --jforce-translatev2 | FileCheck %s --check-prefix=SLICE2SLICE
// RUN : %jforce-opt %t/overlap.mlir --jforce-translatev2 | FileCheck %s --check-prefix=OVERLAP
// RUN : %jforce-opt %t/read-after-slice-write.mlir --jforce-translatev2 | FileCheck %s --check-prefix=READ-AFTER-WRITE
// RUN : %jforce-opt %t/repeated-read.mlir --jforce-translatev2 --cse | FileCheck %s --check-prefix=REPEATED

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
