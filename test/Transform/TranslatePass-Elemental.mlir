// RUN: split-file %s %t
// RUN: %jforce-opt %t/elemental-identity.mlir --jforce-translatev2 | FileCheck %s --check-prefix=ELE-IDENTITY
// RUN: %jforce-opt %t/elemental-scalar-mul.mlir --jforce-translatev2 | FileCheck %s --check-prefix=ELE-SCALAR
// RUN: %jforce-opt %t/elemental-array-add.mlir --jforce-translatev2 | FileCheck %s --check-prefix=ELE-ADD
// RUN: %jforce-opt %t/elemental-chain.mlir --jforce-translatev2 | FileCheck %s --check-prefix=ELE-CHAIN
// RUN: %jforce-opt %t/elemental-slice.mlir --jforce-translatev2 | FileCheck %s --check-prefix=ELE-SLICE

//--- elemental-identity.mlir

func.func @kernel(
    %arg0: !fir.ref<!fir.array<2x3xf64>>,
    %arg1: !fir.ref<!fir.array<2x3xf64>>) {
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %shape = fir.shape %c2, %c3
      : (index, index) -> !fir.shape<2>

  %a:2 = hlfir.declare %arg0(%shape) {uniq_name = ""}
      : (!fir.ref<!fir.array<2x3xf64>>, !fir.shape<2>)
      -> (!fir.ref<!fir.array<2x3xf64>>,
          !fir.ref<!fir.array<2x3xf64>>)

  %out:2 = hlfir.declare %arg1(%shape) {uniq_name = ""}
      : (!fir.ref<!fir.array<2x3xf64>>, !fir.shape<2>)
      -> (!fir.ref<!fir.array<2x3xf64>>,
          !fir.ref<!fir.array<2x3xf64>>)

  %expr = hlfir.elemental %shape
      : (!fir.shape<2>) -> !hlfir.expr<2x3xf64> {
  ^bb0(%i: index, %j: index):
    %ref = hlfir.designate %a#0 (%i, %j)
        : (!fir.ref<!fir.array<2x3xf64>>, index, index)
        -> !fir.ref<f64>
    %value = fir.load %ref : !fir.ref<f64>
    hlfir.yield_element %value : f64
  }

  hlfir.assign %expr to %out#0
      : !hlfir.expr<2x3xf64>,
        !fir.ref<!fir.array<2x3xf64>>
  hlfir.destroy %expr : !hlfir.expr<2x3xf64>
  return
}

// ELE-IDENTITY-LABEL: func.func @main(
// ELE-IDENTITY-SAME: %arg0: tensor<3x2xf64>
// ELE-IDENTITY-SAME: %arg1: tensor<3x2xf64>
// ELE-IDENTITY: return %arg0, %arg0



//--- elemental-scalar-mul.mlir

func.func @kernel(
    %arg0: !fir.ref<!fir.array<2x3xf64>>,
    %arg1: !fir.ref<f64>,
    %arg2: !fir.ref<!fir.array<2x3xf64>>) {
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %shape = fir.shape %c2, %c3
      : (index, index) -> !fir.shape<2>

  %a:2 = hlfir.declare %arg0(%shape) {uniq_name = ""}
      : (!fir.ref<!fir.array<2x3xf64>>, !fir.shape<2>)
      -> (!fir.ref<!fir.array<2x3xf64>>,
          !fir.ref<!fir.array<2x3xf64>>)

  %alpha:2 = hlfir.declare %arg1 {uniq_name = ""}
      : (!fir.ref<f64>) -> (!fir.ref<f64>, !fir.ref<f64>)

  %out:2 = hlfir.declare %arg2(%shape) {uniq_name = ""}
      : (!fir.ref<!fir.array<2x3xf64>>, !fir.shape<2>)
      -> (!fir.ref<!fir.array<2x3xf64>>,
          !fir.ref<!fir.array<2x3xf64>>)

  %scalar = fir.load %alpha#0 : !fir.ref<f64>

  %expr = hlfir.elemental %shape
      : (!fir.shape<2>) -> !hlfir.expr<2x3xf64> {
  ^bb0(%i: index, %j: index):
    %ref = hlfir.designate %a#0 (%i, %j)
        : (!fir.ref<!fir.array<2x3xf64>>, index, index)
        -> !fir.ref<f64>
    %value = fir.load %ref : !fir.ref<f64>
    %result = arith.mulf %value, %scalar : f64
    hlfir.yield_element %result : f64
  }

  hlfir.assign %expr to %out#0
      : !hlfir.expr<2x3xf64>,
        !fir.ref<!fir.array<2x3xf64>>
  hlfir.destroy %expr : !hlfir.expr<2x3xf64>
  return
}

// ELE-SCALAR-LABEL: func.func @main(
// ELE-SCALAR: %[[BROADCAST:.*]] = stablehlo.broadcast_in_dim %arg1, dims = []
// ELE-SCALAR-SAME: (tensor<f64>) -> tensor<3x2xf64>
// ELE-SCALAR: %[[MUL:.*]] = stablehlo.multiply %arg0, %[[BROADCAST]]
// ELE-SCALAR-SAME: tensor<3x2xf64>
// ELE-SCALAR: return %arg0, %arg1, %[[MUL]]



//--- elemental-array-add.mlir

func.func @kernel(
    %arg0: !fir.ref<!fir.array<2x3xf64>>,
    %arg1: !fir.ref<!fir.array<2x3xf64>>,
    %arg2: !fir.ref<!fir.array<2x3xf64>>) {
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %shape = fir.shape %c2, %c3
      : (index, index) -> !fir.shape<2>

  %a:2 = hlfir.declare %arg0(%shape) {uniq_name = ""}
      : (!fir.ref<!fir.array<2x3xf64>>, !fir.shape<2>)
      -> (!fir.ref<!fir.array<2x3xf64>>,
          !fir.ref<!fir.array<2x3xf64>>)

  %b:2 = hlfir.declare %arg1(%shape) {uniq_name = ""}
      : (!fir.ref<!fir.array<2x3xf64>>, !fir.shape<2>)
      -> (!fir.ref<!fir.array<2x3xf64>>,
          !fir.ref<!fir.array<2x3xf64>>)

  %out:2 = hlfir.declare %arg2(%shape) {uniq_name = ""}
      : (!fir.ref<!fir.array<2x3xf64>>, !fir.shape<2>)
      -> (!fir.ref<!fir.array<2x3xf64>>,
          !fir.ref<!fir.array<2x3xf64>>)

  %expr = hlfir.elemental %shape
      : (!fir.shape<2>) -> !hlfir.expr<2x3xf64> {
  ^bb0(%i: index, %j: index):
    %a_ref = hlfir.designate %a#0 (%i, %j)
        : (!fir.ref<!fir.array<2x3xf64>>, index, index)
        -> !fir.ref<f64>
    %a_value = fir.load %a_ref : !fir.ref<f64>

    %b_ref = hlfir.designate %b#0 (%i, %j)
        : (!fir.ref<!fir.array<2x3xf64>>, index, index)
        -> !fir.ref<f64>
    %b_value = fir.load %b_ref : !fir.ref<f64>

    %result = arith.addf %a_value, %b_value : f64
    hlfir.yield_element %result : f64
  }

  hlfir.assign %expr to %out#0
      : !hlfir.expr<2x3xf64>,
        !fir.ref<!fir.array<2x3xf64>>
  hlfir.destroy %expr : !hlfir.expr<2x3xf64>
  return
}

// ELE-ADD-LABEL: func.func @main(
// ELE-ADD-NOT: stablehlo.broadcast_in_dim
// ELE-ADD: %[[ADD:.*]] = stablehlo.add %arg0, %arg1
// ELE-ADD-SAME: tensor<3x2xf64>
// ELE-ADD: return %arg0, %arg1, %[[ADD]]




//--- elemental-chain.mlir

func.func @kernel(
    %arg0: !fir.ref<!fir.array<2x3xf64>>,
    %arg1: !fir.ref<!fir.array<2x3xf64>>,
    %arg2: !fir.ref<!fir.array<2x3xf64>>) {
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %shape = fir.shape %c2, %c3
      : (index, index) -> !fir.shape<2>

  %a:2 = hlfir.declare %arg0(%shape) {uniq_name = ""}
      : (!fir.ref<!fir.array<2x3xf64>>, !fir.shape<2>)
      -> (!fir.ref<!fir.array<2x3xf64>>,
          !fir.ref<!fir.array<2x3xf64>>)

  %b:2 = hlfir.declare %arg1(%shape) {uniq_name = ""}
      : (!fir.ref<!fir.array<2x3xf64>>, !fir.shape<2>)
      -> (!fir.ref<!fir.array<2x3xf64>>,
          !fir.ref<!fir.array<2x3xf64>>)

  %out:2 = hlfir.declare %arg2(%shape) {uniq_name = ""}
      : (!fir.ref<!fir.array<2x3xf64>>, !fir.shape<2>)
      -> (!fir.ref<!fir.array<2x3xf64>>,
          !fir.ref<!fir.array<2x3xf64>>)

  %sum = hlfir.elemental %shape
      : (!fir.shape<2>) -> !hlfir.expr<2x3xf64> {
  ^bb0(%i: index, %j: index):
    %a_ref = hlfir.designate %a#0 (%i, %j)
        : (!fir.ref<!fir.array<2x3xf64>>, index, index)
        -> !fir.ref<f64>
    %a_value = fir.load %a_ref : !fir.ref<f64>

    %b_ref = hlfir.designate %b#0 (%i, %j)
        : (!fir.ref<!fir.array<2x3xf64>>, index, index)
        -> !fir.ref<f64>
    %b_value = fir.load %b_ref : !fir.ref<f64>

    %value = arith.addf %a_value, %b_value : f64
    hlfir.yield_element %value : f64
  }

  %negated = hlfir.elemental %shape
      : (!fir.shape<2>) -> !hlfir.expr<2x3xf64> {
  ^bb0(%i: index, %j: index):
    %value = hlfir.apply %sum, %i, %j
        : (!hlfir.expr<2x3xf64>, index, index) -> f64
    %neg = arith.negf %value : f64
    hlfir.yield_element %neg : f64
  }

  hlfir.assign %negated to %out#0
      : !hlfir.expr<2x3xf64>,
        !fir.ref<!fir.array<2x3xf64>>

  hlfir.destroy %negated : !hlfir.expr<2x3xf64>
  hlfir.destroy %sum : !hlfir.expr<2x3xf64>
  return
}

// ELE-CHAIN-LABEL: func.func @main(
// ELE-CHAIN: %[[ADD:.*]] = stablehlo.add %arg0, %arg1
// ELE-CHAIN: %[[NEG:.*]] = stablehlo.negate %[[ADD]]
// ELE-CHAIN-SAME: tensor<3x2xf64>
// ELE-CHAIN: return %arg0, %arg1, %[[NEG]]




//--- elemental-slice.mlir

func.func @kernel(
    %arg0: !fir.ref<!fir.array<6xf64>>,
    %arg1: !fir.ref<!fir.array<4xf64>>) {
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c4 = arith.constant 4 : index
  %c5 = arith.constant 5 : index
  %c6 = arith.constant 6 : index

  %shape6 = fir.shape %c6 : (index) -> !fir.shape<1>
  %shape4 = fir.shape %c4 : (index) -> !fir.shape<1>

  %a:2 = hlfir.declare %arg0(%shape6) {uniq_name = ""}
      : (!fir.ref<!fir.array<6xf64>>, !fir.shape<1>)
      -> (!fir.ref<!fir.array<6xf64>>,
          !fir.ref<!fir.array<6xf64>>)

  %out:2 = hlfir.declare %arg1(%shape4) {uniq_name = ""}
      : (!fir.ref<!fir.array<4xf64>>, !fir.shape<1>)
      -> (!fir.ref<!fir.array<4xf64>>,
          !fir.ref<!fir.array<4xf64>>)

  // Fortran A(2:5), StableHLO [1:5].
  %slice = hlfir.designate %a#0 (%c2:%c5:%c1)
      shape %shape4
      : (!fir.ref<!fir.array<6xf64>>,
         index, index, index, !fir.shape<1>)
      -> !fir.box<!fir.array<4xf64>>

  %expr = hlfir.elemental %shape4
      : (!fir.shape<1>) -> !hlfir.expr<4xf64> {
  ^bb0(%i: index):
    %ref = hlfir.designate %slice (%i)
        : (!fir.box<!fir.array<4xf64>>, index)
        -> !fir.ref<f64>
    %value = fir.load %ref : !fir.ref<f64>
    %neg = arith.negf %value : f64
    hlfir.yield_element %neg : f64
  }

  hlfir.assign %expr to %out#0
      : !hlfir.expr<4xf64>,
        !fir.ref<!fir.array<4xf64>>

  hlfir.destroy %expr : !hlfir.expr<4xf64>
  return
}

// ELE-SLICE-LABEL: func.func @main(
// ELE-SLICE: %[[SLICE:.*]] = stablehlo.slice %arg0 [1:5]
// ELE-SLICE-SAME: : (tensor<6xf64>) -> tensor<4xf64>
// ELE-SLICE: %[[NEG:.*]] = stablehlo.negate %[[SLICE]]
// ELE-SLICE-SAME: tensor<4xf64>
// ELE-SLICE: return %arg0, %[[NEG]]





