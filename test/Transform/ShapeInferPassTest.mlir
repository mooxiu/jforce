// RUN: split-file %s %t
// RUN: %jforce-opt %t/gemm.mlir --jforce-shape-infer | FileCheck %s --check-prefix=GEMM

//--- gemm.mlir
// GEMM-LABEL: func.func @kernel
module {
  func.func @kernel(%arg0: !fir.ref<!fir.array<?x?xf64>>, %arg1: !fir.ref<!fir.array<?x?xf64>>, %arg2: !fir.ref<!fir.array<?x?xf64>>, %arg3: !fir.ref<f64> {jit.literal_val = 4619567317775286272 : i64}, %arg4: !fir.ref<i32> {jit.literal_val = 3 : i64, jit.shape_arg}, %arg5: !fir.ref<i32> {jit.literal_val = 3 : i64, jit.shape_arg}, %arg6: !fir.ref<i32> {jit.literal_val = 3 : i64, jit.shape_arg}, %arg7: !fir.ref<i32> {jit.literal_val = 3 : i64, jit.shape_arg}, %arg8: !fir.ref<i32> {jit.literal_val = 3 : i64, jit.shape_arg}, %arg9: !fir.ref<i32> {jit.literal_val = 3 : i64, jit.shape_arg}) {
    %c3 = arith.constant 3 : index
    %0 = fir.shape %c3, %c3 : (index, index) -> !fir.shape<2>

// GEMM: (!fir.ref<!fir.array<3x3xf64>>, !fir.shape<2>) -> (!fir.ref<!fir.array<3x3xf64>>, !fir.ref<!fir.array<3x3xf64>>)
// GEMM-NEXT: (!fir.ref<!fir.array<3x3xf64>>, !fir.shape<2>) -> (!fir.ref<!fir.array<3x3xf64>>, !fir.ref<!fir.array<3x3xf64>>)
// GEMM-NEXT: (!fir.ref<!fir.array<3x3xf64>>, !fir.shape<2>) -> (!fir.ref<!fir.array<3x3xf64>>, !fir.ref<!fir.array<3x3xf64>>)

    %1:2 = hlfir.declare %arg0(%0) {uniq_name = "_QFFrun_benchmarkEz"} : (!fir.ref<!fir.array<?x?xf64>>, !fir.shape<2>) -> (!fir.box<!fir.array<?x?xf64>>, !fir.ref<!fir.array<?x?xf64>>)
    %2:2 = hlfir.declare %arg1(%0) {uniq_name = "_QFFrun_benchmarkEx"} : (!fir.ref<!fir.array<?x?xf64>>, !fir.shape<2>) -> (!fir.box<!fir.array<?x?xf64>>, !fir.ref<!fir.array<?x?xf64>>)
    %3:2 = hlfir.declare %arg2(%0) {uniq_name = "_QFFrun_benchmarkEy"} : (!fir.ref<!fir.array<?x?xf64>>, !fir.shape<2>) -> (!fir.box<!fir.array<?x?xf64>>, !fir.ref<!fir.array<?x?xf64>>)
    %4:2 = hlfir.declare %arg3 {uniq_name = "_QFFrun_benchmarkEa"} : (!fir.ref<f64>) -> (!fir.ref<f64>, !fir.ref<f64>)
    omp.teams {
      omp.workdistribute {

// GEMM: (!fir.ref<!fir.array<3x3xf64>>, !fir.ref<!fir.array<3x3xf64>>) -> !hlfir.expr<3x3xf64>
        %5 = hlfir.matmul %2#0 %3#0 {fastmath = #arith.fastmath<contract>} : (!fir.box<!fir.array<?x?xf64>>, !fir.box<!fir.array<?x?xf64>>) -> !hlfir.expr<?x?xf64>
        %6 = fir.load %4#0 : !fir.ref<f64>

// GEMM: hlfir.shape_of
// GEMM-SAME: (!hlfir.expr<3x3xf64>) -> !fir.shape<2>
        %7 = hlfir.shape_of %5 : (!hlfir.expr<?x?xf64>) -> !fir.shape<2>
        %8 = hlfir.elemental %7 unordered : (!fir.shape<2>) -> !hlfir.expr<?x?xf64> {
        ^bb0(%arg10: index, %arg11: index):
          %9 = hlfir.apply %5, %arg10, %arg11 : (!hlfir.expr<?x?xf64>, index, index) -> f64
          %10 = arith.addf %9, %6 fastmath<contract> : f64
          hlfir.yield_element %10 : f64
        }
        hlfir.assign %8 to %1#0 : !hlfir.expr<?x?xf64>, !fir.box<!fir.array<?x?xf64>>
        hlfir.destroy %8 : !hlfir.expr<?x?xf64>
        hlfir.destroy %5 : !hlfir.expr<?x?xf64>
        omp.terminator
      }
      omp.terminator
    }
    omp.terminator
  }
}
