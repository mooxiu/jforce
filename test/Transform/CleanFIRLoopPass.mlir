// TODO: add check
// RUN: split-file %s %t
// RUN: %jforce-opt %t/2d-axpy.mlir --jforce-clean-fir-loop | FileCheck %s --check-prefix=AXPY2D

//--- 2d-axpy.mlir
// AXPY2D-LABEL: func.func @kernel
module {
  func.func @kernel(%arg0: !fir.ref<i32> {jit.arg_type = 1 : ui32, jit.literal_val = 0 : i64}, %arg1: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 128 : i64}, %arg2: !fir.ref<i32> {jit.arg_type = 1 : ui32, jit.literal_val = 0 : i64}, %arg3: !fir.ref<!fir.array<128x128xf64>> {jit.arg_type = 1 : ui32}, %arg4: !fir.ref<!fir.array<128x128xf64>> {jit.arg_type = 1 : ui32}, %arg5: !fir.ref<!fir.array<128x128xf64>> {jit.arg_type = 1 : ui32}, %arg6: !fir.ref<f64> {jit.arg_type = 1 : ui32, jit.literal_val = 4619567317775286272 : i64}, %arg7: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 128 : i64}, %arg8: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 128 : i64}, %arg9: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 128 : i64}, %arg10: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 128 : i64}, %arg11: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 128 : i64}, %arg12: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 128 : i64}, %arg13: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1 : i64}, %arg14: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 128 : i64}, %arg15: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1 : i64}) {
    %c1 = arith.constant 1 : index
    %c128 = arith.constant 128 : index
    %0 = fir.alloca i32
    %1 = fir.load %arg2 : !fir.ref<i32>
    fir.store %1 to %0 : !fir.ref<i32>
    %2 = fir.alloca i32
    %3 = fir.load %arg0 : !fir.ref<i32>
    fir.store %3 to %2 : !fir.ref<i32>
    %4 = fir.declare %2 {uniq_name = "_QFFrun_benchmarkEi"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %5 = fir.declare %arg1 {uniq_name = "_QFFrun_benchmarkEn"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %6 = fir.declare %0 {uniq_name = "_QFFrun_benchmarkEj"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %7 = fir.shape %c128, %c128 : (index, index) -> !fir.shape<2>
    %8 = fir.declare %arg3(%7) {uniq_name = "_QFFrun_benchmarkEz"} : (!fir.ref<!fir.array<128x128xf64>>, !fir.shape<2>) -> !fir.ref<!fir.array<128x128xf64>>
    %9 = fir.declare %arg4(%7) {uniq_name = "_QFFrun_benchmarkEx"} : (!fir.ref<!fir.array<128x128xf64>>, !fir.shape<2>) -> !fir.ref<!fir.array<128x128xf64>>
    %10 = fir.declare %arg5(%7) {uniq_name = "_QFFrun_benchmarkEy"} : (!fir.ref<!fir.array<128x128xf64>>, !fir.shape<2>) -> !fir.ref<!fir.array<128x128xf64>>
    %11 = fir.declare %arg6 {uniq_name = "_QFFrun_benchmarkEa"} : (!fir.ref<f64>) -> !fir.ref<f64>
    %12 = fir.alloca i32
    fir.store %1 to %12 : !fir.ref<i32>
    %13 = fir.alloca i32
    fir.store %3 to %13 : !fir.ref<i32>
    %14 = fir.declare %13 {uniq_name = "_QFFrun_benchmarkEi"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %15 = fir.declare %12 {uniq_name = "_QFFrun_benchmarkEj"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %16 = fir.convert %c1 : (index) -> i32
    fir.do_loop %arg16 = %c1 to %c128 step %c1 {
      %17 = fir.do_loop %arg17 = %c1 to %c128 step %c1 iter_args(%arg18 = %16) -> (i32) {
        %18 = fir.load %11 : !fir.ref<f64>
        %19 = fir.convert %arg18 : (i32) -> i64
        %20 = fir.load %14 : !fir.ref<i32>
        %21 = fir.convert %20 : (i32) -> i64
        %22 = fir.array_coor %9(%7) %19, %21 : (!fir.ref<!fir.array<128x128xf64>>, !fir.shape<2>, i64, i64) -> !fir.ref<f64>
        %23 = fir.load %22 : !fir.ref<f64>
        %24 = arith.mulf %18, %23 fastmath<contract> : f64
        %25 = fir.array_coor %10(%7) %19, %21 : (!fir.ref<!fir.array<128x128xf64>>, !fir.shape<2>, i64, i64) -> !fir.ref<f64>
        %26 = fir.load %25 : !fir.ref<f64>
        %27 = arith.addf %24, %26 fastmath<contract> : f64
        %28 = fir.array_coor %8(%7) %19, %21 : (!fir.ref<!fir.array<128x128xf64>>, !fir.shape<2>, i64, i64) -> !fir.ref<f64>
        fir.store %27 to %28 : !fir.ref<f64>
        %29 = arith.addi %arg18, %16 overflow<nsw> : i32
        fir.result %29 : i32
      }
    }
    omp.terminator
  }
}
