module {
  func.func @kernel(%arg0: !fir.ref<i32> {jit.arg_type = 1 : ui32, jit.literal_val = 0 : i64}, %arg1: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 3 : i64}, %arg2: !fir.ref<i32> {jit.arg_type = 1 : ui32, jit.literal_val = 0 : i64}, %arg3: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 4 : i64}, %arg4: !fir.ref<!fir.array<4x3xf64>> {jit.arg_type = 1 : ui32}, %arg5: !fir.ref<!fir.array<4x3xf64>> {jit.arg_type = 1 : ui32}, %arg6: !fir.ref<!fir.array<4x3xf64>> {jit.arg_type = 1 : ui32}, %arg7: !fir.ref<f64> {jit.arg_type = 1 : ui32, jit.literal_val = 4619567317775286272 : i64}, %arg8: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 3 : i64}, %arg9: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 4 : i64}, %arg10: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 3 : i64}, %arg11: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 4 : i64}, %arg12: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 3 : i64}, %arg13: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 4 : i64}) {
    %c3_i32 = arith.constant 3 : i32
    %c4_i32 = arith.constant 4 : i32
    %c3 = arith.constant 3 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %0 = fir.declare %arg0 {uniq_name = "_QFFrun_benchmarkEj"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %1 = fir.declare %arg1 {uniq_name = "_QFFrun_benchmarkEm"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %2 = fir.declare %arg2 {uniq_name = "_QFFrun_benchmarkEi"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %3 = fir.declare %arg3 {uniq_name = "_QFFrun_benchmarkEn"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %4 = fir.shape %c4, %c3 : (index, index) -> !fir.shape<2>
    %5 = fir.declare %arg4(%4) {uniq_name = "_QFFrun_benchmarkEz"} : (!fir.ref<!fir.array<4x3xf64>>, !fir.shape<2>) -> !fir.ref<!fir.array<4x3xf64>>
    %6 = fir.declare %arg5(%4) {uniq_name = "_QFFrun_benchmarkEx"} : (!fir.ref<!fir.array<4x3xf64>>, !fir.shape<2>) -> !fir.ref<!fir.array<4x3xf64>>
    %7 = fir.declare %arg6(%4) {uniq_name = "_QFFrun_benchmarkEy"} : (!fir.ref<!fir.array<4x3xf64>>, !fir.shape<2>) -> !fir.ref<!fir.array<4x3xf64>>
    %8 = fir.declare %arg7 {uniq_name = "_QFFrun_benchmarkEa"} : (!fir.ref<f64>) -> !fir.ref<f64>
    %9 = fir.convert %c1 : (index) -> i32
    %10 = fir.convert %0 : (!fir.ref<i32>) -> memref<i32>
    %11 = fir.convert %2 : (!fir.ref<i32>) -> memref<i32>
    %12 = fir.convert %8 : (!fir.ref<f64>) -> memref<f64>
    %13 = fir.convert %6 : (!fir.ref<!fir.array<4x3xf64>>) -> memref<3x4xf64>
    %14 = fir.convert %7 : (!fir.ref<!fir.array<4x3xf64>>) -> memref<3x4xf64>
    %15 = fir.convert %5 : (!fir.ref<!fir.array<4x3xf64>>) -> memref<3x4xf64>
    %16 = affine.load %12[] : memref<f64>
    affine.for %arg14 = 1 to 4 {
      affine.parallel (%arg15) = (1) to (5) {
        %19 = affine.load %13[%arg14 - 1, %arg15 - 1] : memref<3x4xf64>
        %20 = arith.mulf %16, %19 fastmath<contract> : f64
        %21 = affine.load %14[%arg14 - 1, %arg15 - 1] : memref<3x4xf64>
        %22 = arith.addf %20, %21 fastmath<contract> : f64
        affine.store %22, %15[%arg14 - 1, %arg15 - 1] : memref<3x4xf64>
      }
    }
    %17 = arith.addi %9, %c4_i32 overflow<nsw> : i32
    affine.store %17, %11[] : memref<i32>
    %18 = arith.addi %9, %c3_i32 overflow<nsw> : i32
    affine.store %18, %10[] : memref<i32>
    omp.terminator
  }
}

