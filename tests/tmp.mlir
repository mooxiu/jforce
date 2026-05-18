module {
  func.func @kernel(%arg0: !fir.ref<i32> {jit.arg_type = 1 : ui32, jit.literal_val = 0 : i64}, %arg1: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 4 : i64}, %arg2: !fir.ref<!fir.array<4xf64>> {jit.arg_type = 1 : ui32}, %arg3: !fir.ref<!fir.array<4xf64>> {jit.arg_type = 1 : ui32}, %arg4: !fir.ref<!fir.array<4xf64>> {jit.arg_type = 1 : ui32}, %arg5: !fir.ref<f64> {jit.arg_type = 1 : ui32, jit.literal_val = 4619567317775286272 : i64}, %arg6: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 4 : i64}, %arg7: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 4 : i64}, %arg8: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 4 : i64}) {
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %0 = fir.declare %arg0 {uniq_name = "_QFFrun_benchmarkEi"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %1 = fir.declare %arg1 {uniq_name = "_QFFrun_benchmarkEn"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %2 = fir.shape %c4 : (index) -> !fir.shape<1>
    %3 = fir.declare %arg2(%2) {uniq_name = "_QFFrun_benchmarkEz"} : (!fir.ref<!fir.array<4xf64>>, !fir.shape<1>) -> !fir.ref<!fir.array<4xf64>>
    %4 = fir.declare %arg3(%2) {uniq_name = "_QFFrun_benchmarkEx"} : (!fir.ref<!fir.array<4xf64>>, !fir.shape<1>) -> !fir.ref<!fir.array<4xf64>>
    %5 = fir.declare %arg4(%2) {uniq_name = "_QFFrun_benchmarkEy"} : (!fir.ref<!fir.array<4xf64>>, !fir.shape<1>) -> !fir.ref<!fir.array<4xf64>>
    %6 = fir.declare %arg5 {uniq_name = "_QFFrun_benchmarkEa"} : (!fir.ref<f64>) -> !fir.ref<f64>
    %7 = fir.convert %c1 : (index) -> i32
    %8 = fir.convert %0 : (!fir.ref<i32>) -> memref<i32>
    %9 = fir.convert %6 : (!fir.ref<f64>) -> memref<f64>
    %10 = fir.convert %4 : (!fir.ref<!fir.array<4xf64>>) -> memref<4xf64>
    %11 = fir.convert %5 : (!fir.ref<!fir.array<4xf64>>) -> memref<4xf64>
    %12 = fir.convert %3 : (!fir.ref<!fir.array<4xf64>>) -> memref<4xf64>
    affine.for %arg9 = 1 to 5 {
      %14 = arith.index_cast %arg9 : index to i32
      memref.store %14, %8[] : memref<i32>
      %15 = memref.load %9[] : memref<f64>
      %16 = memref.load %8[] : memref<i32>
      %17 = arith.index_cast %16 : i32 to index
      %18 = arith.subi %17, %c1 : index
      %19 = memref.load %10[%18] : memref<4xf64>
      %20 = arith.mulf %15, %19 fastmath<contract> : f64
      %21 = memref.load %11[%18] : memref<4xf64>
      %22 = arith.addf %20, %21 fastmath<contract> : f64
      memref.store %22, %12[%18] : memref<4xf64>
      %23 = memref.load %8[] : memref<i32>
      %24 = arith.addi %23, %7 overflow<nsw> : i32
      memref.store %24, %8[] : memref<i32>
    }
    %13 = memref.load %8[] : memref<i32>
    memref.store %13, %8[] : memref<i32>
    omp.terminator
  }
}

