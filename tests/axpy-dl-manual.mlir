module {
  func.func @kernel(%arg0: !fir.ref<i32> {jit.arg_type = 1 : ui32, jit.literal_val = 0 : i64}, %arg1: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 4 : i64}, %arg2: !fir.ref<!fir.array<4xf64>> {jit.arg_type = 1 : ui32}, %arg3: !fir.ref<!fir.array<4xf64>> {jit.arg_type = 1 : ui32}, %arg4: !fir.ref<!fir.array<4xf64>> {jit.arg_type = 1 : ui32}, %arg5: !fir.ref<f64> {jit.arg_type = 1 : ui32, jit.literal_val = 4619567317775286272 : i64}, %arg6: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 4 : i64}, %arg7: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 4 : i64}, %arg8: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 4 : i64}) {
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %0:2 = hlfir.declare %arg0 {uniq_name = "_QFFrun_benchmarkEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
    %1:2 = hlfir.declare %arg1 {uniq_name = "_QFFrun_benchmarkEn"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
    %2 = fir.shape %c4 : (index) -> !fir.shape<1>
    %3:2 = hlfir.declare %arg2(%2) {uniq_name = "_QFFrun_benchmarkEz"} : (!fir.ref<!fir.array<4xf64>>, !fir.shape<1>) -> (!fir.ref<!fir.array<4xf64>>, !fir.ref<!fir.array<4xf64>>)
    %4:2 = hlfir.declare %arg3(%2) {uniq_name = "_QFFrun_benchmarkEx"} : (!fir.ref<!fir.array<4xf64>>, !fir.shape<1>) -> (!fir.ref<!fir.array<4xf64>>, !fir.ref<!fir.array<4xf64>>)
    %5:2 = hlfir.declare %arg4(%2) {uniq_name = "_QFFrun_benchmarkEy"} : (!fir.ref<!fir.array<4xf64>>, !fir.shape<1>) -> (!fir.ref<!fir.array<4xf64>>, !fir.ref<!fir.array<4xf64>>)
    %6:2 = hlfir.declare %arg5 {uniq_name = "_QFFrun_benchmarkEa"} : (!fir.ref<f64>) -> (!fir.ref<f64>, !fir.ref<f64>)
    %7 = fir.convert %c1 : (index) -> i32
    %9 = fir.load %6#0 : !fir.ref<f64>
    fir.do_loop %arg9 = %c1 to %c4 step %c1 {
      %8 = fir.convert %arg9 : (index) -> i32
      %10 = fir.convert %8 : (i32) -> i64
      %11 = hlfir.designate %4#0 (%10)  : (!fir.ref<!fir.array<4xf64>>, i64) -> !fir.ref<f64>
      %12 = fir.load %11 : !fir.ref<f64>
      %13 = arith.mulf %9, %12 fastmath<contract> : f64
      %14 = hlfir.designate %5#0 (%10)  : (!fir.ref<!fir.array<4xf64>>, i64) -> !fir.ref<f64>
      %15 = fir.load %14 : !fir.ref<f64>
      %16 = arith.addf %13, %15 fastmath<contract> : f64
      %17 = hlfir.designate %3#0 (%10)  : (!fir.ref<!fir.array<4xf64>>, i64) -> !fir.ref<f64>
      hlfir.assign %16 to %17 : f64, !fir.ref<f64>
    }
    %final = fir.convert %c4 : (index) -> i32 
    %18 = arith.addi %final, %7 overflow<nsw> : i32
    fir.store %18 to %0#0 : !fir.ref<i32>
    omp.terminator
  }
}

