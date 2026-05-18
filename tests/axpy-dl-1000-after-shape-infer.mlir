// ./build/src/tool/jforce-opt ./tests/axpy-dl.mlir \
//   --jforce-annotate \
//   --jforce-propagate-constants \
//   --canonicalize --sccp --cse --canonicalize \
//   --jforce-shape-infer --canonicalize 

module {
  func.func @kernel(%arg0: !fir.ref<i32> {jit.arg_type = 1 : ui32, jit.literal_val = 0 : i64}, %arg1: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1000 : i64}, %arg2: !fir.ref<!fir.array<1000xf64>> {jit.arg_type = 1 : ui32}, %arg3: !fir.ref<!fir.array<1000xf64>> {jit.arg_type = 1 : ui32}, %arg4: !fir.ref<!fir.array<1000xf64>> {jit.arg_type = 1 : ui32}, %arg5: !fir.ref<f64> {jit.arg_type = 1 : ui32, jit.literal_val = 4619567317775286272 : i64}, %arg6: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1000 : i64}, %arg7: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1000 : i64}, %arg8: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1000 : i64}) {
    %c1000 = arith.constant 1000 : index
    %c1 = arith.constant 1 : index
    %0:2 = hlfir.declare %arg0 {uniq_name = "_QFFrun_benchmarkEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
    %1:2 = hlfir.declare %arg1 {uniq_name = "_QFFrun_benchmarkEn"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
    %2 = fir.shape %c1000 : (index) -> !fir.shape<1>
    %3:2 = hlfir.declare %arg2(%2) {uniq_name = "_QFFrun_benchmarkEz"} : (!fir.ref<!fir.array<1000xf64>>, !fir.shape<1>) -> (!fir.ref<!fir.array<1000xf64>>, !fir.ref<!fir.array<1000xf64>>)
    %4:2 = hlfir.declare %arg3(%2) {uniq_name = "_QFFrun_benchmarkEx"} : (!fir.ref<!fir.array<1000xf64>>, !fir.shape<1>) -> (!fir.ref<!fir.array<1000xf64>>, !fir.ref<!fir.array<1000xf64>>)
    %5:2 = hlfir.declare %arg4(%2) {uniq_name = "_QFFrun_benchmarkEy"} : (!fir.ref<!fir.array<1000xf64>>, !fir.shape<1>) -> (!fir.ref<!fir.array<1000xf64>>, !fir.ref<!fir.array<1000xf64>>)
    %6:2 = hlfir.declare %arg5 {uniq_name = "_QFFrun_benchmarkEa"} : (!fir.ref<f64>) -> (!fir.ref<f64>, !fir.ref<f64>)
    %7 = fir.convert %c1 : (index) -> i32
    %8 = fir.do_loop %arg9 = %c1 to %c1000 step %c1 iter_args(%arg10 = %7) -> (i32) {
      fir.store %arg10 to %0#0 : !fir.ref<i32>
      %9 = fir.load %6#0 : !fir.ref<f64>
      %10 = fir.load %0#0 : !fir.ref<i32>
      %11 = fir.convert %10 : (i32) -> i64
      %12 = hlfir.designate %4#0 (%11)  : (!fir.ref<!fir.array<1000xf64>>, i64) -> !fir.ref<f64>
      %13 = fir.load %12 : !fir.ref<f64>
      %14 = arith.mulf %9, %13 fastmath<contract> : f64
      %15 = hlfir.designate %5#0 (%11)  : (!fir.ref<!fir.array<1000xf64>>, i64) -> !fir.ref<f64>
      %16 = fir.load %15 : !fir.ref<f64>
      %17 = arith.addf %14, %16 fastmath<contract> : f64
      %18 = hlfir.designate %3#0 (%11)  : (!fir.ref<!fir.array<1000xf64>>, i64) -> !fir.ref<f64>
      hlfir.assign %17 to %18 : f64, !fir.ref<f64>
      %19 = fir.load %0#0 : !fir.ref<i32>
      %20 = arith.addi %19, %7 overflow<nsw> : i32
      fir.result %20 : i32
    }
    fir.store %8 to %0#0 : !fir.ref<i32>
    omp.terminator
  }
}

