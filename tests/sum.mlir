func.func @kernel(%arg0: !fir.ref<i32> {jit.literal_val = 0 : i64}, %arg1: !fir.ref<i32> {jit.literal_val = 1024 : i64}, %arg2: !fir.ref<f64> {jit.literal_val = 0 : i64}, %arg3: !fir.ref<!fir.array<?xf64>>, %arg4: !fir.ref<i32> {jit.literal_val = 1024 : i64}) {
  %0 = fir.load %arg4 : !fir.ref<i32>
  %1 = fir.convert %0 : (i32) -> i64
  %c0 = arith.constant 0 : index
  %2 = fir.convert %1 : (i64) -> index
  %3 = arith.cmpi sgt, %2, %c0 : index
  %4 = arith.select %3, %2, %c0 : index
  %5:2 = hlfir.declare %arg0 {uniq_name = "_QFFrun_benchmarkEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %6:2 = hlfir.declare %arg1 {uniq_name = "_QFFrun_benchmarkEn"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %7:2 = hlfir.declare %arg2 {uniq_name = "_QFFrun_benchmarkEsum_x"} : (!fir.ref<f64>) -> (!fir.ref<f64>, !fir.ref<f64>)
  %8 = fir.shape %4 : (index) -> !fir.shape<1>
  %9:2 = hlfir.declare %arg3(%8) {uniq_name = "_QFFrun_benchmarkEx"} : (!fir.ref<!fir.array<?xf64>>, !fir.shape<1>) -> (!fir.box<!fir.array<?xf64>>, !fir.ref<!fir.array<?xf64>>)
  %c1_i32 = arith.constant 1 : i32
  %10 = fir.convert %c1_i32 : (i32) -> index
  %11 = fir.load %6#0 : !fir.ref<i32>
  %12 = fir.convert %11 : (i32) -> index
  %c1 = arith.constant 1 : index
  %13 = fir.convert %10 : (index) -> i32
  %14 = fir.do_loop %arg5 = %10 to %12 step %c1 iter_args(%arg6 = %13) -> (i32) {
    fir.store %arg6 to %5#0 : !fir.ref<i32>
    %15 = fir.load %7#0 : !fir.ref<f64>
    %16 = fir.load %5#0 : !fir.ref<i32>
    %17 = fir.convert %16 : (i32) -> i64
    %18 = hlfir.designate %9#0 (%17)  : (!fir.box<!fir.array<?xf64>>, i64) -> !fir.ref<f64>
    %19 = fir.load %18 : !fir.ref<f64>
    %20 = arith.addf %15, %19 fastmath<contract> : f64
    hlfir.assign %20 to %7#0 : f64, !fir.ref<f64>
    %21 = fir.convert %c1 : (index) -> i32
    %22 = fir.load %5#0 : !fir.ref<i32>
    %23 = arith.addi %22, %21 overflow<nsw> : i32
    fir.result %23 : i32
  }
  fir.store %14 to %5#0 : !fir.ref<i32>
  omp.terminator
}
