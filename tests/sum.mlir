func.func @kernel(%arg0: !fir.ref<i32>, %arg1: !fir.ref<i32>, %arg2: !fir.ref<f64>, %arg3: !fir.ref<!fir.array<?xf64>>, %arg4: !fir.ref<f64>, %arg5: !fir.ref<!fir.array<?xf64>>, %arg6: !fir.ref<i32>, %arg7: !fir.ref<i32>) {
  %0 = fir.load %arg7 : !fir.ref<i32>
  %1 = fir.load %arg6 : !fir.ref<i32>
  %2 = fir.convert %1 : (i32) -> i64
  %3 = fir.convert %0 : (i32) -> i64
  %c0 = arith.constant 0 : index
  %4 = fir.convert %3 : (i64) -> index
  %5 = arith.cmpi sgt, %4, %c0 : index
  %c0_0 = arith.constant 0 : index
  %6 = fir.convert %2 : (i64) -> index
  %7 = arith.cmpi sgt, %6, %c0_0 : index
  %8 = arith.select %7, %6, %c0_0 : index
  %9 = arith.select %5, %4, %c0 : index
  %10:2 = hlfir.declare %arg0 {uniq_name = "_QFFrun_benchmarkEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %11:2 = hlfir.declare %arg1 {uniq_name = "_QFFrun_benchmarkEn"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %12:2 = hlfir.declare %arg2 {uniq_name = "_QFFrun_benchmarkEsum_x"} : (!fir.ref<f64>) -> (!fir.ref<f64>, !fir.ref<f64>)
  %13 = fir.shape %9 : (index) -> !fir.shape<1>
  %14:2 = hlfir.declare %arg3(%13) {uniq_name = "_QFFrun_benchmarkEx"} : (!fir.ref<!fir.array<?xf64>>, !fir.shape<1>) -> (!fir.box<!fir.array<?xf64>>, !fir.ref<!fir.array<?xf64>>)
  %15:2 = hlfir.declare %arg4 {uniq_name = "_QFFrun_benchmarkEsum_y"} : (!fir.ref<f64>) -> (!fir.ref<f64>, !fir.ref<f64>)
  %16 = fir.shape %8 : (index) -> !fir.shape<1>
  %17:2 = hlfir.declare %arg5(%16) {uniq_name = "_QFFrun_benchmarkEy"} : (!fir.ref<!fir.array<?xf64>>, !fir.shape<1>) -> (!fir.box<!fir.array<?xf64>>, !fir.ref<!fir.array<?xf64>>)
  %c1_i32 = arith.constant 1 : i32
  %18 = fir.convert %c1_i32 : (i32) -> index
  %19 = fir.load %11#0 : !fir.ref<i32>
  %20 = fir.convert %19 : (i32) -> index
  %c1 = arith.constant 1 : index
  %21 = fir.convert %18 : (index) -> i32
  %22 = fir.do_loop %arg8 = %18 to %20 step %c1 iter_args(%arg9 = %21) -> (i32) {
    fir.store %arg9 to %10#0 : !fir.ref<i32>
    %23 = fir.load %12#0 : !fir.ref<f64>
    %24 = fir.load %10#0 : !fir.ref<i32>
    %25 = fir.convert %24 : (i32) -> i64
    %26 = hlfir.designate %14#0 (%25)  : (!fir.box<!fir.array<?xf64>>, i64) -> !fir.ref<f64>
    %27 = fir.load %26 : !fir.ref<f64>
    %28 = arith.addf %23, %27 fastmath<contract> : f64
    hlfir.assign %28 to %12#0 : f64, !fir.ref<f64>
    %29 = fir.load %15#0 : !fir.ref<f64>
    %30 = fir.load %10#0 : !fir.ref<i32>
    %31 = fir.convert %30 : (i32) -> i64
    %32 = hlfir.designate %17#0 (%31)  : (!fir.box<!fir.array<?xf64>>, i64) -> !fir.ref<f64>
    %33 = fir.load %32 : !fir.ref<f64>
    %34 = arith.addf %29, %33 fastmath<contract> : f64
    hlfir.assign %34 to %15#0 : f64, !fir.ref<f64>
    %35 = fir.convert %c1 : (index) -> i32
    %36 = fir.load %10#0 : !fir.ref<i32>
    %37 = arith.addi %36, %35 overflow<nsw> : i32
    fir.result %37 : i32
  }
  fir.store %22 to %10#0 : !fir.ref<i32>
  omp.terminator
}

