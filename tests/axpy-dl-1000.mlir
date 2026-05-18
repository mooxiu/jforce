func.func @kernel(%arg0: !fir.ref<i32> {jit.literal_val = 0 : i64}, %arg1: !fir.ref<i32> {jit.literal_val = 1000 : i64}, %arg2: !fir.ref<!fir.array<?xf64>>, %arg3: !fir.ref<!fir.array<?xf64>>, %arg4: !fir.ref<!fir.array<?xf64>>, %arg5: !fir.ref<f64> {jit.literal_val = 4619567317775286272 : i64}, %arg6: !fir.ref<i32> {jit.literal_val = 1000 : i64}, %arg7: !fir.ref<i32> {jit.literal_val = 1000 : i64}, %arg8: !fir.ref<i32> {jit.literal_val = 1000 : i64}) {
  %0 = fir.load %arg8 : !fir.ref<i32>
  %1 = fir.load %arg7 : !fir.ref<i32>
  %2 = fir.load %arg6 : !fir.ref<i32>
  %3 = fir.convert %2 : (i32) -> i64
  %4 = fir.convert %1 : (i32) -> i64
  %5 = fir.convert %0 : (i32) -> i64
  %c0 = arith.constant 0 : index
  %6 = fir.convert %5 : (i64) -> index
  %7 = arith.cmpi sgt, %6, %c0 : index
  %c0_0 = arith.constant 0 : index
  %8 = fir.convert %4 : (i64) -> index
  %9 = arith.cmpi sgt, %8, %c0_0 : index
  %c0_1 = arith.constant 0 : index
  %10 = fir.convert %3 : (i64) -> index
  %11 = arith.cmpi sgt, %10, %c0_1 : index
  %12 = arith.select %11, %10, %c0_1 : index
  %13 = arith.select %9, %8, %c0_0 : index
  %14 = arith.select %7, %6, %c0 : index
  %15:2 = hlfir.declare %arg0 {uniq_name = "_QFFrun_benchmarkEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %16:2 = hlfir.declare %arg1 {uniq_name = "_QFFrun_benchmarkEn"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %17 = fir.shape %14 : (index) -> !fir.shape<1>
  %18:2 = hlfir.declare %arg2(%17) {uniq_name = "_QFFrun_benchmarkEz"} : (!fir.ref<!fir.array<?xf64>>, !fir.shape<1>) -> (!fir.box<!fir.array<?xf64>>, !fir.ref<!fir.array<?xf64>>)
  %19 = fir.shape %13 : (index) -> !fir.shape<1>
  %20:2 = hlfir.declare %arg3(%19) {uniq_name = "_QFFrun_benchmarkEx"} : (!fir.ref<!fir.array<?xf64>>, !fir.shape<1>) -> (!fir.box<!fir.array<?xf64>>, !fir.ref<!fir.array<?xf64>>)
  %21 = fir.shape %12 : (index) -> !fir.shape<1>
  %22:2 = hlfir.declare %arg4(%21) {uniq_name = "_QFFrun_benchmarkEy"} : (!fir.ref<!fir.array<?xf64>>, !fir.shape<1>) -> (!fir.box<!fir.array<?xf64>>, !fir.ref<!fir.array<?xf64>>)
  %23:2 = hlfir.declare %arg5 {uniq_name = "_QFFrun_benchmarkEa"} : (!fir.ref<f64>) -> (!fir.ref<f64>, !fir.ref<f64>)
  %c1_i32 = arith.constant 1 : i32
  %24 = fir.convert %c1_i32 : (i32) -> index
  %25 = fir.load %16#0 : !fir.ref<i32>
  %26 = fir.convert %25 : (i32) -> index
  %c1 = arith.constant 1 : index
  %27 = fir.convert %24 : (index) -> i32
  %28 = fir.do_loop %arg9 = %24 to %26 step %c1 iter_args(%arg10 = %27) -> (i32) {
    fir.store %arg10 to %15#0 : !fir.ref<i32>
    %29 = fir.load %23#0 : !fir.ref<f64>
    %30 = fir.load %15#0 : !fir.ref<i32>
    %31 = fir.convert %30 : (i32) -> i64
    %32 = hlfir.designate %20#0 (%31)  : (!fir.box<!fir.array<?xf64>>, i64) -> !fir.ref<f64>
    %33 = fir.load %32 : !fir.ref<f64>
    %34 = arith.mulf %29, %33 fastmath<contract> : f64
    %35 = fir.load %15#0 : !fir.ref<i32>
    %36 = fir.convert %35 : (i32) -> i64
    %37 = hlfir.designate %22#0 (%36)  : (!fir.box<!fir.array<?xf64>>, i64) -> !fir.ref<f64>
    %38 = fir.load %37 : !fir.ref<f64>
    %39 = arith.addf %34, %38 fastmath<contract> : f64
    %40 = fir.load %15#0 : !fir.ref<i32>
    %41 = fir.convert %40 : (i32) -> i64
    %42 = hlfir.designate %18#0 (%41)  : (!fir.box<!fir.array<?xf64>>, i64) -> !fir.ref<f64>
    hlfir.assign %39 to %42 : f64, !fir.ref<f64>
    %43 = fir.convert %c1 : (index) -> i32
    %44 = fir.load %15#0 : !fir.ref<i32>
    %45 = arith.addi %44, %43 overflow<nsw> : i32
    fir.result %45 : i32
  }
  fir.store %28 to %15#0 : !fir.ref<i32>
  omp.terminator
}
