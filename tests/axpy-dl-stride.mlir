func.func @kernel(%arg0: !fir.ref<i32> {jit.literal_val = 0 : i64}, %arg1: !fir.ref<i32> {jit.literal_val = 1024 : i64}, %arg2: !fir.ref<i32> {jit.literal_val = 3 : i64}, %arg3: !fir.ref<!fir.array<?xf64>>, %arg4: !fir.ref<!fir.array<?xf64>>, %arg5: !fir.ref<!fir.array<?xf64>>, %arg6: !fir.ref<f64> {jit.literal_val = 4619567317775286272 : i64}, %arg7: !fir.ref<i32> {jit.literal_val = 1024 : i64}, %arg8: !fir.ref<i32> {jit.literal_val = 1024 : i64}, %arg9: !fir.ref<i32> {jit.literal_val = 1024 : i64}) {
  %0 = fir.load %arg9 : !fir.ref<i32>
  %1 = fir.load %arg8 : !fir.ref<i32>
  %2 = fir.load %arg7 : !fir.ref<i32>
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
  %17:2 = hlfir.declare %arg2 {uniq_name = "_QFFrun_benchmarkEstride"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %18 = fir.shape %14 : (index) -> !fir.shape<1>
  %19:2 = hlfir.declare %arg3(%18) {uniq_name = "_QFFrun_benchmarkEz"} : (!fir.ref<!fir.array<?xf64>>, !fir.shape<1>) -> (!fir.box<!fir.array<?xf64>>, !fir.ref<!fir.array<?xf64>>)
  %20 = fir.shape %13 : (index) -> !fir.shape<1>
  %21:2 = hlfir.declare %arg4(%20) {uniq_name = "_QFFrun_benchmarkEx"} : (!fir.ref<!fir.array<?xf64>>, !fir.shape<1>) -> (!fir.box<!fir.array<?xf64>>, !fir.ref<!fir.array<?xf64>>)
  %22 = fir.shape %12 : (index) -> !fir.shape<1>
  %23:2 = hlfir.declare %arg5(%22) {uniq_name = "_QFFrun_benchmarkEy"} : (!fir.ref<!fir.array<?xf64>>, !fir.shape<1>) -> (!fir.box<!fir.array<?xf64>>, !fir.ref<!fir.array<?xf64>>)
  %24:2 = hlfir.declare %arg6 {uniq_name = "_QFFrun_benchmarkEa"} : (!fir.ref<f64>) -> (!fir.ref<f64>, !fir.ref<f64>)
  %c1_i32 = arith.constant 1 : i32
  %25 = fir.convert %c1_i32 : (i32) -> index
  %26 = fir.load %16#0 : !fir.ref<i32>
  %27 = fir.convert %26 : (i32) -> index
  %28 = fir.load %17#0 : !fir.ref<i32>
  %29 = fir.convert %28 : (i32) -> index
  %30 = fir.convert %25 : (index) -> i32
  %31 = fir.do_loop %arg10 = %25 to %27 step %29 iter_args(%arg11 = %30) -> (i32) {
    fir.store %arg11 to %15#0 : !fir.ref<i32>
    %32 = fir.load %24#0 : !fir.ref<f64>
    %33 = fir.load %15#0 : !fir.ref<i32>
    %34 = fir.convert %33 : (i32) -> i64
    %35 = hlfir.designate %21#0 (%34)  : (!fir.box<!fir.array<?xf64>>, i64) -> !fir.ref<f64>
    %36 = fir.load %35 : !fir.ref<f64>
    %37 = arith.mulf %32, %36 fastmath<contract> : f64
    %38 = fir.load %15#0 : !fir.ref<i32>
    %39 = fir.convert %38 : (i32) -> i64
    %40 = hlfir.designate %23#0 (%39)  : (!fir.box<!fir.array<?xf64>>, i64) -> !fir.ref<f64>
    %41 = fir.load %40 : !fir.ref<f64>
    %42 = arith.addf %37, %41 fastmath<contract> : f64
    %43 = fir.load %15#0 : !fir.ref<i32>
    %44 = fir.convert %43 : (i32) -> i64
    %45 = hlfir.designate %19#0 (%44)  : (!fir.box<!fir.array<?xf64>>, i64) -> !fir.ref<f64>
    hlfir.assign %42 to %45 : f64, !fir.ref<f64>
    %46 = fir.convert %29 : (index) -> i32
    %47 = fir.load %15#0 : !fir.ref<i32>
    %48 = arith.addi %47, %46 overflow<nsw> : i32
    fir.result %48 : i32
  }
  fir.store %31 to %15#0 : !fir.ref<i32>
  omp.terminator
}
