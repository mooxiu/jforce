func.func @kernel(%arg0: !fir.ref<i32> {jit.literal_val = 0 : i64}, %arg1: !fir.ref<i32> {jit.literal_val = 1024 : i64}, %arg2: !fir.ref<!fir.array<?xf64>>, %arg3: !fir.ref<!fir.array<?xf64>>, %arg4: !fir.ref<i32> {jit.literal_val = 1024 : i64}, %arg5: !fir.ref<i32> {jit.literal_val = 1024 : i64}) {
  %0 = fir.load %arg5 : !fir.ref<i32>
  %1 = fir.load %arg4 : !fir.ref<i32>
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
  %12 = fir.shape %9 : (index) -> !fir.shape<1>
  %13:2 = hlfir.declare %arg2(%12) {uniq_name = "_QFFrun_benchmarkEy"} : (!fir.ref<!fir.array<?xf64>>, !fir.shape<1>) -> (!fir.box<!fir.array<?xf64>>, !fir.ref<!fir.array<?xf64>>)
  %14 = fir.shape %8 : (index) -> !fir.shape<1>
  %15:2 = hlfir.declare %arg3(%14) {uniq_name = "_QFFrun_benchmarkEx"} : (!fir.ref<!fir.array<?xf64>>, !fir.shape<1>) -> (!fir.box<!fir.array<?xf64>>, !fir.ref<!fir.array<?xf64>>)
  %c2_i32 = arith.constant 2 : i32
  %16 = fir.convert %c2_i32 : (i32) -> index
  %17 = fir.load %11#0 : !fir.ref<i32>
  %c1_i32 = arith.constant 1 : i32
  %18 = arith.subi %17, %c1_i32 overflow<nsw> : i32
  %19 = fir.convert %18 : (i32) -> index
  %c1 = arith.constant 1 : index
  %20 = fir.convert %16 : (index) -> i32
  %21 = fir.do_loop %arg6 = %16 to %19 step %c1 iter_args(%arg7 = %20) -> (i32) {
    fir.store %arg7 to %10#0 : !fir.ref<i32>
    %22 = fir.load %10#0 : !fir.ref<i32>
    %c1_i32_1 = arith.constant 1 : i32
    %23 = arith.subi %22, %c1_i32_1 overflow<nsw> : i32
    %24 = fir.convert %23 : (i32) -> i64
    %25 = hlfir.designate %15#0 (%24)  : (!fir.box<!fir.array<?xf64>>, i64) -> !fir.ref<f64>
    %26 = fir.load %25 : !fir.ref<f64>
    %27 = fir.load %10#0 : !fir.ref<i32>
    %c1_i32_2 = arith.constant 1 : i32
    %28 = arith.addi %27, %c1_i32_2 overflow<nsw> : i32
    %29 = fir.convert %28 : (i32) -> i64
    %30 = hlfir.designate %15#0 (%29)  : (!fir.box<!fir.array<?xf64>>, i64) -> !fir.ref<f64>
    %31 = fir.load %30 : !fir.ref<f64>
    %32 = arith.addf %26, %31 fastmath<contract> : f64
    %33 = fir.load %10#0 : !fir.ref<i32>
    %34 = fir.convert %33 : (i32) -> i64
    %35 = hlfir.designate %13#0 (%34)  : (!fir.box<!fir.array<?xf64>>, i64) -> !fir.ref<f64>
    hlfir.assign %32 to %35 : f64, !fir.ref<f64>
    %36 = fir.convert %c1 : (index) -> i32
    %37 = fir.load %10#0 : !fir.ref<i32>
    %38 = arith.addi %37, %36 overflow<nsw> : i32
    fir.result %38 : i32
  }
  fir.store %21 to %10#0 : !fir.ref<i32>
  omp.terminator
}
