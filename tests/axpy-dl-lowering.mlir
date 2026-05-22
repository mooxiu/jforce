// ./build/src/tool/jforce-opt ./tests/stencil-2d.mlir \
//    --jforce-annotate \
//    --jforce-propagate-constants \
//    --canonicalize --sccp --cse --canonicalize \
//    --jforce-shape-infer --canonicalize \
//    --convert-hlfir-to-fir --fir-to-memref --canonicalize \
//    --loop-invariant-code-motion --cse --canonicalize \
//    --jforce-clean-fir-loop --canonicalize --jforce-clean-fir-op  --promote-to-affine --affine-loop-normalize --canonicalize \
//    --jforce-optimize-mem-ops --canonicalize \
//    --jforce-loop-sink \
//    --enzyme-affinecfg --jforce-outline-affine \
//    --enzyme-affine-to-stablehlo --canonicalize --mlir-print-ir-after-all 2> ./tests/stencil-2d-lowering.mlir


// -----// IR Dump After {anonymous}::AnnotatePass (jforce-annotate) //----- //
func.func @kernel(%arg0: !fir.ref<i32> {jit.arg_type = 1 : ui32, jit.literal_val = 0 : i64}, %arg1: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 4 : i64}, %arg2: !fir.ref<!fir.array<?xf64>> {jit.arg_type = 1 : ui32}, %arg3: !fir.ref<!fir.array<?xf64>> {jit.arg_type = 1 : ui32}, %arg4: !fir.ref<!fir.array<?xf64>> {jit.arg_type = 1 : ui32}, %arg5: !fir.ref<f64> {jit.arg_type = 1 : ui32, jit.literal_val = 4619567317775286272 : i64}, %arg6: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 4 : i64}, %arg7: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 4 : i64}, %arg8: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 4 : i64}) {
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

// -----// IR Dump After {anonymous}::PropagateConstantsPass (jforce-propagate-constants) //----- //
func.func @kernel(%arg0: !fir.ref<i32> {jit.arg_type = 1 : ui32, jit.literal_val = 0 : i64}, %arg1: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 4 : i64}, %arg2: !fir.ref<!fir.array<?xf64>> {jit.arg_type = 1 : ui32}, %arg3: !fir.ref<!fir.array<?xf64>> {jit.arg_type = 1 : ui32}, %arg4: !fir.ref<!fir.array<?xf64>> {jit.arg_type = 1 : ui32}, %arg5: !fir.ref<f64> {jit.arg_type = 1 : ui32, jit.literal_val = 4619567317775286272 : i64}, %arg6: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 4 : i64}, %arg7: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 4 : i64}, %arg8: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 4 : i64}) {
  %c4_i32 = arith.constant 4 : i32
  %c4_i32_0 = arith.constant 4 : i32
  %c4_i32_1 = arith.constant 4 : i32
  %0 = fir.convert %c4_i32_1 : (i32) -> i64
  %1 = fir.convert %c4_i32_0 : (i32) -> i64
  %2 = fir.convert %c4_i32 : (i32) -> i64
  %c0 = arith.constant 0 : index
  %3 = fir.convert %2 : (i64) -> index
  %4 = arith.cmpi sgt, %3, %c0 : index
  %c0_2 = arith.constant 0 : index
  %5 = fir.convert %1 : (i64) -> index
  %6 = arith.cmpi sgt, %5, %c0_2 : index
  %c0_3 = arith.constant 0 : index
  %7 = fir.convert %0 : (i64) -> index
  %8 = arith.cmpi sgt, %7, %c0_3 : index
  %9 = arith.select %8, %7, %c0_3 : index
  %10 = arith.select %6, %5, %c0_2 : index
  %11 = arith.select %4, %3, %c0 : index
  %12:2 = hlfir.declare %arg0 {uniq_name = "_QFFrun_benchmarkEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %13:2 = hlfir.declare %arg1 {uniq_name = "_QFFrun_benchmarkEn"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %14 = fir.shape %11 : (index) -> !fir.shape<1>
  %15:2 = hlfir.declare %arg2(%14) {uniq_name = "_QFFrun_benchmarkEz"} : (!fir.ref<!fir.array<?xf64>>, !fir.shape<1>) -> (!fir.box<!fir.array<?xf64>>, !fir.ref<!fir.array<?xf64>>)
  %16 = fir.shape %10 : (index) -> !fir.shape<1>
  %17:2 = hlfir.declare %arg3(%16) {uniq_name = "_QFFrun_benchmarkEx"} : (!fir.ref<!fir.array<?xf64>>, !fir.shape<1>) -> (!fir.box<!fir.array<?xf64>>, !fir.ref<!fir.array<?xf64>>)
  %18 = fir.shape %9 : (index) -> !fir.shape<1>
  %19:2 = hlfir.declare %arg4(%18) {uniq_name = "_QFFrun_benchmarkEy"} : (!fir.ref<!fir.array<?xf64>>, !fir.shape<1>) -> (!fir.box<!fir.array<?xf64>>, !fir.ref<!fir.array<?xf64>>)
  %20:2 = hlfir.declare %arg5 {uniq_name = "_QFFrun_benchmarkEa"} : (!fir.ref<f64>) -> (!fir.ref<f64>, !fir.ref<f64>)
  %c1_i32 = arith.constant 1 : i32
  %21 = fir.convert %c1_i32 : (i32) -> index
  %c4_i32_4 = arith.constant 4 : i32
  %22 = fir.convert %c4_i32_4 : (i32) -> index
  %c1 = arith.constant 1 : index
  %23 = fir.convert %21 : (index) -> i32
  %24 = fir.do_loop %arg9 = %21 to %22 step %c1 iter_args(%arg10 = %23) -> (i32) {
    fir.store %arg10 to %12#0 : !fir.ref<i32>
    %25 = fir.load %20#0 : !fir.ref<f64>
    %26 = fir.load %12#0 : !fir.ref<i32>
    %27 = fir.convert %26 : (i32) -> i64
    %28 = hlfir.designate %17#0 (%27)  : (!fir.box<!fir.array<?xf64>>, i64) -> !fir.ref<f64>
    %29 = fir.load %28 : !fir.ref<f64>
    %30 = arith.mulf %25, %29 fastmath<contract> : f64
    %31 = fir.load %12#0 : !fir.ref<i32>
    %32 = fir.convert %31 : (i32) -> i64
    %33 = hlfir.designate %19#0 (%32)  : (!fir.box<!fir.array<?xf64>>, i64) -> !fir.ref<f64>
    %34 = fir.load %33 : !fir.ref<f64>
    %35 = arith.addf %30, %34 fastmath<contract> : f64
    %36 = fir.load %12#0 : !fir.ref<i32>
    %37 = fir.convert %36 : (i32) -> i64
    %38 = hlfir.designate %15#0 (%37)  : (!fir.box<!fir.array<?xf64>>, i64) -> !fir.ref<f64>
    hlfir.assign %35 to %38 : f64, !fir.ref<f64>
    %39 = fir.convert %c1 : (index) -> i32
    %40 = fir.load %12#0 : !fir.ref<i32>
    %41 = arith.addi %40, %39 overflow<nsw> : i32
    fir.result %41 : i32
  }
  fir.store %24 to %12#0 : !fir.ref<i32>
  omp.terminator
}

// -----// IR Dump After CanonicalizerPass (canonicalize) //----- //
module {
  func.func @kernel(%arg0: !fir.ref<i32> {jit.arg_type = 1 : ui32, jit.literal_val = 0 : i64}, %arg1: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 4 : i64}, %arg2: !fir.ref<!fir.array<?xf64>> {jit.arg_type = 1 : ui32}, %arg3: !fir.ref<!fir.array<?xf64>> {jit.arg_type = 1 : ui32}, %arg4: !fir.ref<!fir.array<?xf64>> {jit.arg_type = 1 : ui32}, %arg5: !fir.ref<f64> {jit.arg_type = 1 : ui32, jit.literal_val = 4619567317775286272 : i64}, %arg6: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 4 : i64}, %arg7: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 4 : i64}, %arg8: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 4 : i64}) {
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %0:2 = hlfir.declare %arg0 {uniq_name = "_QFFrun_benchmarkEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
    %1:2 = hlfir.declare %arg1 {uniq_name = "_QFFrun_benchmarkEn"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
    %2 = fir.shape %c4 : (index) -> !fir.shape<1>
    %3:2 = hlfir.declare %arg2(%2) {uniq_name = "_QFFrun_benchmarkEz"} : (!fir.ref<!fir.array<?xf64>>, !fir.shape<1>) -> (!fir.box<!fir.array<?xf64>>, !fir.ref<!fir.array<?xf64>>)
    %4 = fir.shape %c4 : (index) -> !fir.shape<1>
    %5:2 = hlfir.declare %arg3(%4) {uniq_name = "_QFFrun_benchmarkEx"} : (!fir.ref<!fir.array<?xf64>>, !fir.shape<1>) -> (!fir.box<!fir.array<?xf64>>, !fir.ref<!fir.array<?xf64>>)
    %6 = fir.shape %c4 : (index) -> !fir.shape<1>
    %7:2 = hlfir.declare %arg4(%6) {uniq_name = "_QFFrun_benchmarkEy"} : (!fir.ref<!fir.array<?xf64>>, !fir.shape<1>) -> (!fir.box<!fir.array<?xf64>>, !fir.ref<!fir.array<?xf64>>)
    %8:2 = hlfir.declare %arg5 {uniq_name = "_QFFrun_benchmarkEa"} : (!fir.ref<f64>) -> (!fir.ref<f64>, !fir.ref<f64>)
    %9 = fir.convert %c1 : (index) -> i32
    %10 = fir.do_loop %arg9 = %c1 to %c4 step %c1 iter_args(%arg10 = %9) -> (i32) {
      fir.store %arg10 to %0#0 : !fir.ref<i32>
      %11 = fir.load %8#0 : !fir.ref<f64>
      %12 = fir.load %0#0 : !fir.ref<i32>
      %13 = fir.convert %12 : (i32) -> i64
      %14 = hlfir.designate %5#0 (%13)  : (!fir.box<!fir.array<?xf64>>, i64) -> !fir.ref<f64>
      %15 = fir.load %14 : !fir.ref<f64>
      %16 = arith.mulf %11, %15 fastmath<contract> : f64
      %17 = fir.load %0#0 : !fir.ref<i32>
      %18 = fir.convert %17 : (i32) -> i64
      %19 = hlfir.designate %7#0 (%18)  : (!fir.box<!fir.array<?xf64>>, i64) -> !fir.ref<f64>
      %20 = fir.load %19 : !fir.ref<f64>
      %21 = arith.addf %16, %20 fastmath<contract> : f64
      %22 = fir.load %0#0 : !fir.ref<i32>
      %23 = fir.convert %22 : (i32) -> i64
      %24 = hlfir.designate %3#0 (%23)  : (!fir.box<!fir.array<?xf64>>, i64) -> !fir.ref<f64>
      hlfir.assign %21 to %24 : f64, !fir.ref<f64>
      %25 = fir.convert %c1 : (index) -> i32
      %26 = fir.load %0#0 : !fir.ref<i32>
      %27 = arith.addi %26, %25 overflow<nsw> : i32
      fir.result %27 : i32
    }
    fir.store %10 to %0#0 : !fir.ref<i32>
    omp.terminator
  }
}


// -----// IR Dump After SCCPPass (sccp) //----- //
module {
  func.func @kernel(%arg0: !fir.ref<i32> {jit.arg_type = 1 : ui32, jit.literal_val = 0 : i64}, %arg1: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 4 : i64}, %arg2: !fir.ref<!fir.array<?xf64>> {jit.arg_type = 1 : ui32}, %arg3: !fir.ref<!fir.array<?xf64>> {jit.arg_type = 1 : ui32}, %arg4: !fir.ref<!fir.array<?xf64>> {jit.arg_type = 1 : ui32}, %arg5: !fir.ref<f64> {jit.arg_type = 1 : ui32, jit.literal_val = 4619567317775286272 : i64}, %arg6: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 4 : i64}, %arg7: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 4 : i64}, %arg8: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 4 : i64}) {
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %0:2 = hlfir.declare %arg0 {uniq_name = "_QFFrun_benchmarkEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
    %1:2 = hlfir.declare %arg1 {uniq_name = "_QFFrun_benchmarkEn"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
    %2 = fir.shape %c4 : (index) -> !fir.shape<1>
    %3:2 = hlfir.declare %arg2(%2) {uniq_name = "_QFFrun_benchmarkEz"} : (!fir.ref<!fir.array<?xf64>>, !fir.shape<1>) -> (!fir.box<!fir.array<?xf64>>, !fir.ref<!fir.array<?xf64>>)
    %4 = fir.shape %c4 : (index) -> !fir.shape<1>
    %5:2 = hlfir.declare %arg3(%4) {uniq_name = "_QFFrun_benchmarkEx"} : (!fir.ref<!fir.array<?xf64>>, !fir.shape<1>) -> (!fir.box<!fir.array<?xf64>>, !fir.ref<!fir.array<?xf64>>)
    %6 = fir.shape %c4 : (index) -> !fir.shape<1>
    %7:2 = hlfir.declare %arg4(%6) {uniq_name = "_QFFrun_benchmarkEy"} : (!fir.ref<!fir.array<?xf64>>, !fir.shape<1>) -> (!fir.box<!fir.array<?xf64>>, !fir.ref<!fir.array<?xf64>>)
    %8:2 = hlfir.declare %arg5 {uniq_name = "_QFFrun_benchmarkEa"} : (!fir.ref<f64>) -> (!fir.ref<f64>, !fir.ref<f64>)
    %9 = fir.convert %c1 : (index) -> i32
    %10 = fir.do_loop %arg9 = %c1 to %c4 step %c1 iter_args(%arg10 = %9) -> (i32) {
      fir.store %arg10 to %0#0 : !fir.ref<i32>
      %11 = fir.load %8#0 : !fir.ref<f64>
      %12 = fir.load %0#0 : !fir.ref<i32>
      %13 = fir.convert %12 : (i32) -> i64
      %14 = hlfir.designate %5#0 (%13)  : (!fir.box<!fir.array<?xf64>>, i64) -> !fir.ref<f64>
      %15 = fir.load %14 : !fir.ref<f64>
      %16 = arith.mulf %11, %15 fastmath<contract> : f64
      %17 = fir.load %0#0 : !fir.ref<i32>
      %18 = fir.convert %17 : (i32) -> i64
      %19 = hlfir.designate %7#0 (%18)  : (!fir.box<!fir.array<?xf64>>, i64) -> !fir.ref<f64>
      %20 = fir.load %19 : !fir.ref<f64>
      %21 = arith.addf %16, %20 fastmath<contract> : f64
      %22 = fir.load %0#0 : !fir.ref<i32>
      %23 = fir.convert %22 : (i32) -> i64
      %24 = hlfir.designate %3#0 (%23)  : (!fir.box<!fir.array<?xf64>>, i64) -> !fir.ref<f64>
      hlfir.assign %21 to %24 : f64, !fir.ref<f64>
      %25 = fir.convert %c1 : (index) -> i32
      %26 = fir.load %0#0 : !fir.ref<i32>
      %27 = arith.addi %26, %25 overflow<nsw> : i32
      fir.result %27 : i32
    }
    fir.store %10 to %0#0 : !fir.ref<i32>
    omp.terminator
  }
}


// -----// IR Dump After CSEPass (cse) //----- //
module {
  func.func @kernel(%arg0: !fir.ref<i32> {jit.arg_type = 1 : ui32, jit.literal_val = 0 : i64}, %arg1: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 4 : i64}, %arg2: !fir.ref<!fir.array<?xf64>> {jit.arg_type = 1 : ui32}, %arg3: !fir.ref<!fir.array<?xf64>> {jit.arg_type = 1 : ui32}, %arg4: !fir.ref<!fir.array<?xf64>> {jit.arg_type = 1 : ui32}, %arg5: !fir.ref<f64> {jit.arg_type = 1 : ui32, jit.literal_val = 4619567317775286272 : i64}, %arg6: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 4 : i64}, %arg7: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 4 : i64}, %arg8: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 4 : i64}) {
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %0:2 = hlfir.declare %arg0 {uniq_name = "_QFFrun_benchmarkEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
    %1:2 = hlfir.declare %arg1 {uniq_name = "_QFFrun_benchmarkEn"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
    %2 = fir.shape %c4 : (index) -> !fir.shape<1>
    %3:2 = hlfir.declare %arg2(%2) {uniq_name = "_QFFrun_benchmarkEz"} : (!fir.ref<!fir.array<?xf64>>, !fir.shape<1>) -> (!fir.box<!fir.array<?xf64>>, !fir.ref<!fir.array<?xf64>>)
    %4:2 = hlfir.declare %arg3(%2) {uniq_name = "_QFFrun_benchmarkEx"} : (!fir.ref<!fir.array<?xf64>>, !fir.shape<1>) -> (!fir.box<!fir.array<?xf64>>, !fir.ref<!fir.array<?xf64>>)
    %5:2 = hlfir.declare %arg4(%2) {uniq_name = "_QFFrun_benchmarkEy"} : (!fir.ref<!fir.array<?xf64>>, !fir.shape<1>) -> (!fir.box<!fir.array<?xf64>>, !fir.ref<!fir.array<?xf64>>)
    %6:2 = hlfir.declare %arg5 {uniq_name = "_QFFrun_benchmarkEa"} : (!fir.ref<f64>) -> (!fir.ref<f64>, !fir.ref<f64>)
    %7 = fir.convert %c1 : (index) -> i32
    %8 = fir.do_loop %arg9 = %c1 to %c4 step %c1 iter_args(%arg10 = %7) -> (i32) {
      fir.store %arg10 to %0#0 : !fir.ref<i32>
      %9 = fir.load %6#0 : !fir.ref<f64>
      %10 = fir.load %0#0 : !fir.ref<i32>
      %11 = fir.convert %10 : (i32) -> i64
      %12 = hlfir.designate %4#0 (%11)  : (!fir.box<!fir.array<?xf64>>, i64) -> !fir.ref<f64>
      %13 = fir.load %12 : !fir.ref<f64>
      %14 = arith.mulf %9, %13 fastmath<contract> : f64
      %15 = hlfir.designate %5#0 (%11)  : (!fir.box<!fir.array<?xf64>>, i64) -> !fir.ref<f64>
      %16 = fir.load %15 : !fir.ref<f64>
      %17 = arith.addf %14, %16 fastmath<contract> : f64
      %18 = hlfir.designate %3#0 (%11)  : (!fir.box<!fir.array<?xf64>>, i64) -> !fir.ref<f64>
      hlfir.assign %17 to %18 : f64, !fir.ref<f64>
      %19 = fir.load %0#0 : !fir.ref<i32>
      %20 = arith.addi %19, %7 overflow<nsw> : i32
      fir.result %20 : i32
    }
    fir.store %8 to %0#0 : !fir.ref<i32>
    omp.terminator
  }
}


// -----// IR Dump After CanonicalizerPass (canonicalize) //----- //
module {
  func.func @kernel(%arg0: !fir.ref<i32> {jit.arg_type = 1 : ui32, jit.literal_val = 0 : i64}, %arg1: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 4 : i64}, %arg2: !fir.ref<!fir.array<?xf64>> {jit.arg_type = 1 : ui32}, %arg3: !fir.ref<!fir.array<?xf64>> {jit.arg_type = 1 : ui32}, %arg4: !fir.ref<!fir.array<?xf64>> {jit.arg_type = 1 : ui32}, %arg5: !fir.ref<f64> {jit.arg_type = 1 : ui32, jit.literal_val = 4619567317775286272 : i64}, %arg6: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 4 : i64}, %arg7: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 4 : i64}, %arg8: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 4 : i64}) {
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %0:2 = hlfir.declare %arg0 {uniq_name = "_QFFrun_benchmarkEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
    %1:2 = hlfir.declare %arg1 {uniq_name = "_QFFrun_benchmarkEn"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
    %2 = fir.shape %c4 : (index) -> !fir.shape<1>
    %3:2 = hlfir.declare %arg2(%2) {uniq_name = "_QFFrun_benchmarkEz"} : (!fir.ref<!fir.array<?xf64>>, !fir.shape<1>) -> (!fir.box<!fir.array<?xf64>>, !fir.ref<!fir.array<?xf64>>)
    %4:2 = hlfir.declare %arg3(%2) {uniq_name = "_QFFrun_benchmarkEx"} : (!fir.ref<!fir.array<?xf64>>, !fir.shape<1>) -> (!fir.box<!fir.array<?xf64>>, !fir.ref<!fir.array<?xf64>>)
    %5:2 = hlfir.declare %arg4(%2) {uniq_name = "_QFFrun_benchmarkEy"} : (!fir.ref<!fir.array<?xf64>>, !fir.shape<1>) -> (!fir.box<!fir.array<?xf64>>, !fir.ref<!fir.array<?xf64>>)
    %6:2 = hlfir.declare %arg5 {uniq_name = "_QFFrun_benchmarkEa"} : (!fir.ref<f64>) -> (!fir.ref<f64>, !fir.ref<f64>)
    %7 = fir.convert %c1 : (index) -> i32
    %8 = fir.do_loop %arg9 = %c1 to %c4 step %c1 iter_args(%arg10 = %7) -> (i32) {
      fir.store %arg10 to %0#0 : !fir.ref<i32>
      %9 = fir.load %6#0 : !fir.ref<f64>
      %10 = fir.load %0#0 : !fir.ref<i32>
      %11 = fir.convert %10 : (i32) -> i64
      %12 = hlfir.designate %4#0 (%11)  : (!fir.box<!fir.array<?xf64>>, i64) -> !fir.ref<f64>
      %13 = fir.load %12 : !fir.ref<f64>
      %14 = arith.mulf %9, %13 fastmath<contract> : f64
      %15 = hlfir.designate %5#0 (%11)  : (!fir.box<!fir.array<?xf64>>, i64) -> !fir.ref<f64>
      %16 = fir.load %15 : !fir.ref<f64>
      %17 = arith.addf %14, %16 fastmath<contract> : f64
      %18 = hlfir.designate %3#0 (%11)  : (!fir.box<!fir.array<?xf64>>, i64) -> !fir.ref<f64>
      hlfir.assign %17 to %18 : f64, !fir.ref<f64>
      %19 = fir.load %0#0 : !fir.ref<i32>
      %20 = arith.addi %19, %7 overflow<nsw> : i32
      fir.result %20 : i32
    }
    fir.store %8 to %0#0 : !fir.ref<i32>
    omp.terminator
  }
}


// -----// IR Dump After {anonymous}::ShapeInferPass (jforce-shape-infer) //----- //
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
  %8 = fir.do_loop %arg9 = %c1 to %c4 step %c1 iter_args(%arg10 = %7) -> (i32) {
    fir.store %arg10 to %0#0 : !fir.ref<i32>
    %9 = fir.load %6#0 : !fir.ref<f64>
    %10 = fir.load %0#0 : !fir.ref<i32>
    %11 = fir.convert %10 : (i32) -> i64
    %12 = hlfir.designate %4#0 (%11)  : (!fir.ref<!fir.array<4xf64>>, i64) -> !fir.ref<f64>
    %13 = fir.load %12 : !fir.ref<f64>
    %14 = arith.mulf %9, %13 fastmath<contract> : f64
    %15 = hlfir.designate %5#0 (%11)  : (!fir.ref<!fir.array<4xf64>>, i64) -> !fir.ref<f64>
    %16 = fir.load %15 : !fir.ref<f64>
    %17 = arith.addf %14, %16 fastmath<contract> : f64
    %18 = hlfir.designate %3#0 (%11)  : (!fir.ref<!fir.array<4xf64>>, i64) -> !fir.ref<f64>
    hlfir.assign %17 to %18 : f64, !fir.ref<f64>
    %19 = fir.load %0#0 : !fir.ref<i32>
    %20 = arith.addi %19, %7 overflow<nsw> : i32
    fir.result %20 : i32
  }
  fir.store %8 to %0#0 : !fir.ref<i32>
  omp.terminator
}

// -----// IR Dump After CanonicalizerPass (canonicalize) //----- //
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
    %8 = fir.do_loop %arg9 = %c1 to %c4 step %c1 iter_args(%arg10 = %7) -> (i32) {
      fir.store %arg10 to %0#0 : !fir.ref<i32>
      %9 = fir.load %6#0 : !fir.ref<f64>
      %10 = fir.load %0#0 : !fir.ref<i32>
      %11 = fir.convert %10 : (i32) -> i64
      %12 = hlfir.designate %4#0 (%11)  : (!fir.ref<!fir.array<4xf64>>, i64) -> !fir.ref<f64>
      %13 = fir.load %12 : !fir.ref<f64>
      %14 = arith.mulf %9, %13 fastmath<contract> : f64
      %15 = hlfir.designate %5#0 (%11)  : (!fir.ref<!fir.array<4xf64>>, i64) -> !fir.ref<f64>
      %16 = fir.load %15 : !fir.ref<f64>
      %17 = arith.addf %14, %16 fastmath<contract> : f64
      %18 = hlfir.designate %3#0 (%11)  : (!fir.ref<!fir.array<4xf64>>, i64) -> !fir.ref<f64>
      hlfir.assign %17 to %18 : f64, !fir.ref<f64>
      %19 = fir.load %0#0 : !fir.ref<i32>
      %20 = arith.addi %19, %7 overflow<nsw> : i32
      fir.result %20 : i32
    }
    fir.store %8 to %0#0 : !fir.ref<i32>
    omp.terminator
  }
}


// -----// IR Dump After ConvertHLFIRtoFIR (convert-hlfir-to-fir) //----- //
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
    %8 = fir.do_loop %arg9 = %c1 to %c4 step %c1 iter_args(%arg10 = %7) -> (i32) {
      fir.store %arg10 to %0 : !fir.ref<i32>
      %9 = fir.load %6 : !fir.ref<f64>
      %10 = fir.load %0 : !fir.ref<i32>
      %11 = fir.convert %10 : (i32) -> i64
      %12 = fir.array_coor %4(%2) %11 : (!fir.ref<!fir.array<4xf64>>, !fir.shape<1>, i64) -> !fir.ref<f64>
      %13 = fir.load %12 : !fir.ref<f64>
      %14 = arith.mulf %9, %13 fastmath<contract> : f64
      %15 = fir.array_coor %5(%2) %11 : (!fir.ref<!fir.array<4xf64>>, !fir.shape<1>, i64) -> !fir.ref<f64>
      %16 = fir.load %15 : !fir.ref<f64>
      %17 = arith.addf %14, %16 fastmath<contract> : f64
      %18 = fir.array_coor %3(%2) %11 : (!fir.ref<!fir.array<4xf64>>, !fir.shape<1>, i64) -> !fir.ref<f64>
      fir.store %17 to %18 : !fir.ref<f64>
      %19 = fir.load %0 : !fir.ref<i32>
      %20 = arith.addi %19, %7 overflow<nsw> : i32
      fir.result %20 : i32
    }
    fir.store %8 to %0 : !fir.ref<i32>
    omp.terminator
  }
}


// -----// IR Dump After FIRToMemRef (fir-to-memref) //----- //
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
  %8 = fir.do_loop %arg9 = %c1 to %c4 step %c1 iter_args(%arg10 = %7) -> (i32) {
    %10 = fir.convert %0 : (!fir.ref<i32>) -> memref<i32>
    memref.store %arg10, %10[] : memref<i32>
    %11 = fir.convert %6 : (!fir.ref<f64>) -> memref<f64>
    %12 = memref.load %11[] : memref<f64>
    %13 = fir.convert %0 : (!fir.ref<i32>) -> memref<i32>
    %14 = memref.load %13[] : memref<i32>
    %15 = fir.convert %14 : (i32) -> i64
    %16 = fir.convert %4 : (!fir.ref<!fir.array<4xf64>>) -> memref<4xf64>
    %c1_0 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %17 = arith.index_cast %15 : i64 to index
    %18 = arith.subi %17, %c1_0 : index
    %19 = arith.muli %18, %c1_0 : index
    %20 = arith.subi %c1_0, %c1_0 : index
    %21 = arith.addi %19, %20 : index
    %22 = memref.load %16[%21] : memref<4xf64>
    %23 = arith.mulf %12, %22 fastmath<contract> : f64
    %24 = fir.convert %5 : (!fir.ref<!fir.array<4xf64>>) -> memref<4xf64>
    %c1_1 = arith.constant 1 : index
    %c0_2 = arith.constant 0 : index
    %25 = arith.index_cast %15 : i64 to index
    %26 = arith.subi %25, %c1_1 : index
    %27 = arith.muli %26, %c1_1 : index
    %28 = arith.subi %c1_1, %c1_1 : index
    %29 = arith.addi %27, %28 : index
    %30 = memref.load %24[%29] : memref<4xf64>
    %31 = arith.addf %23, %30 fastmath<contract> : f64
    %32 = fir.convert %3 : (!fir.ref<!fir.array<4xf64>>) -> memref<4xf64>
    %c1_3 = arith.constant 1 : index
    %c0_4 = arith.constant 0 : index
    %33 = arith.index_cast %15 : i64 to index
    %34 = arith.subi %33, %c1_3 : index
    %35 = arith.muli %34, %c1_3 : index
    %36 = arith.subi %c1_3, %c1_3 : index
    %37 = arith.addi %35, %36 : index
    memref.store %31, %32[%37] : memref<4xf64>
    %38 = fir.convert %0 : (!fir.ref<i32>) -> memref<i32>
    %39 = memref.load %38[] : memref<i32>
    %40 = arith.addi %39, %7 overflow<nsw> : i32
    fir.result %40 : i32
  }
  %9 = fir.convert %0 : (!fir.ref<i32>) -> memref<i32>
  memref.store %8, %9[] : memref<i32>
  omp.terminator
}

// -----// IR Dump After CanonicalizerPass (canonicalize) //----- //
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
    %8 = fir.do_loop %arg9 = %c1 to %c4 step %c1 iter_args(%arg10 = %7) -> (i32) {
      %10 = fir.convert %0 : (!fir.ref<i32>) -> memref<i32>
      memref.store %arg10, %10[] : memref<i32>
      %11 = fir.convert %6 : (!fir.ref<f64>) -> memref<f64>
      %12 = memref.load %11[] : memref<f64>
      %13 = fir.convert %0 : (!fir.ref<i32>) -> memref<i32>
      %14 = memref.load %13[] : memref<i32>
      %15 = fir.convert %14 : (i32) -> i64
      %16 = fir.convert %4 : (!fir.ref<!fir.array<4xf64>>) -> memref<4xf64>
      %17 = arith.index_cast %15 : i64 to index
      %18 = arith.subi %17, %c1 : index
      %19 = memref.load %16[%18] : memref<4xf64>
      %20 = arith.mulf %12, %19 fastmath<contract> : f64
      %21 = fir.convert %5 : (!fir.ref<!fir.array<4xf64>>) -> memref<4xf64>
      %22 = arith.index_cast %15 : i64 to index
      %23 = arith.subi %22, %c1 : index
      %24 = memref.load %21[%23] : memref<4xf64>
      %25 = arith.addf %20, %24 fastmath<contract> : f64
      %26 = fir.convert %3 : (!fir.ref<!fir.array<4xf64>>) -> memref<4xf64>
      %27 = arith.index_cast %15 : i64 to index
      %28 = arith.subi %27, %c1 : index
      memref.store %25, %26[%28] : memref<4xf64>
      %29 = fir.convert %0 : (!fir.ref<i32>) -> memref<i32>
      %30 = memref.load %29[] : memref<i32>
      %31 = arith.addi %30, %7 overflow<nsw> : i32
      fir.result %31 : i32
    }
    %9 = fir.convert %0 : (!fir.ref<i32>) -> memref<i32>
    memref.store %8, %9[] : memref<i32>
    omp.terminator
  }
}


// -----// IR Dump After LoopInvariantCodeMotionPass (loop-invariant-code-motion) //----- //
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
    %10 = fir.convert %0 : (!fir.ref<i32>) -> memref<i32>
    %11 = fir.convert %4 : (!fir.ref<!fir.array<4xf64>>) -> memref<4xf64>
    %12 = fir.convert %5 : (!fir.ref<!fir.array<4xf64>>) -> memref<4xf64>
    %13 = fir.convert %3 : (!fir.ref<!fir.array<4xf64>>) -> memref<4xf64>
    %14 = fir.convert %0 : (!fir.ref<i32>) -> memref<i32>
    %15 = fir.do_loop %arg9 = %c1 to %c4 step %c1 iter_args(%arg10 = %7) -> (i32) {
      memref.store %arg10, %8[] : memref<i32>
      %17 = memref.load %9[] : memref<f64>
      %18 = memref.load %10[] : memref<i32>
      %19 = fir.convert %18 : (i32) -> i64
      %20 = arith.index_cast %19 : i64 to index
      %21 = arith.subi %20, %c1 : index
      %22 = memref.load %11[%21] : memref<4xf64>
      %23 = arith.mulf %17, %22 fastmath<contract> : f64
      %24 = arith.index_cast %19 : i64 to index
      %25 = arith.subi %24, %c1 : index
      %26 = memref.load %12[%25] : memref<4xf64>
      %27 = arith.addf %23, %26 fastmath<contract> : f64
      %28 = arith.index_cast %19 : i64 to index
      %29 = arith.subi %28, %c1 : index
      memref.store %27, %13[%29] : memref<4xf64>
      %30 = memref.load %14[] : memref<i32>
      %31 = arith.addi %30, %7 overflow<nsw> : i32
      fir.result %31 : i32
    }
    %16 = fir.convert %0 : (!fir.ref<i32>) -> memref<i32>
    memref.store %15, %16[] : memref<i32>
    omp.terminator
  }
}


// -----// IR Dump After CSEPass (cse) //----- //
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
    %13 = fir.do_loop %arg9 = %c1 to %c4 step %c1 iter_args(%arg10 = %7) -> (i32) {
      memref.store %arg10, %8[] : memref<i32>
      %14 = memref.load %9[] : memref<f64>
      %15 = memref.load %8[] : memref<i32>
      %16 = fir.convert %15 : (i32) -> i64
      %17 = arith.index_cast %16 : i64 to index
      %18 = arith.subi %17, %c1 : index
      %19 = memref.load %10[%18] : memref<4xf64>
      %20 = arith.mulf %14, %19 fastmath<contract> : f64
      %21 = memref.load %11[%18] : memref<4xf64>
      %22 = arith.addf %20, %21 fastmath<contract> : f64
      memref.store %22, %12[%18] : memref<4xf64>
      %23 = memref.load %8[] : memref<i32>
      %24 = arith.addi %23, %7 overflow<nsw> : i32
      fir.result %24 : i32
    }
    memref.store %13, %8[] : memref<i32>
    omp.terminator
  }
}


// -----// IR Dump After CanonicalizerPass (canonicalize) //----- //
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
    %13 = fir.do_loop %arg9 = %c1 to %c4 step %c1 iter_args(%arg10 = %7) -> (i32) {
      memref.store %arg10, %8[] : memref<i32>
      %14 = memref.load %9[] : memref<f64>
      %15 = memref.load %8[] : memref<i32>
      %16 = fir.convert %15 : (i32) -> i64
      %17 = arith.index_cast %16 : i64 to index
      %18 = arith.subi %17, %c1 : index
      %19 = memref.load %10[%18] : memref<4xf64>
      %20 = arith.mulf %14, %19 fastmath<contract> : f64
      %21 = memref.load %11[%18] : memref<4xf64>
      %22 = arith.addf %20, %21 fastmath<contract> : f64
      memref.store %22, %12[%18] : memref<4xf64>
      %23 = memref.load %8[] : memref<i32>
      %24 = arith.addi %23, %7 overflow<nsw> : i32
      fir.result %24 : i32
    }
    memref.store %13, %8[] : memref<i32>
    omp.terminator
  }
}


// -----// IR Dump After {anonymous}::CleanFIRLoopPass (jforce-clean-fir-loop) //----- //
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
  fir.do_loop %arg9 = %c1 to %c4 step %c1 {
    %14 = fir.convert %arg9 : (index) -> i32
    memref.store %14, %8[] : memref<i32>
    %15 = memref.load %9[] : memref<f64>
    %16 = memref.load %8[] : memref<i32>
    %17 = fir.convert %16 : (i32) -> i64
    %18 = arith.index_cast %17 : i64 to index
    %19 = arith.subi %18, %c1 : index
    %20 = memref.load %10[%19] : memref<4xf64>
    %21 = arith.mulf %15, %20 fastmath<contract> : f64
    %22 = memref.load %11[%19] : memref<4xf64>
    %23 = arith.addf %21, %22 fastmath<contract> : f64
    memref.store %23, %12[%19] : memref<4xf64>
    %24 = memref.load %8[] : memref<i32>
    %25 = arith.addi %24, %7 overflow<nsw> : i32
    memref.store %25, %8[] : memref<i32>
  }
  %13 = memref.load %8[] : memref<i32>
  memref.store %13, %8[] : memref<i32>
  omp.terminator
}

// -----// IR Dump After CanonicalizerPass (canonicalize) //----- //
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
    fir.do_loop %arg9 = %c1 to %c4 step %c1 {
      %14 = fir.convert %arg9 : (index) -> i32
      memref.store %14, %8[] : memref<i32>
      %15 = memref.load %9[] : memref<f64>
      %16 = memref.load %8[] : memref<i32>
      %17 = fir.convert %16 : (i32) -> i64
      %18 = arith.index_cast %17 : i64 to index
      %19 = arith.subi %18, %c1 : index
      %20 = memref.load %10[%19] : memref<4xf64>
      %21 = arith.mulf %15, %20 fastmath<contract> : f64
      %22 = memref.load %11[%19] : memref<4xf64>
      %23 = arith.addf %21, %22 fastmath<contract> : f64
      memref.store %23, %12[%19] : memref<4xf64>
      %24 = memref.load %8[] : memref<i32>
      %25 = arith.addi %24, %7 overflow<nsw> : i32
      memref.store %25, %8[] : memref<i32>
    }
    %13 = memref.load %8[] : memref<i32>
    memref.store %13, %8[] : memref<i32>
    omp.terminator
  }
}


// -----// IR Dump After {anonymous}::CleanFIROpsPass (jforce-clean-fir-op) //----- //
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
  fir.do_loop %arg9 = %c1 to %c4 step %c1 {
    %14 = arith.index_cast %arg9 : index to i32
    memref.store %14, %8[] : memref<i32>
    %15 = memref.load %9[] : memref<f64>
    %16 = memref.load %8[] : memref<i32>
    %17 = arith.extsi %16 : i32 to i64
    %18 = arith.index_cast %17 : i64 to index
    %19 = arith.subi %18, %c1 : index
    %20 = memref.load %10[%19] : memref<4xf64>
    %21 = arith.mulf %15, %20 fastmath<contract> : f64
    %22 = memref.load %11[%19] : memref<4xf64>
    %23 = arith.addf %21, %22 fastmath<contract> : f64
    memref.store %23, %12[%19] : memref<4xf64>
    %24 = memref.load %8[] : memref<i32>
    %25 = arith.addi %24, %7 overflow<nsw> : i32
    memref.store %25, %8[] : memref<i32>
  }
  %13 = memref.load %8[] : memref<i32>
  memref.store %13, %8[] : memref<i32>
  omp.terminator
}

// -----// IR Dump After AffineDialectPromotion (promote-to-affine) //----- //
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
  affine.for %arg9 = %c1 to affine_map<()[s0] -> (s0 + 1)>()[%c4] {
    %14 = arith.index_cast %arg9 : index to i32
    memref.store %14, %8[] : memref<i32>
    %15 = memref.load %9[] : memref<f64>
    %16 = memref.load %8[] : memref<i32>
    %17 = arith.extsi %16 : i32 to i64
    %18 = arith.index_cast %17 : i64 to index
    %19 = arith.subi %18, %c1 : index
    %20 = memref.load %10[%19] : memref<4xf64>
    %21 = arith.mulf %15, %20 fastmath<contract> : f64
    %22 = memref.load %11[%19] : memref<4xf64>
    %23 = arith.addf %21, %22 fastmath<contract> : f64
    memref.store %23, %12[%19] : memref<4xf64>
    %24 = memref.load %8[] : memref<i32>
    %25 = arith.addi %24, %7 overflow<nsw> : i32
    memref.store %25, %8[] : memref<i32>
  }
  %13 = memref.load %8[] : memref<i32>
  memref.store %13, %8[] : memref<i32>
  omp.terminator
}

// -----// IR Dump After AffineLoopNormalize (affine-loop-normalize) //----- //
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
  affine.for %arg9 = 0 to 4 {
    %14 = affine.apply affine_map<(d0) -> (d0 + 1)>(%arg9)
    %15 = arith.index_cast %14 : index to i32
    memref.store %15, %8[] : memref<i32>
    %16 = memref.load %9[] : memref<f64>
    %17 = memref.load %8[] : memref<i32>
    %18 = arith.extsi %17 : i32 to i64
    %19 = arith.index_cast %18 : i64 to index
    %20 = arith.subi %19, %c1 : index
    %21 = memref.load %10[%20] : memref<4xf64>
    %22 = arith.mulf %16, %21 fastmath<contract> : f64
    %23 = memref.load %11[%20] : memref<4xf64>
    %24 = arith.addf %22, %23 fastmath<contract> : f64
    memref.store %24, %12[%20] : memref<4xf64>
    %25 = memref.load %8[] : memref<i32>
    %26 = arith.addi %25, %7 overflow<nsw> : i32
    memref.store %26, %8[] : memref<i32>
  }
  %13 = memref.load %8[] : memref<i32>
  memref.store %13, %8[] : memref<i32>
  omp.terminator
}

// -----// IR Dump After CanonicalizerPass (canonicalize) //----- //
#map = affine_map<(d0) -> (d0 + 1)>
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
    affine.for %arg9 = 0 to 4 {
      %14 = affine.apply #map(%arg9)
      %15 = arith.index_cast %14 : index to i32
      memref.store %15, %8[] : memref<i32>
      %16 = memref.load %9[] : memref<f64>
      %17 = memref.load %8[] : memref<i32>
      %18 = arith.index_cast %17 : i32 to index
      %19 = arith.subi %18, %c1 : index
      %20 = memref.load %10[%19] : memref<4xf64>
      %21 = arith.mulf %16, %20 fastmath<contract> : f64
      %22 = memref.load %11[%19] : memref<4xf64>
      %23 = arith.addf %21, %22 fastmath<contract> : f64
      memref.store %23, %12[%19] : memref<4xf64>
      %24 = memref.load %8[] : memref<i32>
      %25 = arith.addi %24, %7 overflow<nsw> : i32
      memref.store %25, %8[] : memref<i32>
    }
    %13 = memref.load %8[] : memref<i32>
    memref.store %13, %8[] : memref<i32>
    omp.terminator
  }
}


[DEBUG]compare reading and storeOp
[DEBUG] memref.store %13, %8[] : memref<i32> 
[DEBUG] %13 = memref.load %8[] : memref<i32> 
[DEBUG]compare reading and storeOp
[DEBUG] memref.store %24, %8[] : memref<i32> 
[DEBUG] %24 = arith.addi %23, %7 overflow<nsw> : i32 
[DEBUG]Unexpected readOp: 
[DEBUG] %24 = arith.addi %23, %7 overflow<nsw> : i32 
[DEBUG]compare reading and storeOp
[DEBUG] memref.store %22, %12[%18] : memref<4xf64> 
[DEBUG] %22 = arith.addf %20, %21 fastmath<contract> : f64 
[DEBUG]Unexpected readOp: 
[DEBUG] %22 = arith.addf %20, %21 fastmath<contract> : f64 
[DEBUG]Dependency chain relies on LoadOp!
[DEBUG]compare reading and storeOp
[DEBUG] memref.store %15, %8[] : memref<i32> 
[DEBUG] %15 = arith.index_cast %14 : index to i32 
[DEBUG]Unexpected readOp: 
[DEBUG] %15 = arith.index_cast %14 : index to i32 
[DEBUG]compare reading and storeOp
[DEBUG] memref.store %22, %8[] : memref<i32> 
[DEBUG] %22 = arith.addi %15, %7 overflow<nsw> : i32 
[DEBUG]Unexpected readOp: 
[DEBUG] %22 = arith.addi %15, %7 overflow<nsw> : i32 
[DEBUG]compare reading and storeOp
[DEBUG] memref.store %21, %8[] : memref<i32> 
[DEBUG] %21 = arith.addi %20, %7 overflow<nsw> : i32 
[DEBUG]Unexpected readOp: 
[DEBUG] %21 = arith.addi %20, %7 overflow<nsw> : i32 
[DEBUG]compare reading and storeOp
[DEBUG] memref.store %22, %12[%18] : memref<4xf64> 
[DEBUG] %22 = arith.addf %20, %21 fastmath<contract> : f64 
[DEBUG]Unexpected readOp: 
[DEBUG] %22 = arith.addf %20, %21 fastmath<contract> : f64 
[DEBUG]Dependency chain relies on LoadOp!
[DEBUG]compare reading and storeOp
[DEBUG] memref.store %14, %8[] : memref<i32> 
[DEBUG] %14 = arith.addi %7, %c4_i32 overflow<nsw> : i32 
[DEBUG]Unexpected readOp: 
[DEBUG] %14 = arith.addi %7, %c4_i32 overflow<nsw> : i32 
[DEBUG]compare reading and storeOp
[DEBUG] memref.store %22, %12[%18] : memref<4xf64> 
[DEBUG] %22 = arith.addf %20, %21 fastmath<contract> : f64 
[DEBUG]Unexpected readOp: 
[DEBUG] %22 = arith.addf %20, %21 fastmath<contract> : f64 
[DEBUG]Dependency chain relies on LoadOp!
// -----// IR Dump After {anonymous}::OptimizeMemOpsPass (jforce-optimize-mem-ops) //----- //
func.func @kernel(%arg0: !fir.ref<i32> {jit.arg_type = 1 : ui32, jit.literal_val = 0 : i64}, %arg1: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 4 : i64}, %arg2: !fir.ref<!fir.array<4xf64>> {jit.arg_type = 1 : ui32}, %arg3: !fir.ref<!fir.array<4xf64>> {jit.arg_type = 1 : ui32}, %arg4: !fir.ref<!fir.array<4xf64>> {jit.arg_type = 1 : ui32}, %arg5: !fir.ref<f64> {jit.arg_type = 1 : ui32, jit.literal_val = 4619567317775286272 : i64}, %arg6: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 4 : i64}, %arg7: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 4 : i64}, %arg8: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 4 : i64}) {
  %c4_i32 = arith.constant 4 : i32
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
  %13 = memref.load %9[] : memref<f64>
  affine.for %arg9 = 0 to 4 {
    %15 = affine.apply affine_map<(d0) -> (d0 + 1)>(%arg9)
    %16 = arith.index_cast %15 : index to i32
    %17 = arith.index_cast %16 : i32 to index
    %18 = arith.subi %17, %c1 : index
    %19 = memref.load %10[%18] : memref<4xf64>
    %20 = arith.mulf %13, %19 fastmath<contract> : f64
    %21 = memref.load %11[%18] : memref<4xf64>
    %22 = arith.addf %20, %21 fastmath<contract> : f64
    memref.store %22, %12[%18] : memref<4xf64>
  }
  %14 = arith.addi %7, %c4_i32 overflow<nsw> : i32
  memref.store %14, %8[] : memref<i32>
  omp.terminator
}

// -----// IR Dump After CanonicalizerPass (canonicalize) //----- //
#map = affine_map<(d0) -> (d0 + 1)>
module {
  func.func @kernel(%arg0: !fir.ref<i32> {jit.arg_type = 1 : ui32, jit.literal_val = 0 : i64}, %arg1: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 4 : i64}, %arg2: !fir.ref<!fir.array<4xf64>> {jit.arg_type = 1 : ui32}, %arg3: !fir.ref<!fir.array<4xf64>> {jit.arg_type = 1 : ui32}, %arg4: !fir.ref<!fir.array<4xf64>> {jit.arg_type = 1 : ui32}, %arg5: !fir.ref<f64> {jit.arg_type = 1 : ui32, jit.literal_val = 4619567317775286272 : i64}, %arg6: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 4 : i64}, %arg7: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 4 : i64}, %arg8: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 4 : i64}) {
    %c4_i32 = arith.constant 4 : i32
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
    %13 = memref.load %9[] : memref<f64>
    affine.for %arg9 = 0 to 4 {
      %15 = affine.apply #map(%arg9)
      %16 = arith.index_cast %15 : index to i32
      %17 = arith.index_cast %16 : i32 to index
      %18 = arith.subi %17, %c1 : index
      %19 = memref.load %10[%18] : memref<4xf64>
      %20 = arith.mulf %13, %19 fastmath<contract> : f64
      %21 = memref.load %11[%18] : memref<4xf64>
      %22 = arith.addf %20, %21 fastmath<contract> : f64
      memref.store %22, %12[%18] : memref<4xf64>
    }
    %14 = arith.addi %7, %c4_i32 overflow<nsw> : i32
    memref.store %14, %8[] : memref<i32>
    omp.terminator
  }
}


func.func @kernel(%arg0: !fir.ref<i32> {jit.arg_type = 1 : ui32, jit.literal_val = 0 : i64}, %arg1: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 4 : i64}, %arg2: !fir.ref<!fir.array<4xf64>> {jit.arg_type = 1 : ui32}, %arg3: !fir.ref<!fir.array<4xf64>> {jit.arg_type = 1 : ui32}, %arg4: !fir.ref<!fir.array<4xf64>> {jit.arg_type = 1 : ui32}, %arg5: !fir.ref<f64> {jit.arg_type = 1 : ui32, jit.literal_val = 4619567317775286272 : i64}, %arg6: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 4 : i64}, %arg7: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 4 : i64}, %arg8: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 4 : i64}) {
  %c4_i32 = arith.constant 4 : i32
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
  %13 = memref.load %9[] : memref<f64>
  affine.for %arg9 = 0 to 4 {
    %15 = affine.apply affine_map<(d0) -> (d0 + 1)>(%arg9)
    %16 = arith.index_cast %15 : index to i32
    %17 = arith.index_cast %16 : i32 to index
    %18 = arith.subi %17, %c1 : index
    %19 = memref.load %10[%18] : memref<4xf64>
    %20 = arith.mulf %13, %19 fastmath<contract> : f64
    %21 = memref.load %11[%18] : memref<4xf64>
    %22 = arith.addf %20, %21 fastmath<contract> : f64
    memref.store %22, %12[%18] : memref<4xf64>
  }
  %14 = arith.addi %7, %c4_i32 overflow<nsw> : i32
  memref.store %14, %8[] : memref<i32>
  omp.terminator
}
// -----// IR Dump After {anonymous}::LoopSinkingPass (jforce-loop-sink) //----- //
func.func @kernel(%arg0: !fir.ref<i32> {jit.arg_type = 1 : ui32, jit.literal_val = 0 : i64}, %arg1: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 4 : i64}, %arg2: !fir.ref<!fir.array<4xf64>> {jit.arg_type = 1 : ui32}, %arg3: !fir.ref<!fir.array<4xf64>> {jit.arg_type = 1 : ui32}, %arg4: !fir.ref<!fir.array<4xf64>> {jit.arg_type = 1 : ui32}, %arg5: !fir.ref<f64> {jit.arg_type = 1 : ui32, jit.literal_val = 4619567317775286272 : i64}, %arg6: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 4 : i64}, %arg7: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 4 : i64}, %arg8: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 4 : i64}) {
  %c4_i32 = arith.constant 4 : i32
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
  %13 = memref.load %9[] : memref<f64>
  affine.for %arg9 = 0 to 4 {
    %15 = affine.apply affine_map<(d0) -> (d0 + 1)>(%arg9)
    %16 = arith.index_cast %15 : index to i32
    %17 = arith.index_cast %16 : i32 to index
    %18 = arith.subi %17, %c1 : index
    %19 = memref.load %10[%18] : memref<4xf64>
    %20 = arith.mulf %13, %19 fastmath<contract> : f64
    %21 = memref.load %11[%18] : memref<4xf64>
    %22 = arith.addf %20, %21 fastmath<contract> : f64
    memref.store %22, %12[%18] : memref<4xf64>
  }
  %14 = arith.addi %7, %c4_i32 overflow<nsw> : i32
  memref.store %14, %8[] : memref<i32>
  omp.terminator
}

// -----// IR Dump After {anonymous}::AffineCFGPass (enzyme-affinecfg) //----- //
module {
  func.func @kernel(%arg0: !fir.ref<i32> {jit.arg_type = 1 : ui32, jit.literal_val = 0 : i64}, %arg1: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 4 : i64}, %arg2: !fir.ref<!fir.array<4xf64>> {jit.arg_type = 1 : ui32}, %arg3: !fir.ref<!fir.array<4xf64>> {jit.arg_type = 1 : ui32}, %arg4: !fir.ref<!fir.array<4xf64>> {jit.arg_type = 1 : ui32}, %arg5: !fir.ref<f64> {jit.arg_type = 1 : ui32, jit.literal_val = 4619567317775286272 : i64}, %arg6: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 4 : i64}, %arg7: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 4 : i64}, %arg8: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 4 : i64}) {
    %c4_i32 = arith.constant 4 : i32
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
    %13 = affine.load %9[] : memref<f64>
    affine.parallel (%arg9) = (0) to (4) {
      %15 = affine.load %10[%arg9] : memref<4xf64>
      %16 = arith.mulf %13, %15 fastmath<contract> : f64
      %17 = affine.load %11[%arg9] : memref<4xf64>
      %18 = arith.addf %16, %17 fastmath<contract> : f64
      affine.store %18, %12[%arg9] : memref<4xf64>
    }
    %14 = arith.addi %7, %c4_i32 overflow<nsw> : i32
    affine.store %14, %8[] : memref<i32>
    omp.terminator
  }
}


module {
  func.func @outlined_affinefor_104018181802624(%arg0: memref<4xf64>, %arg1: memref<f64>, %arg2: memref<4xf64>, %arg3: memref<4xf64>) {
    %0 = affine.load %arg1[] : memref<f64>
    affine.parallel (%arg4) = (0) to (4) {
      %1 = affine.load %arg0[%arg4] : memref<4xf64>
      %2 = arith.mulf %0, %1 fastmath<contract> : f64
      %3 = affine.load %arg2[%arg4] : memref<4xf64>
      %4 = arith.addf %2, %3 fastmath<contract> : f64
      affine.store %4, %arg3[%arg4] : memref<4xf64>
    }
    return
  }
  func.func @kernel(%arg0: !fir.ref<i32> {jit.arg_type = 1 : ui32, jit.literal_val = 0 : i64}, %arg1: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 4 : i64}, %arg2: !fir.ref<!fir.array<4xf64>> {jit.arg_type = 1 : ui32}, %arg3: !fir.ref<!fir.array<4xf64>> {jit.arg_type = 1 : ui32}, %arg4: !fir.ref<!fir.array<4xf64>> {jit.arg_type = 1 : ui32}, %arg5: !fir.ref<f64> {jit.arg_type = 1 : ui32, jit.literal_val = 4619567317775286272 : i64}, %arg6: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 4 : i64}, %arg7: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 4 : i64}, %arg8: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 4 : i64}) {
    %c4_i32 = arith.constant 4 : i32
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
    %13 = affine.load %9[] : memref<f64>
    %alloca = memref.alloca() : memref<f64>
    memref.store %13, %alloca[] : memref<f64>
    call @outlined_affinefor_104018181802624(%10, %alloca, %11, %12) : (memref<4xf64>, memref<f64>, memref<4xf64>, memref<4xf64>) -> ()
    %14 = arith.addi %7, %c4_i32 overflow<nsw> : i32
    affine.store %14, %8[] : memref<i32>
    omp.terminator
  }
}
// -----// IR Dump After {anonymous}::OutlineAffinePass (jforce-outline-affine) //----- //
module {
  func.func @outlined_affinefor_104018181802624(%arg0: memref<4xf64>, %arg1: memref<f64>, %arg2: memref<4xf64>, %arg3: memref<4xf64>) {
    %0 = affine.load %arg1[] : memref<f64>
    affine.parallel (%arg4) = (0) to (4) {
      %1 = affine.load %arg0[%arg4] : memref<4xf64>
      %2 = arith.mulf %0, %1 fastmath<contract> : f64
      %3 = affine.load %arg2[%arg4] : memref<4xf64>
      %4 = arith.addf %2, %3 fastmath<contract> : f64
      affine.store %4, %arg3[%arg4] : memref<4xf64>
    }
    return
  }
  func.func @kernel(%arg0: !fir.ref<i32> {jit.arg_type = 1 : ui32, jit.literal_val = 0 : i64}, %arg1: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 4 : i64}, %arg2: !fir.ref<!fir.array<4xf64>> {jit.arg_type = 1 : ui32}, %arg3: !fir.ref<!fir.array<4xf64>> {jit.arg_type = 1 : ui32}, %arg4: !fir.ref<!fir.array<4xf64>> {jit.arg_type = 1 : ui32}, %arg5: !fir.ref<f64> {jit.arg_type = 1 : ui32, jit.literal_val = 4619567317775286272 : i64}, %arg6: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 4 : i64}, %arg7: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 4 : i64}, %arg8: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 4 : i64}) {
    %c4_i32 = arith.constant 4 : i32
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
    %13 = affine.load %9[] : memref<f64>
    %alloca = memref.alloca() : memref<f64>
    memref.store %13, %alloca[] : memref<f64>
    call @outlined_affinefor_104018181802624(%10, %alloca, %11, %12) : (memref<4xf64>, memref<f64>, memref<4xf64>, memref<4xf64>) -> ()
    %14 = arith.addi %7, %c4_i32 overflow<nsw> : i32
    affine.store %14, %8[] : memref<i32>
    omp.terminator
  }
}


// -----// IR Dump After {anonymous}::AffineToStableHLORaisingPass (enzyme-affine-to-stablehlo) //----- //
func.func @kernel(%arg0: !fir.ref<i32> {jit.arg_type = 1 : ui32, jit.literal_val = 0 : i64}, %arg1: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 4 : i64}, %arg2: !fir.ref<!fir.array<4xf64>> {jit.arg_type = 1 : ui32}, %arg3: !fir.ref<!fir.array<4xf64>> {jit.arg_type = 1 : ui32}, %arg4: !fir.ref<!fir.array<4xf64>> {jit.arg_type = 1 : ui32}, %arg5: !fir.ref<f64> {jit.arg_type = 1 : ui32, jit.literal_val = 4619567317775286272 : i64}, %arg6: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 4 : i64}, %arg7: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 4 : i64}, %arg8: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 4 : i64}) {
  %c4_i32 = arith.constant 4 : i32
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
  %13 = affine.load %9[] : memref<f64>
  %alloca = memref.alloca() : memref<f64>
  memref.store %13, %alloca[] : memref<f64>
  call @outlined_affinefor_104018181802624(%10, %alloca, %11, %12) : (memref<4xf64>, memref<f64>, memref<4xf64>, memref<4xf64>) -> ()
  %14 = arith.addi %7, %c4_i32 overflow<nsw> : i32
  affine.store %14, %8[] : memref<i32>
  omp.terminator
}

// -----// IR Dump After {anonymous}::AffineToStableHLORaisingPass (enzyme-affine-to-stablehlo) //----- //
func.func @outlined_affinefor_104018181802624(%arg0: memref<4xf64>, %arg1: memref<f64>, %arg2: memref<4xf64>, %arg3: memref<4xf64>) {
  %0 = affine.load %arg1[] : memref<f64>
  affine.parallel (%arg4) = (0) to (4) {
    %1 = affine.load %arg0[%arg4] : memref<4xf64>
    %2 = arith.mulf %0, %1 fastmath<contract> : f64
    %3 = affine.load %arg2[%arg4] : memref<4xf64>
    %4 = arith.addf %2, %3 fastmath<contract> : f64
    affine.store %4, %arg3[%arg4] : memref<4xf64>
  }
  return
}

// -----// IR Dump After CanonicalizerPass (canonicalize) //----- //
module {
  func.func @outlined_affinefor_104018181802624(%arg0: memref<4xf64>, %arg1: memref<f64>, %arg2: memref<4xf64>, %arg3: memref<4xf64>) {
    %0 = affine.load %arg1[] : memref<f64>
    affine.parallel (%arg4) = (0) to (4) {
      %1 = affine.load %arg0[%arg4] : memref<4xf64>
      %2 = arith.mulf %0, %1 fastmath<contract> : f64
      %3 = affine.load %arg2[%arg4] : memref<4xf64>
      %4 = arith.addf %2, %3 fastmath<contract> : f64
      affine.store %4, %arg3[%arg4] : memref<4xf64>
    }
    return
  }
  func.func @kernel(%arg0: !fir.ref<i32> {jit.arg_type = 1 : ui32, jit.literal_val = 0 : i64}, %arg1: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 4 : i64}, %arg2: !fir.ref<!fir.array<4xf64>> {jit.arg_type = 1 : ui32}, %arg3: !fir.ref<!fir.array<4xf64>> {jit.arg_type = 1 : ui32}, %arg4: !fir.ref<!fir.array<4xf64>> {jit.arg_type = 1 : ui32}, %arg5: !fir.ref<f64> {jit.arg_type = 1 : ui32, jit.literal_val = 4619567317775286272 : i64}, %arg6: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 4 : i64}, %arg7: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 4 : i64}, %arg8: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 4 : i64}) {
    %c4_i32 = arith.constant 4 : i32
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
    %13 = affine.load %9[] : memref<f64>
    %alloca = memref.alloca() : memref<f64>
    memref.store %13, %alloca[] : memref<f64>
    call @outlined_affinefor_104018181802624(%10, %alloca, %11, %12) : (memref<4xf64>, memref<f64>, memref<4xf64>, memref<4xf64>) -> ()
    %14 = arith.addi %7, %c4_i32 overflow<nsw> : i32
    affine.store %14, %8[] : memref<i32>
    omp.terminator
  }
  func.func private @outlined_affinefor_104018181802624_raised(%arg0: tensor<4xf64>, %arg1: tensor<f64>, %arg2: tensor<4xf64>, %arg3: tensor<4xf64>) -> (tensor<4xf64>, tensor<f64>, tensor<4xf64>, tensor<4xf64>) {
    %c = stablehlo.constant dense<0> : tensor<i64>
    %0 = stablehlo.reshape %arg1 : (tensor<f64>) -> tensor<f64>
    %1 = stablehlo.reshape %arg0 : (tensor<4xf64>) -> tensor<4xf64>
    %2 = stablehlo.broadcast_in_dim %0, dims = [] : (tensor<f64>) -> tensor<4xf64>
    %3 = arith.mulf %2, %1 fastmath<contract> : tensor<4xf64>
    %4 = stablehlo.reshape %arg2 : (tensor<4xf64>) -> tensor<4xf64>
    %5 = arith.addf %3, %4 fastmath<contract> : tensor<4xf64>
    %6 = stablehlo.broadcast_in_dim %5, dims = [0] : (tensor<4xf64>) -> tensor<4xf64>
    %7 = stablehlo.dynamic_update_slice %arg3, %6, %c : (tensor<4xf64>, tensor<4xf64>, tensor<i64>) -> tensor<4xf64>
    return %arg0, %arg1, %arg2, %7 : tensor<4xf64>, tensor<f64>, tensor<4xf64>, tensor<4xf64>
  }
}


