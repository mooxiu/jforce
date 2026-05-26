// ./build/src/tool/jforce-opt ./tests/axpy-dl-stride.mlir \
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
//    --enzyme-affine-to-stablehlo --canonicalize \
//    --jforce-switch-outline-function \
//    --jforce-translate --inline \
//    --stablehlo-aggressive-simplification --jforce-trim-args \
//    --mlir-print-ir-after-all 2> ./tests/axpy-dl-stride-lowering.mlir


// -----// IR Dump After (anonymous namespace)::AnnotatePass (jforce-annotate) //----- //
func.func @kernel(%arg0: !fir.ref<i32> {jit.arg_type = 1 : ui32, jit.literal_val = 0 : i64}, %arg1: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg2: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 3 : i64}, %arg3: !fir.ref<!fir.array<?xf64>> {jit.arg_type = 1 : ui32}, %arg4: !fir.ref<!fir.array<?xf64>> {jit.arg_type = 1 : ui32}, %arg5: !fir.ref<!fir.array<?xf64>> {jit.arg_type = 1 : ui32}, %arg6: !fir.ref<f64> {jit.arg_type = 1 : ui32, jit.literal_val = 4619567317775286272 : i64}, %arg7: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg8: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg9: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}) {
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

// -----// IR Dump After (anonymous namespace)::PropagateConstantsPass (jforce-propagate-constants) //----- //
func.func @kernel(%arg0: !fir.ref<i32> {jit.arg_type = 1 : ui32, jit.literal_val = 0 : i64}, %arg1: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg2: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 3 : i64}, %arg3: !fir.ref<!fir.array<?xf64>> {jit.arg_type = 1 : ui32}, %arg4: !fir.ref<!fir.array<?xf64>> {jit.arg_type = 1 : ui32}, %arg5: !fir.ref<!fir.array<?xf64>> {jit.arg_type = 1 : ui32}, %arg6: !fir.ref<f64> {jit.arg_type = 1 : ui32, jit.literal_val = 4619567317775286272 : i64}, %arg7: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg8: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg9: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}) {
  %c1024_i32 = arith.constant 1024 : i32
  %c1024_i32_0 = arith.constant 1024 : i32
  %c1024_i32_1 = arith.constant 1024 : i32
  %0 = fir.convert %c1024_i32_1 : (i32) -> i64
  %1 = fir.convert %c1024_i32_0 : (i32) -> i64
  %2 = fir.convert %c1024_i32 : (i32) -> i64
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
  %14:2 = hlfir.declare %arg2 {uniq_name = "_QFFrun_benchmarkEstride"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %15 = fir.shape %11 : (index) -> !fir.shape<1>
  %16:2 = hlfir.declare %arg3(%15) {uniq_name = "_QFFrun_benchmarkEz"} : (!fir.ref<!fir.array<?xf64>>, !fir.shape<1>) -> (!fir.box<!fir.array<?xf64>>, !fir.ref<!fir.array<?xf64>>)
  %17 = fir.shape %10 : (index) -> !fir.shape<1>
  %18:2 = hlfir.declare %arg4(%17) {uniq_name = "_QFFrun_benchmarkEx"} : (!fir.ref<!fir.array<?xf64>>, !fir.shape<1>) -> (!fir.box<!fir.array<?xf64>>, !fir.ref<!fir.array<?xf64>>)
  %19 = fir.shape %9 : (index) -> !fir.shape<1>
  %20:2 = hlfir.declare %arg5(%19) {uniq_name = "_QFFrun_benchmarkEy"} : (!fir.ref<!fir.array<?xf64>>, !fir.shape<1>) -> (!fir.box<!fir.array<?xf64>>, !fir.ref<!fir.array<?xf64>>)
  %21:2 = hlfir.declare %arg6 {uniq_name = "_QFFrun_benchmarkEa"} : (!fir.ref<f64>) -> (!fir.ref<f64>, !fir.ref<f64>)
  %c1_i32 = arith.constant 1 : i32
  %22 = fir.convert %c1_i32 : (i32) -> index
  %c1024_i32_4 = arith.constant 1024 : i32
  %23 = fir.convert %c1024_i32_4 : (i32) -> index
  %c3_i32 = arith.constant 3 : i32
  %24 = fir.convert %c3_i32 : (i32) -> index
  %25 = fir.convert %22 : (index) -> i32
  %26 = fir.do_loop %arg10 = %22 to %23 step %24 iter_args(%arg11 = %25) -> (i32) {
    fir.store %arg11 to %12#0 : !fir.ref<i32>
    %27 = fir.load %21#0 : !fir.ref<f64>
    %28 = fir.load %12#0 : !fir.ref<i32>
    %29 = fir.convert %28 : (i32) -> i64
    %30 = hlfir.designate %18#0 (%29)  : (!fir.box<!fir.array<?xf64>>, i64) -> !fir.ref<f64>
    %31 = fir.load %30 : !fir.ref<f64>
    %32 = arith.mulf %27, %31 fastmath<contract> : f64
    %33 = fir.load %12#0 : !fir.ref<i32>
    %34 = fir.convert %33 : (i32) -> i64
    %35 = hlfir.designate %20#0 (%34)  : (!fir.box<!fir.array<?xf64>>, i64) -> !fir.ref<f64>
    %36 = fir.load %35 : !fir.ref<f64>
    %37 = arith.addf %32, %36 fastmath<contract> : f64
    %38 = fir.load %12#0 : !fir.ref<i32>
    %39 = fir.convert %38 : (i32) -> i64
    %40 = hlfir.designate %16#0 (%39)  : (!fir.box<!fir.array<?xf64>>, i64) -> !fir.ref<f64>
    hlfir.assign %37 to %40 : f64, !fir.ref<f64>
    %41 = fir.convert %24 : (index) -> i32
    %42 = fir.load %12#0 : !fir.ref<i32>
    %43 = arith.addi %42, %41 overflow<nsw> : i32
    fir.result %43 : i32
  }
  fir.store %26 to %12#0 : !fir.ref<i32>
  omp.terminator
}

// -----// IR Dump After CanonicalizerPass (canonicalize) //----- //
module {
  func.func @kernel(%arg0: !fir.ref<i32> {jit.arg_type = 1 : ui32, jit.literal_val = 0 : i64}, %arg1: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg2: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 3 : i64}, %arg3: !fir.ref<!fir.array<?xf64>> {jit.arg_type = 1 : ui32}, %arg4: !fir.ref<!fir.array<?xf64>> {jit.arg_type = 1 : ui32}, %arg5: !fir.ref<!fir.array<?xf64>> {jit.arg_type = 1 : ui32}, %arg6: !fir.ref<f64> {jit.arg_type = 1 : ui32, jit.literal_val = 4619567317775286272 : i64}, %arg7: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg8: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg9: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}) {
    %c3 = arith.constant 3 : index
    %c1 = arith.constant 1 : index
    %c1024 = arith.constant 1024 : index
    %0:2 = hlfir.declare %arg0 {uniq_name = "_QFFrun_benchmarkEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
    %1:2 = hlfir.declare %arg1 {uniq_name = "_QFFrun_benchmarkEn"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
    %2:2 = hlfir.declare %arg2 {uniq_name = "_QFFrun_benchmarkEstride"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
    %3 = fir.shape %c1024 : (index) -> !fir.shape<1>
    %4:2 = hlfir.declare %arg3(%3) {uniq_name = "_QFFrun_benchmarkEz"} : (!fir.ref<!fir.array<?xf64>>, !fir.shape<1>) -> (!fir.box<!fir.array<?xf64>>, !fir.ref<!fir.array<?xf64>>)
    %5 = fir.shape %c1024 : (index) -> !fir.shape<1>
    %6:2 = hlfir.declare %arg4(%5) {uniq_name = "_QFFrun_benchmarkEx"} : (!fir.ref<!fir.array<?xf64>>, !fir.shape<1>) -> (!fir.box<!fir.array<?xf64>>, !fir.ref<!fir.array<?xf64>>)
    %7 = fir.shape %c1024 : (index) -> !fir.shape<1>
    %8:2 = hlfir.declare %arg5(%7) {uniq_name = "_QFFrun_benchmarkEy"} : (!fir.ref<!fir.array<?xf64>>, !fir.shape<1>) -> (!fir.box<!fir.array<?xf64>>, !fir.ref<!fir.array<?xf64>>)
    %9:2 = hlfir.declare %arg6 {uniq_name = "_QFFrun_benchmarkEa"} : (!fir.ref<f64>) -> (!fir.ref<f64>, !fir.ref<f64>)
    %10 = fir.convert %c1 : (index) -> i32
    %11 = fir.do_loop %arg10 = %c1 to %c1024 step %c3 iter_args(%arg11 = %10) -> (i32) {
      fir.store %arg11 to %0#0 : !fir.ref<i32>
      %12 = fir.load %9#0 : !fir.ref<f64>
      %13 = fir.load %0#0 : !fir.ref<i32>
      %14 = fir.convert %13 : (i32) -> i64
      %15 = hlfir.designate %6#0 (%14)  : (!fir.box<!fir.array<?xf64>>, i64) -> !fir.ref<f64>
      %16 = fir.load %15 : !fir.ref<f64>
      %17 = arith.mulf %12, %16 fastmath<contract> : f64
      %18 = fir.load %0#0 : !fir.ref<i32>
      %19 = fir.convert %18 : (i32) -> i64
      %20 = hlfir.designate %8#0 (%19)  : (!fir.box<!fir.array<?xf64>>, i64) -> !fir.ref<f64>
      %21 = fir.load %20 : !fir.ref<f64>
      %22 = arith.addf %17, %21 fastmath<contract> : f64
      %23 = fir.load %0#0 : !fir.ref<i32>
      %24 = fir.convert %23 : (i32) -> i64
      %25 = hlfir.designate %4#0 (%24)  : (!fir.box<!fir.array<?xf64>>, i64) -> !fir.ref<f64>
      hlfir.assign %22 to %25 : f64, !fir.ref<f64>
      %26 = fir.convert %c3 : (index) -> i32
      %27 = fir.load %0#0 : !fir.ref<i32>
      %28 = arith.addi %27, %26 overflow<nsw> : i32
      fir.result %28 : i32
    }
    fir.store %11 to %0#0 : !fir.ref<i32>
    omp.terminator
  }
}


// -----// IR Dump After SCCPPass (sccp) //----- //
module {
  func.func @kernel(%arg0: !fir.ref<i32> {jit.arg_type = 1 : ui32, jit.literal_val = 0 : i64}, %arg1: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg2: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 3 : i64}, %arg3: !fir.ref<!fir.array<?xf64>> {jit.arg_type = 1 : ui32}, %arg4: !fir.ref<!fir.array<?xf64>> {jit.arg_type = 1 : ui32}, %arg5: !fir.ref<!fir.array<?xf64>> {jit.arg_type = 1 : ui32}, %arg6: !fir.ref<f64> {jit.arg_type = 1 : ui32, jit.literal_val = 4619567317775286272 : i64}, %arg7: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg8: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg9: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}) {
    %c1024 = arith.constant 1024 : index
    %c1 = arith.constant 1 : index
    %c3 = arith.constant 3 : index
    %0:2 = hlfir.declare %arg0 {uniq_name = "_QFFrun_benchmarkEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
    %1:2 = hlfir.declare %arg1 {uniq_name = "_QFFrun_benchmarkEn"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
    %2:2 = hlfir.declare %arg2 {uniq_name = "_QFFrun_benchmarkEstride"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
    %3 = fir.shape %c1024 : (index) -> !fir.shape<1>
    %4:2 = hlfir.declare %arg3(%3) {uniq_name = "_QFFrun_benchmarkEz"} : (!fir.ref<!fir.array<?xf64>>, !fir.shape<1>) -> (!fir.box<!fir.array<?xf64>>, !fir.ref<!fir.array<?xf64>>)
    %5 = fir.shape %c1024 : (index) -> !fir.shape<1>
    %6:2 = hlfir.declare %arg4(%5) {uniq_name = "_QFFrun_benchmarkEx"} : (!fir.ref<!fir.array<?xf64>>, !fir.shape<1>) -> (!fir.box<!fir.array<?xf64>>, !fir.ref<!fir.array<?xf64>>)
    %7 = fir.shape %c1024 : (index) -> !fir.shape<1>
    %8:2 = hlfir.declare %arg5(%7) {uniq_name = "_QFFrun_benchmarkEy"} : (!fir.ref<!fir.array<?xf64>>, !fir.shape<1>) -> (!fir.box<!fir.array<?xf64>>, !fir.ref<!fir.array<?xf64>>)
    %9:2 = hlfir.declare %arg6 {uniq_name = "_QFFrun_benchmarkEa"} : (!fir.ref<f64>) -> (!fir.ref<f64>, !fir.ref<f64>)
    %10 = fir.convert %c1 : (index) -> i32
    %11 = fir.do_loop %arg10 = %c1 to %c1024 step %c3 iter_args(%arg11 = %10) -> (i32) {
      fir.store %arg11 to %0#0 : !fir.ref<i32>
      %12 = fir.load %9#0 : !fir.ref<f64>
      %13 = fir.load %0#0 : !fir.ref<i32>
      %14 = fir.convert %13 : (i32) -> i64
      %15 = hlfir.designate %6#0 (%14)  : (!fir.box<!fir.array<?xf64>>, i64) -> !fir.ref<f64>
      %16 = fir.load %15 : !fir.ref<f64>
      %17 = arith.mulf %12, %16 fastmath<contract> : f64
      %18 = fir.load %0#0 : !fir.ref<i32>
      %19 = fir.convert %18 : (i32) -> i64
      %20 = hlfir.designate %8#0 (%19)  : (!fir.box<!fir.array<?xf64>>, i64) -> !fir.ref<f64>
      %21 = fir.load %20 : !fir.ref<f64>
      %22 = arith.addf %17, %21 fastmath<contract> : f64
      %23 = fir.load %0#0 : !fir.ref<i32>
      %24 = fir.convert %23 : (i32) -> i64
      %25 = hlfir.designate %4#0 (%24)  : (!fir.box<!fir.array<?xf64>>, i64) -> !fir.ref<f64>
      hlfir.assign %22 to %25 : f64, !fir.ref<f64>
      %26 = fir.convert %c3 : (index) -> i32
      %27 = fir.load %0#0 : !fir.ref<i32>
      %28 = arith.addi %27, %26 overflow<nsw> : i32
      fir.result %28 : i32
    }
    fir.store %11 to %0#0 : !fir.ref<i32>
    omp.terminator
  }
}


// -----// IR Dump After CSEPass (cse) //----- //
module {
  func.func @kernel(%arg0: !fir.ref<i32> {jit.arg_type = 1 : ui32, jit.literal_val = 0 : i64}, %arg1: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg2: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 3 : i64}, %arg3: !fir.ref<!fir.array<?xf64>> {jit.arg_type = 1 : ui32}, %arg4: !fir.ref<!fir.array<?xf64>> {jit.arg_type = 1 : ui32}, %arg5: !fir.ref<!fir.array<?xf64>> {jit.arg_type = 1 : ui32}, %arg6: !fir.ref<f64> {jit.arg_type = 1 : ui32, jit.literal_val = 4619567317775286272 : i64}, %arg7: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg8: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg9: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}) {
    %c1024 = arith.constant 1024 : index
    %c1 = arith.constant 1 : index
    %c3 = arith.constant 3 : index
    %0:2 = hlfir.declare %arg0 {uniq_name = "_QFFrun_benchmarkEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
    %1:2 = hlfir.declare %arg1 {uniq_name = "_QFFrun_benchmarkEn"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
    %2:2 = hlfir.declare %arg2 {uniq_name = "_QFFrun_benchmarkEstride"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
    %3 = fir.shape %c1024 : (index) -> !fir.shape<1>
    %4:2 = hlfir.declare %arg3(%3) {uniq_name = "_QFFrun_benchmarkEz"} : (!fir.ref<!fir.array<?xf64>>, !fir.shape<1>) -> (!fir.box<!fir.array<?xf64>>, !fir.ref<!fir.array<?xf64>>)
    %5:2 = hlfir.declare %arg4(%3) {uniq_name = "_QFFrun_benchmarkEx"} : (!fir.ref<!fir.array<?xf64>>, !fir.shape<1>) -> (!fir.box<!fir.array<?xf64>>, !fir.ref<!fir.array<?xf64>>)
    %6:2 = hlfir.declare %arg5(%3) {uniq_name = "_QFFrun_benchmarkEy"} : (!fir.ref<!fir.array<?xf64>>, !fir.shape<1>) -> (!fir.box<!fir.array<?xf64>>, !fir.ref<!fir.array<?xf64>>)
    %7:2 = hlfir.declare %arg6 {uniq_name = "_QFFrun_benchmarkEa"} : (!fir.ref<f64>) -> (!fir.ref<f64>, !fir.ref<f64>)
    %8 = fir.convert %c1 : (index) -> i32
    %9 = fir.do_loop %arg10 = %c1 to %c1024 step %c3 iter_args(%arg11 = %8) -> (i32) {
      fir.store %arg11 to %0#0 : !fir.ref<i32>
      %10 = fir.load %7#0 : !fir.ref<f64>
      %11 = fir.load %0#0 : !fir.ref<i32>
      %12 = fir.convert %11 : (i32) -> i64
      %13 = hlfir.designate %5#0 (%12)  : (!fir.box<!fir.array<?xf64>>, i64) -> !fir.ref<f64>
      %14 = fir.load %13 : !fir.ref<f64>
      %15 = arith.mulf %10, %14 fastmath<contract> : f64
      %16 = hlfir.designate %6#0 (%12)  : (!fir.box<!fir.array<?xf64>>, i64) -> !fir.ref<f64>
      %17 = fir.load %16 : !fir.ref<f64>
      %18 = arith.addf %15, %17 fastmath<contract> : f64
      %19 = hlfir.designate %4#0 (%12)  : (!fir.box<!fir.array<?xf64>>, i64) -> !fir.ref<f64>
      hlfir.assign %18 to %19 : f64, !fir.ref<f64>
      %20 = fir.convert %c3 : (index) -> i32
      %21 = fir.load %0#0 : !fir.ref<i32>
      %22 = arith.addi %21, %20 overflow<nsw> : i32
      fir.result %22 : i32
    }
    fir.store %9 to %0#0 : !fir.ref<i32>
    omp.terminator
  }
}


// -----// IR Dump After CanonicalizerPass (canonicalize) //----- //
module {
  func.func @kernel(%arg0: !fir.ref<i32> {jit.arg_type = 1 : ui32, jit.literal_val = 0 : i64}, %arg1: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg2: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 3 : i64}, %arg3: !fir.ref<!fir.array<?xf64>> {jit.arg_type = 1 : ui32}, %arg4: !fir.ref<!fir.array<?xf64>> {jit.arg_type = 1 : ui32}, %arg5: !fir.ref<!fir.array<?xf64>> {jit.arg_type = 1 : ui32}, %arg6: !fir.ref<f64> {jit.arg_type = 1 : ui32, jit.literal_val = 4619567317775286272 : i64}, %arg7: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg8: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg9: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}) {
    %c1024 = arith.constant 1024 : index
    %c1 = arith.constant 1 : index
    %c3 = arith.constant 3 : index
    %0:2 = hlfir.declare %arg0 {uniq_name = "_QFFrun_benchmarkEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
    %1:2 = hlfir.declare %arg1 {uniq_name = "_QFFrun_benchmarkEn"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
    %2:2 = hlfir.declare %arg2 {uniq_name = "_QFFrun_benchmarkEstride"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
    %3 = fir.shape %c1024 : (index) -> !fir.shape<1>
    %4:2 = hlfir.declare %arg3(%3) {uniq_name = "_QFFrun_benchmarkEz"} : (!fir.ref<!fir.array<?xf64>>, !fir.shape<1>) -> (!fir.box<!fir.array<?xf64>>, !fir.ref<!fir.array<?xf64>>)
    %5:2 = hlfir.declare %arg4(%3) {uniq_name = "_QFFrun_benchmarkEx"} : (!fir.ref<!fir.array<?xf64>>, !fir.shape<1>) -> (!fir.box<!fir.array<?xf64>>, !fir.ref<!fir.array<?xf64>>)
    %6:2 = hlfir.declare %arg5(%3) {uniq_name = "_QFFrun_benchmarkEy"} : (!fir.ref<!fir.array<?xf64>>, !fir.shape<1>) -> (!fir.box<!fir.array<?xf64>>, !fir.ref<!fir.array<?xf64>>)
    %7:2 = hlfir.declare %arg6 {uniq_name = "_QFFrun_benchmarkEa"} : (!fir.ref<f64>) -> (!fir.ref<f64>, !fir.ref<f64>)
    %8 = fir.convert %c1 : (index) -> i32
    %9 = fir.do_loop %arg10 = %c1 to %c1024 step %c3 iter_args(%arg11 = %8) -> (i32) {
      fir.store %arg11 to %0#0 : !fir.ref<i32>
      %10 = fir.load %7#0 : !fir.ref<f64>
      %11 = fir.load %0#0 : !fir.ref<i32>
      %12 = fir.convert %11 : (i32) -> i64
      %13 = hlfir.designate %5#0 (%12)  : (!fir.box<!fir.array<?xf64>>, i64) -> !fir.ref<f64>
      %14 = fir.load %13 : !fir.ref<f64>
      %15 = arith.mulf %10, %14 fastmath<contract> : f64
      %16 = hlfir.designate %6#0 (%12)  : (!fir.box<!fir.array<?xf64>>, i64) -> !fir.ref<f64>
      %17 = fir.load %16 : !fir.ref<f64>
      %18 = arith.addf %15, %17 fastmath<contract> : f64
      %19 = hlfir.designate %4#0 (%12)  : (!fir.box<!fir.array<?xf64>>, i64) -> !fir.ref<f64>
      hlfir.assign %18 to %19 : f64, !fir.ref<f64>
      %20 = fir.convert %c3 : (index) -> i32
      %21 = fir.load %0#0 : !fir.ref<i32>
      %22 = arith.addi %21, %20 overflow<nsw> : i32
      fir.result %22 : i32
    }
    fir.store %9 to %0#0 : !fir.ref<i32>
    omp.terminator
  }
}


// -----// IR Dump After (anonymous namespace)::ShapeInferPass (jforce-shape-infer) //----- //
func.func @kernel(%arg0: !fir.ref<i32> {jit.arg_type = 1 : ui32, jit.literal_val = 0 : i64}, %arg1: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg2: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 3 : i64}, %arg3: !fir.ref<!fir.array<1024xf64>> {jit.arg_type = 1 : ui32}, %arg4: !fir.ref<!fir.array<1024xf64>> {jit.arg_type = 1 : ui32}, %arg5: !fir.ref<!fir.array<1024xf64>> {jit.arg_type = 1 : ui32}, %arg6: !fir.ref<f64> {jit.arg_type = 1 : ui32, jit.literal_val = 4619567317775286272 : i64}, %arg7: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg8: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg9: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}) {
  %c1024 = arith.constant 1024 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %0:2 = hlfir.declare %arg0 {uniq_name = "_QFFrun_benchmarkEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %1:2 = hlfir.declare %arg1 {uniq_name = "_QFFrun_benchmarkEn"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %2:2 = hlfir.declare %arg2 {uniq_name = "_QFFrun_benchmarkEstride"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %3 = fir.shape %c1024 : (index) -> !fir.shape<1>
  %4:2 = hlfir.declare %arg3(%3) {uniq_name = "_QFFrun_benchmarkEz"} : (!fir.ref<!fir.array<1024xf64>>, !fir.shape<1>) -> (!fir.ref<!fir.array<1024xf64>>, !fir.ref<!fir.array<1024xf64>>)
  %5:2 = hlfir.declare %arg4(%3) {uniq_name = "_QFFrun_benchmarkEx"} : (!fir.ref<!fir.array<1024xf64>>, !fir.shape<1>) -> (!fir.ref<!fir.array<1024xf64>>, !fir.ref<!fir.array<1024xf64>>)
  %6:2 = hlfir.declare %arg5(%3) {uniq_name = "_QFFrun_benchmarkEy"} : (!fir.ref<!fir.array<1024xf64>>, !fir.shape<1>) -> (!fir.ref<!fir.array<1024xf64>>, !fir.ref<!fir.array<1024xf64>>)
  %7:2 = hlfir.declare %arg6 {uniq_name = "_QFFrun_benchmarkEa"} : (!fir.ref<f64>) -> (!fir.ref<f64>, !fir.ref<f64>)
  %8 = fir.convert %c1 : (index) -> i32
  %9 = fir.do_loop %arg10 = %c1 to %c1024 step %c3 iter_args(%arg11 = %8) -> (i32) {
    fir.store %arg11 to %0#0 : !fir.ref<i32>
    %10 = fir.load %7#0 : !fir.ref<f64>
    %11 = fir.load %0#0 : !fir.ref<i32>
    %12 = fir.convert %11 : (i32) -> i64
    %13 = hlfir.designate %5#0 (%12)  : (!fir.ref<!fir.array<1024xf64>>, i64) -> !fir.ref<f64>
    %14 = fir.load %13 : !fir.ref<f64>
    %15 = arith.mulf %10, %14 fastmath<contract> : f64
    %16 = hlfir.designate %6#0 (%12)  : (!fir.ref<!fir.array<1024xf64>>, i64) -> !fir.ref<f64>
    %17 = fir.load %16 : !fir.ref<f64>
    %18 = arith.addf %15, %17 fastmath<contract> : f64
    %19 = hlfir.designate %4#0 (%12)  : (!fir.ref<!fir.array<1024xf64>>, i64) -> !fir.ref<f64>
    hlfir.assign %18 to %19 : f64, !fir.ref<f64>
    %20 = fir.convert %c3 : (index) -> i32
    %21 = fir.load %0#0 : !fir.ref<i32>
    %22 = arith.addi %21, %20 overflow<nsw> : i32
    fir.result %22 : i32
  }
  fir.store %9 to %0#0 : !fir.ref<i32>
  omp.terminator
}

// -----// IR Dump After CanonicalizerPass (canonicalize) //----- //
module {
  func.func @kernel(%arg0: !fir.ref<i32> {jit.arg_type = 1 : ui32, jit.literal_val = 0 : i64}, %arg1: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg2: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 3 : i64}, %arg3: !fir.ref<!fir.array<1024xf64>> {jit.arg_type = 1 : ui32}, %arg4: !fir.ref<!fir.array<1024xf64>> {jit.arg_type = 1 : ui32}, %arg5: !fir.ref<!fir.array<1024xf64>> {jit.arg_type = 1 : ui32}, %arg6: !fir.ref<f64> {jit.arg_type = 1 : ui32, jit.literal_val = 4619567317775286272 : i64}, %arg7: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg8: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg9: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}) {
    %c1024 = arith.constant 1024 : index
    %c1 = arith.constant 1 : index
    %c3 = arith.constant 3 : index
    %0:2 = hlfir.declare %arg0 {uniq_name = "_QFFrun_benchmarkEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
    %1:2 = hlfir.declare %arg1 {uniq_name = "_QFFrun_benchmarkEn"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
    %2:2 = hlfir.declare %arg2 {uniq_name = "_QFFrun_benchmarkEstride"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
    %3 = fir.shape %c1024 : (index) -> !fir.shape<1>
    %4:2 = hlfir.declare %arg3(%3) {uniq_name = "_QFFrun_benchmarkEz"} : (!fir.ref<!fir.array<1024xf64>>, !fir.shape<1>) -> (!fir.ref<!fir.array<1024xf64>>, !fir.ref<!fir.array<1024xf64>>)
    %5:2 = hlfir.declare %arg4(%3) {uniq_name = "_QFFrun_benchmarkEx"} : (!fir.ref<!fir.array<1024xf64>>, !fir.shape<1>) -> (!fir.ref<!fir.array<1024xf64>>, !fir.ref<!fir.array<1024xf64>>)
    %6:2 = hlfir.declare %arg5(%3) {uniq_name = "_QFFrun_benchmarkEy"} : (!fir.ref<!fir.array<1024xf64>>, !fir.shape<1>) -> (!fir.ref<!fir.array<1024xf64>>, !fir.ref<!fir.array<1024xf64>>)
    %7:2 = hlfir.declare %arg6 {uniq_name = "_QFFrun_benchmarkEa"} : (!fir.ref<f64>) -> (!fir.ref<f64>, !fir.ref<f64>)
    %8 = fir.convert %c1 : (index) -> i32
    %9 = fir.do_loop %arg10 = %c1 to %c1024 step %c3 iter_args(%arg11 = %8) -> (i32) {
      fir.store %arg11 to %0#0 : !fir.ref<i32>
      %10 = fir.load %7#0 : !fir.ref<f64>
      %11 = fir.load %0#0 : !fir.ref<i32>
      %12 = fir.convert %11 : (i32) -> i64
      %13 = hlfir.designate %5#0 (%12)  : (!fir.ref<!fir.array<1024xf64>>, i64) -> !fir.ref<f64>
      %14 = fir.load %13 : !fir.ref<f64>
      %15 = arith.mulf %10, %14 fastmath<contract> : f64
      %16 = hlfir.designate %6#0 (%12)  : (!fir.ref<!fir.array<1024xf64>>, i64) -> !fir.ref<f64>
      %17 = fir.load %16 : !fir.ref<f64>
      %18 = arith.addf %15, %17 fastmath<contract> : f64
      %19 = hlfir.designate %4#0 (%12)  : (!fir.ref<!fir.array<1024xf64>>, i64) -> !fir.ref<f64>
      hlfir.assign %18 to %19 : f64, !fir.ref<f64>
      %20 = fir.convert %c3 : (index) -> i32
      %21 = fir.load %0#0 : !fir.ref<i32>
      %22 = arith.addi %21, %20 overflow<nsw> : i32
      fir.result %22 : i32
    }
    fir.store %9 to %0#0 : !fir.ref<i32>
    omp.terminator
  }
}


// -----// IR Dump After ConvertHLFIRtoFIR (convert-hlfir-to-fir) //----- //
module {
  func.func @kernel(%arg0: !fir.ref<i32> {jit.arg_type = 1 : ui32, jit.literal_val = 0 : i64}, %arg1: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg2: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 3 : i64}, %arg3: !fir.ref<!fir.array<1024xf64>> {jit.arg_type = 1 : ui32}, %arg4: !fir.ref<!fir.array<1024xf64>> {jit.arg_type = 1 : ui32}, %arg5: !fir.ref<!fir.array<1024xf64>> {jit.arg_type = 1 : ui32}, %arg6: !fir.ref<f64> {jit.arg_type = 1 : ui32, jit.literal_val = 4619567317775286272 : i64}, %arg7: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg8: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg9: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}) {
    %c1024 = arith.constant 1024 : index
    %c1 = arith.constant 1 : index
    %c3 = arith.constant 3 : index
    %0 = fir.declare %arg0 {uniq_name = "_QFFrun_benchmarkEi"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %1 = fir.declare %arg1 {uniq_name = "_QFFrun_benchmarkEn"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %2 = fir.declare %arg2 {uniq_name = "_QFFrun_benchmarkEstride"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %3 = fir.shape %c1024 : (index) -> !fir.shape<1>
    %4 = fir.declare %arg3(%3) {uniq_name = "_QFFrun_benchmarkEz"} : (!fir.ref<!fir.array<1024xf64>>, !fir.shape<1>) -> !fir.ref<!fir.array<1024xf64>>
    %5 = fir.declare %arg4(%3) {uniq_name = "_QFFrun_benchmarkEx"} : (!fir.ref<!fir.array<1024xf64>>, !fir.shape<1>) -> !fir.ref<!fir.array<1024xf64>>
    %6 = fir.declare %arg5(%3) {uniq_name = "_QFFrun_benchmarkEy"} : (!fir.ref<!fir.array<1024xf64>>, !fir.shape<1>) -> !fir.ref<!fir.array<1024xf64>>
    %7 = fir.declare %arg6 {uniq_name = "_QFFrun_benchmarkEa"} : (!fir.ref<f64>) -> !fir.ref<f64>
    %8 = fir.convert %c1 : (index) -> i32
    %9 = fir.do_loop %arg10 = %c1 to %c1024 step %c3 iter_args(%arg11 = %8) -> (i32) {
      fir.store %arg11 to %0 : !fir.ref<i32>
      %10 = fir.load %7 : !fir.ref<f64>
      %11 = fir.load %0 : !fir.ref<i32>
      %12 = fir.convert %11 : (i32) -> i64
      %13 = fir.array_coor %5(%3) %12 : (!fir.ref<!fir.array<1024xf64>>, !fir.shape<1>, i64) -> !fir.ref<f64>
      %14 = fir.load %13 : !fir.ref<f64>
      %15 = arith.mulf %10, %14 fastmath<contract> : f64
      %16 = fir.array_coor %6(%3) %12 : (!fir.ref<!fir.array<1024xf64>>, !fir.shape<1>, i64) -> !fir.ref<f64>
      %17 = fir.load %16 : !fir.ref<f64>
      %18 = arith.addf %15, %17 fastmath<contract> : f64
      %19 = fir.array_coor %4(%3) %12 : (!fir.ref<!fir.array<1024xf64>>, !fir.shape<1>, i64) -> !fir.ref<f64>
      fir.store %18 to %19 : !fir.ref<f64>
      %20 = fir.convert %c3 : (index) -> i32
      %21 = fir.load %0 : !fir.ref<i32>
      %22 = arith.addi %21, %20 overflow<nsw> : i32
      fir.result %22 : i32
    }
    fir.store %9 to %0 : !fir.ref<i32>
    omp.terminator
  }
}


// -----// IR Dump After FIRToMemRef (fir-to-memref) //----- //
func.func @kernel(%arg0: !fir.ref<i32> {jit.arg_type = 1 : ui32, jit.literal_val = 0 : i64}, %arg1: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg2: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 3 : i64}, %arg3: !fir.ref<!fir.array<1024xf64>> {jit.arg_type = 1 : ui32}, %arg4: !fir.ref<!fir.array<1024xf64>> {jit.arg_type = 1 : ui32}, %arg5: !fir.ref<!fir.array<1024xf64>> {jit.arg_type = 1 : ui32}, %arg6: !fir.ref<f64> {jit.arg_type = 1 : ui32, jit.literal_val = 4619567317775286272 : i64}, %arg7: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg8: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg9: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}) {
  %c1024 = arith.constant 1024 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %0 = fir.declare %arg0 {uniq_name = "_QFFrun_benchmarkEi"} : (!fir.ref<i32>) -> !fir.ref<i32>
  %1 = fir.declare %arg1 {uniq_name = "_QFFrun_benchmarkEn"} : (!fir.ref<i32>) -> !fir.ref<i32>
  %2 = fir.declare %arg2 {uniq_name = "_QFFrun_benchmarkEstride"} : (!fir.ref<i32>) -> !fir.ref<i32>
  %3 = fir.shape %c1024 : (index) -> !fir.shape<1>
  %4 = fir.declare %arg3(%3) {uniq_name = "_QFFrun_benchmarkEz"} : (!fir.ref<!fir.array<1024xf64>>, !fir.shape<1>) -> !fir.ref<!fir.array<1024xf64>>
  %5 = fir.declare %arg4(%3) {uniq_name = "_QFFrun_benchmarkEx"} : (!fir.ref<!fir.array<1024xf64>>, !fir.shape<1>) -> !fir.ref<!fir.array<1024xf64>>
  %6 = fir.declare %arg5(%3) {uniq_name = "_QFFrun_benchmarkEy"} : (!fir.ref<!fir.array<1024xf64>>, !fir.shape<1>) -> !fir.ref<!fir.array<1024xf64>>
  %7 = fir.declare %arg6 {uniq_name = "_QFFrun_benchmarkEa"} : (!fir.ref<f64>) -> !fir.ref<f64>
  %8 = fir.convert %c1 : (index) -> i32
  %9 = fir.do_loop %arg10 = %c1 to %c1024 step %c3 iter_args(%arg11 = %8) -> (i32) {
    %11 = fir.convert %0 : (!fir.ref<i32>) -> memref<i32>
    memref.store %arg11, %11[] : memref<i32>
    %12 = fir.convert %7 : (!fir.ref<f64>) -> memref<f64>
    %13 = memref.load %12[] : memref<f64>
    %14 = fir.convert %0 : (!fir.ref<i32>) -> memref<i32>
    %15 = memref.load %14[] : memref<i32>
    %16 = fir.convert %15 : (i32) -> i64
    %17 = fir.convert %5 : (!fir.ref<!fir.array<1024xf64>>) -> memref<1024xf64>
    %c1_0 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %18 = arith.index_cast %16 : i64 to index
    %19 = arith.subi %18, %c1_0 : index
    %20 = arith.muli %19, %c1_0 : index
    %21 = arith.subi %c1_0, %c1_0 : index
    %22 = arith.addi %20, %21 : index
    %23 = memref.load %17[%22] : memref<1024xf64>
    %24 = arith.mulf %13, %23 fastmath<contract> : f64
    %25 = fir.convert %6 : (!fir.ref<!fir.array<1024xf64>>) -> memref<1024xf64>
    %c1_1 = arith.constant 1 : index
    %c0_2 = arith.constant 0 : index
    %26 = arith.index_cast %16 : i64 to index
    %27 = arith.subi %26, %c1_1 : index
    %28 = arith.muli %27, %c1_1 : index
    %29 = arith.subi %c1_1, %c1_1 : index
    %30 = arith.addi %28, %29 : index
    %31 = memref.load %25[%30] : memref<1024xf64>
    %32 = arith.addf %24, %31 fastmath<contract> : f64
    %33 = fir.convert %4 : (!fir.ref<!fir.array<1024xf64>>) -> memref<1024xf64>
    %c1_3 = arith.constant 1 : index
    %c0_4 = arith.constant 0 : index
    %34 = arith.index_cast %16 : i64 to index
    %35 = arith.subi %34, %c1_3 : index
    %36 = arith.muli %35, %c1_3 : index
    %37 = arith.subi %c1_3, %c1_3 : index
    %38 = arith.addi %36, %37 : index
    memref.store %32, %33[%38] : memref<1024xf64>
    %39 = fir.convert %c3 : (index) -> i32
    %40 = fir.convert %0 : (!fir.ref<i32>) -> memref<i32>
    %41 = memref.load %40[] : memref<i32>
    %42 = arith.addi %41, %39 overflow<nsw> : i32
    fir.result %42 : i32
  }
  %10 = fir.convert %0 : (!fir.ref<i32>) -> memref<i32>
  memref.store %9, %10[] : memref<i32>
  omp.terminator
}

// -----// IR Dump After CanonicalizerPass (canonicalize) //----- //
module {
  func.func @kernel(%arg0: !fir.ref<i32> {jit.arg_type = 1 : ui32, jit.literal_val = 0 : i64}, %arg1: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg2: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 3 : i64}, %arg3: !fir.ref<!fir.array<1024xf64>> {jit.arg_type = 1 : ui32}, %arg4: !fir.ref<!fir.array<1024xf64>> {jit.arg_type = 1 : ui32}, %arg5: !fir.ref<!fir.array<1024xf64>> {jit.arg_type = 1 : ui32}, %arg6: !fir.ref<f64> {jit.arg_type = 1 : ui32, jit.literal_val = 4619567317775286272 : i64}, %arg7: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg8: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg9: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}) {
    %c1024 = arith.constant 1024 : index
    %c1 = arith.constant 1 : index
    %c3 = arith.constant 3 : index
    %0 = fir.declare %arg0 {uniq_name = "_QFFrun_benchmarkEi"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %1 = fir.declare %arg1 {uniq_name = "_QFFrun_benchmarkEn"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %2 = fir.declare %arg2 {uniq_name = "_QFFrun_benchmarkEstride"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %3 = fir.shape %c1024 : (index) -> !fir.shape<1>
    %4 = fir.declare %arg3(%3) {uniq_name = "_QFFrun_benchmarkEz"} : (!fir.ref<!fir.array<1024xf64>>, !fir.shape<1>) -> !fir.ref<!fir.array<1024xf64>>
    %5 = fir.declare %arg4(%3) {uniq_name = "_QFFrun_benchmarkEx"} : (!fir.ref<!fir.array<1024xf64>>, !fir.shape<1>) -> !fir.ref<!fir.array<1024xf64>>
    %6 = fir.declare %arg5(%3) {uniq_name = "_QFFrun_benchmarkEy"} : (!fir.ref<!fir.array<1024xf64>>, !fir.shape<1>) -> !fir.ref<!fir.array<1024xf64>>
    %7 = fir.declare %arg6 {uniq_name = "_QFFrun_benchmarkEa"} : (!fir.ref<f64>) -> !fir.ref<f64>
    %8 = fir.convert %c1 : (index) -> i32
    %9 = fir.do_loop %arg10 = %c1 to %c1024 step %c3 iter_args(%arg11 = %8) -> (i32) {
      %11 = fir.convert %0 : (!fir.ref<i32>) -> memref<i32>
      memref.store %arg11, %11[] : memref<i32>
      %12 = fir.convert %7 : (!fir.ref<f64>) -> memref<f64>
      %13 = memref.load %12[] : memref<f64>
      %14 = fir.convert %0 : (!fir.ref<i32>) -> memref<i32>
      %15 = memref.load %14[] : memref<i32>
      %16 = fir.convert %15 : (i32) -> i64
      %17 = fir.convert %5 : (!fir.ref<!fir.array<1024xf64>>) -> memref<1024xf64>
      %18 = arith.index_cast %16 : i64 to index
      %19 = arith.subi %18, %c1 : index
      %20 = memref.load %17[%19] : memref<1024xf64>
      %21 = arith.mulf %13, %20 fastmath<contract> : f64
      %22 = fir.convert %6 : (!fir.ref<!fir.array<1024xf64>>) -> memref<1024xf64>
      %23 = arith.index_cast %16 : i64 to index
      %24 = arith.subi %23, %c1 : index
      %25 = memref.load %22[%24] : memref<1024xf64>
      %26 = arith.addf %21, %25 fastmath<contract> : f64
      %27 = fir.convert %4 : (!fir.ref<!fir.array<1024xf64>>) -> memref<1024xf64>
      %28 = arith.index_cast %16 : i64 to index
      %29 = arith.subi %28, %c1 : index
      memref.store %26, %27[%29] : memref<1024xf64>
      %30 = fir.convert %c3 : (index) -> i32
      %31 = fir.convert %0 : (!fir.ref<i32>) -> memref<i32>
      %32 = memref.load %31[] : memref<i32>
      %33 = arith.addi %32, %30 overflow<nsw> : i32
      fir.result %33 : i32
    }
    %10 = fir.convert %0 : (!fir.ref<i32>) -> memref<i32>
    memref.store %9, %10[] : memref<i32>
    omp.terminator
  }
}


// -----// IR Dump After LoopInvariantCodeMotionPass (loop-invariant-code-motion) //----- //
module {
  func.func @kernel(%arg0: !fir.ref<i32> {jit.arg_type = 1 : ui32, jit.literal_val = 0 : i64}, %arg1: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg2: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 3 : i64}, %arg3: !fir.ref<!fir.array<1024xf64>> {jit.arg_type = 1 : ui32}, %arg4: !fir.ref<!fir.array<1024xf64>> {jit.arg_type = 1 : ui32}, %arg5: !fir.ref<!fir.array<1024xf64>> {jit.arg_type = 1 : ui32}, %arg6: !fir.ref<f64> {jit.arg_type = 1 : ui32, jit.literal_val = 4619567317775286272 : i64}, %arg7: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg8: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg9: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}) {
    %c1024 = arith.constant 1024 : index
    %c1 = arith.constant 1 : index
    %c3 = arith.constant 3 : index
    %0 = fir.declare %arg0 {uniq_name = "_QFFrun_benchmarkEi"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %1 = fir.declare %arg1 {uniq_name = "_QFFrun_benchmarkEn"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %2 = fir.declare %arg2 {uniq_name = "_QFFrun_benchmarkEstride"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %3 = fir.shape %c1024 : (index) -> !fir.shape<1>
    %4 = fir.declare %arg3(%3) {uniq_name = "_QFFrun_benchmarkEz"} : (!fir.ref<!fir.array<1024xf64>>, !fir.shape<1>) -> !fir.ref<!fir.array<1024xf64>>
    %5 = fir.declare %arg4(%3) {uniq_name = "_QFFrun_benchmarkEx"} : (!fir.ref<!fir.array<1024xf64>>, !fir.shape<1>) -> !fir.ref<!fir.array<1024xf64>>
    %6 = fir.declare %arg5(%3) {uniq_name = "_QFFrun_benchmarkEy"} : (!fir.ref<!fir.array<1024xf64>>, !fir.shape<1>) -> !fir.ref<!fir.array<1024xf64>>
    %7 = fir.declare %arg6 {uniq_name = "_QFFrun_benchmarkEa"} : (!fir.ref<f64>) -> !fir.ref<f64>
    %8 = fir.convert %c1 : (index) -> i32
    %9 = fir.convert %0 : (!fir.ref<i32>) -> memref<i32>
    %10 = fir.convert %7 : (!fir.ref<f64>) -> memref<f64>
    %11 = fir.convert %0 : (!fir.ref<i32>) -> memref<i32>
    %12 = fir.convert %5 : (!fir.ref<!fir.array<1024xf64>>) -> memref<1024xf64>
    %13 = fir.convert %6 : (!fir.ref<!fir.array<1024xf64>>) -> memref<1024xf64>
    %14 = fir.convert %4 : (!fir.ref<!fir.array<1024xf64>>) -> memref<1024xf64>
    %15 = fir.convert %c3 : (index) -> i32
    %16 = fir.convert %0 : (!fir.ref<i32>) -> memref<i32>
    %17 = fir.do_loop %arg10 = %c1 to %c1024 step %c3 iter_args(%arg11 = %8) -> (i32) {
      memref.store %arg11, %9[] : memref<i32>
      %19 = memref.load %10[] : memref<f64>
      %20 = memref.load %11[] : memref<i32>
      %21 = fir.convert %20 : (i32) -> i64
      %22 = arith.index_cast %21 : i64 to index
      %23 = arith.subi %22, %c1 : index
      %24 = memref.load %12[%23] : memref<1024xf64>
      %25 = arith.mulf %19, %24 fastmath<contract> : f64
      %26 = arith.index_cast %21 : i64 to index
      %27 = arith.subi %26, %c1 : index
      %28 = memref.load %13[%27] : memref<1024xf64>
      %29 = arith.addf %25, %28 fastmath<contract> : f64
      %30 = arith.index_cast %21 : i64 to index
      %31 = arith.subi %30, %c1 : index
      memref.store %29, %14[%31] : memref<1024xf64>
      %32 = memref.load %16[] : memref<i32>
      %33 = arith.addi %32, %15 overflow<nsw> : i32
      fir.result %33 : i32
    }
    %18 = fir.convert %0 : (!fir.ref<i32>) -> memref<i32>
    memref.store %17, %18[] : memref<i32>
    omp.terminator
  }
}


// -----// IR Dump After CSEPass (cse) //----- //
module {
  func.func @kernel(%arg0: !fir.ref<i32> {jit.arg_type = 1 : ui32, jit.literal_val = 0 : i64}, %arg1: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg2: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 3 : i64}, %arg3: !fir.ref<!fir.array<1024xf64>> {jit.arg_type = 1 : ui32}, %arg4: !fir.ref<!fir.array<1024xf64>> {jit.arg_type = 1 : ui32}, %arg5: !fir.ref<!fir.array<1024xf64>> {jit.arg_type = 1 : ui32}, %arg6: !fir.ref<f64> {jit.arg_type = 1 : ui32, jit.literal_val = 4619567317775286272 : i64}, %arg7: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg8: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg9: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}) {
    %c1024 = arith.constant 1024 : index
    %c1 = arith.constant 1 : index
    %c3 = arith.constant 3 : index
    %0 = fir.declare %arg0 {uniq_name = "_QFFrun_benchmarkEi"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %1 = fir.declare %arg1 {uniq_name = "_QFFrun_benchmarkEn"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %2 = fir.declare %arg2 {uniq_name = "_QFFrun_benchmarkEstride"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %3 = fir.shape %c1024 : (index) -> !fir.shape<1>
    %4 = fir.declare %arg3(%3) {uniq_name = "_QFFrun_benchmarkEz"} : (!fir.ref<!fir.array<1024xf64>>, !fir.shape<1>) -> !fir.ref<!fir.array<1024xf64>>
    %5 = fir.declare %arg4(%3) {uniq_name = "_QFFrun_benchmarkEx"} : (!fir.ref<!fir.array<1024xf64>>, !fir.shape<1>) -> !fir.ref<!fir.array<1024xf64>>
    %6 = fir.declare %arg5(%3) {uniq_name = "_QFFrun_benchmarkEy"} : (!fir.ref<!fir.array<1024xf64>>, !fir.shape<1>) -> !fir.ref<!fir.array<1024xf64>>
    %7 = fir.declare %arg6 {uniq_name = "_QFFrun_benchmarkEa"} : (!fir.ref<f64>) -> !fir.ref<f64>
    %8 = fir.convert %c1 : (index) -> i32
    %9 = fir.convert %0 : (!fir.ref<i32>) -> memref<i32>
    %10 = fir.convert %7 : (!fir.ref<f64>) -> memref<f64>
    %11 = fir.convert %5 : (!fir.ref<!fir.array<1024xf64>>) -> memref<1024xf64>
    %12 = fir.convert %6 : (!fir.ref<!fir.array<1024xf64>>) -> memref<1024xf64>
    %13 = fir.convert %4 : (!fir.ref<!fir.array<1024xf64>>) -> memref<1024xf64>
    %14 = fir.convert %c3 : (index) -> i32
    %15 = fir.do_loop %arg10 = %c1 to %c1024 step %c3 iter_args(%arg11 = %8) -> (i32) {
      memref.store %arg11, %9[] : memref<i32>
      %16 = memref.load %10[] : memref<f64>
      %17 = memref.load %9[] : memref<i32>
      %18 = fir.convert %17 : (i32) -> i64
      %19 = arith.index_cast %18 : i64 to index
      %20 = arith.subi %19, %c1 : index
      %21 = memref.load %11[%20] : memref<1024xf64>
      %22 = arith.mulf %16, %21 fastmath<contract> : f64
      %23 = memref.load %12[%20] : memref<1024xf64>
      %24 = arith.addf %22, %23 fastmath<contract> : f64
      memref.store %24, %13[%20] : memref<1024xf64>
      %25 = memref.load %9[] : memref<i32>
      %26 = arith.addi %25, %14 overflow<nsw> : i32
      fir.result %26 : i32
    }
    memref.store %15, %9[] : memref<i32>
    omp.terminator
  }
}


// -----// IR Dump After CanonicalizerPass (canonicalize) //----- //
module {
  func.func @kernel(%arg0: !fir.ref<i32> {jit.arg_type = 1 : ui32, jit.literal_val = 0 : i64}, %arg1: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg2: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 3 : i64}, %arg3: !fir.ref<!fir.array<1024xf64>> {jit.arg_type = 1 : ui32}, %arg4: !fir.ref<!fir.array<1024xf64>> {jit.arg_type = 1 : ui32}, %arg5: !fir.ref<!fir.array<1024xf64>> {jit.arg_type = 1 : ui32}, %arg6: !fir.ref<f64> {jit.arg_type = 1 : ui32, jit.literal_val = 4619567317775286272 : i64}, %arg7: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg8: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg9: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}) {
    %c1024 = arith.constant 1024 : index
    %c1 = arith.constant 1 : index
    %c3 = arith.constant 3 : index
    %0 = fir.declare %arg0 {uniq_name = "_QFFrun_benchmarkEi"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %1 = fir.declare %arg1 {uniq_name = "_QFFrun_benchmarkEn"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %2 = fir.declare %arg2 {uniq_name = "_QFFrun_benchmarkEstride"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %3 = fir.shape %c1024 : (index) -> !fir.shape<1>
    %4 = fir.declare %arg3(%3) {uniq_name = "_QFFrun_benchmarkEz"} : (!fir.ref<!fir.array<1024xf64>>, !fir.shape<1>) -> !fir.ref<!fir.array<1024xf64>>
    %5 = fir.declare %arg4(%3) {uniq_name = "_QFFrun_benchmarkEx"} : (!fir.ref<!fir.array<1024xf64>>, !fir.shape<1>) -> !fir.ref<!fir.array<1024xf64>>
    %6 = fir.declare %arg5(%3) {uniq_name = "_QFFrun_benchmarkEy"} : (!fir.ref<!fir.array<1024xf64>>, !fir.shape<1>) -> !fir.ref<!fir.array<1024xf64>>
    %7 = fir.declare %arg6 {uniq_name = "_QFFrun_benchmarkEa"} : (!fir.ref<f64>) -> !fir.ref<f64>
    %8 = fir.convert %c1 : (index) -> i32
    %9 = fir.convert %0 : (!fir.ref<i32>) -> memref<i32>
    %10 = fir.convert %7 : (!fir.ref<f64>) -> memref<f64>
    %11 = fir.convert %5 : (!fir.ref<!fir.array<1024xf64>>) -> memref<1024xf64>
    %12 = fir.convert %6 : (!fir.ref<!fir.array<1024xf64>>) -> memref<1024xf64>
    %13 = fir.convert %4 : (!fir.ref<!fir.array<1024xf64>>) -> memref<1024xf64>
    %14 = fir.convert %c3 : (index) -> i32
    %15 = fir.do_loop %arg10 = %c1 to %c1024 step %c3 iter_args(%arg11 = %8) -> (i32) {
      memref.store %arg11, %9[] : memref<i32>
      %16 = memref.load %10[] : memref<f64>
      %17 = memref.load %9[] : memref<i32>
      %18 = fir.convert %17 : (i32) -> i64
      %19 = arith.index_cast %18 : i64 to index
      %20 = arith.subi %19, %c1 : index
      %21 = memref.load %11[%20] : memref<1024xf64>
      %22 = arith.mulf %16, %21 fastmath<contract> : f64
      %23 = memref.load %12[%20] : memref<1024xf64>
      %24 = arith.addf %22, %23 fastmath<contract> : f64
      memref.store %24, %13[%20] : memref<1024xf64>
      %25 = memref.load %9[] : memref<i32>
      %26 = arith.addi %25, %14 overflow<nsw> : i32
      fir.result %26 : i32
    }
    memref.store %15, %9[] : memref<i32>
    omp.terminator
  }
}


// -----// IR Dump After (anonymous namespace)::CleanFIRLoopPass (jforce-clean-fir-loop) //----- //
func.func @kernel(%arg0: !fir.ref<i32> {jit.arg_type = 1 : ui32, jit.literal_val = 0 : i64}, %arg1: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg2: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 3 : i64}, %arg3: !fir.ref<!fir.array<1024xf64>> {jit.arg_type = 1 : ui32}, %arg4: !fir.ref<!fir.array<1024xf64>> {jit.arg_type = 1 : ui32}, %arg5: !fir.ref<!fir.array<1024xf64>> {jit.arg_type = 1 : ui32}, %arg6: !fir.ref<f64> {jit.arg_type = 1 : ui32, jit.literal_val = 4619567317775286272 : i64}, %arg7: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg8: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg9: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}) {
  %c1024 = arith.constant 1024 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %0 = fir.declare %arg0 {uniq_name = "_QFFrun_benchmarkEi"} : (!fir.ref<i32>) -> !fir.ref<i32>
  %1 = fir.declare %arg1 {uniq_name = "_QFFrun_benchmarkEn"} : (!fir.ref<i32>) -> !fir.ref<i32>
  %2 = fir.declare %arg2 {uniq_name = "_QFFrun_benchmarkEstride"} : (!fir.ref<i32>) -> !fir.ref<i32>
  %3 = fir.shape %c1024 : (index) -> !fir.shape<1>
  %4 = fir.declare %arg3(%3) {uniq_name = "_QFFrun_benchmarkEz"} : (!fir.ref<!fir.array<1024xf64>>, !fir.shape<1>) -> !fir.ref<!fir.array<1024xf64>>
  %5 = fir.declare %arg4(%3) {uniq_name = "_QFFrun_benchmarkEx"} : (!fir.ref<!fir.array<1024xf64>>, !fir.shape<1>) -> !fir.ref<!fir.array<1024xf64>>
  %6 = fir.declare %arg5(%3) {uniq_name = "_QFFrun_benchmarkEy"} : (!fir.ref<!fir.array<1024xf64>>, !fir.shape<1>) -> !fir.ref<!fir.array<1024xf64>>
  %7 = fir.declare %arg6 {uniq_name = "_QFFrun_benchmarkEa"} : (!fir.ref<f64>) -> !fir.ref<f64>
  %8 = fir.convert %c1 : (index) -> i32
  %9 = fir.convert %0 : (!fir.ref<i32>) -> memref<i32>
  %10 = fir.convert %7 : (!fir.ref<f64>) -> memref<f64>
  %11 = fir.convert %5 : (!fir.ref<!fir.array<1024xf64>>) -> memref<1024xf64>
  %12 = fir.convert %6 : (!fir.ref<!fir.array<1024xf64>>) -> memref<1024xf64>
  %13 = fir.convert %4 : (!fir.ref<!fir.array<1024xf64>>) -> memref<1024xf64>
  %14 = fir.convert %c3 : (index) -> i32
  fir.do_loop %arg10 = %c1 to %c1024 step %c3 {
    %16 = fir.convert %arg10 : (index) -> i32
    memref.store %16, %9[] : memref<i32>
    %17 = memref.load %10[] : memref<f64>
    %18 = memref.load %9[] : memref<i32>
    %19 = fir.convert %18 : (i32) -> i64
    %20 = arith.index_cast %19 : i64 to index
    %21 = arith.subi %20, %c1 : index
    %22 = memref.load %11[%21] : memref<1024xf64>
    %23 = arith.mulf %17, %22 fastmath<contract> : f64
    %24 = memref.load %12[%21] : memref<1024xf64>
    %25 = arith.addf %23, %24 fastmath<contract> : f64
    memref.store %25, %13[%21] : memref<1024xf64>
    %26 = memref.load %9[] : memref<i32>
    %27 = arith.addi %26, %14 overflow<nsw> : i32
    memref.store %27, %9[] : memref<i32>
  }
  %15 = memref.load %9[] : memref<i32>
  memref.store %15, %9[] : memref<i32>
  omp.terminator
}

// -----// IR Dump After CanonicalizerPass (canonicalize) //----- //
module {
  func.func @kernel(%arg0: !fir.ref<i32> {jit.arg_type = 1 : ui32, jit.literal_val = 0 : i64}, %arg1: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg2: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 3 : i64}, %arg3: !fir.ref<!fir.array<1024xf64>> {jit.arg_type = 1 : ui32}, %arg4: !fir.ref<!fir.array<1024xf64>> {jit.arg_type = 1 : ui32}, %arg5: !fir.ref<!fir.array<1024xf64>> {jit.arg_type = 1 : ui32}, %arg6: !fir.ref<f64> {jit.arg_type = 1 : ui32, jit.literal_val = 4619567317775286272 : i64}, %arg7: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg8: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg9: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}) {
    %c1024 = arith.constant 1024 : index
    %c1 = arith.constant 1 : index
    %c3 = arith.constant 3 : index
    %0 = fir.declare %arg0 {uniq_name = "_QFFrun_benchmarkEi"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %1 = fir.declare %arg1 {uniq_name = "_QFFrun_benchmarkEn"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %2 = fir.declare %arg2 {uniq_name = "_QFFrun_benchmarkEstride"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %3 = fir.shape %c1024 : (index) -> !fir.shape<1>
    %4 = fir.declare %arg3(%3) {uniq_name = "_QFFrun_benchmarkEz"} : (!fir.ref<!fir.array<1024xf64>>, !fir.shape<1>) -> !fir.ref<!fir.array<1024xf64>>
    %5 = fir.declare %arg4(%3) {uniq_name = "_QFFrun_benchmarkEx"} : (!fir.ref<!fir.array<1024xf64>>, !fir.shape<1>) -> !fir.ref<!fir.array<1024xf64>>
    %6 = fir.declare %arg5(%3) {uniq_name = "_QFFrun_benchmarkEy"} : (!fir.ref<!fir.array<1024xf64>>, !fir.shape<1>) -> !fir.ref<!fir.array<1024xf64>>
    %7 = fir.declare %arg6 {uniq_name = "_QFFrun_benchmarkEa"} : (!fir.ref<f64>) -> !fir.ref<f64>
    %8 = fir.convert %0 : (!fir.ref<i32>) -> memref<i32>
    %9 = fir.convert %7 : (!fir.ref<f64>) -> memref<f64>
    %10 = fir.convert %5 : (!fir.ref<!fir.array<1024xf64>>) -> memref<1024xf64>
    %11 = fir.convert %6 : (!fir.ref<!fir.array<1024xf64>>) -> memref<1024xf64>
    %12 = fir.convert %4 : (!fir.ref<!fir.array<1024xf64>>) -> memref<1024xf64>
    %13 = fir.convert %c3 : (index) -> i32
    fir.do_loop %arg10 = %c1 to %c1024 step %c3 {
      %15 = fir.convert %arg10 : (index) -> i32
      memref.store %15, %8[] : memref<i32>
      %16 = memref.load %9[] : memref<f64>
      %17 = memref.load %8[] : memref<i32>
      %18 = fir.convert %17 : (i32) -> i64
      %19 = arith.index_cast %18 : i64 to index
      %20 = arith.subi %19, %c1 : index
      %21 = memref.load %10[%20] : memref<1024xf64>
      %22 = arith.mulf %16, %21 fastmath<contract> : f64
      %23 = memref.load %11[%20] : memref<1024xf64>
      %24 = arith.addf %22, %23 fastmath<contract> : f64
      memref.store %24, %12[%20] : memref<1024xf64>
      %25 = memref.load %8[] : memref<i32>
      %26 = arith.addi %25, %13 overflow<nsw> : i32
      memref.store %26, %8[] : memref<i32>
    }
    %14 = memref.load %8[] : memref<i32>
    memref.store %14, %8[] : memref<i32>
    omp.terminator
  }
}


// -----// IR Dump After (anonymous namespace)::CleanFIROpsPass (jforce-clean-fir-op) //----- //
func.func @kernel(%arg0: !fir.ref<i32> {jit.arg_type = 1 : ui32, jit.literal_val = 0 : i64}, %arg1: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg2: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 3 : i64}, %arg3: !fir.ref<!fir.array<1024xf64>> {jit.arg_type = 1 : ui32}, %arg4: !fir.ref<!fir.array<1024xf64>> {jit.arg_type = 1 : ui32}, %arg5: !fir.ref<!fir.array<1024xf64>> {jit.arg_type = 1 : ui32}, %arg6: !fir.ref<f64> {jit.arg_type = 1 : ui32, jit.literal_val = 4619567317775286272 : i64}, %arg7: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg8: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg9: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}) {
  %c1024 = arith.constant 1024 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %0 = fir.declare %arg0 {uniq_name = "_QFFrun_benchmarkEi"} : (!fir.ref<i32>) -> !fir.ref<i32>
  %1 = fir.declare %arg1 {uniq_name = "_QFFrun_benchmarkEn"} : (!fir.ref<i32>) -> !fir.ref<i32>
  %2 = fir.declare %arg2 {uniq_name = "_QFFrun_benchmarkEstride"} : (!fir.ref<i32>) -> !fir.ref<i32>
  %3 = fir.shape %c1024 : (index) -> !fir.shape<1>
  %4 = fir.declare %arg3(%3) {uniq_name = "_QFFrun_benchmarkEz"} : (!fir.ref<!fir.array<1024xf64>>, !fir.shape<1>) -> !fir.ref<!fir.array<1024xf64>>
  %5 = fir.declare %arg4(%3) {uniq_name = "_QFFrun_benchmarkEx"} : (!fir.ref<!fir.array<1024xf64>>, !fir.shape<1>) -> !fir.ref<!fir.array<1024xf64>>
  %6 = fir.declare %arg5(%3) {uniq_name = "_QFFrun_benchmarkEy"} : (!fir.ref<!fir.array<1024xf64>>, !fir.shape<1>) -> !fir.ref<!fir.array<1024xf64>>
  %7 = fir.declare %arg6 {uniq_name = "_QFFrun_benchmarkEa"} : (!fir.ref<f64>) -> !fir.ref<f64>
  %8 = fir.convert %0 : (!fir.ref<i32>) -> memref<i32>
  %9 = fir.convert %7 : (!fir.ref<f64>) -> memref<f64>
  %10 = fir.convert %5 : (!fir.ref<!fir.array<1024xf64>>) -> memref<1024xf64>
  %11 = fir.convert %6 : (!fir.ref<!fir.array<1024xf64>>) -> memref<1024xf64>
  %12 = fir.convert %4 : (!fir.ref<!fir.array<1024xf64>>) -> memref<1024xf64>
  %13 = fir.convert %c3 : (index) -> i32
  fir.do_loop %arg10 = %c1 to %c1024 step %c3 {
    %15 = arith.index_cast %arg10 : index to i32
    memref.store %15, %8[] : memref<i32>
    %16 = memref.load %9[] : memref<f64>
    %17 = memref.load %8[] : memref<i32>
    %18 = arith.extsi %17 : i32 to i64
    %19 = arith.index_cast %18 : i64 to index
    %20 = arith.subi %19, %c1 : index
    %21 = memref.load %10[%20] : memref<1024xf64>
    %22 = arith.mulf %16, %21 fastmath<contract> : f64
    %23 = memref.load %11[%20] : memref<1024xf64>
    %24 = arith.addf %22, %23 fastmath<contract> : f64
    memref.store %24, %12[%20] : memref<1024xf64>
    %25 = memref.load %8[] : memref<i32>
    %26 = arith.addi %25, %13 overflow<nsw> : i32
    memref.store %26, %8[] : memref<i32>
  }
  %14 = memref.load %8[] : memref<i32>
  memref.store %14, %8[] : memref<i32>
  omp.terminator
}

// -----// IR Dump After AffineDialectPromotion (promote-to-affine) //----- //
func.func @kernel(%arg0: !fir.ref<i32> {jit.arg_type = 1 : ui32, jit.literal_val = 0 : i64}, %arg1: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg2: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 3 : i64}, %arg3: !fir.ref<!fir.array<1024xf64>> {jit.arg_type = 1 : ui32}, %arg4: !fir.ref<!fir.array<1024xf64>> {jit.arg_type = 1 : ui32}, %arg5: !fir.ref<!fir.array<1024xf64>> {jit.arg_type = 1 : ui32}, %arg6: !fir.ref<f64> {jit.arg_type = 1 : ui32, jit.literal_val = 4619567317775286272 : i64}, %arg7: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg8: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg9: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}) {
  %c1024 = arith.constant 1024 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %0 = fir.declare %arg0 {uniq_name = "_QFFrun_benchmarkEi"} : (!fir.ref<i32>) -> !fir.ref<i32>
  %1 = fir.declare %arg1 {uniq_name = "_QFFrun_benchmarkEn"} : (!fir.ref<i32>) -> !fir.ref<i32>
  %2 = fir.declare %arg2 {uniq_name = "_QFFrun_benchmarkEstride"} : (!fir.ref<i32>) -> !fir.ref<i32>
  %3 = fir.shape %c1024 : (index) -> !fir.shape<1>
  %4 = fir.declare %arg3(%3) {uniq_name = "_QFFrun_benchmarkEz"} : (!fir.ref<!fir.array<1024xf64>>, !fir.shape<1>) -> !fir.ref<!fir.array<1024xf64>>
  %5 = fir.declare %arg4(%3) {uniq_name = "_QFFrun_benchmarkEx"} : (!fir.ref<!fir.array<1024xf64>>, !fir.shape<1>) -> !fir.ref<!fir.array<1024xf64>>
  %6 = fir.declare %arg5(%3) {uniq_name = "_QFFrun_benchmarkEy"} : (!fir.ref<!fir.array<1024xf64>>, !fir.shape<1>) -> !fir.ref<!fir.array<1024xf64>>
  %7 = fir.declare %arg6 {uniq_name = "_QFFrun_benchmarkEa"} : (!fir.ref<f64>) -> !fir.ref<f64>
  %8 = fir.convert %0 : (!fir.ref<i32>) -> memref<i32>
  %9 = fir.convert %7 : (!fir.ref<f64>) -> memref<f64>
  %10 = fir.convert %5 : (!fir.ref<!fir.array<1024xf64>>) -> memref<1024xf64>
  %11 = fir.convert %6 : (!fir.ref<!fir.array<1024xf64>>) -> memref<1024xf64>
  %12 = fir.convert %4 : (!fir.ref<!fir.array<1024xf64>>) -> memref<1024xf64>
  %13 = fir.convert %c3 : (index) -> i32
  affine.for %arg10 = %c1 to affine_map<()[s0] -> (s0 + 1)>()[%c1024] step 3 {
    %15 = arith.index_cast %arg10 : index to i32
    memref.store %15, %8[] : memref<i32>
    %16 = memref.load %9[] : memref<f64>
    %17 = memref.load %8[] : memref<i32>
    %18 = arith.extsi %17 : i32 to i64
    %19 = arith.index_cast %18 : i64 to index
    %20 = arith.subi %19, %c1 : index
    %21 = memref.load %10[%20] : memref<1024xf64>
    %22 = arith.mulf %16, %21 fastmath<contract> : f64
    %23 = memref.load %11[%20] : memref<1024xf64>
    %24 = arith.addf %22, %23 fastmath<contract> : f64
    memref.store %24, %12[%20] : memref<1024xf64>
    %25 = memref.load %8[] : memref<i32>
    %26 = arith.addi %25, %13 overflow<nsw> : i32
    memref.store %26, %8[] : memref<i32>
  }
  %14 = memref.load %8[] : memref<i32>
  memref.store %14, %8[] : memref<i32>
  omp.terminator
}

// -----// IR Dump After AffineLoopNormalize (affine-loop-normalize) //----- //
func.func @kernel(%arg0: !fir.ref<i32> {jit.arg_type = 1 : ui32, jit.literal_val = 0 : i64}, %arg1: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg2: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 3 : i64}, %arg3: !fir.ref<!fir.array<1024xf64>> {jit.arg_type = 1 : ui32}, %arg4: !fir.ref<!fir.array<1024xf64>> {jit.arg_type = 1 : ui32}, %arg5: !fir.ref<!fir.array<1024xf64>> {jit.arg_type = 1 : ui32}, %arg6: !fir.ref<f64> {jit.arg_type = 1 : ui32, jit.literal_val = 4619567317775286272 : i64}, %arg7: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg8: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg9: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}) {
  %c1024 = arith.constant 1024 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %0 = fir.declare %arg0 {uniq_name = "_QFFrun_benchmarkEi"} : (!fir.ref<i32>) -> !fir.ref<i32>
  %1 = fir.declare %arg1 {uniq_name = "_QFFrun_benchmarkEn"} : (!fir.ref<i32>) -> !fir.ref<i32>
  %2 = fir.declare %arg2 {uniq_name = "_QFFrun_benchmarkEstride"} : (!fir.ref<i32>) -> !fir.ref<i32>
  %3 = fir.shape %c1024 : (index) -> !fir.shape<1>
  %4 = fir.declare %arg3(%3) {uniq_name = "_QFFrun_benchmarkEz"} : (!fir.ref<!fir.array<1024xf64>>, !fir.shape<1>) -> !fir.ref<!fir.array<1024xf64>>
  %5 = fir.declare %arg4(%3) {uniq_name = "_QFFrun_benchmarkEx"} : (!fir.ref<!fir.array<1024xf64>>, !fir.shape<1>) -> !fir.ref<!fir.array<1024xf64>>
  %6 = fir.declare %arg5(%3) {uniq_name = "_QFFrun_benchmarkEy"} : (!fir.ref<!fir.array<1024xf64>>, !fir.shape<1>) -> !fir.ref<!fir.array<1024xf64>>
  %7 = fir.declare %arg6 {uniq_name = "_QFFrun_benchmarkEa"} : (!fir.ref<f64>) -> !fir.ref<f64>
  %8 = fir.convert %0 : (!fir.ref<i32>) -> memref<i32>
  %9 = fir.convert %7 : (!fir.ref<f64>) -> memref<f64>
  %10 = fir.convert %5 : (!fir.ref<!fir.array<1024xf64>>) -> memref<1024xf64>
  %11 = fir.convert %6 : (!fir.ref<!fir.array<1024xf64>>) -> memref<1024xf64>
  %12 = fir.convert %4 : (!fir.ref<!fir.array<1024xf64>>) -> memref<1024xf64>
  %13 = fir.convert %c3 : (index) -> i32
  affine.for %arg10 = 0 to 342 {
    %15 = affine.apply affine_map<(d0) -> (d0 * 3 + 1)>(%arg10)
    %16 = arith.index_cast %15 : index to i32
    memref.store %16, %8[] : memref<i32>
    %17 = memref.load %9[] : memref<f64>
    %18 = memref.load %8[] : memref<i32>
    %19 = arith.extsi %18 : i32 to i64
    %20 = arith.index_cast %19 : i64 to index
    %21 = arith.subi %20, %c1 : index
    %22 = memref.load %10[%21] : memref<1024xf64>
    %23 = arith.mulf %17, %22 fastmath<contract> : f64
    %24 = memref.load %11[%21] : memref<1024xf64>
    %25 = arith.addf %23, %24 fastmath<contract> : f64
    memref.store %25, %12[%21] : memref<1024xf64>
    %26 = memref.load %8[] : memref<i32>
    %27 = arith.addi %26, %13 overflow<nsw> : i32
    memref.store %27, %8[] : memref<i32>
  }
  %14 = memref.load %8[] : memref<i32>
  memref.store %14, %8[] : memref<i32>
  omp.terminator
}

// -----// IR Dump After CanonicalizerPass (canonicalize) //----- //
#map = affine_map<(d0) -> (d0 * 3 + 1)>
module {
  func.func @kernel(%arg0: !fir.ref<i32> {jit.arg_type = 1 : ui32, jit.literal_val = 0 : i64}, %arg1: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg2: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 3 : i64}, %arg3: !fir.ref<!fir.array<1024xf64>> {jit.arg_type = 1 : ui32}, %arg4: !fir.ref<!fir.array<1024xf64>> {jit.arg_type = 1 : ui32}, %arg5: !fir.ref<!fir.array<1024xf64>> {jit.arg_type = 1 : ui32}, %arg6: !fir.ref<f64> {jit.arg_type = 1 : ui32, jit.literal_val = 4619567317775286272 : i64}, %arg7: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg8: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg9: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}) {
    %c1024 = arith.constant 1024 : index
    %c1 = arith.constant 1 : index
    %c3 = arith.constant 3 : index
    %0 = fir.declare %arg0 {uniq_name = "_QFFrun_benchmarkEi"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %1 = fir.declare %arg1 {uniq_name = "_QFFrun_benchmarkEn"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %2 = fir.declare %arg2 {uniq_name = "_QFFrun_benchmarkEstride"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %3 = fir.shape %c1024 : (index) -> !fir.shape<1>
    %4 = fir.declare %arg3(%3) {uniq_name = "_QFFrun_benchmarkEz"} : (!fir.ref<!fir.array<1024xf64>>, !fir.shape<1>) -> !fir.ref<!fir.array<1024xf64>>
    %5 = fir.declare %arg4(%3) {uniq_name = "_QFFrun_benchmarkEx"} : (!fir.ref<!fir.array<1024xf64>>, !fir.shape<1>) -> !fir.ref<!fir.array<1024xf64>>
    %6 = fir.declare %arg5(%3) {uniq_name = "_QFFrun_benchmarkEy"} : (!fir.ref<!fir.array<1024xf64>>, !fir.shape<1>) -> !fir.ref<!fir.array<1024xf64>>
    %7 = fir.declare %arg6 {uniq_name = "_QFFrun_benchmarkEa"} : (!fir.ref<f64>) -> !fir.ref<f64>
    %8 = fir.convert %0 : (!fir.ref<i32>) -> memref<i32>
    %9 = fir.convert %7 : (!fir.ref<f64>) -> memref<f64>
    %10 = fir.convert %5 : (!fir.ref<!fir.array<1024xf64>>) -> memref<1024xf64>
    %11 = fir.convert %6 : (!fir.ref<!fir.array<1024xf64>>) -> memref<1024xf64>
    %12 = fir.convert %4 : (!fir.ref<!fir.array<1024xf64>>) -> memref<1024xf64>
    %13 = fir.convert %c3 : (index) -> i32
    affine.for %arg10 = 0 to 342 {
      %15 = affine.apply #map(%arg10)
      %16 = arith.index_cast %15 : index to i32
      memref.store %16, %8[] : memref<i32>
      %17 = memref.load %9[] : memref<f64>
      %18 = memref.load %8[] : memref<i32>
      %19 = arith.index_cast %18 : i32 to index
      %20 = arith.subi %19, %c1 : index
      %21 = memref.load %10[%20] : memref<1024xf64>
      %22 = arith.mulf %17, %21 fastmath<contract> : f64
      %23 = memref.load %11[%20] : memref<1024xf64>
      %24 = arith.addf %22, %23 fastmath<contract> : f64
      memref.store %24, %12[%20] : memref<1024xf64>
      %25 = memref.load %8[] : memref<i32>
      %26 = arith.addi %25, %13 overflow<nsw> : i32
      memref.store %26, %8[] : memref<i32>
    }
    %14 = memref.load %8[] : memref<i32>
    memref.store %14, %8[] : memref<i32>
    omp.terminator
  }
}


[DEBUG]compare reading and storeOp
[DEBUG] memref.store %14, %8[] : memref<i32> 
[DEBUG] %14 = memref.load %8[] : memref<i32> 
[DEBUG]compare reading and storeOp
[DEBUG] memref.store %25, %8[] : memref<i32> 
[DEBUG] %25 = arith.addi %24, %13 overflow<nsw> : i32 
[DEBUG]Unexpected readOp: 
[DEBUG] %25 = arith.addi %24, %13 overflow<nsw> : i32 
[DEBUG]compare reading and storeOp
[DEBUG] memref.store %23, %12[%19] : memref<1024xf64> 
[DEBUG] %23 = arith.addf %21, %22 fastmath<contract> : f64 
[DEBUG]Unexpected readOp: 
[DEBUG] %23 = arith.addf %21, %22 fastmath<contract> : f64 
[DEBUG]Dependency chain relies on LoadOp!
[DEBUG]compare reading and storeOp
[DEBUG] memref.store %16, %8[] : memref<i32> 
[DEBUG] %16 = arith.index_cast %15 : index to i32 
[DEBUG]Unexpected readOp: 
[DEBUG] %16 = arith.index_cast %15 : index to i32 
[DEBUG]compare reading and storeOp
[DEBUG] memref.store %23, %8[] : memref<i32> 
[DEBUG] %23 = arith.addi %16, %13 overflow<nsw> : i32 
[DEBUG]Unexpected readOp: 
[DEBUG] %23 = arith.addi %16, %13 overflow<nsw> : i32 
[DEBUG]compare reading and storeOp
[DEBUG] memref.store %22, %8[] : memref<i32> 
[DEBUG] %22 = arith.addi %21, %13 overflow<nsw> : i32 
[DEBUG]Unexpected readOp: 
[DEBUG] %22 = arith.addi %21, %13 overflow<nsw> : i32 
[DEBUG]compare reading and storeOp
[DEBUG] memref.store %23, %12[%19] : memref<1024xf64> 
[DEBUG] %23 = arith.addf %21, %22 fastmath<contract> : f64 
[DEBUG]Unexpected readOp: 
[DEBUG] %23 = arith.addf %21, %22 fastmath<contract> : f64 
[DEBUG]Dependency chain relies on LoadOp!
[DEBUG]compare reading and storeOp
[DEBUG] memref.store %15, %8[] : memref<i32> 
[DEBUG] %15 = arith.addi %13, %c1024_i32 overflow<nsw> : i32 
[DEBUG]Unexpected readOp: 
[DEBUG] %15 = arith.addi %13, %c1024_i32 overflow<nsw> : i32 
[DEBUG]compare reading and storeOp
[DEBUG] memref.store %23, %12[%19] : memref<1024xf64> 
[DEBUG] %23 = arith.addf %21, %22 fastmath<contract> : f64 
[DEBUG]Unexpected readOp: 
[DEBUG] %23 = arith.addf %21, %22 fastmath<contract> : f64 
[DEBUG]Dependency chain relies on LoadOp!
// -----// IR Dump After (anonymous namespace)::OptimizeMemOpsPass (jforce-optimize-mem-ops) //----- //
func.func @kernel(%arg0: !fir.ref<i32> {jit.arg_type = 1 : ui32, jit.literal_val = 0 : i64}, %arg1: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg2: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 3 : i64}, %arg3: !fir.ref<!fir.array<1024xf64>> {jit.arg_type = 1 : ui32}, %arg4: !fir.ref<!fir.array<1024xf64>> {jit.arg_type = 1 : ui32}, %arg5: !fir.ref<!fir.array<1024xf64>> {jit.arg_type = 1 : ui32}, %arg6: !fir.ref<f64> {jit.arg_type = 1 : ui32, jit.literal_val = 4619567317775286272 : i64}, %arg7: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg8: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg9: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}) {
  %c1024_i32 = arith.constant 1024 : i32
  %c1024 = arith.constant 1024 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %0 = fir.declare %arg0 {uniq_name = "_QFFrun_benchmarkEi"} : (!fir.ref<i32>) -> !fir.ref<i32>
  %1 = fir.declare %arg1 {uniq_name = "_QFFrun_benchmarkEn"} : (!fir.ref<i32>) -> !fir.ref<i32>
  %2 = fir.declare %arg2 {uniq_name = "_QFFrun_benchmarkEstride"} : (!fir.ref<i32>) -> !fir.ref<i32>
  %3 = fir.shape %c1024 : (index) -> !fir.shape<1>
  %4 = fir.declare %arg3(%3) {uniq_name = "_QFFrun_benchmarkEz"} : (!fir.ref<!fir.array<1024xf64>>, !fir.shape<1>) -> !fir.ref<!fir.array<1024xf64>>
  %5 = fir.declare %arg4(%3) {uniq_name = "_QFFrun_benchmarkEx"} : (!fir.ref<!fir.array<1024xf64>>, !fir.shape<1>) -> !fir.ref<!fir.array<1024xf64>>
  %6 = fir.declare %arg5(%3) {uniq_name = "_QFFrun_benchmarkEy"} : (!fir.ref<!fir.array<1024xf64>>, !fir.shape<1>) -> !fir.ref<!fir.array<1024xf64>>
  %7 = fir.declare %arg6 {uniq_name = "_QFFrun_benchmarkEa"} : (!fir.ref<f64>) -> !fir.ref<f64>
  %8 = fir.convert %0 : (!fir.ref<i32>) -> memref<i32>
  %9 = fir.convert %7 : (!fir.ref<f64>) -> memref<f64>
  %10 = fir.convert %5 : (!fir.ref<!fir.array<1024xf64>>) -> memref<1024xf64>
  %11 = fir.convert %6 : (!fir.ref<!fir.array<1024xf64>>) -> memref<1024xf64>
  %12 = fir.convert %4 : (!fir.ref<!fir.array<1024xf64>>) -> memref<1024xf64>
  %13 = fir.convert %c3 : (index) -> i32
  %14 = memref.load %9[] : memref<f64>
  affine.for %arg10 = 0 to 342 {
    %16 = affine.apply affine_map<(d0) -> (d0 * 3 + 1)>(%arg10)
    %17 = arith.index_cast %16 : index to i32
    %18 = arith.index_cast %17 : i32 to index
    %19 = arith.subi %18, %c1 : index
    %20 = memref.load %10[%19] : memref<1024xf64>
    %21 = arith.mulf %14, %20 fastmath<contract> : f64
    %22 = memref.load %11[%19] : memref<1024xf64>
    %23 = arith.addf %21, %22 fastmath<contract> : f64
    memref.store %23, %12[%19] : memref<1024xf64>
  }
  %15 = arith.addi %13, %c1024_i32 overflow<nsw> : i32
  memref.store %15, %8[] : memref<i32>
  omp.terminator
}

// -----// IR Dump After CanonicalizerPass (canonicalize) //----- //
#map = affine_map<(d0) -> (d0 * 3 + 1)>
module {
  func.func @kernel(%arg0: !fir.ref<i32> {jit.arg_type = 1 : ui32, jit.literal_val = 0 : i64}, %arg1: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg2: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 3 : i64}, %arg3: !fir.ref<!fir.array<1024xf64>> {jit.arg_type = 1 : ui32}, %arg4: !fir.ref<!fir.array<1024xf64>> {jit.arg_type = 1 : ui32}, %arg5: !fir.ref<!fir.array<1024xf64>> {jit.arg_type = 1 : ui32}, %arg6: !fir.ref<f64> {jit.arg_type = 1 : ui32, jit.literal_val = 4619567317775286272 : i64}, %arg7: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg8: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg9: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}) {
    %c1024_i32 = arith.constant 1024 : i32
    %c1024 = arith.constant 1024 : index
    %c1 = arith.constant 1 : index
    %c3 = arith.constant 3 : index
    %0 = fir.declare %arg0 {uniq_name = "_QFFrun_benchmarkEi"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %1 = fir.declare %arg1 {uniq_name = "_QFFrun_benchmarkEn"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %2 = fir.declare %arg2 {uniq_name = "_QFFrun_benchmarkEstride"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %3 = fir.shape %c1024 : (index) -> !fir.shape<1>
    %4 = fir.declare %arg3(%3) {uniq_name = "_QFFrun_benchmarkEz"} : (!fir.ref<!fir.array<1024xf64>>, !fir.shape<1>) -> !fir.ref<!fir.array<1024xf64>>
    %5 = fir.declare %arg4(%3) {uniq_name = "_QFFrun_benchmarkEx"} : (!fir.ref<!fir.array<1024xf64>>, !fir.shape<1>) -> !fir.ref<!fir.array<1024xf64>>
    %6 = fir.declare %arg5(%3) {uniq_name = "_QFFrun_benchmarkEy"} : (!fir.ref<!fir.array<1024xf64>>, !fir.shape<1>) -> !fir.ref<!fir.array<1024xf64>>
    %7 = fir.declare %arg6 {uniq_name = "_QFFrun_benchmarkEa"} : (!fir.ref<f64>) -> !fir.ref<f64>
    %8 = fir.convert %0 : (!fir.ref<i32>) -> memref<i32>
    %9 = fir.convert %7 : (!fir.ref<f64>) -> memref<f64>
    %10 = fir.convert %5 : (!fir.ref<!fir.array<1024xf64>>) -> memref<1024xf64>
    %11 = fir.convert %6 : (!fir.ref<!fir.array<1024xf64>>) -> memref<1024xf64>
    %12 = fir.convert %4 : (!fir.ref<!fir.array<1024xf64>>) -> memref<1024xf64>
    %13 = fir.convert %c3 : (index) -> i32
    %14 = memref.load %9[] : memref<f64>
    affine.for %arg10 = 0 to 342 {
      %16 = affine.apply #map(%arg10)
      %17 = arith.index_cast %16 : index to i32
      %18 = arith.index_cast %17 : i32 to index
      %19 = arith.subi %18, %c1 : index
      %20 = memref.load %10[%19] : memref<1024xf64>
      %21 = arith.mulf %14, %20 fastmath<contract> : f64
      %22 = memref.load %11[%19] : memref<1024xf64>
      %23 = arith.addf %21, %22 fastmath<contract> : f64
      memref.store %23, %12[%19] : memref<1024xf64>
    }
    %15 = arith.addi %13, %c1024_i32 overflow<nsw> : i32
    memref.store %15, %8[] : memref<i32>
    omp.terminator
  }
}


func.func @kernel(%arg0: !fir.ref<i32> {jit.arg_type = 1 : ui32, jit.literal_val = 0 : i64}, %arg1: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg2: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 3 : i64}, %arg3: !fir.ref<!fir.array<1024xf64>> {jit.arg_type = 1 : ui32}, %arg4: !fir.ref<!fir.array<1024xf64>> {jit.arg_type = 1 : ui32}, %arg5: !fir.ref<!fir.array<1024xf64>> {jit.arg_type = 1 : ui32}, %arg6: !fir.ref<f64> {jit.arg_type = 1 : ui32, jit.literal_val = 4619567317775286272 : i64}, %arg7: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg8: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg9: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}) {
  %c1024_i32 = arith.constant 1024 : i32
  %c1024 = arith.constant 1024 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %0 = fir.declare %arg0 {uniq_name = "_QFFrun_benchmarkEi"} : (!fir.ref<i32>) -> !fir.ref<i32>
  %1 = fir.declare %arg1 {uniq_name = "_QFFrun_benchmarkEn"} : (!fir.ref<i32>) -> !fir.ref<i32>
  %2 = fir.declare %arg2 {uniq_name = "_QFFrun_benchmarkEstride"} : (!fir.ref<i32>) -> !fir.ref<i32>
  %3 = fir.shape %c1024 : (index) -> !fir.shape<1>
  %4 = fir.declare %arg3(%3) {uniq_name = "_QFFrun_benchmarkEz"} : (!fir.ref<!fir.array<1024xf64>>, !fir.shape<1>) -> !fir.ref<!fir.array<1024xf64>>
  %5 = fir.declare %arg4(%3) {uniq_name = "_QFFrun_benchmarkEx"} : (!fir.ref<!fir.array<1024xf64>>, !fir.shape<1>) -> !fir.ref<!fir.array<1024xf64>>
  %6 = fir.declare %arg5(%3) {uniq_name = "_QFFrun_benchmarkEy"} : (!fir.ref<!fir.array<1024xf64>>, !fir.shape<1>) -> !fir.ref<!fir.array<1024xf64>>
  %7 = fir.declare %arg6 {uniq_name = "_QFFrun_benchmarkEa"} : (!fir.ref<f64>) -> !fir.ref<f64>
  %8 = fir.convert %0 : (!fir.ref<i32>) -> memref<i32>
  %9 = fir.convert %7 : (!fir.ref<f64>) -> memref<f64>
  %10 = fir.convert %5 : (!fir.ref<!fir.array<1024xf64>>) -> memref<1024xf64>
  %11 = fir.convert %6 : (!fir.ref<!fir.array<1024xf64>>) -> memref<1024xf64>
  %12 = fir.convert %4 : (!fir.ref<!fir.array<1024xf64>>) -> memref<1024xf64>
  %13 = fir.convert %c3 : (index) -> i32
  %14 = memref.load %9[] : memref<f64>
  affine.for %arg10 = 0 to 342 {
    %16 = affine.apply affine_map<(d0) -> (d0 * 3 + 1)>(%arg10)
    %17 = arith.index_cast %16 : index to i32
    %18 = arith.index_cast %17 : i32 to index
    %19 = arith.subi %18, %c1 : index
    %20 = memref.load %10[%19] : memref<1024xf64>
    %21 = arith.mulf %14, %20 fastmath<contract> : f64
    %22 = memref.load %11[%19] : memref<1024xf64>
    %23 = arith.addf %21, %22 fastmath<contract> : f64
    memref.store %23, %12[%19] : memref<1024xf64>
  }
  %15 = arith.addi %13, %c1024_i32 overflow<nsw> : i32
  memref.store %15, %8[] : memref<i32>
  omp.terminator
}
// -----// IR Dump After (anonymous namespace)::LoopSinkingPass (jforce-loop-sink) //----- //
func.func @kernel(%arg0: !fir.ref<i32> {jit.arg_type = 1 : ui32, jit.literal_val = 0 : i64}, %arg1: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg2: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 3 : i64}, %arg3: !fir.ref<!fir.array<1024xf64>> {jit.arg_type = 1 : ui32}, %arg4: !fir.ref<!fir.array<1024xf64>> {jit.arg_type = 1 : ui32}, %arg5: !fir.ref<!fir.array<1024xf64>> {jit.arg_type = 1 : ui32}, %arg6: !fir.ref<f64> {jit.arg_type = 1 : ui32, jit.literal_val = 4619567317775286272 : i64}, %arg7: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg8: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg9: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}) {
  %c1024_i32 = arith.constant 1024 : i32
  %c1024 = arith.constant 1024 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %0 = fir.declare %arg0 {uniq_name = "_QFFrun_benchmarkEi"} : (!fir.ref<i32>) -> !fir.ref<i32>
  %1 = fir.declare %arg1 {uniq_name = "_QFFrun_benchmarkEn"} : (!fir.ref<i32>) -> !fir.ref<i32>
  %2 = fir.declare %arg2 {uniq_name = "_QFFrun_benchmarkEstride"} : (!fir.ref<i32>) -> !fir.ref<i32>
  %3 = fir.shape %c1024 : (index) -> !fir.shape<1>
  %4 = fir.declare %arg3(%3) {uniq_name = "_QFFrun_benchmarkEz"} : (!fir.ref<!fir.array<1024xf64>>, !fir.shape<1>) -> !fir.ref<!fir.array<1024xf64>>
  %5 = fir.declare %arg4(%3) {uniq_name = "_QFFrun_benchmarkEx"} : (!fir.ref<!fir.array<1024xf64>>, !fir.shape<1>) -> !fir.ref<!fir.array<1024xf64>>
  %6 = fir.declare %arg5(%3) {uniq_name = "_QFFrun_benchmarkEy"} : (!fir.ref<!fir.array<1024xf64>>, !fir.shape<1>) -> !fir.ref<!fir.array<1024xf64>>
  %7 = fir.declare %arg6 {uniq_name = "_QFFrun_benchmarkEa"} : (!fir.ref<f64>) -> !fir.ref<f64>
  %8 = fir.convert %0 : (!fir.ref<i32>) -> memref<i32>
  %9 = fir.convert %7 : (!fir.ref<f64>) -> memref<f64>
  %10 = fir.convert %5 : (!fir.ref<!fir.array<1024xf64>>) -> memref<1024xf64>
  %11 = fir.convert %6 : (!fir.ref<!fir.array<1024xf64>>) -> memref<1024xf64>
  %12 = fir.convert %4 : (!fir.ref<!fir.array<1024xf64>>) -> memref<1024xf64>
  %13 = fir.convert %c3 : (index) -> i32
  %14 = memref.load %9[] : memref<f64>
  affine.for %arg10 = 0 to 342 {
    %16 = affine.apply affine_map<(d0) -> (d0 * 3 + 1)>(%arg10)
    %17 = arith.index_cast %16 : index to i32
    %18 = arith.index_cast %17 : i32 to index
    %19 = arith.subi %18, %c1 : index
    %20 = memref.load %10[%19] : memref<1024xf64>
    %21 = arith.mulf %14, %20 fastmath<contract> : f64
    %22 = memref.load %11[%19] : memref<1024xf64>
    %23 = arith.addf %21, %22 fastmath<contract> : f64
    memref.store %23, %12[%19] : memref<1024xf64>
  }
  %15 = arith.addi %13, %c1024_i32 overflow<nsw> : i32
  memref.store %15, %8[] : memref<i32>
  omp.terminator
}

// -----// IR Dump After (anonymous namespace)::AffineCFGPass (enzyme-affinecfg) //----- //
module {
  func.func @kernel(%arg0: !fir.ref<i32> {jit.arg_type = 1 : ui32, jit.literal_val = 0 : i64}, %arg1: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg2: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 3 : i64}, %arg3: !fir.ref<!fir.array<1024xf64>> {jit.arg_type = 1 : ui32}, %arg4: !fir.ref<!fir.array<1024xf64>> {jit.arg_type = 1 : ui32}, %arg5: !fir.ref<!fir.array<1024xf64>> {jit.arg_type = 1 : ui32}, %arg6: !fir.ref<f64> {jit.arg_type = 1 : ui32, jit.literal_val = 4619567317775286272 : i64}, %arg7: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg8: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg9: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}) {
    %c1024_i32 = arith.constant 1024 : i32
    %c1024 = arith.constant 1024 : index
    %c3 = arith.constant 3 : index
    %0 = fir.declare %arg0 {uniq_name = "_QFFrun_benchmarkEi"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %1 = fir.declare %arg1 {uniq_name = "_QFFrun_benchmarkEn"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %2 = fir.declare %arg2 {uniq_name = "_QFFrun_benchmarkEstride"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %3 = fir.shape %c1024 : (index) -> !fir.shape<1>
    %4 = fir.declare %arg3(%3) {uniq_name = "_QFFrun_benchmarkEz"} : (!fir.ref<!fir.array<1024xf64>>, !fir.shape<1>) -> !fir.ref<!fir.array<1024xf64>>
    %5 = fir.declare %arg4(%3) {uniq_name = "_QFFrun_benchmarkEx"} : (!fir.ref<!fir.array<1024xf64>>, !fir.shape<1>) -> !fir.ref<!fir.array<1024xf64>>
    %6 = fir.declare %arg5(%3) {uniq_name = "_QFFrun_benchmarkEy"} : (!fir.ref<!fir.array<1024xf64>>, !fir.shape<1>) -> !fir.ref<!fir.array<1024xf64>>
    %7 = fir.declare %arg6 {uniq_name = "_QFFrun_benchmarkEa"} : (!fir.ref<f64>) -> !fir.ref<f64>
    %8 = fir.convert %0 : (!fir.ref<i32>) -> memref<i32>
    %9 = fir.convert %7 : (!fir.ref<f64>) -> memref<f64>
    %10 = fir.convert %5 : (!fir.ref<!fir.array<1024xf64>>) -> memref<1024xf64>
    %11 = fir.convert %6 : (!fir.ref<!fir.array<1024xf64>>) -> memref<1024xf64>
    %12 = fir.convert %4 : (!fir.ref<!fir.array<1024xf64>>) -> memref<1024xf64>
    %13 = fir.convert %c3 : (index) -> i32
    %14 = affine.load %9[] : memref<f64>
    affine.parallel (%arg10) = (0) to (342) {
      %16 = affine.load %10[%arg10 * 3] : memref<1024xf64>
      %17 = arith.mulf %14, %16 fastmath<contract> : f64
      %18 = affine.load %11[%arg10 * 3] : memref<1024xf64>
      %19 = arith.addf %17, %18 fastmath<contract> : f64
      affine.store %19, %12[%arg10 * 3] : memref<1024xf64>
    }
    %15 = arith.addi %13, %c1024_i32 overflow<nsw> : i32
    affine.store %15, %8[] : memref<i32>
    omp.terminator
  }
}


module {
  func.func @outlined_affinefor_94167852449568(%arg0: memref<1024xf64>, %arg1: memref<f64>, %arg2: memref<1024xf64>, %arg3: memref<1024xf64>) {
    %0 = affine.load %arg1[] : memref<f64>
    affine.parallel (%arg4) = (0) to (342) {
      %1 = affine.load %arg0[%arg4 * 3] : memref<1024xf64>
      %2 = arith.mulf %0, %1 fastmath<contract> : f64
      %3 = affine.load %arg2[%arg4 * 3] : memref<1024xf64>
      %4 = arith.addf %2, %3 fastmath<contract> : f64
      affine.store %4, %arg3[%arg4 * 3] : memref<1024xf64>
    }
    return
  }
  func.func @kernel(%arg0: !fir.ref<i32> {jit.arg_type = 1 : ui32, jit.literal_val = 0 : i64}, %arg1: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg2: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 3 : i64}, %arg3: !fir.ref<!fir.array<1024xf64>> {jit.arg_type = 1 : ui32}, %arg4: !fir.ref<!fir.array<1024xf64>> {jit.arg_type = 1 : ui32}, %arg5: !fir.ref<!fir.array<1024xf64>> {jit.arg_type = 1 : ui32}, %arg6: !fir.ref<f64> {jit.arg_type = 1 : ui32, jit.literal_val = 4619567317775286272 : i64}, %arg7: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg8: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg9: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}) {
    %c1024_i32 = arith.constant 1024 : i32
    %c1024 = arith.constant 1024 : index
    %c3 = arith.constant 3 : index
    %0 = fir.declare %arg0 {uniq_name = "_QFFrun_benchmarkEi"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %1 = fir.declare %arg1 {uniq_name = "_QFFrun_benchmarkEn"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %2 = fir.declare %arg2 {uniq_name = "_QFFrun_benchmarkEstride"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %3 = fir.shape %c1024 : (index) -> !fir.shape<1>
    %4 = fir.declare %arg3(%3) {uniq_name = "_QFFrun_benchmarkEz"} : (!fir.ref<!fir.array<1024xf64>>, !fir.shape<1>) -> !fir.ref<!fir.array<1024xf64>>
    %5 = fir.declare %arg4(%3) {uniq_name = "_QFFrun_benchmarkEx"} : (!fir.ref<!fir.array<1024xf64>>, !fir.shape<1>) -> !fir.ref<!fir.array<1024xf64>>
    %6 = fir.declare %arg5(%3) {uniq_name = "_QFFrun_benchmarkEy"} : (!fir.ref<!fir.array<1024xf64>>, !fir.shape<1>) -> !fir.ref<!fir.array<1024xf64>>
    %7 = fir.declare %arg6 {uniq_name = "_QFFrun_benchmarkEa"} : (!fir.ref<f64>) -> !fir.ref<f64>
    %8 = fir.convert %0 : (!fir.ref<i32>) -> memref<i32>
    %9 = fir.convert %7 : (!fir.ref<f64>) -> memref<f64>
    %10 = fir.convert %5 : (!fir.ref<!fir.array<1024xf64>>) -> memref<1024xf64>
    %11 = fir.convert %6 : (!fir.ref<!fir.array<1024xf64>>) -> memref<1024xf64>
    %12 = fir.convert %4 : (!fir.ref<!fir.array<1024xf64>>) -> memref<1024xf64>
    %13 = fir.convert %c3 : (index) -> i32
    %14 = affine.load %9[] : memref<f64>
    %alloca = memref.alloca() : memref<f64>
    memref.store %14, %alloca[] : memref<f64>
    call @outlined_affinefor_94167852449568(%10, %alloca, %11, %12) : (memref<1024xf64>, memref<f64>, memref<1024xf64>, memref<1024xf64>) -> ()
    %15 = arith.addi %13, %c1024_i32 overflow<nsw> : i32
    affine.store %15, %8[] : memref<i32>
    omp.terminator
  }
}
// -----// IR Dump After (anonymous namespace)::OutlineAffinePass (jforce-outline-affine) //----- //
module {
  func.func @outlined_affinefor_94167852449568(%arg0: memref<1024xf64>, %arg1: memref<f64>, %arg2: memref<1024xf64>, %arg3: memref<1024xf64>) {
    %0 = affine.load %arg1[] : memref<f64>
    affine.parallel (%arg4) = (0) to (342) {
      %1 = affine.load %arg0[%arg4 * 3] : memref<1024xf64>
      %2 = arith.mulf %0, %1 fastmath<contract> : f64
      %3 = affine.load %arg2[%arg4 * 3] : memref<1024xf64>
      %4 = arith.addf %2, %3 fastmath<contract> : f64
      affine.store %4, %arg3[%arg4 * 3] : memref<1024xf64>
    }
    return
  }
  func.func @kernel(%arg0: !fir.ref<i32> {jit.arg_type = 1 : ui32, jit.literal_val = 0 : i64}, %arg1: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg2: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 3 : i64}, %arg3: !fir.ref<!fir.array<1024xf64>> {jit.arg_type = 1 : ui32}, %arg4: !fir.ref<!fir.array<1024xf64>> {jit.arg_type = 1 : ui32}, %arg5: !fir.ref<!fir.array<1024xf64>> {jit.arg_type = 1 : ui32}, %arg6: !fir.ref<f64> {jit.arg_type = 1 : ui32, jit.literal_val = 4619567317775286272 : i64}, %arg7: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg8: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg9: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}) {
    %c1024_i32 = arith.constant 1024 : i32
    %c1024 = arith.constant 1024 : index
    %c3 = arith.constant 3 : index
    %0 = fir.declare %arg0 {uniq_name = "_QFFrun_benchmarkEi"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %1 = fir.declare %arg1 {uniq_name = "_QFFrun_benchmarkEn"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %2 = fir.declare %arg2 {uniq_name = "_QFFrun_benchmarkEstride"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %3 = fir.shape %c1024 : (index) -> !fir.shape<1>
    %4 = fir.declare %arg3(%3) {uniq_name = "_QFFrun_benchmarkEz"} : (!fir.ref<!fir.array<1024xf64>>, !fir.shape<1>) -> !fir.ref<!fir.array<1024xf64>>
    %5 = fir.declare %arg4(%3) {uniq_name = "_QFFrun_benchmarkEx"} : (!fir.ref<!fir.array<1024xf64>>, !fir.shape<1>) -> !fir.ref<!fir.array<1024xf64>>
    %6 = fir.declare %arg5(%3) {uniq_name = "_QFFrun_benchmarkEy"} : (!fir.ref<!fir.array<1024xf64>>, !fir.shape<1>) -> !fir.ref<!fir.array<1024xf64>>
    %7 = fir.declare %arg6 {uniq_name = "_QFFrun_benchmarkEa"} : (!fir.ref<f64>) -> !fir.ref<f64>
    %8 = fir.convert %0 : (!fir.ref<i32>) -> memref<i32>
    %9 = fir.convert %7 : (!fir.ref<f64>) -> memref<f64>
    %10 = fir.convert %5 : (!fir.ref<!fir.array<1024xf64>>) -> memref<1024xf64>
    %11 = fir.convert %6 : (!fir.ref<!fir.array<1024xf64>>) -> memref<1024xf64>
    %12 = fir.convert %4 : (!fir.ref<!fir.array<1024xf64>>) -> memref<1024xf64>
    %13 = fir.convert %c3 : (index) -> i32
    %14 = affine.load %9[] : memref<f64>
    %alloca = memref.alloca() : memref<f64>
    memref.store %14, %alloca[] : memref<f64>
    call @outlined_affinefor_94167852449568(%10, %alloca, %11, %12) : (memref<1024xf64>, memref<f64>, memref<1024xf64>, memref<1024xf64>) -> ()
    %15 = arith.addi %13, %c1024_i32 overflow<nsw> : i32
    affine.store %15, %8[] : memref<i32>
    omp.terminator
  }
}


// -----// IR Dump After (anonymous namespace)::AffineToStableHLORaisingPass (enzyme-affine-to-stablehlo) //----- //
func.func @kernel(%arg0: !fir.ref<i32> {jit.arg_type = 1 : ui32, jit.literal_val = 0 : i64}, %arg1: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg2: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 3 : i64}, %arg3: !fir.ref<!fir.array<1024xf64>> {jit.arg_type = 1 : ui32}, %arg4: !fir.ref<!fir.array<1024xf64>> {jit.arg_type = 1 : ui32}, %arg5: !fir.ref<!fir.array<1024xf64>> {jit.arg_type = 1 : ui32}, %arg6: !fir.ref<f64> {jit.arg_type = 1 : ui32, jit.literal_val = 4619567317775286272 : i64}, %arg7: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg8: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg9: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}) {
  %c1024_i32 = arith.constant 1024 : i32
  %c1024 = arith.constant 1024 : index
  %c3 = arith.constant 3 : index
  %0 = fir.declare %arg0 {uniq_name = "_QFFrun_benchmarkEi"} : (!fir.ref<i32>) -> !fir.ref<i32>
  %1 = fir.declare %arg1 {uniq_name = "_QFFrun_benchmarkEn"} : (!fir.ref<i32>) -> !fir.ref<i32>
  %2 = fir.declare %arg2 {uniq_name = "_QFFrun_benchmarkEstride"} : (!fir.ref<i32>) -> !fir.ref<i32>
  %3 = fir.shape %c1024 : (index) -> !fir.shape<1>
  %4 = fir.declare %arg3(%3) {uniq_name = "_QFFrun_benchmarkEz"} : (!fir.ref<!fir.array<1024xf64>>, !fir.shape<1>) -> !fir.ref<!fir.array<1024xf64>>
  %5 = fir.declare %arg4(%3) {uniq_name = "_QFFrun_benchmarkEx"} : (!fir.ref<!fir.array<1024xf64>>, !fir.shape<1>) -> !fir.ref<!fir.array<1024xf64>>
  %6 = fir.declare %arg5(%3) {uniq_name = "_QFFrun_benchmarkEy"} : (!fir.ref<!fir.array<1024xf64>>, !fir.shape<1>) -> !fir.ref<!fir.array<1024xf64>>
  %7 = fir.declare %arg6 {uniq_name = "_QFFrun_benchmarkEa"} : (!fir.ref<f64>) -> !fir.ref<f64>
  %8 = fir.convert %0 : (!fir.ref<i32>) -> memref<i32>
  %9 = fir.convert %7 : (!fir.ref<f64>) -> memref<f64>
  %10 = fir.convert %5 : (!fir.ref<!fir.array<1024xf64>>) -> memref<1024xf64>
  %11 = fir.convert %6 : (!fir.ref<!fir.array<1024xf64>>) -> memref<1024xf64>
  %12 = fir.convert %4 : (!fir.ref<!fir.array<1024xf64>>) -> memref<1024xf64>
  %13 = fir.convert %c3 : (index) -> i32
  %14 = affine.load %9[] : memref<f64>
  %alloca = memref.alloca() : memref<f64>
  memref.store %14, %alloca[] : memref<f64>
  call @outlined_affinefor_94167852449568(%10, %alloca, %11, %12) : (memref<1024xf64>, memref<f64>, memref<1024xf64>, memref<1024xf64>) -> ()
  %15 = arith.addi %13, %c1024_i32 overflow<nsw> : i32
  affine.store %15, %8[] : memref<i32>
  omp.terminator
}

// -----// IR Dump After (anonymous namespace)::AffineToStableHLORaisingPass (enzyme-affine-to-stablehlo) //----- //
func.func @outlined_affinefor_94167852449568(%arg0: memref<1024xf64>, %arg1: memref<f64>, %arg2: memref<1024xf64>, %arg3: memref<1024xf64>) {
  %0 = affine.load %arg1[] : memref<f64>
  affine.parallel (%arg4) = (0) to (342) {
    %1 = affine.load %arg0[%arg4 * 3] : memref<1024xf64>
    %2 = arith.mulf %0, %1 fastmath<contract> : f64
    %3 = affine.load %arg2[%arg4 * 3] : memref<1024xf64>
    %4 = arith.addf %2, %3 fastmath<contract> : f64
    affine.store %4, %arg3[%arg4 * 3] : memref<1024xf64>
  }
  return
}

// -----// IR Dump After CanonicalizerPass (canonicalize) //----- //
module {
  func.func @outlined_affinefor_94167852449568(%arg0: memref<1024xf64>, %arg1: memref<f64>, %arg2: memref<1024xf64>, %arg3: memref<1024xf64>) {
    %0 = affine.load %arg1[] : memref<f64>
    affine.parallel (%arg4) = (0) to (342) {
      %1 = affine.load %arg0[%arg4 * 3] : memref<1024xf64>
      %2 = arith.mulf %0, %1 fastmath<contract> : f64
      %3 = affine.load %arg2[%arg4 * 3] : memref<1024xf64>
      %4 = arith.addf %2, %3 fastmath<contract> : f64
      affine.store %4, %arg3[%arg4 * 3] : memref<1024xf64>
    }
    return
  }
  func.func @kernel(%arg0: !fir.ref<i32> {jit.arg_type = 1 : ui32, jit.literal_val = 0 : i64}, %arg1: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg2: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 3 : i64}, %arg3: !fir.ref<!fir.array<1024xf64>> {jit.arg_type = 1 : ui32}, %arg4: !fir.ref<!fir.array<1024xf64>> {jit.arg_type = 1 : ui32}, %arg5: !fir.ref<!fir.array<1024xf64>> {jit.arg_type = 1 : ui32}, %arg6: !fir.ref<f64> {jit.arg_type = 1 : ui32, jit.literal_val = 4619567317775286272 : i64}, %arg7: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg8: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg9: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}) {
    %c1024_i32 = arith.constant 1024 : i32
    %c1024 = arith.constant 1024 : index
    %c3 = arith.constant 3 : index
    %0 = fir.declare %arg0 {uniq_name = "_QFFrun_benchmarkEi"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %1 = fir.declare %arg1 {uniq_name = "_QFFrun_benchmarkEn"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %2 = fir.declare %arg2 {uniq_name = "_QFFrun_benchmarkEstride"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %3 = fir.shape %c1024 : (index) -> !fir.shape<1>
    %4 = fir.declare %arg3(%3) {uniq_name = "_QFFrun_benchmarkEz"} : (!fir.ref<!fir.array<1024xf64>>, !fir.shape<1>) -> !fir.ref<!fir.array<1024xf64>>
    %5 = fir.declare %arg4(%3) {uniq_name = "_QFFrun_benchmarkEx"} : (!fir.ref<!fir.array<1024xf64>>, !fir.shape<1>) -> !fir.ref<!fir.array<1024xf64>>
    %6 = fir.declare %arg5(%3) {uniq_name = "_QFFrun_benchmarkEy"} : (!fir.ref<!fir.array<1024xf64>>, !fir.shape<1>) -> !fir.ref<!fir.array<1024xf64>>
    %7 = fir.declare %arg6 {uniq_name = "_QFFrun_benchmarkEa"} : (!fir.ref<f64>) -> !fir.ref<f64>
    %8 = fir.convert %0 : (!fir.ref<i32>) -> memref<i32>
    %9 = fir.convert %7 : (!fir.ref<f64>) -> memref<f64>
    %10 = fir.convert %5 : (!fir.ref<!fir.array<1024xf64>>) -> memref<1024xf64>
    %11 = fir.convert %6 : (!fir.ref<!fir.array<1024xf64>>) -> memref<1024xf64>
    %12 = fir.convert %4 : (!fir.ref<!fir.array<1024xf64>>) -> memref<1024xf64>
    %13 = fir.convert %c3 : (index) -> i32
    %14 = affine.load %9[] : memref<f64>
    %alloca = memref.alloca() : memref<f64>
    memref.store %14, %alloca[] : memref<f64>
    call @outlined_affinefor_94167852449568(%10, %alloca, %11, %12) : (memref<1024xf64>, memref<f64>, memref<1024xf64>, memref<1024xf64>) -> ()
    %15 = arith.addi %13, %c1024_i32 overflow<nsw> : i32
    affine.store %15, %8[] : memref<i32>
    omp.terminator
  }
  func.func private @outlined_affinefor_94167852449568_raised(%arg0: tensor<1024xf64>, %arg1: tensor<f64>, %arg2: tensor<1024xf64>, %arg3: tensor<1024xf64>) -> (tensor<1024xf64>, tensor<f64>, tensor<1024xf64>, tensor<1024xf64>) {
    %c = stablehlo.constant dense<3> : tensor<i64>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %c_0 = stablehlo.constant dense<1> : tensor<342xi64>
    %c_1 = stablehlo.constant dense<0> : tensor<342xi64>
    %0 = stablehlo.reshape %arg1 : (tensor<f64>) -> tensor<f64>
    %1 = stablehlo.iota dim = 0 : tensor<342xi64>
    %2 = stablehlo.add %1, %c_1 : tensor<342xi64>
    %3 = stablehlo.multiply %2, %c_0 : tensor<342xi64>
    %4 = stablehlo.pad %arg0, %cst, low = [0], high = [2], interior = [0] : (tensor<1024xf64>, tensor<f64>) -> tensor<1026xf64>
    %5 = stablehlo.slice %4 [0:1026:3] : (tensor<1026xf64>) -> tensor<342xf64>
    %6 = stablehlo.reshape %5 : (tensor<342xf64>) -> tensor<342xf64>
    %7 = stablehlo.broadcast_in_dim %0, dims = [] : (tensor<f64>) -> tensor<342xf64>
    %8 = arith.mulf %7, %6 fastmath<contract> : tensor<342xf64>
    %9 = stablehlo.pad %arg2, %cst, low = [0], high = [2], interior = [0] : (tensor<1024xf64>, tensor<f64>) -> tensor<1026xf64>
    %10 = stablehlo.slice %9 [0:1026:3] : (tensor<1026xf64>) -> tensor<342xf64>
    %11 = stablehlo.reshape %10 : (tensor<342xf64>) -> tensor<342xf64>
    %12 = arith.addf %8, %11 fastmath<contract> : tensor<342xf64>
    %13 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i64>) -> tensor<342xi64>
    %14 = stablehlo.multiply %3, %13 : tensor<342xi64>
    %15 = stablehlo.reshape %14 : (tensor<342xi64>) -> tensor<342x1xi64>
    %16 = stablehlo.broadcast_in_dim %12, dims = [0] : (tensor<342xf64>) -> tensor<342xf64>
    %17 = stablehlo.reshape %16 : (tensor<342xf64>) -> tensor<342xf64>
    %18 = "stablehlo.scatter"(%arg3, %15, %17) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = true}> ({
    ^bb0(%arg4: tensor<f64>, %arg5: tensor<f64>):
      stablehlo.return %arg5 : tensor<f64>
    }) : (tensor<1024xf64>, tensor<342x1xi64>, tensor<342xf64>) -> tensor<1024xf64>
    return %arg0, %arg1, %arg2, %18 : tensor<1024xf64>, tensor<f64>, tensor<1024xf64>, tensor<1024xf64>
  }
}


The outlinedFuncName is: outlined_affinefor_94167852449568_raised, which does not match the pattern. Skip this callee.
// -----// IR Dump After (anonymous namespace)::SwitchOutlineFuncPass (jforce-switch-outline-function) //----- //
module {
  func.func @kernel(%arg0: !fir.ref<i32> {jit.arg_type = 1 : ui32, jit.literal_val = 0 : i64}, %arg1: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg2: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 3 : i64}, %arg3: !fir.ref<!fir.array<1024xf64>> {jit.arg_type = 1 : ui32}, %arg4: !fir.ref<!fir.array<1024xf64>> {jit.arg_type = 1 : ui32}, %arg5: !fir.ref<!fir.array<1024xf64>> {jit.arg_type = 1 : ui32}, %arg6: !fir.ref<f64> {jit.arg_type = 1 : ui32, jit.literal_val = 4619567317775286272 : i64}, %arg7: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg8: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg9: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}) {
    %c1024_i32 = arith.constant 1024 : i32
    %c1024 = arith.constant 1024 : index
    %c3 = arith.constant 3 : index
    %0 = fir.declare %arg0 {uniq_name = "_QFFrun_benchmarkEi"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %1 = fir.declare %arg1 {uniq_name = "_QFFrun_benchmarkEn"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %2 = fir.declare %arg2 {uniq_name = "_QFFrun_benchmarkEstride"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %3 = fir.shape %c1024 : (index) -> !fir.shape<1>
    %4 = fir.declare %arg3(%3) {uniq_name = "_QFFrun_benchmarkEz"} : (!fir.ref<!fir.array<1024xf64>>, !fir.shape<1>) -> !fir.ref<!fir.array<1024xf64>>
    %5 = fir.declare %arg4(%3) {uniq_name = "_QFFrun_benchmarkEx"} : (!fir.ref<!fir.array<1024xf64>>, !fir.shape<1>) -> !fir.ref<!fir.array<1024xf64>>
    %6 = fir.declare %arg5(%3) {uniq_name = "_QFFrun_benchmarkEy"} : (!fir.ref<!fir.array<1024xf64>>, !fir.shape<1>) -> !fir.ref<!fir.array<1024xf64>>
    %7 = fir.declare %arg6 {uniq_name = "_QFFrun_benchmarkEa"} : (!fir.ref<f64>) -> !fir.ref<f64>
    %8 = fir.convert %0 : (!fir.ref<i32>) -> memref<i32>
    %9 = fir.convert %7 : (!fir.ref<f64>) -> memref<f64>
    %10 = fir.convert %5 : (!fir.ref<!fir.array<1024xf64>>) -> memref<1024xf64>
    %11 = fir.convert %6 : (!fir.ref<!fir.array<1024xf64>>) -> memref<1024xf64>
    %12 = fir.convert %4 : (!fir.ref<!fir.array<1024xf64>>) -> memref<1024xf64>
    %13 = fir.convert %c3 : (index) -> i32
    %14 = affine.load %9[] : memref<f64>
    %alloca = memref.alloca() : memref<f64>
    memref.store %14, %alloca[] : memref<f64>
    %15 = bufferization.to_tensor %10 : memref<1024xf64> to tensor<1024xf64>
    %16 = bufferization.to_tensor %alloca : memref<f64> to tensor<f64>
    %17 = bufferization.to_tensor %11 : memref<1024xf64> to tensor<1024xf64>
    %18 = bufferization.to_tensor %12 : memref<1024xf64> to tensor<1024xf64>
    %19:4 = call @outlined_affinefor_94167852449568_raised(%15, %16, %17, %18) : (tensor<1024xf64>, tensor<f64>, tensor<1024xf64>, tensor<1024xf64>) -> (tensor<1024xf64>, tensor<f64>, tensor<1024xf64>, tensor<1024xf64>)
    %20 = arith.addi %13, %c1024_i32 overflow<nsw> : i32
    affine.store %20, %8[] : memref<i32>
    omp.terminator
  }
  func.func private @outlined_affinefor_94167852449568_raised(%arg0: tensor<1024xf64>, %arg1: tensor<f64>, %arg2: tensor<1024xf64>, %arg3: tensor<1024xf64>) -> (tensor<1024xf64>, tensor<f64>, tensor<1024xf64>, tensor<1024xf64>) {
    %c = stablehlo.constant dense<3> : tensor<i64>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %c_0 = stablehlo.constant dense<1> : tensor<342xi64>
    %c_1 = stablehlo.constant dense<0> : tensor<342xi64>
    %0 = stablehlo.reshape %arg1 : (tensor<f64>) -> tensor<f64>
    %1 = stablehlo.iota dim = 0 : tensor<342xi64>
    %2 = stablehlo.add %1, %c_1 : tensor<342xi64>
    %3 = stablehlo.multiply %2, %c_0 : tensor<342xi64>
    %4 = stablehlo.pad %arg0, %cst, low = [0], high = [2], interior = [0] : (tensor<1024xf64>, tensor<f64>) -> tensor<1026xf64>
    %5 = stablehlo.slice %4 [0:1026:3] : (tensor<1026xf64>) -> tensor<342xf64>
    %6 = stablehlo.reshape %5 : (tensor<342xf64>) -> tensor<342xf64>
    %7 = stablehlo.broadcast_in_dim %0, dims = [] : (tensor<f64>) -> tensor<342xf64>
    %8 = arith.mulf %7, %6 fastmath<contract> : tensor<342xf64>
    %9 = stablehlo.pad %arg2, %cst, low = [0], high = [2], interior = [0] : (tensor<1024xf64>, tensor<f64>) -> tensor<1026xf64>
    %10 = stablehlo.slice %9 [0:1026:3] : (tensor<1026xf64>) -> tensor<342xf64>
    %11 = stablehlo.reshape %10 : (tensor<342xf64>) -> tensor<342xf64>
    %12 = arith.addf %8, %11 fastmath<contract> : tensor<342xf64>
    %13 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i64>) -> tensor<342xi64>
    %14 = stablehlo.multiply %3, %13 : tensor<342xi64>
    %15 = stablehlo.reshape %14 : (tensor<342xi64>) -> tensor<342x1xi64>
    %16 = stablehlo.broadcast_in_dim %12, dims = [0] : (tensor<342xf64>) -> tensor<342xf64>
    %17 = stablehlo.reshape %16 : (tensor<342xf64>) -> tensor<342xf64>
    %18 = "stablehlo.scatter"(%arg3, %15, %17) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = true}> ({
    ^bb0(%arg4: tensor<f64>, %arg5: tensor<f64>):
      stablehlo.return %arg5 : tensor<f64>
    }) : (tensor<1024xf64>, tensor<342x1xi64>, tensor<342xf64>) -> tensor<1024xf64>
    return %arg0, %arg1, %arg2, %18 : tensor<1024xf64>, tensor<f64>, tensor<1024xf64>, tensor<1024xf64>
  }
}


[DEBUG]Unhandled operation:
[DEBUG] %3 = fir.shape %c1024 : (index) -> !fir.shape<1> 
[DEBUG]Unhandled operation:
[DEBUG] omp.terminator 
// -----// IR Dump After (anonymous namespace)::WorkdistributeToStableHLOPass (jforce-translate) //----- //
module {
  func.func @main(%arg0: tensor<i32> {jit.literal_val = 0 : i64}, %arg1: tensor<i32> {jit.literal_val = 1024 : i64}, %arg2: tensor<i32> {jit.literal_val = 3 : i64}, %arg3: tensor<1024xf64>, %arg4: tensor<1024xf64>, %arg5: tensor<1024xf64>, %arg6: tensor<f64> {jit.literal_val = 4619567317775286272 : i64}, %arg7: tensor<i32> {jit.literal_val = 1024 : i64}, %arg8: tensor<i32> {jit.literal_val = 1024 : i64}, %arg9: tensor<i32> {jit.literal_val = 1024 : i64}) -> (tensor<i32>, tensor<i32>, tensor<i32>, tensor<1024xf64>, tensor<1024xf64>, tensor<1024xf64>, tensor<f64>, tensor<i32>, tensor<i32>, tensor<i32>) {
    %c = stablehlo.constant dense<1024> : tensor<i32>
    %c_0 = stablehlo.constant dense<1024> : tensor<i64>
    %c_1 = stablehlo.constant dense<3> : tensor<i64>
    %0 = stablehlo.convert %c_1 : (tensor<i64>) -> tensor<i32>
    %1:4 = call @outlined_affinefor_94167852449568_raised(%arg4, %arg6, %arg5, %arg3) : (tensor<1024xf64>, tensor<f64>, tensor<1024xf64>, tensor<1024xf64>) -> (tensor<1024xf64>, tensor<f64>, tensor<1024xf64>, tensor<1024xf64>)
    %2 = stablehlo.add %0, %c : tensor<i32>
    return %arg0, %arg1, %arg2, %1#3, %1#0, %1#2, %1#1, %arg7, %arg8, %arg9 : tensor<i32>, tensor<i32>, tensor<i32>, tensor<1024xf64>, tensor<1024xf64>, tensor<1024xf64>, tensor<f64>, tensor<i32>, tensor<i32>, tensor<i32>
  }
  func.func private @outlined_affinefor_94167852449568_raised(%arg0: tensor<1024xf64>, %arg1: tensor<f64>, %arg2: tensor<1024xf64>, %arg3: tensor<1024xf64>) -> (tensor<1024xf64>, tensor<f64>, tensor<1024xf64>, tensor<1024xf64>) {
    %c = stablehlo.constant dense<3> : tensor<i64>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %c_0 = stablehlo.constant dense<1> : tensor<342xi64>
    %c_1 = stablehlo.constant dense<0> : tensor<342xi64>
    %0 = stablehlo.reshape %arg1 : (tensor<f64>) -> tensor<f64>
    %1 = stablehlo.iota dim = 0 : tensor<342xi64>
    %2 = stablehlo.add %1, %c_1 : tensor<342xi64>
    %3 = stablehlo.multiply %2, %c_0 : tensor<342xi64>
    %4 = stablehlo.pad %arg0, %cst, low = [0], high = [2], interior = [0] : (tensor<1024xf64>, tensor<f64>) -> tensor<1026xf64>
    %5 = stablehlo.slice %4 [0:1026:3] : (tensor<1026xf64>) -> tensor<342xf64>
    %6 = stablehlo.reshape %5 : (tensor<342xf64>) -> tensor<342xf64>
    %7 = stablehlo.broadcast_in_dim %0, dims = [] : (tensor<f64>) -> tensor<342xf64>
    %8 = arith.mulf %7, %6 fastmath<contract> : tensor<342xf64>
    %9 = stablehlo.pad %arg2, %cst, low = [0], high = [2], interior = [0] : (tensor<1024xf64>, tensor<f64>) -> tensor<1026xf64>
    %10 = stablehlo.slice %9 [0:1026:3] : (tensor<1026xf64>) -> tensor<342xf64>
    %11 = stablehlo.reshape %10 : (tensor<342xf64>) -> tensor<342xf64>
    %12 = arith.addf %8, %11 fastmath<contract> : tensor<342xf64>
    %13 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i64>) -> tensor<342xi64>
    %14 = stablehlo.multiply %3, %13 : tensor<342xi64>
    %15 = stablehlo.reshape %14 : (tensor<342xi64>) -> tensor<342x1xi64>
    %16 = stablehlo.broadcast_in_dim %12, dims = [0] : (tensor<342xf64>) -> tensor<342xf64>
    %17 = stablehlo.reshape %16 : (tensor<342xf64>) -> tensor<342xf64>
    %18 = "stablehlo.scatter"(%arg3, %15, %17) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = true}> ({
    ^bb0(%arg4: tensor<f64>, %arg5: tensor<f64>):
      stablehlo.return %arg5 : tensor<f64>
    }) : (tensor<1024xf64>, tensor<342x1xi64>, tensor<342xf64>) -> tensor<1024xf64>
    return %arg0, %arg1, %arg2, %18 : tensor<1024xf64>, tensor<f64>, tensor<1024xf64>, tensor<1024xf64>
  }
}


// -----// IR Dump After CanonicalizerPass (canonicalize) //----- //
func.func private @outlined_affinefor_94167852449568_raised(%arg0: tensor<1024xf64>, %arg1: tensor<f64>, %arg2: tensor<1024xf64>, %arg3: tensor<1024xf64>) -> (tensor<1024xf64>, tensor<f64>, tensor<1024xf64>, tensor<1024xf64>) {
  %c = stablehlo.constant dense<3> : tensor<i64>
  %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
  %c_0 = stablehlo.constant dense<1> : tensor<342xi64>
  %c_1 = stablehlo.constant dense<0> : tensor<342xi64>
  %0 = stablehlo.reshape %arg1 : (tensor<f64>) -> tensor<f64>
  %1 = stablehlo.iota dim = 0 : tensor<342xi64>
  %2 = stablehlo.add %1, %c_1 : tensor<342xi64>
  %3 = stablehlo.multiply %2, %c_0 : tensor<342xi64>
  %4 = stablehlo.pad %arg0, %cst, low = [0], high = [2], interior = [0] : (tensor<1024xf64>, tensor<f64>) -> tensor<1026xf64>
  %5 = stablehlo.slice %4 [0:1026:3] : (tensor<1026xf64>) -> tensor<342xf64>
  %6 = stablehlo.reshape %5 : (tensor<342xf64>) -> tensor<342xf64>
  %7 = stablehlo.broadcast_in_dim %0, dims = [] : (tensor<f64>) -> tensor<342xf64>
  %8 = arith.mulf %7, %6 fastmath<contract> : tensor<342xf64>
  %9 = stablehlo.pad %arg2, %cst, low = [0], high = [2], interior = [0] : (tensor<1024xf64>, tensor<f64>) -> tensor<1026xf64>
  %10 = stablehlo.slice %9 [0:1026:3] : (tensor<1026xf64>) -> tensor<342xf64>
  %11 = stablehlo.reshape %10 : (tensor<342xf64>) -> tensor<342xf64>
  %12 = arith.addf %8, %11 fastmath<contract> : tensor<342xf64>
  %13 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i64>) -> tensor<342xi64>
  %14 = stablehlo.multiply %3, %13 : tensor<342xi64>
  %15 = stablehlo.reshape %14 : (tensor<342xi64>) -> tensor<342x1xi64>
  %16 = stablehlo.broadcast_in_dim %12, dims = [0] : (tensor<342xf64>) -> tensor<342xf64>
  %17 = stablehlo.reshape %16 : (tensor<342xf64>) -> tensor<342xf64>
  %18 = "stablehlo.scatter"(%arg3, %15, %17) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = true}> ({
  ^bb0(%arg4: tensor<f64>, %arg5: tensor<f64>):
    stablehlo.return %arg5 : tensor<f64>
  }) : (tensor<1024xf64>, tensor<342x1xi64>, tensor<342xf64>) -> tensor<1024xf64>
  return %arg0, %arg1, %arg2, %18 : tensor<1024xf64>, tensor<f64>, tensor<1024xf64>, tensor<1024xf64>
}

// -----// IR Dump After CanonicalizerPass (canonicalize) //----- //
func.func @main(%arg0: tensor<i32> {jit.literal_val = 0 : i64}, %arg1: tensor<i32> {jit.literal_val = 1024 : i64}, %arg2: tensor<i32> {jit.literal_val = 3 : i64}, %arg3: tensor<1024xf64>, %arg4: tensor<1024xf64>, %arg5: tensor<1024xf64>, %arg6: tensor<f64> {jit.literal_val = 4619567317775286272 : i64}, %arg7: tensor<i32> {jit.literal_val = 1024 : i64}, %arg8: tensor<i32> {jit.literal_val = 1024 : i64}, %arg9: tensor<i32> {jit.literal_val = 1024 : i64}) -> (tensor<i32>, tensor<i32>, tensor<i32>, tensor<1024xf64>, tensor<1024xf64>, tensor<1024xf64>, tensor<f64>, tensor<i32>, tensor<i32>, tensor<i32>) {
  %0:4 = call @outlined_affinefor_94167852449568_raised(%arg4, %arg6, %arg5, %arg3) : (tensor<1024xf64>, tensor<f64>, tensor<1024xf64>, tensor<1024xf64>) -> (tensor<1024xf64>, tensor<f64>, tensor<1024xf64>, tensor<1024xf64>)
  return %arg0, %arg1, %arg2, %0#3, %0#0, %0#2, %0#1, %arg7, %arg8, %arg9 : tensor<i32>, tensor<i32>, tensor<i32>, tensor<1024xf64>, tensor<1024xf64>, tensor<1024xf64>, tensor<f64>, tensor<i32>, tensor<i32>, tensor<i32>
}

// -----// IR Dump After CanonicalizerPass (canonicalize) //----- //
func.func @main(%arg0: tensor<i32> {jit.literal_val = 0 : i64}, %arg1: tensor<i32> {jit.literal_val = 1024 : i64}, %arg2: tensor<i32> {jit.literal_val = 3 : i64}, %arg3: tensor<1024xf64>, %arg4: tensor<1024xf64>, %arg5: tensor<1024xf64>, %arg6: tensor<f64> {jit.literal_val = 4619567317775286272 : i64}, %arg7: tensor<i32> {jit.literal_val = 1024 : i64}, %arg8: tensor<i32> {jit.literal_val = 1024 : i64}, %arg9: tensor<i32> {jit.literal_val = 1024 : i64}) -> (tensor<i32>, tensor<i32>, tensor<i32>, tensor<1024xf64>, tensor<1024xf64>, tensor<1024xf64>, tensor<f64>, tensor<i32>, tensor<i32>, tensor<i32>) {
  %c = stablehlo.constant dense<3> : tensor<i64>
  %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
  %c_0 = stablehlo.constant dense<1> : tensor<342xi64>
  %c_1 = stablehlo.constant dense<0> : tensor<342xi64>
  %0 = stablehlo.reshape %arg6 : (tensor<f64>) -> tensor<f64>
  %1 = stablehlo.iota dim = 0 : tensor<342xi64>
  %2 = stablehlo.add %1, %c_1 : tensor<342xi64>
  %3 = stablehlo.multiply %2, %c_0 : tensor<342xi64>
  %4 = stablehlo.pad %arg4, %cst, low = [0], high = [2], interior = [0] : (tensor<1024xf64>, tensor<f64>) -> tensor<1026xf64>
  %5 = stablehlo.slice %4 [0:1026:3] : (tensor<1026xf64>) -> tensor<342xf64>
  %6 = stablehlo.reshape %5 : (tensor<342xf64>) -> tensor<342xf64>
  %7 = stablehlo.broadcast_in_dim %0, dims = [] : (tensor<f64>) -> tensor<342xf64>
  %8 = arith.mulf %7, %6 fastmath<contract> : tensor<342xf64>
  %9 = stablehlo.pad %arg5, %cst, low = [0], high = [2], interior = [0] : (tensor<1024xf64>, tensor<f64>) -> tensor<1026xf64>
  %10 = stablehlo.slice %9 [0:1026:3] : (tensor<1026xf64>) -> tensor<342xf64>
  %11 = stablehlo.reshape %10 : (tensor<342xf64>) -> tensor<342xf64>
  %12 = arith.addf %8, %11 fastmath<contract> : tensor<342xf64>
  %13 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i64>) -> tensor<342xi64>
  %14 = stablehlo.multiply %3, %13 : tensor<342xi64>
  %15 = stablehlo.reshape %14 : (tensor<342xi64>) -> tensor<342x1xi64>
  %16 = stablehlo.broadcast_in_dim %12, dims = [0] : (tensor<342xf64>) -> tensor<342xf64>
  %17 = stablehlo.reshape %16 : (tensor<342xf64>) -> tensor<342xf64>
  %18 = "stablehlo.scatter"(%arg3, %15, %17) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = true}> ({
  ^bb0(%arg10: tensor<f64>, %arg11: tensor<f64>):
    stablehlo.return %arg11 : tensor<f64>
  }) : (tensor<1024xf64>, tensor<342x1xi64>, tensor<342xf64>) -> tensor<1024xf64>
  return %arg0, %arg1, %arg2, %18, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9 : tensor<i32>, tensor<i32>, tensor<i32>, tensor<1024xf64>, tensor<1024xf64>, tensor<1024xf64>, tensor<f64>, tensor<i32>, tensor<i32>, tensor<i32>
}

// -----// IR Dump After InlinerPass (inline) //----- //
module {
  func.func @main(%arg0: tensor<i32> {jit.literal_val = 0 : i64}, %arg1: tensor<i32> {jit.literal_val = 1024 : i64}, %arg2: tensor<i32> {jit.literal_val = 3 : i64}, %arg3: tensor<1024xf64>, %arg4: tensor<1024xf64>, %arg5: tensor<1024xf64>, %arg6: tensor<f64> {jit.literal_val = 4619567317775286272 : i64}, %arg7: tensor<i32> {jit.literal_val = 1024 : i64}, %arg8: tensor<i32> {jit.literal_val = 1024 : i64}, %arg9: tensor<i32> {jit.literal_val = 1024 : i64}) -> (tensor<i32>, tensor<i32>, tensor<i32>, tensor<1024xf64>, tensor<1024xf64>, tensor<1024xf64>, tensor<f64>, tensor<i32>, tensor<i32>, tensor<i32>) {
    %c = stablehlo.constant dense<3> : tensor<i64>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %c_0 = stablehlo.constant dense<1> : tensor<342xi64>
    %c_1 = stablehlo.constant dense<0> : tensor<342xi64>
    %0 = stablehlo.reshape %arg6 : (tensor<f64>) -> tensor<f64>
    %1 = stablehlo.iota dim = 0 : tensor<342xi64>
    %2 = stablehlo.add %1, %c_1 : tensor<342xi64>
    %3 = stablehlo.multiply %2, %c_0 : tensor<342xi64>
    %4 = stablehlo.pad %arg4, %cst, low = [0], high = [2], interior = [0] : (tensor<1024xf64>, tensor<f64>) -> tensor<1026xf64>
    %5 = stablehlo.slice %4 [0:1026:3] : (tensor<1026xf64>) -> tensor<342xf64>
    %6 = stablehlo.reshape %5 : (tensor<342xf64>) -> tensor<342xf64>
    %7 = stablehlo.broadcast_in_dim %0, dims = [] : (tensor<f64>) -> tensor<342xf64>
    %8 = arith.mulf %7, %6 fastmath<contract> : tensor<342xf64>
    %9 = stablehlo.pad %arg5, %cst, low = [0], high = [2], interior = [0] : (tensor<1024xf64>, tensor<f64>) -> tensor<1026xf64>
    %10 = stablehlo.slice %9 [0:1026:3] : (tensor<1026xf64>) -> tensor<342xf64>
    %11 = stablehlo.reshape %10 : (tensor<342xf64>) -> tensor<342xf64>
    %12 = arith.addf %8, %11 fastmath<contract> : tensor<342xf64>
    %13 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i64>) -> tensor<342xi64>
    %14 = stablehlo.multiply %3, %13 : tensor<342xi64>
    %15 = stablehlo.reshape %14 : (tensor<342xi64>) -> tensor<342x1xi64>
    %16 = stablehlo.broadcast_in_dim %12, dims = [0] : (tensor<342xf64>) -> tensor<342xf64>
    %17 = stablehlo.reshape %16 : (tensor<342xf64>) -> tensor<342xf64>
    %18 = "stablehlo.scatter"(%arg3, %15, %17) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = true}> ({
    ^bb0(%arg10: tensor<f64>, %arg11: tensor<f64>):
      stablehlo.return %arg11 : tensor<f64>
    }) : (tensor<1024xf64>, tensor<342x1xi64>, tensor<342xf64>) -> tensor<1024xf64>
    return %arg0, %arg1, %arg2, %18, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9 : tensor<i32>, tensor<i32>, tensor<i32>, tensor<1024xf64>, tensor<1024xf64>, tensor<1024xf64>, tensor<f64>, tensor<i32>, tensor<i32>, tensor<i32>
  }
}


// -----// IR Dump After StablehloAggressiveSimplificationPass (stablehlo-aggressive-simplification) //----- //
func.func @main(%arg0: tensor<i32> {jit.literal_val = 0 : i64}, %arg1: tensor<i32> {jit.literal_val = 1024 : i64}, %arg2: tensor<i32> {jit.literal_val = 3 : i64}, %arg3: tensor<1024xf64>, %arg4: tensor<1024xf64>, %arg5: tensor<1024xf64>, %arg6: tensor<f64> {jit.literal_val = 4619567317775286272 : i64}, %arg7: tensor<i32> {jit.literal_val = 1024 : i64}, %arg8: tensor<i32> {jit.literal_val = 1024 : i64}, %arg9: tensor<i32> {jit.literal_val = 1024 : i64}) -> (tensor<i32>, tensor<i32>, tensor<i32>, tensor<1024xf64>, tensor<1024xf64>, tensor<1024xf64>, tensor<f64>, tensor<i32>, tensor<i32>, tensor<i32>) {
  %c = stablehlo.constant dense<3> : tensor<i64>
  %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
  %0 = stablehlo.iota dim = 0 : tensor<342xi64>
  %1 = stablehlo.pad %arg4, %cst, low = [0], high = [2], interior = [0] : (tensor<1024xf64>, tensor<f64>) -> tensor<1026xf64>
  %2 = stablehlo.slice %1 [0:1026:3] : (tensor<1026xf64>) -> tensor<342xf64>
  %3 = stablehlo.broadcast_in_dim %arg6, dims = [] : (tensor<f64>) -> tensor<342xf64>
  %4 = arith.mulf %3, %2 fastmath<contract> : tensor<342xf64>
  %5 = stablehlo.pad %arg5, %cst, low = [0], high = [2], interior = [0] : (tensor<1024xf64>, tensor<f64>) -> tensor<1026xf64>
  %6 = stablehlo.slice %5 [0:1026:3] : (tensor<1026xf64>) -> tensor<342xf64>
  %7 = arith.addf %4, %6 fastmath<contract> : tensor<342xf64>
  %8 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i64>) -> tensor<342xi64>
  %9 = stablehlo.multiply %0, %8 : tensor<342xi64>
  %10 = stablehlo.reshape %9 : (tensor<342xi64>) -> tensor<342x1xi64>
  %11 = "stablehlo.scatter"(%arg3, %10, %7) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = true}> ({
  ^bb0(%arg10: tensor<f64>, %arg11: tensor<f64>):
    stablehlo.return %arg11 : tensor<f64>
  }) : (tensor<1024xf64>, tensor<342x1xi64>, tensor<342xf64>) -> tensor<1024xf64>
  return %arg0, %arg1, %arg2, %11, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9 : tensor<i32>, tensor<i32>, tensor<i32>, tensor<1024xf64>, tensor<1024xf64>, tensor<1024xf64>, tensor<f64>, tensor<i32>, tensor<i32>, tensor<i32>
}

// -----// IR Dump After (anonymous namespace)::TrimArgsPass (jforce-trim-args) //----- //
func.func @main(%arg0: tensor<1024xf64> {jit.literal_val = 0 : i64}, %arg1: tensor<1024xf64> {jit.literal_val = 1024 : i64}, %arg2: tensor<1024xf64> {jit.literal_val = 3 : i64}, %arg3: tensor<f64>) -> (tensor<1024xf64>, tensor<1024xf64>, tensor<1024xf64>, tensor<f64>) attributes {jit.args_mapping = [4 : ui32, 1 : ui32, 6 : ui32, 3 : ui32, 3 : ui32, 0 : ui32, 5 : ui32, 2 : ui32]} {
  %c = stablehlo.constant dense<3> : tensor<i64>
  %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
  %0 = stablehlo.iota dim = 0 : tensor<342xi64>
  %1 = stablehlo.pad %arg1, %cst, low = [0], high = [2], interior = [0] : (tensor<1024xf64>, tensor<f64>) -> tensor<1026xf64>
  %2 = stablehlo.slice %1 [0:1026:3] : (tensor<1026xf64>) -> tensor<342xf64>
  %3 = stablehlo.broadcast_in_dim %arg3, dims = [] : (tensor<f64>) -> tensor<342xf64>
  %4 = arith.mulf %3, %2 fastmath<contract> : tensor<342xf64>
  %5 = stablehlo.pad %arg2, %cst, low = [0], high = [2], interior = [0] : (tensor<1024xf64>, tensor<f64>) -> tensor<1026xf64>
  %6 = stablehlo.slice %5 [0:1026:3] : (tensor<1026xf64>) -> tensor<342xf64>
  %7 = arith.addf %4, %6 fastmath<contract> : tensor<342xf64>
  %8 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i64>) -> tensor<342xi64>
  %9 = stablehlo.multiply %0, %8 : tensor<342xi64>
  %10 = stablehlo.reshape %9 : (tensor<342xi64>) -> tensor<342x1xi64>
  %11 = "stablehlo.scatter"(%arg0, %10, %7) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = true}> ({
  ^bb0(%arg4: tensor<f64>, %arg5: tensor<f64>):
    stablehlo.return %arg5 : tensor<f64>
  }) : (tensor<1024xf64>, tensor<342x1xi64>, tensor<342xf64>) -> tensor<1024xf64>
  return %11, %arg1, %arg2, %arg3 : tensor<1024xf64>, tensor<1024xf64>, tensor<1024xf64>, tensor<f64>
}

