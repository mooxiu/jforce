// ./build/src/tool/jforce-opt ./tests/stencil-1d.mlir \
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
//    --mlir-print-ir-after-all 2> ./tests/stencil-1d-lowering.mlir


// -----// IR Dump After (anonymous namespace)::AnnotatePass (jforce-annotate) //----- //
func.func @kernel(%arg0: !fir.ref<i32> {jit.arg_type = 1 : ui32, jit.literal_val = 0 : i64}, %arg1: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg2: !fir.ref<!fir.array<?xf64>> {jit.arg_type = 1 : ui32}, %arg3: !fir.ref<!fir.array<?xf64>> {jit.arg_type = 1 : ui32}, %arg4: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg5: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}) {
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

// -----// IR Dump After (anonymous namespace)::PropagateConstantsPass (jforce-propagate-constants) //----- //
func.func @kernel(%arg0: !fir.ref<i32> {jit.arg_type = 1 : ui32, jit.literal_val = 0 : i64}, %arg1: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg2: !fir.ref<!fir.array<?xf64>> {jit.arg_type = 1 : ui32}, %arg3: !fir.ref<!fir.array<?xf64>> {jit.arg_type = 1 : ui32}, %arg4: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg5: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}) {
  %c1024_i32 = arith.constant 1024 : i32
  %c1024_i32_0 = arith.constant 1024 : i32
  %0 = fir.convert %c1024_i32_0 : (i32) -> i64
  %1 = fir.convert %c1024_i32 : (i32) -> i64
  %c0 = arith.constant 0 : index
  %2 = fir.convert %1 : (i64) -> index
  %3 = arith.cmpi sgt, %2, %c0 : index
  %c0_1 = arith.constant 0 : index
  %4 = fir.convert %0 : (i64) -> index
  %5 = arith.cmpi sgt, %4, %c0_1 : index
  %6 = arith.select %5, %4, %c0_1 : index
  %7 = arith.select %3, %2, %c0 : index
  %8:2 = hlfir.declare %arg0 {uniq_name = "_QFFrun_benchmarkEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %9:2 = hlfir.declare %arg1 {uniq_name = "_QFFrun_benchmarkEn"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %10 = fir.shape %7 : (index) -> !fir.shape<1>
  %11:2 = hlfir.declare %arg2(%10) {uniq_name = "_QFFrun_benchmarkEy"} : (!fir.ref<!fir.array<?xf64>>, !fir.shape<1>) -> (!fir.box<!fir.array<?xf64>>, !fir.ref<!fir.array<?xf64>>)
  %12 = fir.shape %6 : (index) -> !fir.shape<1>
  %13:2 = hlfir.declare %arg3(%12) {uniq_name = "_QFFrun_benchmarkEx"} : (!fir.ref<!fir.array<?xf64>>, !fir.shape<1>) -> (!fir.box<!fir.array<?xf64>>, !fir.ref<!fir.array<?xf64>>)
  %c2_i32 = arith.constant 2 : i32
  %14 = fir.convert %c2_i32 : (i32) -> index
  %c1024_i32_2 = arith.constant 1024 : i32
  %c1_i32 = arith.constant 1 : i32
  %15 = arith.subi %c1024_i32_2, %c1_i32 overflow<nsw> : i32
  %16 = fir.convert %15 : (i32) -> index
  %c1 = arith.constant 1 : index
  %17 = fir.convert %14 : (index) -> i32
  %18 = fir.do_loop %arg6 = %14 to %16 step %c1 iter_args(%arg7 = %17) -> (i32) {
    fir.store %arg7 to %8#0 : !fir.ref<i32>
    %19 = fir.load %8#0 : !fir.ref<i32>
    %c1_i32_3 = arith.constant 1 : i32
    %20 = arith.subi %19, %c1_i32_3 overflow<nsw> : i32
    %21 = fir.convert %20 : (i32) -> i64
    %22 = hlfir.designate %13#0 (%21)  : (!fir.box<!fir.array<?xf64>>, i64) -> !fir.ref<f64>
    %23 = fir.load %22 : !fir.ref<f64>
    %24 = fir.load %8#0 : !fir.ref<i32>
    %c1_i32_4 = arith.constant 1 : i32
    %25 = arith.addi %24, %c1_i32_4 overflow<nsw> : i32
    %26 = fir.convert %25 : (i32) -> i64
    %27 = hlfir.designate %13#0 (%26)  : (!fir.box<!fir.array<?xf64>>, i64) -> !fir.ref<f64>
    %28 = fir.load %27 : !fir.ref<f64>
    %29 = arith.addf %23, %28 fastmath<contract> : f64
    %30 = fir.load %8#0 : !fir.ref<i32>
    %31 = fir.convert %30 : (i32) -> i64
    %32 = hlfir.designate %11#0 (%31)  : (!fir.box<!fir.array<?xf64>>, i64) -> !fir.ref<f64>
    hlfir.assign %29 to %32 : f64, !fir.ref<f64>
    %33 = fir.convert %c1 : (index) -> i32
    %34 = fir.load %8#0 : !fir.ref<i32>
    %35 = arith.addi %34, %33 overflow<nsw> : i32
    fir.result %35 : i32
  }
  fir.store %18 to %8#0 : !fir.ref<i32>
  omp.terminator
}

// -----// IR Dump After CanonicalizerPass (canonicalize) //----- //
module {
  func.func @kernel(%arg0: !fir.ref<i32> {jit.arg_type = 1 : ui32, jit.literal_val = 0 : i64}, %arg1: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg2: !fir.ref<!fir.array<?xf64>> {jit.arg_type = 1 : ui32}, %arg3: !fir.ref<!fir.array<?xf64>> {jit.arg_type = 1 : ui32}, %arg4: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg5: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}) {
    %c1023 = arith.constant 1023 : index
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %c1_i32 = arith.constant 1 : i32
    %c1024 = arith.constant 1024 : index
    %0:2 = hlfir.declare %arg0 {uniq_name = "_QFFrun_benchmarkEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
    %1:2 = hlfir.declare %arg1 {uniq_name = "_QFFrun_benchmarkEn"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
    %2 = fir.shape %c1024 : (index) -> !fir.shape<1>
    %3:2 = hlfir.declare %arg2(%2) {uniq_name = "_QFFrun_benchmarkEy"} : (!fir.ref<!fir.array<?xf64>>, !fir.shape<1>) -> (!fir.box<!fir.array<?xf64>>, !fir.ref<!fir.array<?xf64>>)
    %4 = fir.shape %c1024 : (index) -> !fir.shape<1>
    %5:2 = hlfir.declare %arg3(%4) {uniq_name = "_QFFrun_benchmarkEx"} : (!fir.ref<!fir.array<?xf64>>, !fir.shape<1>) -> (!fir.box<!fir.array<?xf64>>, !fir.ref<!fir.array<?xf64>>)
    %6 = fir.convert %c2 : (index) -> i32
    %7 = fir.do_loop %arg6 = %c2 to %c1023 step %c1 iter_args(%arg7 = %6) -> (i32) {
      fir.store %arg7 to %0#0 : !fir.ref<i32>
      %8 = fir.load %0#0 : !fir.ref<i32>
      %9 = arith.subi %8, %c1_i32 overflow<nsw> : i32
      %10 = fir.convert %9 : (i32) -> i64
      %11 = hlfir.designate %5#0 (%10)  : (!fir.box<!fir.array<?xf64>>, i64) -> !fir.ref<f64>
      %12 = fir.load %11 : !fir.ref<f64>
      %13 = fir.load %0#0 : !fir.ref<i32>
      %14 = arith.addi %13, %c1_i32 overflow<nsw> : i32
      %15 = fir.convert %14 : (i32) -> i64
      %16 = hlfir.designate %5#0 (%15)  : (!fir.box<!fir.array<?xf64>>, i64) -> !fir.ref<f64>
      %17 = fir.load %16 : !fir.ref<f64>
      %18 = arith.addf %12, %17 fastmath<contract> : f64
      %19 = fir.load %0#0 : !fir.ref<i32>
      %20 = fir.convert %19 : (i32) -> i64
      %21 = hlfir.designate %3#0 (%20)  : (!fir.box<!fir.array<?xf64>>, i64) -> !fir.ref<f64>
      hlfir.assign %18 to %21 : f64, !fir.ref<f64>
      %22 = fir.convert %c1 : (index) -> i32
      %23 = fir.load %0#0 : !fir.ref<i32>
      %24 = arith.addi %23, %22 overflow<nsw> : i32
      fir.result %24 : i32
    }
    fir.store %7 to %0#0 : !fir.ref<i32>
    omp.terminator
  }
}


// -----// IR Dump After SCCPPass (sccp) //----- //
module {
  func.func @kernel(%arg0: !fir.ref<i32> {jit.arg_type = 1 : ui32, jit.literal_val = 0 : i64}, %arg1: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg2: !fir.ref<!fir.array<?xf64>> {jit.arg_type = 1 : ui32}, %arg3: !fir.ref<!fir.array<?xf64>> {jit.arg_type = 1 : ui32}, %arg4: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg5: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}) {
    %c1024 = arith.constant 1024 : index
    %c1_i32 = arith.constant 1 : i32
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c1023 = arith.constant 1023 : index
    %0:2 = hlfir.declare %arg0 {uniq_name = "_QFFrun_benchmarkEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
    %1:2 = hlfir.declare %arg1 {uniq_name = "_QFFrun_benchmarkEn"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
    %2 = fir.shape %c1024 : (index) -> !fir.shape<1>
    %3:2 = hlfir.declare %arg2(%2) {uniq_name = "_QFFrun_benchmarkEy"} : (!fir.ref<!fir.array<?xf64>>, !fir.shape<1>) -> (!fir.box<!fir.array<?xf64>>, !fir.ref<!fir.array<?xf64>>)
    %4 = fir.shape %c1024 : (index) -> !fir.shape<1>
    %5:2 = hlfir.declare %arg3(%4) {uniq_name = "_QFFrun_benchmarkEx"} : (!fir.ref<!fir.array<?xf64>>, !fir.shape<1>) -> (!fir.box<!fir.array<?xf64>>, !fir.ref<!fir.array<?xf64>>)
    %6 = fir.convert %c2 : (index) -> i32
    %7 = fir.do_loop %arg6 = %c2 to %c1023 step %c1 iter_args(%arg7 = %6) -> (i32) {
      fir.store %arg7 to %0#0 : !fir.ref<i32>
      %8 = fir.load %0#0 : !fir.ref<i32>
      %9 = arith.subi %8, %c1_i32 overflow<nsw> : i32
      %10 = fir.convert %9 : (i32) -> i64
      %11 = hlfir.designate %5#0 (%10)  : (!fir.box<!fir.array<?xf64>>, i64) -> !fir.ref<f64>
      %12 = fir.load %11 : !fir.ref<f64>
      %13 = fir.load %0#0 : !fir.ref<i32>
      %14 = arith.addi %13, %c1_i32 overflow<nsw> : i32
      %15 = fir.convert %14 : (i32) -> i64
      %16 = hlfir.designate %5#0 (%15)  : (!fir.box<!fir.array<?xf64>>, i64) -> !fir.ref<f64>
      %17 = fir.load %16 : !fir.ref<f64>
      %18 = arith.addf %12, %17 fastmath<contract> : f64
      %19 = fir.load %0#0 : !fir.ref<i32>
      %20 = fir.convert %19 : (i32) -> i64
      %21 = hlfir.designate %3#0 (%20)  : (!fir.box<!fir.array<?xf64>>, i64) -> !fir.ref<f64>
      hlfir.assign %18 to %21 : f64, !fir.ref<f64>
      %22 = fir.convert %c1 : (index) -> i32
      %23 = fir.load %0#0 : !fir.ref<i32>
      %24 = arith.addi %23, %22 overflow<nsw> : i32
      fir.result %24 : i32
    }
    fir.store %7 to %0#0 : !fir.ref<i32>
    omp.terminator
  }
}


// -----// IR Dump After CSEPass (cse) //----- //
module {
  func.func @kernel(%arg0: !fir.ref<i32> {jit.arg_type = 1 : ui32, jit.literal_val = 0 : i64}, %arg1: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg2: !fir.ref<!fir.array<?xf64>> {jit.arg_type = 1 : ui32}, %arg3: !fir.ref<!fir.array<?xf64>> {jit.arg_type = 1 : ui32}, %arg4: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg5: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}) {
    %c1024 = arith.constant 1024 : index
    %c1_i32 = arith.constant 1 : i32
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c1023 = arith.constant 1023 : index
    %0:2 = hlfir.declare %arg0 {uniq_name = "_QFFrun_benchmarkEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
    %1:2 = hlfir.declare %arg1 {uniq_name = "_QFFrun_benchmarkEn"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
    %2 = fir.shape %c1024 : (index) -> !fir.shape<1>
    %3:2 = hlfir.declare %arg2(%2) {uniq_name = "_QFFrun_benchmarkEy"} : (!fir.ref<!fir.array<?xf64>>, !fir.shape<1>) -> (!fir.box<!fir.array<?xf64>>, !fir.ref<!fir.array<?xf64>>)
    %4:2 = hlfir.declare %arg3(%2) {uniq_name = "_QFFrun_benchmarkEx"} : (!fir.ref<!fir.array<?xf64>>, !fir.shape<1>) -> (!fir.box<!fir.array<?xf64>>, !fir.ref<!fir.array<?xf64>>)
    %5 = fir.convert %c2 : (index) -> i32
    %6 = fir.do_loop %arg6 = %c2 to %c1023 step %c1 iter_args(%arg7 = %5) -> (i32) {
      fir.store %arg7 to %0#0 : !fir.ref<i32>
      %7 = fir.load %0#0 : !fir.ref<i32>
      %8 = arith.subi %7, %c1_i32 overflow<nsw> : i32
      %9 = fir.convert %8 : (i32) -> i64
      %10 = hlfir.designate %4#0 (%9)  : (!fir.box<!fir.array<?xf64>>, i64) -> !fir.ref<f64>
      %11 = fir.load %10 : !fir.ref<f64>
      %12 = arith.addi %7, %c1_i32 overflow<nsw> : i32
      %13 = fir.convert %12 : (i32) -> i64
      %14 = hlfir.designate %4#0 (%13)  : (!fir.box<!fir.array<?xf64>>, i64) -> !fir.ref<f64>
      %15 = fir.load %14 : !fir.ref<f64>
      %16 = arith.addf %11, %15 fastmath<contract> : f64
      %17 = fir.convert %7 : (i32) -> i64
      %18 = hlfir.designate %3#0 (%17)  : (!fir.box<!fir.array<?xf64>>, i64) -> !fir.ref<f64>
      hlfir.assign %16 to %18 : f64, !fir.ref<f64>
      %19 = fir.convert %c1 : (index) -> i32
      %20 = fir.load %0#0 : !fir.ref<i32>
      %21 = arith.addi %20, %19 overflow<nsw> : i32
      fir.result %21 : i32
    }
    fir.store %6 to %0#0 : !fir.ref<i32>
    omp.terminator
  }
}


// -----// IR Dump After CanonicalizerPass (canonicalize) //----- //
module {
  func.func @kernel(%arg0: !fir.ref<i32> {jit.arg_type = 1 : ui32, jit.literal_val = 0 : i64}, %arg1: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg2: !fir.ref<!fir.array<?xf64>> {jit.arg_type = 1 : ui32}, %arg3: !fir.ref<!fir.array<?xf64>> {jit.arg_type = 1 : ui32}, %arg4: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg5: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}) {
    %c1024 = arith.constant 1024 : index
    %c1_i32 = arith.constant 1 : i32
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c1023 = arith.constant 1023 : index
    %0:2 = hlfir.declare %arg0 {uniq_name = "_QFFrun_benchmarkEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
    %1:2 = hlfir.declare %arg1 {uniq_name = "_QFFrun_benchmarkEn"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
    %2 = fir.shape %c1024 : (index) -> !fir.shape<1>
    %3:2 = hlfir.declare %arg2(%2) {uniq_name = "_QFFrun_benchmarkEy"} : (!fir.ref<!fir.array<?xf64>>, !fir.shape<1>) -> (!fir.box<!fir.array<?xf64>>, !fir.ref<!fir.array<?xf64>>)
    %4:2 = hlfir.declare %arg3(%2) {uniq_name = "_QFFrun_benchmarkEx"} : (!fir.ref<!fir.array<?xf64>>, !fir.shape<1>) -> (!fir.box<!fir.array<?xf64>>, !fir.ref<!fir.array<?xf64>>)
    %5 = fir.convert %c2 : (index) -> i32
    %6 = fir.do_loop %arg6 = %c2 to %c1023 step %c1 iter_args(%arg7 = %5) -> (i32) {
      fir.store %arg7 to %0#0 : !fir.ref<i32>
      %7 = fir.load %0#0 : !fir.ref<i32>
      %8 = arith.subi %7, %c1_i32 overflow<nsw> : i32
      %9 = fir.convert %8 : (i32) -> i64
      %10 = hlfir.designate %4#0 (%9)  : (!fir.box<!fir.array<?xf64>>, i64) -> !fir.ref<f64>
      %11 = fir.load %10 : !fir.ref<f64>
      %12 = arith.addi %7, %c1_i32 overflow<nsw> : i32
      %13 = fir.convert %12 : (i32) -> i64
      %14 = hlfir.designate %4#0 (%13)  : (!fir.box<!fir.array<?xf64>>, i64) -> !fir.ref<f64>
      %15 = fir.load %14 : !fir.ref<f64>
      %16 = arith.addf %11, %15 fastmath<contract> : f64
      %17 = fir.convert %7 : (i32) -> i64
      %18 = hlfir.designate %3#0 (%17)  : (!fir.box<!fir.array<?xf64>>, i64) -> !fir.ref<f64>
      hlfir.assign %16 to %18 : f64, !fir.ref<f64>
      %19 = fir.convert %c1 : (index) -> i32
      %20 = fir.load %0#0 : !fir.ref<i32>
      %21 = arith.addi %20, %19 overflow<nsw> : i32
      fir.result %21 : i32
    }
    fir.store %6 to %0#0 : !fir.ref<i32>
    omp.terminator
  }
}


// -----// IR Dump After (anonymous namespace)::ShapeInferPass (jforce-shape-infer) //----- //
func.func @kernel(%arg0: !fir.ref<i32> {jit.arg_type = 1 : ui32, jit.literal_val = 0 : i64}, %arg1: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg2: !fir.ref<!fir.array<1024xf64>> {jit.arg_type = 1 : ui32}, %arg3: !fir.ref<!fir.array<1024xf64>> {jit.arg_type = 1 : ui32}, %arg4: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg5: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}) {
  %c1024 = arith.constant 1024 : index
  %c1_i32 = arith.constant 1 : i32
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c1023 = arith.constant 1023 : index
  %0:2 = hlfir.declare %arg0 {uniq_name = "_QFFrun_benchmarkEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %1:2 = hlfir.declare %arg1 {uniq_name = "_QFFrun_benchmarkEn"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %2 = fir.shape %c1024 : (index) -> !fir.shape<1>
  %3:2 = hlfir.declare %arg2(%2) {uniq_name = "_QFFrun_benchmarkEy"} : (!fir.ref<!fir.array<1024xf64>>, !fir.shape<1>) -> (!fir.ref<!fir.array<1024xf64>>, !fir.ref<!fir.array<1024xf64>>)
  %4:2 = hlfir.declare %arg3(%2) {uniq_name = "_QFFrun_benchmarkEx"} : (!fir.ref<!fir.array<1024xf64>>, !fir.shape<1>) -> (!fir.ref<!fir.array<1024xf64>>, !fir.ref<!fir.array<1024xf64>>)
  %5 = fir.convert %c2 : (index) -> i32
  %6 = fir.do_loop %arg6 = %c2 to %c1023 step %c1 iter_args(%arg7 = %5) -> (i32) {
    fir.store %arg7 to %0#0 : !fir.ref<i32>
    %7 = fir.load %0#0 : !fir.ref<i32>
    %8 = arith.subi %7, %c1_i32 overflow<nsw> : i32
    %9 = fir.convert %8 : (i32) -> i64
    %10 = hlfir.designate %4#0 (%9)  : (!fir.ref<!fir.array<1024xf64>>, i64) -> !fir.ref<f64>
    %11 = fir.load %10 : !fir.ref<f64>
    %12 = arith.addi %7, %c1_i32 overflow<nsw> : i32
    %13 = fir.convert %12 : (i32) -> i64
    %14 = hlfir.designate %4#0 (%13)  : (!fir.ref<!fir.array<1024xf64>>, i64) -> !fir.ref<f64>
    %15 = fir.load %14 : !fir.ref<f64>
    %16 = arith.addf %11, %15 fastmath<contract> : f64
    %17 = fir.convert %7 : (i32) -> i64
    %18 = hlfir.designate %3#0 (%17)  : (!fir.ref<!fir.array<1024xf64>>, i64) -> !fir.ref<f64>
    hlfir.assign %16 to %18 : f64, !fir.ref<f64>
    %19 = fir.convert %c1 : (index) -> i32
    %20 = fir.load %0#0 : !fir.ref<i32>
    %21 = arith.addi %20, %19 overflow<nsw> : i32
    fir.result %21 : i32
  }
  fir.store %6 to %0#0 : !fir.ref<i32>
  omp.terminator
}

// -----// IR Dump After CanonicalizerPass (canonicalize) //----- //
module {
  func.func @kernel(%arg0: !fir.ref<i32> {jit.arg_type = 1 : ui32, jit.literal_val = 0 : i64}, %arg1: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg2: !fir.ref<!fir.array<1024xf64>> {jit.arg_type = 1 : ui32}, %arg3: !fir.ref<!fir.array<1024xf64>> {jit.arg_type = 1 : ui32}, %arg4: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg5: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}) {
    %c1024 = arith.constant 1024 : index
    %c1_i32 = arith.constant 1 : i32
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c1023 = arith.constant 1023 : index
    %0:2 = hlfir.declare %arg0 {uniq_name = "_QFFrun_benchmarkEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
    %1:2 = hlfir.declare %arg1 {uniq_name = "_QFFrun_benchmarkEn"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
    %2 = fir.shape %c1024 : (index) -> !fir.shape<1>
    %3:2 = hlfir.declare %arg2(%2) {uniq_name = "_QFFrun_benchmarkEy"} : (!fir.ref<!fir.array<1024xf64>>, !fir.shape<1>) -> (!fir.ref<!fir.array<1024xf64>>, !fir.ref<!fir.array<1024xf64>>)
    %4:2 = hlfir.declare %arg3(%2) {uniq_name = "_QFFrun_benchmarkEx"} : (!fir.ref<!fir.array<1024xf64>>, !fir.shape<1>) -> (!fir.ref<!fir.array<1024xf64>>, !fir.ref<!fir.array<1024xf64>>)
    %5 = fir.convert %c2 : (index) -> i32
    %6 = fir.do_loop %arg6 = %c2 to %c1023 step %c1 iter_args(%arg7 = %5) -> (i32) {
      fir.store %arg7 to %0#0 : !fir.ref<i32>
      %7 = fir.load %0#0 : !fir.ref<i32>
      %8 = arith.subi %7, %c1_i32 overflow<nsw> : i32
      %9 = fir.convert %8 : (i32) -> i64
      %10 = hlfir.designate %4#0 (%9)  : (!fir.ref<!fir.array<1024xf64>>, i64) -> !fir.ref<f64>
      %11 = fir.load %10 : !fir.ref<f64>
      %12 = arith.addi %7, %c1_i32 overflow<nsw> : i32
      %13 = fir.convert %12 : (i32) -> i64
      %14 = hlfir.designate %4#0 (%13)  : (!fir.ref<!fir.array<1024xf64>>, i64) -> !fir.ref<f64>
      %15 = fir.load %14 : !fir.ref<f64>
      %16 = arith.addf %11, %15 fastmath<contract> : f64
      %17 = fir.convert %7 : (i32) -> i64
      %18 = hlfir.designate %3#0 (%17)  : (!fir.ref<!fir.array<1024xf64>>, i64) -> !fir.ref<f64>
      hlfir.assign %16 to %18 : f64, !fir.ref<f64>
      %19 = fir.convert %c1 : (index) -> i32
      %20 = fir.load %0#0 : !fir.ref<i32>
      %21 = arith.addi %20, %19 overflow<nsw> : i32
      fir.result %21 : i32
    }
    fir.store %6 to %0#0 : !fir.ref<i32>
    omp.terminator
  }
}


// -----// IR Dump After ConvertHLFIRtoFIR (convert-hlfir-to-fir) //----- //
module {
  func.func @kernel(%arg0: !fir.ref<i32> {jit.arg_type = 1 : ui32, jit.literal_val = 0 : i64}, %arg1: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg2: !fir.ref<!fir.array<1024xf64>> {jit.arg_type = 1 : ui32}, %arg3: !fir.ref<!fir.array<1024xf64>> {jit.arg_type = 1 : ui32}, %arg4: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg5: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}) {
    %c1024 = arith.constant 1024 : index
    %c1_i32 = arith.constant 1 : i32
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c1023 = arith.constant 1023 : index
    %0 = fir.declare %arg0 {uniq_name = "_QFFrun_benchmarkEi"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %1 = fir.declare %arg1 {uniq_name = "_QFFrun_benchmarkEn"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %2 = fir.shape %c1024 : (index) -> !fir.shape<1>
    %3 = fir.declare %arg2(%2) {uniq_name = "_QFFrun_benchmarkEy"} : (!fir.ref<!fir.array<1024xf64>>, !fir.shape<1>) -> !fir.ref<!fir.array<1024xf64>>
    %4 = fir.declare %arg3(%2) {uniq_name = "_QFFrun_benchmarkEx"} : (!fir.ref<!fir.array<1024xf64>>, !fir.shape<1>) -> !fir.ref<!fir.array<1024xf64>>
    %5 = fir.convert %c2 : (index) -> i32
    %6 = fir.do_loop %arg6 = %c2 to %c1023 step %c1 iter_args(%arg7 = %5) -> (i32) {
      fir.store %arg7 to %0 : !fir.ref<i32>
      %7 = fir.load %0 : !fir.ref<i32>
      %8 = arith.subi %7, %c1_i32 overflow<nsw> : i32
      %9 = fir.convert %8 : (i32) -> i64
      %10 = fir.array_coor %4(%2) %9 : (!fir.ref<!fir.array<1024xf64>>, !fir.shape<1>, i64) -> !fir.ref<f64>
      %11 = fir.load %10 : !fir.ref<f64>
      %12 = arith.addi %7, %c1_i32 overflow<nsw> : i32
      %13 = fir.convert %12 : (i32) -> i64
      %14 = fir.array_coor %4(%2) %13 : (!fir.ref<!fir.array<1024xf64>>, !fir.shape<1>, i64) -> !fir.ref<f64>
      %15 = fir.load %14 : !fir.ref<f64>
      %16 = arith.addf %11, %15 fastmath<contract> : f64
      %17 = fir.convert %7 : (i32) -> i64
      %18 = fir.array_coor %3(%2) %17 : (!fir.ref<!fir.array<1024xf64>>, !fir.shape<1>, i64) -> !fir.ref<f64>
      fir.store %16 to %18 : !fir.ref<f64>
      %19 = fir.convert %c1 : (index) -> i32
      %20 = fir.load %0 : !fir.ref<i32>
      %21 = arith.addi %20, %19 overflow<nsw> : i32
      fir.result %21 : i32
    }
    fir.store %6 to %0 : !fir.ref<i32>
    omp.terminator
  }
}


// -----// IR Dump After FIRToMemRef (fir-to-memref) //----- //
func.func @kernel(%arg0: !fir.ref<i32> {jit.arg_type = 1 : ui32, jit.literal_val = 0 : i64}, %arg1: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg2: !fir.ref<!fir.array<1024xf64>> {jit.arg_type = 1 : ui32}, %arg3: !fir.ref<!fir.array<1024xf64>> {jit.arg_type = 1 : ui32}, %arg4: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg5: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}) {
  %c1024 = arith.constant 1024 : index
  %c1_i32 = arith.constant 1 : i32
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c1023 = arith.constant 1023 : index
  %0 = fir.declare %arg0 {uniq_name = "_QFFrun_benchmarkEi"} : (!fir.ref<i32>) -> !fir.ref<i32>
  %1 = fir.declare %arg1 {uniq_name = "_QFFrun_benchmarkEn"} : (!fir.ref<i32>) -> !fir.ref<i32>
  %2 = fir.shape %c1024 : (index) -> !fir.shape<1>
  %3 = fir.declare %arg2(%2) {uniq_name = "_QFFrun_benchmarkEy"} : (!fir.ref<!fir.array<1024xf64>>, !fir.shape<1>) -> !fir.ref<!fir.array<1024xf64>>
  %4 = fir.declare %arg3(%2) {uniq_name = "_QFFrun_benchmarkEx"} : (!fir.ref<!fir.array<1024xf64>>, !fir.shape<1>) -> !fir.ref<!fir.array<1024xf64>>
  %5 = fir.convert %c2 : (index) -> i32
  %6 = fir.do_loop %arg6 = %c2 to %c1023 step %c1 iter_args(%arg7 = %5) -> (i32) {
    %8 = fir.convert %0 : (!fir.ref<i32>) -> memref<i32>
    memref.store %arg7, %8[] : memref<i32>
    %9 = fir.convert %0 : (!fir.ref<i32>) -> memref<i32>
    %10 = memref.load %9[] : memref<i32>
    %11 = arith.subi %10, %c1_i32 overflow<nsw> : i32
    %12 = fir.convert %11 : (i32) -> i64
    %13 = fir.convert %4 : (!fir.ref<!fir.array<1024xf64>>) -> memref<1024xf64>
    %c1_0 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %14 = arith.index_cast %12 : i64 to index
    %15 = arith.subi %14, %c1_0 : index
    %16 = arith.muli %15, %c1_0 : index
    %17 = arith.subi %c1_0, %c1_0 : index
    %18 = arith.addi %16, %17 : index
    %19 = memref.load %13[%18] : memref<1024xf64>
    %20 = arith.addi %10, %c1_i32 overflow<nsw> : i32
    %21 = fir.convert %20 : (i32) -> i64
    %22 = fir.convert %4 : (!fir.ref<!fir.array<1024xf64>>) -> memref<1024xf64>
    %c1_1 = arith.constant 1 : index
    %c0_2 = arith.constant 0 : index
    %23 = arith.index_cast %21 : i64 to index
    %24 = arith.subi %23, %c1_1 : index
    %25 = arith.muli %24, %c1_1 : index
    %26 = arith.subi %c1_1, %c1_1 : index
    %27 = arith.addi %25, %26 : index
    %28 = memref.load %22[%27] : memref<1024xf64>
    %29 = arith.addf %19, %28 fastmath<contract> : f64
    %30 = fir.convert %10 : (i32) -> i64
    %31 = fir.convert %3 : (!fir.ref<!fir.array<1024xf64>>) -> memref<1024xf64>
    %c1_3 = arith.constant 1 : index
    %c0_4 = arith.constant 0 : index
    %32 = arith.index_cast %30 : i64 to index
    %33 = arith.subi %32, %c1_3 : index
    %34 = arith.muli %33, %c1_3 : index
    %35 = arith.subi %c1_3, %c1_3 : index
    %36 = arith.addi %34, %35 : index
    memref.store %29, %31[%36] : memref<1024xf64>
    %37 = fir.convert %c1 : (index) -> i32
    %38 = fir.convert %0 : (!fir.ref<i32>) -> memref<i32>
    %39 = memref.load %38[] : memref<i32>
    %40 = arith.addi %39, %37 overflow<nsw> : i32
    fir.result %40 : i32
  }
  %7 = fir.convert %0 : (!fir.ref<i32>) -> memref<i32>
  memref.store %6, %7[] : memref<i32>
  omp.terminator
}

// -----// IR Dump After CanonicalizerPass (canonicalize) //----- //
module {
  func.func @kernel(%arg0: !fir.ref<i32> {jit.arg_type = 1 : ui32, jit.literal_val = 0 : i64}, %arg1: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg2: !fir.ref<!fir.array<1024xf64>> {jit.arg_type = 1 : ui32}, %arg3: !fir.ref<!fir.array<1024xf64>> {jit.arg_type = 1 : ui32}, %arg4: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg5: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}) {
    %c1024 = arith.constant 1024 : index
    %c1_i32 = arith.constant 1 : i32
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c1023 = arith.constant 1023 : index
    %0 = fir.declare %arg0 {uniq_name = "_QFFrun_benchmarkEi"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %1 = fir.declare %arg1 {uniq_name = "_QFFrun_benchmarkEn"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %2 = fir.shape %c1024 : (index) -> !fir.shape<1>
    %3 = fir.declare %arg2(%2) {uniq_name = "_QFFrun_benchmarkEy"} : (!fir.ref<!fir.array<1024xf64>>, !fir.shape<1>) -> !fir.ref<!fir.array<1024xf64>>
    %4 = fir.declare %arg3(%2) {uniq_name = "_QFFrun_benchmarkEx"} : (!fir.ref<!fir.array<1024xf64>>, !fir.shape<1>) -> !fir.ref<!fir.array<1024xf64>>
    %5 = fir.convert %c2 : (index) -> i32
    %6 = fir.do_loop %arg6 = %c2 to %c1023 step %c1 iter_args(%arg7 = %5) -> (i32) {
      %8 = fir.convert %0 : (!fir.ref<i32>) -> memref<i32>
      memref.store %arg7, %8[] : memref<i32>
      %9 = fir.convert %0 : (!fir.ref<i32>) -> memref<i32>
      %10 = memref.load %9[] : memref<i32>
      %11 = arith.subi %10, %c1_i32 overflow<nsw> : i32
      %12 = fir.convert %11 : (i32) -> i64
      %13 = fir.convert %4 : (!fir.ref<!fir.array<1024xf64>>) -> memref<1024xf64>
      %14 = arith.index_cast %12 : i64 to index
      %15 = arith.subi %14, %c1 : index
      %16 = memref.load %13[%15] : memref<1024xf64>
      %17 = arith.addi %10, %c1_i32 overflow<nsw> : i32
      %18 = fir.convert %17 : (i32) -> i64
      %19 = fir.convert %4 : (!fir.ref<!fir.array<1024xf64>>) -> memref<1024xf64>
      %20 = arith.index_cast %18 : i64 to index
      %21 = arith.subi %20, %c1 : index
      %22 = memref.load %19[%21] : memref<1024xf64>
      %23 = arith.addf %16, %22 fastmath<contract> : f64
      %24 = fir.convert %10 : (i32) -> i64
      %25 = fir.convert %3 : (!fir.ref<!fir.array<1024xf64>>) -> memref<1024xf64>
      %26 = arith.index_cast %24 : i64 to index
      %27 = arith.subi %26, %c1 : index
      memref.store %23, %25[%27] : memref<1024xf64>
      %28 = fir.convert %c1 : (index) -> i32
      %29 = fir.convert %0 : (!fir.ref<i32>) -> memref<i32>
      %30 = memref.load %29[] : memref<i32>
      %31 = arith.addi %30, %28 overflow<nsw> : i32
      fir.result %31 : i32
    }
    %7 = fir.convert %0 : (!fir.ref<i32>) -> memref<i32>
    memref.store %6, %7[] : memref<i32>
    omp.terminator
  }
}


// -----// IR Dump After LoopInvariantCodeMotionPass (loop-invariant-code-motion) //----- //
module {
  func.func @kernel(%arg0: !fir.ref<i32> {jit.arg_type = 1 : ui32, jit.literal_val = 0 : i64}, %arg1: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg2: !fir.ref<!fir.array<1024xf64>> {jit.arg_type = 1 : ui32}, %arg3: !fir.ref<!fir.array<1024xf64>> {jit.arg_type = 1 : ui32}, %arg4: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg5: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}) {
    %c1024 = arith.constant 1024 : index
    %c1_i32 = arith.constant 1 : i32
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c1023 = arith.constant 1023 : index
    %0 = fir.declare %arg0 {uniq_name = "_QFFrun_benchmarkEi"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %1 = fir.declare %arg1 {uniq_name = "_QFFrun_benchmarkEn"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %2 = fir.shape %c1024 : (index) -> !fir.shape<1>
    %3 = fir.declare %arg2(%2) {uniq_name = "_QFFrun_benchmarkEy"} : (!fir.ref<!fir.array<1024xf64>>, !fir.shape<1>) -> !fir.ref<!fir.array<1024xf64>>
    %4 = fir.declare %arg3(%2) {uniq_name = "_QFFrun_benchmarkEx"} : (!fir.ref<!fir.array<1024xf64>>, !fir.shape<1>) -> !fir.ref<!fir.array<1024xf64>>
    %5 = fir.convert %c2 : (index) -> i32
    %6 = fir.convert %0 : (!fir.ref<i32>) -> memref<i32>
    %7 = fir.convert %0 : (!fir.ref<i32>) -> memref<i32>
    %8 = fir.convert %4 : (!fir.ref<!fir.array<1024xf64>>) -> memref<1024xf64>
    %9 = fir.convert %4 : (!fir.ref<!fir.array<1024xf64>>) -> memref<1024xf64>
    %10 = fir.convert %3 : (!fir.ref<!fir.array<1024xf64>>) -> memref<1024xf64>
    %11 = fir.convert %c1 : (index) -> i32
    %12 = fir.convert %0 : (!fir.ref<i32>) -> memref<i32>
    %13 = fir.do_loop %arg6 = %c2 to %c1023 step %c1 iter_args(%arg7 = %5) -> (i32) {
      memref.store %arg7, %6[] : memref<i32>
      %15 = memref.load %7[] : memref<i32>
      %16 = arith.subi %15, %c1_i32 overflow<nsw> : i32
      %17 = fir.convert %16 : (i32) -> i64
      %18 = arith.index_cast %17 : i64 to index
      %19 = arith.subi %18, %c1 : index
      %20 = memref.load %8[%19] : memref<1024xf64>
      %21 = arith.addi %15, %c1_i32 overflow<nsw> : i32
      %22 = fir.convert %21 : (i32) -> i64
      %23 = arith.index_cast %22 : i64 to index
      %24 = arith.subi %23, %c1 : index
      %25 = memref.load %9[%24] : memref<1024xf64>
      %26 = arith.addf %20, %25 fastmath<contract> : f64
      %27 = fir.convert %15 : (i32) -> i64
      %28 = arith.index_cast %27 : i64 to index
      %29 = arith.subi %28, %c1 : index
      memref.store %26, %10[%29] : memref<1024xf64>
      %30 = memref.load %12[] : memref<i32>
      %31 = arith.addi %30, %11 overflow<nsw> : i32
      fir.result %31 : i32
    }
    %14 = fir.convert %0 : (!fir.ref<i32>) -> memref<i32>
    memref.store %13, %14[] : memref<i32>
    omp.terminator
  }
}


// -----// IR Dump After CSEPass (cse) //----- //
module {
  func.func @kernel(%arg0: !fir.ref<i32> {jit.arg_type = 1 : ui32, jit.literal_val = 0 : i64}, %arg1: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg2: !fir.ref<!fir.array<1024xf64>> {jit.arg_type = 1 : ui32}, %arg3: !fir.ref<!fir.array<1024xf64>> {jit.arg_type = 1 : ui32}, %arg4: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg5: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}) {
    %c1024 = arith.constant 1024 : index
    %c1_i32 = arith.constant 1 : i32
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c1023 = arith.constant 1023 : index
    %0 = fir.declare %arg0 {uniq_name = "_QFFrun_benchmarkEi"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %1 = fir.declare %arg1 {uniq_name = "_QFFrun_benchmarkEn"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %2 = fir.shape %c1024 : (index) -> !fir.shape<1>
    %3 = fir.declare %arg2(%2) {uniq_name = "_QFFrun_benchmarkEy"} : (!fir.ref<!fir.array<1024xf64>>, !fir.shape<1>) -> !fir.ref<!fir.array<1024xf64>>
    %4 = fir.declare %arg3(%2) {uniq_name = "_QFFrun_benchmarkEx"} : (!fir.ref<!fir.array<1024xf64>>, !fir.shape<1>) -> !fir.ref<!fir.array<1024xf64>>
    %5 = fir.convert %c2 : (index) -> i32
    %6 = fir.convert %0 : (!fir.ref<i32>) -> memref<i32>
    %7 = fir.convert %4 : (!fir.ref<!fir.array<1024xf64>>) -> memref<1024xf64>
    %8 = fir.convert %3 : (!fir.ref<!fir.array<1024xf64>>) -> memref<1024xf64>
    %9 = fir.convert %c1 : (index) -> i32
    %10 = fir.do_loop %arg6 = %c2 to %c1023 step %c1 iter_args(%arg7 = %5) -> (i32) {
      memref.store %arg7, %6[] : memref<i32>
      %11 = memref.load %6[] : memref<i32>
      %12 = arith.subi %11, %c1_i32 overflow<nsw> : i32
      %13 = fir.convert %12 : (i32) -> i64
      %14 = arith.index_cast %13 : i64 to index
      %15 = arith.subi %14, %c1 : index
      %16 = memref.load %7[%15] : memref<1024xf64>
      %17 = arith.addi %11, %c1_i32 overflow<nsw> : i32
      %18 = fir.convert %17 : (i32) -> i64
      %19 = arith.index_cast %18 : i64 to index
      %20 = arith.subi %19, %c1 : index
      %21 = memref.load %7[%20] : memref<1024xf64>
      %22 = arith.addf %16, %21 fastmath<contract> : f64
      %23 = fir.convert %11 : (i32) -> i64
      %24 = arith.index_cast %23 : i64 to index
      %25 = arith.subi %24, %c1 : index
      memref.store %22, %8[%25] : memref<1024xf64>
      %26 = memref.load %6[] : memref<i32>
      %27 = arith.addi %26, %9 overflow<nsw> : i32
      fir.result %27 : i32
    }
    memref.store %10, %6[] : memref<i32>
    omp.terminator
  }
}


// -----// IR Dump After CanonicalizerPass (canonicalize) //----- //
module {
  func.func @kernel(%arg0: !fir.ref<i32> {jit.arg_type = 1 : ui32, jit.literal_val = 0 : i64}, %arg1: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg2: !fir.ref<!fir.array<1024xf64>> {jit.arg_type = 1 : ui32}, %arg3: !fir.ref<!fir.array<1024xf64>> {jit.arg_type = 1 : ui32}, %arg4: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg5: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}) {
    %c1024 = arith.constant 1024 : index
    %c1_i32 = arith.constant 1 : i32
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c1023 = arith.constant 1023 : index
    %0 = fir.declare %arg0 {uniq_name = "_QFFrun_benchmarkEi"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %1 = fir.declare %arg1 {uniq_name = "_QFFrun_benchmarkEn"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %2 = fir.shape %c1024 : (index) -> !fir.shape<1>
    %3 = fir.declare %arg2(%2) {uniq_name = "_QFFrun_benchmarkEy"} : (!fir.ref<!fir.array<1024xf64>>, !fir.shape<1>) -> !fir.ref<!fir.array<1024xf64>>
    %4 = fir.declare %arg3(%2) {uniq_name = "_QFFrun_benchmarkEx"} : (!fir.ref<!fir.array<1024xf64>>, !fir.shape<1>) -> !fir.ref<!fir.array<1024xf64>>
    %5 = fir.convert %c2 : (index) -> i32
    %6 = fir.convert %0 : (!fir.ref<i32>) -> memref<i32>
    %7 = fir.convert %4 : (!fir.ref<!fir.array<1024xf64>>) -> memref<1024xf64>
    %8 = fir.convert %3 : (!fir.ref<!fir.array<1024xf64>>) -> memref<1024xf64>
    %9 = fir.convert %c1 : (index) -> i32
    %10 = fir.do_loop %arg6 = %c2 to %c1023 step %c1 iter_args(%arg7 = %5) -> (i32) {
      memref.store %arg7, %6[] : memref<i32>
      %11 = memref.load %6[] : memref<i32>
      %12 = arith.subi %11, %c1_i32 overflow<nsw> : i32
      %13 = fir.convert %12 : (i32) -> i64
      %14 = arith.index_cast %13 : i64 to index
      %15 = arith.subi %14, %c1 : index
      %16 = memref.load %7[%15] : memref<1024xf64>
      %17 = arith.addi %11, %c1_i32 overflow<nsw> : i32
      %18 = fir.convert %17 : (i32) -> i64
      %19 = arith.index_cast %18 : i64 to index
      %20 = arith.subi %19, %c1 : index
      %21 = memref.load %7[%20] : memref<1024xf64>
      %22 = arith.addf %16, %21 fastmath<contract> : f64
      %23 = fir.convert %11 : (i32) -> i64
      %24 = arith.index_cast %23 : i64 to index
      %25 = arith.subi %24, %c1 : index
      memref.store %22, %8[%25] : memref<1024xf64>
      %26 = memref.load %6[] : memref<i32>
      %27 = arith.addi %26, %9 overflow<nsw> : i32
      fir.result %27 : i32
    }
    memref.store %10, %6[] : memref<i32>
    omp.terminator
  }
}


// -----// IR Dump After (anonymous namespace)::CleanFIRLoopPass (jforce-clean-fir-loop) //----- //
func.func @kernel(%arg0: !fir.ref<i32> {jit.arg_type = 1 : ui32, jit.literal_val = 0 : i64}, %arg1: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg2: !fir.ref<!fir.array<1024xf64>> {jit.arg_type = 1 : ui32}, %arg3: !fir.ref<!fir.array<1024xf64>> {jit.arg_type = 1 : ui32}, %arg4: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg5: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}) {
  %c1024 = arith.constant 1024 : index
  %c1_i32 = arith.constant 1 : i32
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c1023 = arith.constant 1023 : index
  %0 = fir.declare %arg0 {uniq_name = "_QFFrun_benchmarkEi"} : (!fir.ref<i32>) -> !fir.ref<i32>
  %1 = fir.declare %arg1 {uniq_name = "_QFFrun_benchmarkEn"} : (!fir.ref<i32>) -> !fir.ref<i32>
  %2 = fir.shape %c1024 : (index) -> !fir.shape<1>
  %3 = fir.declare %arg2(%2) {uniq_name = "_QFFrun_benchmarkEy"} : (!fir.ref<!fir.array<1024xf64>>, !fir.shape<1>) -> !fir.ref<!fir.array<1024xf64>>
  %4 = fir.declare %arg3(%2) {uniq_name = "_QFFrun_benchmarkEx"} : (!fir.ref<!fir.array<1024xf64>>, !fir.shape<1>) -> !fir.ref<!fir.array<1024xf64>>
  %5 = fir.convert %c2 : (index) -> i32
  %6 = fir.convert %0 : (!fir.ref<i32>) -> memref<i32>
  %7 = fir.convert %4 : (!fir.ref<!fir.array<1024xf64>>) -> memref<1024xf64>
  %8 = fir.convert %3 : (!fir.ref<!fir.array<1024xf64>>) -> memref<1024xf64>
  %9 = fir.convert %c1 : (index) -> i32
  fir.do_loop %arg6 = %c2 to %c1023 step %c1 {
    %11 = fir.convert %arg6 : (index) -> i32
    memref.store %11, %6[] : memref<i32>
    %12 = memref.load %6[] : memref<i32>
    %13 = arith.subi %12, %c1_i32 overflow<nsw> : i32
    %14 = fir.convert %13 : (i32) -> i64
    %15 = arith.index_cast %14 : i64 to index
    %16 = arith.subi %15, %c1 : index
    %17 = memref.load %7[%16] : memref<1024xf64>
    %18 = arith.addi %12, %c1_i32 overflow<nsw> : i32
    %19 = fir.convert %18 : (i32) -> i64
    %20 = arith.index_cast %19 : i64 to index
    %21 = arith.subi %20, %c1 : index
    %22 = memref.load %7[%21] : memref<1024xf64>
    %23 = arith.addf %17, %22 fastmath<contract> : f64
    %24 = fir.convert %12 : (i32) -> i64
    %25 = arith.index_cast %24 : i64 to index
    %26 = arith.subi %25, %c1 : index
    memref.store %23, %8[%26] : memref<1024xf64>
    %27 = memref.load %6[] : memref<i32>
    %28 = arith.addi %27, %9 overflow<nsw> : i32
    memref.store %28, %6[] : memref<i32>
  }
  %10 = memref.load %6[] : memref<i32>
  memref.store %10, %6[] : memref<i32>
  omp.terminator
}

// -----// IR Dump After CanonicalizerPass (canonicalize) //----- //
module {
  func.func @kernel(%arg0: !fir.ref<i32> {jit.arg_type = 1 : ui32, jit.literal_val = 0 : i64}, %arg1: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg2: !fir.ref<!fir.array<1024xf64>> {jit.arg_type = 1 : ui32}, %arg3: !fir.ref<!fir.array<1024xf64>> {jit.arg_type = 1 : ui32}, %arg4: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg5: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}) {
    %c1024 = arith.constant 1024 : index
    %c1_i32 = arith.constant 1 : i32
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c1023 = arith.constant 1023 : index
    %0 = fir.declare %arg0 {uniq_name = "_QFFrun_benchmarkEi"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %1 = fir.declare %arg1 {uniq_name = "_QFFrun_benchmarkEn"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %2 = fir.shape %c1024 : (index) -> !fir.shape<1>
    %3 = fir.declare %arg2(%2) {uniq_name = "_QFFrun_benchmarkEy"} : (!fir.ref<!fir.array<1024xf64>>, !fir.shape<1>) -> !fir.ref<!fir.array<1024xf64>>
    %4 = fir.declare %arg3(%2) {uniq_name = "_QFFrun_benchmarkEx"} : (!fir.ref<!fir.array<1024xf64>>, !fir.shape<1>) -> !fir.ref<!fir.array<1024xf64>>
    %5 = fir.convert %0 : (!fir.ref<i32>) -> memref<i32>
    %6 = fir.convert %4 : (!fir.ref<!fir.array<1024xf64>>) -> memref<1024xf64>
    %7 = fir.convert %3 : (!fir.ref<!fir.array<1024xf64>>) -> memref<1024xf64>
    %8 = fir.convert %c1 : (index) -> i32
    fir.do_loop %arg6 = %c2 to %c1023 step %c1 {
      %10 = fir.convert %arg6 : (index) -> i32
      memref.store %10, %5[] : memref<i32>
      %11 = memref.load %5[] : memref<i32>
      %12 = arith.subi %11, %c1_i32 overflow<nsw> : i32
      %13 = fir.convert %12 : (i32) -> i64
      %14 = arith.index_cast %13 : i64 to index
      %15 = arith.subi %14, %c1 : index
      %16 = memref.load %6[%15] : memref<1024xf64>
      %17 = arith.addi %11, %c1_i32 overflow<nsw> : i32
      %18 = fir.convert %17 : (i32) -> i64
      %19 = arith.index_cast %18 : i64 to index
      %20 = arith.subi %19, %c1 : index
      %21 = memref.load %6[%20] : memref<1024xf64>
      %22 = arith.addf %16, %21 fastmath<contract> : f64
      %23 = fir.convert %11 : (i32) -> i64
      %24 = arith.index_cast %23 : i64 to index
      %25 = arith.subi %24, %c1 : index
      memref.store %22, %7[%25] : memref<1024xf64>
      %26 = memref.load %5[] : memref<i32>
      %27 = arith.addi %26, %8 overflow<nsw> : i32
      memref.store %27, %5[] : memref<i32>
    }
    %9 = memref.load %5[] : memref<i32>
    memref.store %9, %5[] : memref<i32>
    omp.terminator
  }
}


// -----// IR Dump After (anonymous namespace)::CleanFIROpsPass (jforce-clean-fir-op) //----- //
func.func @kernel(%arg0: !fir.ref<i32> {jit.arg_type = 1 : ui32, jit.literal_val = 0 : i64}, %arg1: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg2: !fir.ref<!fir.array<1024xf64>> {jit.arg_type = 1 : ui32}, %arg3: !fir.ref<!fir.array<1024xf64>> {jit.arg_type = 1 : ui32}, %arg4: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg5: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}) {
  %c1024 = arith.constant 1024 : index
  %c1_i32 = arith.constant 1 : i32
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c1023 = arith.constant 1023 : index
  %0 = fir.declare %arg0 {uniq_name = "_QFFrun_benchmarkEi"} : (!fir.ref<i32>) -> !fir.ref<i32>
  %1 = fir.declare %arg1 {uniq_name = "_QFFrun_benchmarkEn"} : (!fir.ref<i32>) -> !fir.ref<i32>
  %2 = fir.shape %c1024 : (index) -> !fir.shape<1>
  %3 = fir.declare %arg2(%2) {uniq_name = "_QFFrun_benchmarkEy"} : (!fir.ref<!fir.array<1024xf64>>, !fir.shape<1>) -> !fir.ref<!fir.array<1024xf64>>
  %4 = fir.declare %arg3(%2) {uniq_name = "_QFFrun_benchmarkEx"} : (!fir.ref<!fir.array<1024xf64>>, !fir.shape<1>) -> !fir.ref<!fir.array<1024xf64>>
  %5 = fir.convert %0 : (!fir.ref<i32>) -> memref<i32>
  %6 = fir.convert %4 : (!fir.ref<!fir.array<1024xf64>>) -> memref<1024xf64>
  %7 = fir.convert %3 : (!fir.ref<!fir.array<1024xf64>>) -> memref<1024xf64>
  %8 = fir.convert %c1 : (index) -> i32
  fir.do_loop %arg6 = %c2 to %c1023 step %c1 {
    %10 = arith.index_cast %arg6 : index to i32
    memref.store %10, %5[] : memref<i32>
    %11 = memref.load %5[] : memref<i32>
    %12 = arith.subi %11, %c1_i32 overflow<nsw> : i32
    %13 = arith.extsi %12 : i32 to i64
    %14 = arith.index_cast %13 : i64 to index
    %15 = arith.subi %14, %c1 : index
    %16 = memref.load %6[%15] : memref<1024xf64>
    %17 = arith.addi %11, %c1_i32 overflow<nsw> : i32
    %18 = arith.extsi %17 : i32 to i64
    %19 = arith.index_cast %18 : i64 to index
    %20 = arith.subi %19, %c1 : index
    %21 = memref.load %6[%20] : memref<1024xf64>
    %22 = arith.addf %16, %21 fastmath<contract> : f64
    %23 = arith.extsi %11 : i32 to i64
    %24 = arith.index_cast %23 : i64 to index
    %25 = arith.subi %24, %c1 : index
    memref.store %22, %7[%25] : memref<1024xf64>
    %26 = memref.load %5[] : memref<i32>
    %27 = arith.addi %26, %8 overflow<nsw> : i32
    memref.store %27, %5[] : memref<i32>
  }
  %9 = memref.load %5[] : memref<i32>
  memref.store %9, %5[] : memref<i32>
  omp.terminator
}

// -----// IR Dump After AffineDialectPromotion (promote-to-affine) //----- //
func.func @kernel(%arg0: !fir.ref<i32> {jit.arg_type = 1 : ui32, jit.literal_val = 0 : i64}, %arg1: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg2: !fir.ref<!fir.array<1024xf64>> {jit.arg_type = 1 : ui32}, %arg3: !fir.ref<!fir.array<1024xf64>> {jit.arg_type = 1 : ui32}, %arg4: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg5: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}) {
  %c1024 = arith.constant 1024 : index
  %c1_i32 = arith.constant 1 : i32
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c1023 = arith.constant 1023 : index
  %0 = fir.declare %arg0 {uniq_name = "_QFFrun_benchmarkEi"} : (!fir.ref<i32>) -> !fir.ref<i32>
  %1 = fir.declare %arg1 {uniq_name = "_QFFrun_benchmarkEn"} : (!fir.ref<i32>) -> !fir.ref<i32>
  %2 = fir.shape %c1024 : (index) -> !fir.shape<1>
  %3 = fir.declare %arg2(%2) {uniq_name = "_QFFrun_benchmarkEy"} : (!fir.ref<!fir.array<1024xf64>>, !fir.shape<1>) -> !fir.ref<!fir.array<1024xf64>>
  %4 = fir.declare %arg3(%2) {uniq_name = "_QFFrun_benchmarkEx"} : (!fir.ref<!fir.array<1024xf64>>, !fir.shape<1>) -> !fir.ref<!fir.array<1024xf64>>
  %5 = fir.convert %0 : (!fir.ref<i32>) -> memref<i32>
  %6 = fir.convert %4 : (!fir.ref<!fir.array<1024xf64>>) -> memref<1024xf64>
  %7 = fir.convert %3 : (!fir.ref<!fir.array<1024xf64>>) -> memref<1024xf64>
  %8 = fir.convert %c1 : (index) -> i32
  affine.for %arg6 = %c2 to affine_map<()[s0] -> (s0 + 1)>()[%c1023] {
    %10 = arith.index_cast %arg6 : index to i32
    memref.store %10, %5[] : memref<i32>
    %11 = memref.load %5[] : memref<i32>
    %12 = arith.subi %11, %c1_i32 overflow<nsw> : i32
    %13 = arith.extsi %12 : i32 to i64
    %14 = arith.index_cast %13 : i64 to index
    %15 = arith.subi %14, %c1 : index
    %16 = memref.load %6[%15] : memref<1024xf64>
    %17 = arith.addi %11, %c1_i32 overflow<nsw> : i32
    %18 = arith.extsi %17 : i32 to i64
    %19 = arith.index_cast %18 : i64 to index
    %20 = arith.subi %19, %c1 : index
    %21 = memref.load %6[%20] : memref<1024xf64>
    %22 = arith.addf %16, %21 fastmath<contract> : f64
    %23 = arith.extsi %11 : i32 to i64
    %24 = arith.index_cast %23 : i64 to index
    %25 = arith.subi %24, %c1 : index
    memref.store %22, %7[%25] : memref<1024xf64>
    %26 = memref.load %5[] : memref<i32>
    %27 = arith.addi %26, %8 overflow<nsw> : i32
    memref.store %27, %5[] : memref<i32>
  }
  %9 = memref.load %5[] : memref<i32>
  memref.store %9, %5[] : memref<i32>
  omp.terminator
}

// -----// IR Dump After AffineLoopNormalize (affine-loop-normalize) //----- //
func.func @kernel(%arg0: !fir.ref<i32> {jit.arg_type = 1 : ui32, jit.literal_val = 0 : i64}, %arg1: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg2: !fir.ref<!fir.array<1024xf64>> {jit.arg_type = 1 : ui32}, %arg3: !fir.ref<!fir.array<1024xf64>> {jit.arg_type = 1 : ui32}, %arg4: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg5: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}) {
  %c1024 = arith.constant 1024 : index
  %c1_i32 = arith.constant 1 : i32
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c1023 = arith.constant 1023 : index
  %0 = fir.declare %arg0 {uniq_name = "_QFFrun_benchmarkEi"} : (!fir.ref<i32>) -> !fir.ref<i32>
  %1 = fir.declare %arg1 {uniq_name = "_QFFrun_benchmarkEn"} : (!fir.ref<i32>) -> !fir.ref<i32>
  %2 = fir.shape %c1024 : (index) -> !fir.shape<1>
  %3 = fir.declare %arg2(%2) {uniq_name = "_QFFrun_benchmarkEy"} : (!fir.ref<!fir.array<1024xf64>>, !fir.shape<1>) -> !fir.ref<!fir.array<1024xf64>>
  %4 = fir.declare %arg3(%2) {uniq_name = "_QFFrun_benchmarkEx"} : (!fir.ref<!fir.array<1024xf64>>, !fir.shape<1>) -> !fir.ref<!fir.array<1024xf64>>
  %5 = fir.convert %0 : (!fir.ref<i32>) -> memref<i32>
  %6 = fir.convert %4 : (!fir.ref<!fir.array<1024xf64>>) -> memref<1024xf64>
  %7 = fir.convert %3 : (!fir.ref<!fir.array<1024xf64>>) -> memref<1024xf64>
  %8 = fir.convert %c1 : (index) -> i32
  affine.for %arg6 = 0 to 1022 {
    %10 = affine.apply affine_map<(d0) -> (d0 + 2)>(%arg6)
    %11 = arith.index_cast %10 : index to i32
    memref.store %11, %5[] : memref<i32>
    %12 = memref.load %5[] : memref<i32>
    %13 = arith.subi %12, %c1_i32 overflow<nsw> : i32
    %14 = arith.extsi %13 : i32 to i64
    %15 = arith.index_cast %14 : i64 to index
    %16 = arith.subi %15, %c1 : index
    %17 = memref.load %6[%16] : memref<1024xf64>
    %18 = arith.addi %12, %c1_i32 overflow<nsw> : i32
    %19 = arith.extsi %18 : i32 to i64
    %20 = arith.index_cast %19 : i64 to index
    %21 = arith.subi %20, %c1 : index
    %22 = memref.load %6[%21] : memref<1024xf64>
    %23 = arith.addf %17, %22 fastmath<contract> : f64
    %24 = arith.extsi %12 : i32 to i64
    %25 = arith.index_cast %24 : i64 to index
    %26 = arith.subi %25, %c1 : index
    memref.store %23, %7[%26] : memref<1024xf64>
    %27 = memref.load %5[] : memref<i32>
    %28 = arith.addi %27, %8 overflow<nsw> : i32
    memref.store %28, %5[] : memref<i32>
  }
  %9 = memref.load %5[] : memref<i32>
  memref.store %9, %5[] : memref<i32>
  omp.terminator
}

// -----// IR Dump After CanonicalizerPass (canonicalize) //----- //
#map = affine_map<(d0) -> (d0 + 2)>
module {
  func.func @kernel(%arg0: !fir.ref<i32> {jit.arg_type = 1 : ui32, jit.literal_val = 0 : i64}, %arg1: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg2: !fir.ref<!fir.array<1024xf64>> {jit.arg_type = 1 : ui32}, %arg3: !fir.ref<!fir.array<1024xf64>> {jit.arg_type = 1 : ui32}, %arg4: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg5: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}) {
    %c1024 = arith.constant 1024 : index
    %c1_i32 = arith.constant 1 : i32
    %c1 = arith.constant 1 : index
    %0 = fir.declare %arg0 {uniq_name = "_QFFrun_benchmarkEi"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %1 = fir.declare %arg1 {uniq_name = "_QFFrun_benchmarkEn"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %2 = fir.shape %c1024 : (index) -> !fir.shape<1>
    %3 = fir.declare %arg2(%2) {uniq_name = "_QFFrun_benchmarkEy"} : (!fir.ref<!fir.array<1024xf64>>, !fir.shape<1>) -> !fir.ref<!fir.array<1024xf64>>
    %4 = fir.declare %arg3(%2) {uniq_name = "_QFFrun_benchmarkEx"} : (!fir.ref<!fir.array<1024xf64>>, !fir.shape<1>) -> !fir.ref<!fir.array<1024xf64>>
    %5 = fir.convert %0 : (!fir.ref<i32>) -> memref<i32>
    %6 = fir.convert %4 : (!fir.ref<!fir.array<1024xf64>>) -> memref<1024xf64>
    %7 = fir.convert %3 : (!fir.ref<!fir.array<1024xf64>>) -> memref<1024xf64>
    %8 = fir.convert %c1 : (index) -> i32
    affine.for %arg6 = 0 to 1022 {
      %10 = affine.apply #map(%arg6)
      %11 = arith.index_cast %10 : index to i32
      memref.store %11, %5[] : memref<i32>
      %12 = memref.load %5[] : memref<i32>
      %13 = arith.subi %12, %c1_i32 overflow<nsw> : i32
      %14 = arith.index_cast %13 : i32 to index
      %15 = arith.subi %14, %c1 : index
      %16 = memref.load %6[%15] : memref<1024xf64>
      %17 = arith.addi %12, %c1_i32 overflow<nsw> : i32
      %18 = arith.index_cast %17 : i32 to index
      %19 = arith.subi %18, %c1 : index
      %20 = memref.load %6[%19] : memref<1024xf64>
      %21 = arith.addf %16, %20 fastmath<contract> : f64
      %22 = arith.index_cast %12 : i32 to index
      %23 = arith.subi %22, %c1 : index
      memref.store %21, %7[%23] : memref<1024xf64>
      %24 = memref.load %5[] : memref<i32>
      %25 = arith.addi %24, %8 overflow<nsw> : i32
      memref.store %25, %5[] : memref<i32>
    }
    %9 = memref.load %5[] : memref<i32>
    memref.store %9, %5[] : memref<i32>
    omp.terminator
  }
}


[DEBUG]compare reading and storeOp
[DEBUG] memref.store %9, %5[] : memref<i32> 
[DEBUG] %9 = memref.load %5[] : memref<i32> 
[DEBUG]compare reading and storeOp
[DEBUG] memref.store %24, %5[] : memref<i32> 
[DEBUG] %24 = arith.addi %23, %8 overflow<nsw> : i32 
[DEBUG]Unexpected readOp: 
[DEBUG] %24 = arith.addi %23, %8 overflow<nsw> : i32 
[DEBUG]compare reading and storeOp
[DEBUG] memref.store %20, %7[%22] : memref<1024xf64> 
[DEBUG] %20 = arith.addf %15, %19 fastmath<contract> : f64 
[DEBUG]Unexpected readOp: 
[DEBUG] %20 = arith.addf %15, %19 fastmath<contract> : f64 
[DEBUG]Dependency chain relies on LoadOp!
[DEBUG]compare reading and storeOp
[DEBUG] memref.store %10, %5[] : memref<i32> 
[DEBUG] %10 = arith.index_cast %9 : index to i32 
[DEBUG]Unexpected readOp: 
[DEBUG] %10 = arith.index_cast %9 : index to i32 
[DEBUG]compare reading and storeOp
[DEBUG] memref.store %22, %5[] : memref<i32> 
[DEBUG] %22 = arith.addi %10, %8 overflow<nsw> : i32 
[DEBUG]Unexpected readOp: 
[DEBUG] %22 = arith.addi %10, %8 overflow<nsw> : i32 
[DEBUG]compare reading and storeOp
[DEBUG] memref.store %16, %5[] : memref<i32> 
[DEBUG] %16 = arith.addi %15, %8 overflow<nsw> : i32 
[DEBUG]Unexpected readOp: 
[DEBUG] %16 = arith.addi %15, %8 overflow<nsw> : i32 
[DEBUG]compare reading and storeOp
[DEBUG] memref.store %20, %7[%22] : memref<1024xf64> 
[DEBUG] %20 = arith.addf %15, %19 fastmath<contract> : f64 
[DEBUG]Unexpected readOp: 
[DEBUG] %20 = arith.addf %15, %19 fastmath<contract> : f64 
[DEBUG]Dependency chain relies on LoadOp!
[DEBUG]compare reading and storeOp
[DEBUG] memref.store %9, %5[] : memref<i32> 
[DEBUG] %9 = arith.addi %8, %c1023_i32 overflow<nsw> : i32 
[DEBUG]Unexpected readOp: 
[DEBUG] %9 = arith.addi %8, %c1023_i32 overflow<nsw> : i32 
[DEBUG]compare reading and storeOp
[DEBUG] memref.store %20, %7[%22] : memref<1024xf64> 
[DEBUG] %20 = arith.addf %15, %19 fastmath<contract> : f64 
[DEBUG]Unexpected readOp: 
[DEBUG] %20 = arith.addf %15, %19 fastmath<contract> : f64 
[DEBUG]Dependency chain relies on LoadOp!
// -----// IR Dump After (anonymous namespace)::OptimizeMemOpsPass (jforce-optimize-mem-ops) //----- //
func.func @kernel(%arg0: !fir.ref<i32> {jit.arg_type = 1 : ui32, jit.literal_val = 0 : i64}, %arg1: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg2: !fir.ref<!fir.array<1024xf64>> {jit.arg_type = 1 : ui32}, %arg3: !fir.ref<!fir.array<1024xf64>> {jit.arg_type = 1 : ui32}, %arg4: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg5: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}) {
  %c1023_i32 = arith.constant 1023 : i32
  %c1024 = arith.constant 1024 : index
  %c1_i32 = arith.constant 1 : i32
  %c1 = arith.constant 1 : index
  %0 = fir.declare %arg0 {uniq_name = "_QFFrun_benchmarkEi"} : (!fir.ref<i32>) -> !fir.ref<i32>
  %1 = fir.declare %arg1 {uniq_name = "_QFFrun_benchmarkEn"} : (!fir.ref<i32>) -> !fir.ref<i32>
  %2 = fir.shape %c1024 : (index) -> !fir.shape<1>
  %3 = fir.declare %arg2(%2) {uniq_name = "_QFFrun_benchmarkEy"} : (!fir.ref<!fir.array<1024xf64>>, !fir.shape<1>) -> !fir.ref<!fir.array<1024xf64>>
  %4 = fir.declare %arg3(%2) {uniq_name = "_QFFrun_benchmarkEx"} : (!fir.ref<!fir.array<1024xf64>>, !fir.shape<1>) -> !fir.ref<!fir.array<1024xf64>>
  %5 = fir.convert %0 : (!fir.ref<i32>) -> memref<i32>
  %6 = fir.convert %4 : (!fir.ref<!fir.array<1024xf64>>) -> memref<1024xf64>
  %7 = fir.convert %3 : (!fir.ref<!fir.array<1024xf64>>) -> memref<1024xf64>
  %8 = fir.convert %c1 : (index) -> i32
  affine.for %arg6 = 0 to 1022 {
    %10 = affine.apply affine_map<(d0) -> (d0 + 2)>(%arg6)
    %11 = arith.index_cast %10 : index to i32
    %12 = arith.subi %11, %c1_i32 overflow<nsw> : i32
    %13 = arith.index_cast %12 : i32 to index
    %14 = arith.subi %13, %c1 : index
    %15 = memref.load %6[%14] : memref<1024xf64>
    %16 = arith.addi %11, %c1_i32 overflow<nsw> : i32
    %17 = arith.index_cast %16 : i32 to index
    %18 = arith.subi %17, %c1 : index
    %19 = memref.load %6[%18] : memref<1024xf64>
    %20 = arith.addf %15, %19 fastmath<contract> : f64
    %21 = arith.index_cast %11 : i32 to index
    %22 = arith.subi %21, %c1 : index
    memref.store %20, %7[%22] : memref<1024xf64>
  }
  %9 = arith.addi %8, %c1023_i32 overflow<nsw> : i32
  memref.store %9, %5[] : memref<i32>
  omp.terminator
}

// -----// IR Dump After CanonicalizerPass (canonicalize) //----- //
#map = affine_map<(d0) -> (d0 + 2)>
module {
  func.func @kernel(%arg0: !fir.ref<i32> {jit.arg_type = 1 : ui32, jit.literal_val = 0 : i64}, %arg1: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg2: !fir.ref<!fir.array<1024xf64>> {jit.arg_type = 1 : ui32}, %arg3: !fir.ref<!fir.array<1024xf64>> {jit.arg_type = 1 : ui32}, %arg4: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg5: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}) {
    %c1023_i32 = arith.constant 1023 : i32
    %c1024 = arith.constant 1024 : index
    %c1_i32 = arith.constant 1 : i32
    %c1 = arith.constant 1 : index
    %0 = fir.declare %arg0 {uniq_name = "_QFFrun_benchmarkEi"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %1 = fir.declare %arg1 {uniq_name = "_QFFrun_benchmarkEn"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %2 = fir.shape %c1024 : (index) -> !fir.shape<1>
    %3 = fir.declare %arg2(%2) {uniq_name = "_QFFrun_benchmarkEy"} : (!fir.ref<!fir.array<1024xf64>>, !fir.shape<1>) -> !fir.ref<!fir.array<1024xf64>>
    %4 = fir.declare %arg3(%2) {uniq_name = "_QFFrun_benchmarkEx"} : (!fir.ref<!fir.array<1024xf64>>, !fir.shape<1>) -> !fir.ref<!fir.array<1024xf64>>
    %5 = fir.convert %0 : (!fir.ref<i32>) -> memref<i32>
    %6 = fir.convert %4 : (!fir.ref<!fir.array<1024xf64>>) -> memref<1024xf64>
    %7 = fir.convert %3 : (!fir.ref<!fir.array<1024xf64>>) -> memref<1024xf64>
    %8 = fir.convert %c1 : (index) -> i32
    affine.for %arg6 = 0 to 1022 {
      %10 = affine.apply #map(%arg6)
      %11 = arith.index_cast %10 : index to i32
      %12 = arith.subi %11, %c1_i32 overflow<nsw> : i32
      %13 = arith.index_cast %12 : i32 to index
      %14 = arith.subi %13, %c1 : index
      %15 = memref.load %6[%14] : memref<1024xf64>
      %16 = arith.addi %11, %c1_i32 overflow<nsw> : i32
      %17 = arith.index_cast %16 : i32 to index
      %18 = arith.subi %17, %c1 : index
      %19 = memref.load %6[%18] : memref<1024xf64>
      %20 = arith.addf %15, %19 fastmath<contract> : f64
      %21 = arith.index_cast %11 : i32 to index
      %22 = arith.subi %21, %c1 : index
      memref.store %20, %7[%22] : memref<1024xf64>
    }
    %9 = arith.addi %8, %c1023_i32 overflow<nsw> : i32
    memref.store %9, %5[] : memref<i32>
    omp.terminator
  }
}


func.func @kernel(%arg0: !fir.ref<i32> {jit.arg_type = 1 : ui32, jit.literal_val = 0 : i64}, %arg1: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg2: !fir.ref<!fir.array<1024xf64>> {jit.arg_type = 1 : ui32}, %arg3: !fir.ref<!fir.array<1024xf64>> {jit.arg_type = 1 : ui32}, %arg4: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg5: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}) {
  %c1023_i32 = arith.constant 1023 : i32
  %c1024 = arith.constant 1024 : index
  %c1_i32 = arith.constant 1 : i32
  %c1 = arith.constant 1 : index
  %0 = fir.declare %arg0 {uniq_name = "_QFFrun_benchmarkEi"} : (!fir.ref<i32>) -> !fir.ref<i32>
  %1 = fir.declare %arg1 {uniq_name = "_QFFrun_benchmarkEn"} : (!fir.ref<i32>) -> !fir.ref<i32>
  %2 = fir.shape %c1024 : (index) -> !fir.shape<1>
  %3 = fir.declare %arg2(%2) {uniq_name = "_QFFrun_benchmarkEy"} : (!fir.ref<!fir.array<1024xf64>>, !fir.shape<1>) -> !fir.ref<!fir.array<1024xf64>>
  %4 = fir.declare %arg3(%2) {uniq_name = "_QFFrun_benchmarkEx"} : (!fir.ref<!fir.array<1024xf64>>, !fir.shape<1>) -> !fir.ref<!fir.array<1024xf64>>
  %5 = fir.convert %0 : (!fir.ref<i32>) -> memref<i32>
  %6 = fir.convert %4 : (!fir.ref<!fir.array<1024xf64>>) -> memref<1024xf64>
  %7 = fir.convert %3 : (!fir.ref<!fir.array<1024xf64>>) -> memref<1024xf64>
  %8 = fir.convert %c1 : (index) -> i32
  affine.for %arg6 = 0 to 1022 {
    %10 = affine.apply affine_map<(d0) -> (d0 + 2)>(%arg6)
    %11 = arith.index_cast %10 : index to i32
    %12 = arith.subi %11, %c1_i32 overflow<nsw> : i32
    %13 = arith.index_cast %12 : i32 to index
    %14 = arith.subi %13, %c1 : index
    %15 = memref.load %6[%14] : memref<1024xf64>
    %16 = arith.addi %11, %c1_i32 overflow<nsw> : i32
    %17 = arith.index_cast %16 : i32 to index
    %18 = arith.subi %17, %c1 : index
    %19 = memref.load %6[%18] : memref<1024xf64>
    %20 = arith.addf %15, %19 fastmath<contract> : f64
    %21 = arith.index_cast %11 : i32 to index
    %22 = arith.subi %21, %c1 : index
    memref.store %20, %7[%22] : memref<1024xf64>
  }
  %9 = arith.addi %8, %c1023_i32 overflow<nsw> : i32
  memref.store %9, %5[] : memref<i32>
  omp.terminator
}
// -----// IR Dump After (anonymous namespace)::LoopSinkingPass (jforce-loop-sink) //----- //
func.func @kernel(%arg0: !fir.ref<i32> {jit.arg_type = 1 : ui32, jit.literal_val = 0 : i64}, %arg1: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg2: !fir.ref<!fir.array<1024xf64>> {jit.arg_type = 1 : ui32}, %arg3: !fir.ref<!fir.array<1024xf64>> {jit.arg_type = 1 : ui32}, %arg4: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg5: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}) {
  %c1023_i32 = arith.constant 1023 : i32
  %c1024 = arith.constant 1024 : index
  %c1_i32 = arith.constant 1 : i32
  %c1 = arith.constant 1 : index
  %0 = fir.declare %arg0 {uniq_name = "_QFFrun_benchmarkEi"} : (!fir.ref<i32>) -> !fir.ref<i32>
  %1 = fir.declare %arg1 {uniq_name = "_QFFrun_benchmarkEn"} : (!fir.ref<i32>) -> !fir.ref<i32>
  %2 = fir.shape %c1024 : (index) -> !fir.shape<1>
  %3 = fir.declare %arg2(%2) {uniq_name = "_QFFrun_benchmarkEy"} : (!fir.ref<!fir.array<1024xf64>>, !fir.shape<1>) -> !fir.ref<!fir.array<1024xf64>>
  %4 = fir.declare %arg3(%2) {uniq_name = "_QFFrun_benchmarkEx"} : (!fir.ref<!fir.array<1024xf64>>, !fir.shape<1>) -> !fir.ref<!fir.array<1024xf64>>
  %5 = fir.convert %0 : (!fir.ref<i32>) -> memref<i32>
  %6 = fir.convert %4 : (!fir.ref<!fir.array<1024xf64>>) -> memref<1024xf64>
  %7 = fir.convert %3 : (!fir.ref<!fir.array<1024xf64>>) -> memref<1024xf64>
  %8 = fir.convert %c1 : (index) -> i32
  affine.for %arg6 = 0 to 1022 {
    %10 = affine.apply affine_map<(d0) -> (d0 + 2)>(%arg6)
    %11 = arith.index_cast %10 : index to i32
    %12 = arith.subi %11, %c1_i32 overflow<nsw> : i32
    %13 = arith.index_cast %12 : i32 to index
    %14 = arith.subi %13, %c1 : index
    %15 = memref.load %6[%14] : memref<1024xf64>
    %16 = arith.addi %11, %c1_i32 overflow<nsw> : i32
    %17 = arith.index_cast %16 : i32 to index
    %18 = arith.subi %17, %c1 : index
    %19 = memref.load %6[%18] : memref<1024xf64>
    %20 = arith.addf %15, %19 fastmath<contract> : f64
    %21 = arith.index_cast %11 : i32 to index
    %22 = arith.subi %21, %c1 : index
    memref.store %20, %7[%22] : memref<1024xf64>
  }
  %9 = arith.addi %8, %c1023_i32 overflow<nsw> : i32
  memref.store %9, %5[] : memref<i32>
  omp.terminator
}

// -----// IR Dump After (anonymous namespace)::AffineCFGPass (enzyme-affinecfg) //----- //
module {
  func.func @kernel(%arg0: !fir.ref<i32> {jit.arg_type = 1 : ui32, jit.literal_val = 0 : i64}, %arg1: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg2: !fir.ref<!fir.array<1024xf64>> {jit.arg_type = 1 : ui32}, %arg3: !fir.ref<!fir.array<1024xf64>> {jit.arg_type = 1 : ui32}, %arg4: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg5: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}) {
    %c1023_i32 = arith.constant 1023 : i32
    %c1024 = arith.constant 1024 : index
    %c1 = arith.constant 1 : index
    %0 = fir.declare %arg0 {uniq_name = "_QFFrun_benchmarkEi"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %1 = fir.declare %arg1 {uniq_name = "_QFFrun_benchmarkEn"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %2 = fir.shape %c1024 : (index) -> !fir.shape<1>
    %3 = fir.declare %arg2(%2) {uniq_name = "_QFFrun_benchmarkEy"} : (!fir.ref<!fir.array<1024xf64>>, !fir.shape<1>) -> !fir.ref<!fir.array<1024xf64>>
    %4 = fir.declare %arg3(%2) {uniq_name = "_QFFrun_benchmarkEx"} : (!fir.ref<!fir.array<1024xf64>>, !fir.shape<1>) -> !fir.ref<!fir.array<1024xf64>>
    %5 = fir.convert %0 : (!fir.ref<i32>) -> memref<i32>
    %6 = fir.convert %4 : (!fir.ref<!fir.array<1024xf64>>) -> memref<1024xf64>
    %7 = fir.convert %3 : (!fir.ref<!fir.array<1024xf64>>) -> memref<1024xf64>
    %8 = fir.convert %c1 : (index) -> i32
    affine.parallel (%arg6) = (0) to (1022) {
      %10 = affine.load %6[%arg6] : memref<1024xf64>
      %11 = affine.load %6[%arg6 + 2] : memref<1024xf64>
      %12 = arith.addf %10, %11 fastmath<contract> : f64
      affine.store %12, %7[%arg6 + 1] : memref<1024xf64>
    }
    %9 = arith.addi %8, %c1023_i32 overflow<nsw> : i32
    affine.store %9, %5[] : memref<i32>
    omp.terminator
  }
}


module {
  func.func @outlined_affinefor_94104755733728(%arg0: memref<1024xf64>, %arg1: memref<1024xf64>) {
    affine.parallel (%arg2) = (0) to (1022) {
      %0 = affine.load %arg0[%arg2] : memref<1024xf64>
      %1 = affine.load %arg0[%arg2 + 2] : memref<1024xf64>
      %2 = arith.addf %0, %1 fastmath<contract> : f64
      affine.store %2, %arg1[%arg2 + 1] : memref<1024xf64>
    }
    return
  }
  func.func @kernel(%arg0: !fir.ref<i32> {jit.arg_type = 1 : ui32, jit.literal_val = 0 : i64}, %arg1: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg2: !fir.ref<!fir.array<1024xf64>> {jit.arg_type = 1 : ui32}, %arg3: !fir.ref<!fir.array<1024xf64>> {jit.arg_type = 1 : ui32}, %arg4: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg5: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}) {
    %c1023_i32 = arith.constant 1023 : i32
    %c1024 = arith.constant 1024 : index
    %c1 = arith.constant 1 : index
    %0 = fir.declare %arg0 {uniq_name = "_QFFrun_benchmarkEi"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %1 = fir.declare %arg1 {uniq_name = "_QFFrun_benchmarkEn"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %2 = fir.shape %c1024 : (index) -> !fir.shape<1>
    %3 = fir.declare %arg2(%2) {uniq_name = "_QFFrun_benchmarkEy"} : (!fir.ref<!fir.array<1024xf64>>, !fir.shape<1>) -> !fir.ref<!fir.array<1024xf64>>
    %4 = fir.declare %arg3(%2) {uniq_name = "_QFFrun_benchmarkEx"} : (!fir.ref<!fir.array<1024xf64>>, !fir.shape<1>) -> !fir.ref<!fir.array<1024xf64>>
    %5 = fir.convert %0 : (!fir.ref<i32>) -> memref<i32>
    %6 = fir.convert %4 : (!fir.ref<!fir.array<1024xf64>>) -> memref<1024xf64>
    %7 = fir.convert %3 : (!fir.ref<!fir.array<1024xf64>>) -> memref<1024xf64>
    %8 = fir.convert %c1 : (index) -> i32
    call @outlined_affinefor_94104755733728(%6, %7) : (memref<1024xf64>, memref<1024xf64>) -> ()
    %9 = arith.addi %8, %c1023_i32 overflow<nsw> : i32
    affine.store %9, %5[] : memref<i32>
    omp.terminator
  }
}
// -----// IR Dump After (anonymous namespace)::OutlineAffinePass (jforce-outline-affine) //----- //
module {
  func.func @outlined_affinefor_94104755733728(%arg0: memref<1024xf64>, %arg1: memref<1024xf64>) {
    affine.parallel (%arg2) = (0) to (1022) {
      %0 = affine.load %arg0[%arg2] : memref<1024xf64>
      %1 = affine.load %arg0[%arg2 + 2] : memref<1024xf64>
      %2 = arith.addf %0, %1 fastmath<contract> : f64
      affine.store %2, %arg1[%arg2 + 1] : memref<1024xf64>
    }
    return
  }
  func.func @kernel(%arg0: !fir.ref<i32> {jit.arg_type = 1 : ui32, jit.literal_val = 0 : i64}, %arg1: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg2: !fir.ref<!fir.array<1024xf64>> {jit.arg_type = 1 : ui32}, %arg3: !fir.ref<!fir.array<1024xf64>> {jit.arg_type = 1 : ui32}, %arg4: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg5: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}) {
    %c1023_i32 = arith.constant 1023 : i32
    %c1024 = arith.constant 1024 : index
    %c1 = arith.constant 1 : index
    %0 = fir.declare %arg0 {uniq_name = "_QFFrun_benchmarkEi"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %1 = fir.declare %arg1 {uniq_name = "_QFFrun_benchmarkEn"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %2 = fir.shape %c1024 : (index) -> !fir.shape<1>
    %3 = fir.declare %arg2(%2) {uniq_name = "_QFFrun_benchmarkEy"} : (!fir.ref<!fir.array<1024xf64>>, !fir.shape<1>) -> !fir.ref<!fir.array<1024xf64>>
    %4 = fir.declare %arg3(%2) {uniq_name = "_QFFrun_benchmarkEx"} : (!fir.ref<!fir.array<1024xf64>>, !fir.shape<1>) -> !fir.ref<!fir.array<1024xf64>>
    %5 = fir.convert %0 : (!fir.ref<i32>) -> memref<i32>
    %6 = fir.convert %4 : (!fir.ref<!fir.array<1024xf64>>) -> memref<1024xf64>
    %7 = fir.convert %3 : (!fir.ref<!fir.array<1024xf64>>) -> memref<1024xf64>
    %8 = fir.convert %c1 : (index) -> i32
    call @outlined_affinefor_94104755733728(%6, %7) : (memref<1024xf64>, memref<1024xf64>) -> ()
    %9 = arith.addi %8, %c1023_i32 overflow<nsw> : i32
    affine.store %9, %5[] : memref<i32>
    omp.terminator
  }
}


// -----// IR Dump After (anonymous namespace)::AffineToStableHLORaisingPass (enzyme-affine-to-stablehlo) //----- //
func.func @kernel(%arg0: !fir.ref<i32> {jit.arg_type = 1 : ui32, jit.literal_val = 0 : i64}, %arg1: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg2: !fir.ref<!fir.array<1024xf64>> {jit.arg_type = 1 : ui32}, %arg3: !fir.ref<!fir.array<1024xf64>> {jit.arg_type = 1 : ui32}, %arg4: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg5: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}) {
  %c1023_i32 = arith.constant 1023 : i32
  %c1024 = arith.constant 1024 : index
  %c1 = arith.constant 1 : index
  %0 = fir.declare %arg0 {uniq_name = "_QFFrun_benchmarkEi"} : (!fir.ref<i32>) -> !fir.ref<i32>
  %1 = fir.declare %arg1 {uniq_name = "_QFFrun_benchmarkEn"} : (!fir.ref<i32>) -> !fir.ref<i32>
  %2 = fir.shape %c1024 : (index) -> !fir.shape<1>
  %3 = fir.declare %arg2(%2) {uniq_name = "_QFFrun_benchmarkEy"} : (!fir.ref<!fir.array<1024xf64>>, !fir.shape<1>) -> !fir.ref<!fir.array<1024xf64>>
  %4 = fir.declare %arg3(%2) {uniq_name = "_QFFrun_benchmarkEx"} : (!fir.ref<!fir.array<1024xf64>>, !fir.shape<1>) -> !fir.ref<!fir.array<1024xf64>>
  %5 = fir.convert %0 : (!fir.ref<i32>) -> memref<i32>
  %6 = fir.convert %4 : (!fir.ref<!fir.array<1024xf64>>) -> memref<1024xf64>
  %7 = fir.convert %3 : (!fir.ref<!fir.array<1024xf64>>) -> memref<1024xf64>
  %8 = fir.convert %c1 : (index) -> i32
  call @outlined_affinefor_94104755733728(%6, %7) : (memref<1024xf64>, memref<1024xf64>) -> ()
  %9 = arith.addi %8, %c1023_i32 overflow<nsw> : i32
  affine.store %9, %5[] : memref<i32>
  omp.terminator
}

// -----// IR Dump After (anonymous namespace)::AffineToStableHLORaisingPass (enzyme-affine-to-stablehlo) //----- //
func.func @outlined_affinefor_94104755733728(%arg0: memref<1024xf64>, %arg1: memref<1024xf64>) {
  affine.parallel (%arg2) = (0) to (1022) {
    %0 = affine.load %arg0[%arg2] : memref<1024xf64>
    %1 = affine.load %arg0[%arg2 + 2] : memref<1024xf64>
    %2 = arith.addf %0, %1 fastmath<contract> : f64
    affine.store %2, %arg1[%arg2 + 1] : memref<1024xf64>
  }
  return
}

// -----// IR Dump After CanonicalizerPass (canonicalize) //----- //
module {
  func.func @outlined_affinefor_94104755733728(%arg0: memref<1024xf64>, %arg1: memref<1024xf64>) {
    affine.parallel (%arg2) = (0) to (1022) {
      %0 = affine.load %arg0[%arg2] : memref<1024xf64>
      %1 = affine.load %arg0[%arg2 + 2] : memref<1024xf64>
      %2 = arith.addf %0, %1 fastmath<contract> : f64
      affine.store %2, %arg1[%arg2 + 1] : memref<1024xf64>
    }
    return
  }
  func.func @kernel(%arg0: !fir.ref<i32> {jit.arg_type = 1 : ui32, jit.literal_val = 0 : i64}, %arg1: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg2: !fir.ref<!fir.array<1024xf64>> {jit.arg_type = 1 : ui32}, %arg3: !fir.ref<!fir.array<1024xf64>> {jit.arg_type = 1 : ui32}, %arg4: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg5: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}) {
    %c1023_i32 = arith.constant 1023 : i32
    %c1024 = arith.constant 1024 : index
    %c1 = arith.constant 1 : index
    %0 = fir.declare %arg0 {uniq_name = "_QFFrun_benchmarkEi"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %1 = fir.declare %arg1 {uniq_name = "_QFFrun_benchmarkEn"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %2 = fir.shape %c1024 : (index) -> !fir.shape<1>
    %3 = fir.declare %arg2(%2) {uniq_name = "_QFFrun_benchmarkEy"} : (!fir.ref<!fir.array<1024xf64>>, !fir.shape<1>) -> !fir.ref<!fir.array<1024xf64>>
    %4 = fir.declare %arg3(%2) {uniq_name = "_QFFrun_benchmarkEx"} : (!fir.ref<!fir.array<1024xf64>>, !fir.shape<1>) -> !fir.ref<!fir.array<1024xf64>>
    %5 = fir.convert %0 : (!fir.ref<i32>) -> memref<i32>
    %6 = fir.convert %4 : (!fir.ref<!fir.array<1024xf64>>) -> memref<1024xf64>
    %7 = fir.convert %3 : (!fir.ref<!fir.array<1024xf64>>) -> memref<1024xf64>
    %8 = fir.convert %c1 : (index) -> i32
    call @outlined_affinefor_94104755733728(%6, %7) : (memref<1024xf64>, memref<1024xf64>) -> ()
    %9 = arith.addi %8, %c1023_i32 overflow<nsw> : i32
    affine.store %9, %5[] : memref<i32>
    omp.terminator
  }
  func.func private @outlined_affinefor_94104755733728_raised(%arg0: tensor<1024xf64>, %arg1: tensor<1024xf64>) -> (tensor<1024xf64>, tensor<1024xf64>) {
    %c = stablehlo.constant dense<1> : tensor<i64>
    %0 = stablehlo.slice %arg0 [0:1022] : (tensor<1024xf64>) -> tensor<1022xf64>
    %1 = stablehlo.reshape %0 : (tensor<1022xf64>) -> tensor<1022xf64>
    %2 = stablehlo.slice %arg0 [2:1024] : (tensor<1024xf64>) -> tensor<1022xf64>
    %3 = stablehlo.reshape %2 : (tensor<1022xf64>) -> tensor<1022xf64>
    %4 = arith.addf %1, %3 fastmath<contract> : tensor<1022xf64>
    %5 = stablehlo.broadcast_in_dim %4, dims = [0] : (tensor<1022xf64>) -> tensor<1022xf64>
    %6 = stablehlo.dynamic_update_slice %arg1, %5, %c : (tensor<1024xf64>, tensor<1022xf64>, tensor<i64>) -> tensor<1024xf64>
    return %arg0, %6 : tensor<1024xf64>, tensor<1024xf64>
  }
}


The outlinedFuncName is: outlined_affinefor_94104755733728_raised, which does not match the pattern. Skip this callee.
// -----// IR Dump After (anonymous namespace)::SwitchOutlineFuncPass (jforce-switch-outline-function) //----- //
module {
  func.func @kernel(%arg0: !fir.ref<i32> {jit.arg_type = 1 : ui32, jit.literal_val = 0 : i64}, %arg1: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg2: !fir.ref<!fir.array<1024xf64>> {jit.arg_type = 1 : ui32}, %arg3: !fir.ref<!fir.array<1024xf64>> {jit.arg_type = 1 : ui32}, %arg4: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg5: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}) {
    %c1023_i32 = arith.constant 1023 : i32
    %c1024 = arith.constant 1024 : index
    %c1 = arith.constant 1 : index
    %0 = fir.declare %arg0 {uniq_name = "_QFFrun_benchmarkEi"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %1 = fir.declare %arg1 {uniq_name = "_QFFrun_benchmarkEn"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %2 = fir.shape %c1024 : (index) -> !fir.shape<1>
    %3 = fir.declare %arg2(%2) {uniq_name = "_QFFrun_benchmarkEy"} : (!fir.ref<!fir.array<1024xf64>>, !fir.shape<1>) -> !fir.ref<!fir.array<1024xf64>>
    %4 = fir.declare %arg3(%2) {uniq_name = "_QFFrun_benchmarkEx"} : (!fir.ref<!fir.array<1024xf64>>, !fir.shape<1>) -> !fir.ref<!fir.array<1024xf64>>
    %5 = fir.convert %0 : (!fir.ref<i32>) -> memref<i32>
    %6 = fir.convert %4 : (!fir.ref<!fir.array<1024xf64>>) -> memref<1024xf64>
    %7 = fir.convert %3 : (!fir.ref<!fir.array<1024xf64>>) -> memref<1024xf64>
    %8 = fir.convert %c1 : (index) -> i32
    %9 = bufferization.to_tensor %6 : memref<1024xf64> to tensor<1024xf64>
    %10 = bufferization.to_tensor %7 : memref<1024xf64> to tensor<1024xf64>
    %11:2 = call @outlined_affinefor_94104755733728_raised(%9, %10) : (tensor<1024xf64>, tensor<1024xf64>) -> (tensor<1024xf64>, tensor<1024xf64>)
    %12 = arith.addi %8, %c1023_i32 overflow<nsw> : i32
    affine.store %12, %5[] : memref<i32>
    omp.terminator
  }
  func.func private @outlined_affinefor_94104755733728_raised(%arg0: tensor<1024xf64>, %arg1: tensor<1024xf64>) -> (tensor<1024xf64>, tensor<1024xf64>) {
    %c = stablehlo.constant dense<1> : tensor<i64>
    %0 = stablehlo.slice %arg0 [0:1022] : (tensor<1024xf64>) -> tensor<1022xf64>
    %1 = stablehlo.reshape %0 : (tensor<1022xf64>) -> tensor<1022xf64>
    %2 = stablehlo.slice %arg0 [2:1024] : (tensor<1024xf64>) -> tensor<1022xf64>
    %3 = stablehlo.reshape %2 : (tensor<1022xf64>) -> tensor<1022xf64>
    %4 = arith.addf %1, %3 fastmath<contract> : tensor<1022xf64>
    %5 = stablehlo.broadcast_in_dim %4, dims = [0] : (tensor<1022xf64>) -> tensor<1022xf64>
    %6 = stablehlo.dynamic_update_slice %arg1, %5, %c : (tensor<1024xf64>, tensor<1022xf64>, tensor<i64>) -> tensor<1024xf64>
    return %arg0, %6 : tensor<1024xf64>, tensor<1024xf64>
  }
}


[DEBUG]Unhandled operation:
[DEBUG] %2 = fir.shape %c1024 : (index) -> !fir.shape<1> 
[DEBUG]Unhandled operation:
[DEBUG] omp.terminator 
// -----// IR Dump After (anonymous namespace)::WorkdistributeToStableHLOPass (jforce-translate) //----- //
module {
  func.func @main(%arg0: tensor<i32> {jit.literal_val = 0 : i64}, %arg1: tensor<i32> {jit.literal_val = 1024 : i64}, %arg2: tensor<1024xf64>, %arg3: tensor<1024xf64>, %arg4: tensor<i32> {jit.literal_val = 1024 : i64}, %arg5: tensor<i32> {jit.literal_val = 1024 : i64}) -> (tensor<i32>, tensor<i32>, tensor<1024xf64>, tensor<1024xf64>, tensor<i32>, tensor<i32>) {
    %c = stablehlo.constant dense<1023> : tensor<i32>
    %c_0 = stablehlo.constant dense<1024> : tensor<i64>
    %c_1 = stablehlo.constant dense<1> : tensor<i64>
    %0 = stablehlo.convert %c_1 : (tensor<i64>) -> tensor<i32>
    %1:2 = call @outlined_affinefor_94104755733728_raised(%arg3, %arg2) : (tensor<1024xf64>, tensor<1024xf64>) -> (tensor<1024xf64>, tensor<1024xf64>)
    %2 = stablehlo.add %0, %c : tensor<i32>
    return %arg0, %arg1, %1#1, %1#0, %arg4, %arg5 : tensor<i32>, tensor<i32>, tensor<1024xf64>, tensor<1024xf64>, tensor<i32>, tensor<i32>
  }
  func.func private @outlined_affinefor_94104755733728_raised(%arg0: tensor<1024xf64>, %arg1: tensor<1024xf64>) -> (tensor<1024xf64>, tensor<1024xf64>) {
    %c = stablehlo.constant dense<1> : tensor<i64>
    %0 = stablehlo.slice %arg0 [0:1022] : (tensor<1024xf64>) -> tensor<1022xf64>
    %1 = stablehlo.reshape %0 : (tensor<1022xf64>) -> tensor<1022xf64>
    %2 = stablehlo.slice %arg0 [2:1024] : (tensor<1024xf64>) -> tensor<1022xf64>
    %3 = stablehlo.reshape %2 : (tensor<1022xf64>) -> tensor<1022xf64>
    %4 = arith.addf %1, %3 fastmath<contract> : tensor<1022xf64>
    %5 = stablehlo.broadcast_in_dim %4, dims = [0] : (tensor<1022xf64>) -> tensor<1022xf64>
    %6 = stablehlo.dynamic_update_slice %arg1, %5, %c : (tensor<1024xf64>, tensor<1022xf64>, tensor<i64>) -> tensor<1024xf64>
    return %arg0, %6 : tensor<1024xf64>, tensor<1024xf64>
  }
}


// -----// IR Dump After CanonicalizerPass (canonicalize) //----- //
func.func private @outlined_affinefor_94104755733728_raised(%arg0: tensor<1024xf64>, %arg1: tensor<1024xf64>) -> (tensor<1024xf64>, tensor<1024xf64>) {
  %c = stablehlo.constant dense<1> : tensor<i64>
  %0 = stablehlo.slice %arg0 [0:1022] : (tensor<1024xf64>) -> tensor<1022xf64>
  %1 = stablehlo.reshape %0 : (tensor<1022xf64>) -> tensor<1022xf64>
  %2 = stablehlo.slice %arg0 [2:1024] : (tensor<1024xf64>) -> tensor<1022xf64>
  %3 = stablehlo.reshape %2 : (tensor<1022xf64>) -> tensor<1022xf64>
  %4 = arith.addf %1, %3 fastmath<contract> : tensor<1022xf64>
  %5 = stablehlo.broadcast_in_dim %4, dims = [0] : (tensor<1022xf64>) -> tensor<1022xf64>
  %6 = stablehlo.dynamic_update_slice %arg1, %5, %c : (tensor<1024xf64>, tensor<1022xf64>, tensor<i64>) -> tensor<1024xf64>
  return %arg0, %6 : tensor<1024xf64>, tensor<1024xf64>
}

// -----// IR Dump After CanonicalizerPass (canonicalize) //----- //
func.func @main(%arg0: tensor<i32> {jit.literal_val = 0 : i64}, %arg1: tensor<i32> {jit.literal_val = 1024 : i64}, %arg2: tensor<1024xf64>, %arg3: tensor<1024xf64>, %arg4: tensor<i32> {jit.literal_val = 1024 : i64}, %arg5: tensor<i32> {jit.literal_val = 1024 : i64}) -> (tensor<i32>, tensor<i32>, tensor<1024xf64>, tensor<1024xf64>, tensor<i32>, tensor<i32>) {
  %0:2 = call @outlined_affinefor_94104755733728_raised(%arg3, %arg2) : (tensor<1024xf64>, tensor<1024xf64>) -> (tensor<1024xf64>, tensor<1024xf64>)
  return %arg0, %arg1, %0#1, %0#0, %arg4, %arg5 : tensor<i32>, tensor<i32>, tensor<1024xf64>, tensor<1024xf64>, tensor<i32>, tensor<i32>
}

// -----// IR Dump After CanonicalizerPass (canonicalize) //----- //
func.func @main(%arg0: tensor<i32> {jit.literal_val = 0 : i64}, %arg1: tensor<i32> {jit.literal_val = 1024 : i64}, %arg2: tensor<1024xf64>, %arg3: tensor<1024xf64>, %arg4: tensor<i32> {jit.literal_val = 1024 : i64}, %arg5: tensor<i32> {jit.literal_val = 1024 : i64}) -> (tensor<i32>, tensor<i32>, tensor<1024xf64>, tensor<1024xf64>, tensor<i32>, tensor<i32>) {
  %c = stablehlo.constant dense<1> : tensor<i64>
  %0 = stablehlo.slice %arg3 [0:1022] : (tensor<1024xf64>) -> tensor<1022xf64>
  %1 = stablehlo.reshape %0 : (tensor<1022xf64>) -> tensor<1022xf64>
  %2 = stablehlo.slice %arg3 [2:1024] : (tensor<1024xf64>) -> tensor<1022xf64>
  %3 = stablehlo.reshape %2 : (tensor<1022xf64>) -> tensor<1022xf64>
  %4 = arith.addf %1, %3 fastmath<contract> : tensor<1022xf64>
  %5 = stablehlo.broadcast_in_dim %4, dims = [0] : (tensor<1022xf64>) -> tensor<1022xf64>
  %6 = stablehlo.dynamic_update_slice %arg2, %5, %c : (tensor<1024xf64>, tensor<1022xf64>, tensor<i64>) -> tensor<1024xf64>
  return %arg0, %arg1, %6, %arg3, %arg4, %arg5 : tensor<i32>, tensor<i32>, tensor<1024xf64>, tensor<1024xf64>, tensor<i32>, tensor<i32>
}

// -----// IR Dump After InlinerPass (inline) //----- //
module {
  func.func @main(%arg0: tensor<i32> {jit.literal_val = 0 : i64}, %arg1: tensor<i32> {jit.literal_val = 1024 : i64}, %arg2: tensor<1024xf64>, %arg3: tensor<1024xf64>, %arg4: tensor<i32> {jit.literal_val = 1024 : i64}, %arg5: tensor<i32> {jit.literal_val = 1024 : i64}) -> (tensor<i32>, tensor<i32>, tensor<1024xf64>, tensor<1024xf64>, tensor<i32>, tensor<i32>) {
    %c = stablehlo.constant dense<1> : tensor<i64>
    %0 = stablehlo.slice %arg3 [0:1022] : (tensor<1024xf64>) -> tensor<1022xf64>
    %1 = stablehlo.reshape %0 : (tensor<1022xf64>) -> tensor<1022xf64>
    %2 = stablehlo.slice %arg3 [2:1024] : (tensor<1024xf64>) -> tensor<1022xf64>
    %3 = stablehlo.reshape %2 : (tensor<1022xf64>) -> tensor<1022xf64>
    %4 = arith.addf %1, %3 fastmath<contract> : tensor<1022xf64>
    %5 = stablehlo.broadcast_in_dim %4, dims = [0] : (tensor<1022xf64>) -> tensor<1022xf64>
    %6 = stablehlo.dynamic_update_slice %arg2, %5, %c : (tensor<1024xf64>, tensor<1022xf64>, tensor<i64>) -> tensor<1024xf64>
    return %arg0, %arg1, %6, %arg3, %arg4, %arg5 : tensor<i32>, tensor<i32>, tensor<1024xf64>, tensor<1024xf64>, tensor<i32>, tensor<i32>
  }
}


// -----// IR Dump After StablehloAggressiveSimplificationPass (stablehlo-aggressive-simplification) //----- //
func.func @main(%arg0: tensor<i32> {jit.literal_val = 0 : i64}, %arg1: tensor<i32> {jit.literal_val = 1024 : i64}, %arg2: tensor<1024xf64>, %arg3: tensor<1024xf64>, %arg4: tensor<i32> {jit.literal_val = 1024 : i64}, %arg5: tensor<i32> {jit.literal_val = 1024 : i64}) -> (tensor<i32>, tensor<i32>, tensor<1024xf64>, tensor<1024xf64>, tensor<i32>, tensor<i32>) {
  %c = stablehlo.constant dense<1> : tensor<i64>
  %0 = stablehlo.slice %arg3 [0:1022] : (tensor<1024xf64>) -> tensor<1022xf64>
  %1 = stablehlo.slice %arg3 [2:1024] : (tensor<1024xf64>) -> tensor<1022xf64>
  %2 = arith.addf %0, %1 fastmath<contract> : tensor<1022xf64>
  %3 = stablehlo.dynamic_update_slice %arg2, %2, %c : (tensor<1024xf64>, tensor<1022xf64>, tensor<i64>) -> tensor<1024xf64>
  return %arg0, %arg1, %3, %arg3, %arg4, %arg5 : tensor<i32>, tensor<i32>, tensor<1024xf64>, tensor<1024xf64>, tensor<i32>, tensor<i32>
}

// -----// IR Dump After (anonymous namespace)::TrimArgsPass (jforce-trim-args) //----- //
func.func @main(%arg0: tensor<1024xf64> {jit.literal_val = 0 : i64}, %arg1: tensor<1024xf64> {jit.literal_val = 1024 : i64}) -> (tensor<1024xf64>, tensor<1024xf64>) attributes {jit.args_mapping = [2 : ui32, 0 : ui32, 3 : ui32, 1 : ui32]} {
  %c = stablehlo.constant dense<1> : tensor<i64>
  %0 = stablehlo.slice %arg1 [0:1022] : (tensor<1024xf64>) -> tensor<1022xf64>
  %1 = stablehlo.slice %arg1 [2:1024] : (tensor<1024xf64>) -> tensor<1022xf64>
  %2 = arith.addf %0, %1 fastmath<contract> : tensor<1022xf64>
  %3 = stablehlo.dynamic_update_slice %arg0, %2, %c : (tensor<1024xf64>, tensor<1022xf64>, tensor<i64>) -> tensor<1024xf64>
  return %3, %arg1 : tensor<1024xf64>, tensor<1024xf64>
}

