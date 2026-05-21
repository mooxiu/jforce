// ./build/src/tool/jforce-opt ./tests/sum.mlir \
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
//    --enzyme-affine-to-stablehlo --canonicalize 


// -----// IR Dump After {anonymous}::AnnotatePass (jforce-annotate) //----- //
func.func @kernel(%arg0: !fir.ref<i32> {jit.arg_type = 1 : ui32, jit.literal_val = 0 : i64}, %arg1: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg2: !fir.ref<f64> {jit.arg_type = 1 : ui32, jit.literal_val = 0 : i64}, %arg3: !fir.ref<!fir.array<?xf64>> {jit.arg_type = 1 : ui32}, %arg4: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}) {
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

// -----// IR Dump After {anonymous}::PropagateConstantsPass (jforce-propagate-constants) //----- //
func.func @kernel(%arg0: !fir.ref<i32> {jit.arg_type = 1 : ui32, jit.literal_val = 0 : i64}, %arg1: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg2: !fir.ref<f64> {jit.arg_type = 1 : ui32, jit.literal_val = 0 : i64}, %arg3: !fir.ref<!fir.array<?xf64>> {jit.arg_type = 1 : ui32}, %arg4: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}) {
  %c1024_i32 = arith.constant 1024 : i32
  %0 = fir.convert %c1024_i32 : (i32) -> i64
  %c0 = arith.constant 0 : index
  %1 = fir.convert %0 : (i64) -> index
  %2 = arith.cmpi sgt, %1, %c0 : index
  %3 = arith.select %2, %1, %c0 : index
  %4:2 = hlfir.declare %arg0 {uniq_name = "_QFFrun_benchmarkEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %5:2 = hlfir.declare %arg1 {uniq_name = "_QFFrun_benchmarkEn"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %6:2 = hlfir.declare %arg2 {uniq_name = "_QFFrun_benchmarkEsum_x"} : (!fir.ref<f64>) -> (!fir.ref<f64>, !fir.ref<f64>)
  %7 = fir.shape %3 : (index) -> !fir.shape<1>
  %8:2 = hlfir.declare %arg3(%7) {uniq_name = "_QFFrun_benchmarkEx"} : (!fir.ref<!fir.array<?xf64>>, !fir.shape<1>) -> (!fir.box<!fir.array<?xf64>>, !fir.ref<!fir.array<?xf64>>)
  %c1_i32 = arith.constant 1 : i32
  %9 = fir.convert %c1_i32 : (i32) -> index
  %c1024_i32_0 = arith.constant 1024 : i32
  %10 = fir.convert %c1024_i32_0 : (i32) -> index
  %c1 = arith.constant 1 : index
  %11 = fir.convert %9 : (index) -> i32
  %12 = fir.do_loop %arg5 = %9 to %10 step %c1 iter_args(%arg6 = %11) -> (i32) {
    fir.store %arg6 to %4#0 : !fir.ref<i32>
    %13 = fir.load %6#0 : !fir.ref<f64>
    %14 = fir.load %4#0 : !fir.ref<i32>
    %15 = fir.convert %14 : (i32) -> i64
    %16 = hlfir.designate %8#0 (%15)  : (!fir.box<!fir.array<?xf64>>, i64) -> !fir.ref<f64>
    %17 = fir.load %16 : !fir.ref<f64>
    %18 = arith.addf %13, %17 fastmath<contract> : f64
    hlfir.assign %18 to %6#0 : f64, !fir.ref<f64>
    %19 = fir.convert %c1 : (index) -> i32
    %20 = fir.load %4#0 : !fir.ref<i32>
    %21 = arith.addi %20, %19 overflow<nsw> : i32
    fir.result %21 : i32
  }
  fir.store %12 to %4#0 : !fir.ref<i32>
  omp.terminator
}

// -----// IR Dump After CanonicalizerPass (canonicalize) //----- //
module {
  func.func @kernel(%arg0: !fir.ref<i32> {jit.arg_type = 1 : ui32, jit.literal_val = 0 : i64}, %arg1: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg2: !fir.ref<f64> {jit.arg_type = 1 : ui32, jit.literal_val = 0 : i64}, %arg3: !fir.ref<!fir.array<?xf64>> {jit.arg_type = 1 : ui32}, %arg4: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}) {
    %c1 = arith.constant 1 : index
    %c1024 = arith.constant 1024 : index
    %0:2 = hlfir.declare %arg0 {uniq_name = "_QFFrun_benchmarkEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
    %1:2 = hlfir.declare %arg1 {uniq_name = "_QFFrun_benchmarkEn"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
    %2:2 = hlfir.declare %arg2 {uniq_name = "_QFFrun_benchmarkEsum_x"} : (!fir.ref<f64>) -> (!fir.ref<f64>, !fir.ref<f64>)
    %3 = fir.shape %c1024 : (index) -> !fir.shape<1>
    %4:2 = hlfir.declare %arg3(%3) {uniq_name = "_QFFrun_benchmarkEx"} : (!fir.ref<!fir.array<?xf64>>, !fir.shape<1>) -> (!fir.box<!fir.array<?xf64>>, !fir.ref<!fir.array<?xf64>>)
    %5 = fir.convert %c1 : (index) -> i32
    %6 = fir.do_loop %arg5 = %c1 to %c1024 step %c1 iter_args(%arg6 = %5) -> (i32) {
      fir.store %arg6 to %0#0 : !fir.ref<i32>
      %7 = fir.load %2#0 : !fir.ref<f64>
      %8 = fir.load %0#0 : !fir.ref<i32>
      %9 = fir.convert %8 : (i32) -> i64
      %10 = hlfir.designate %4#0 (%9)  : (!fir.box<!fir.array<?xf64>>, i64) -> !fir.ref<f64>
      %11 = fir.load %10 : !fir.ref<f64>
      %12 = arith.addf %7, %11 fastmath<contract> : f64
      hlfir.assign %12 to %2#0 : f64, !fir.ref<f64>
      %13 = fir.convert %c1 : (index) -> i32
      %14 = fir.load %0#0 : !fir.ref<i32>
      %15 = arith.addi %14, %13 overflow<nsw> : i32
      fir.result %15 : i32
    }
    fir.store %6 to %0#0 : !fir.ref<i32>
    omp.terminator
  }
}


// -----// IR Dump After SCCPPass (sccp) //----- //
module {
  func.func @kernel(%arg0: !fir.ref<i32> {jit.arg_type = 1 : ui32, jit.literal_val = 0 : i64}, %arg1: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg2: !fir.ref<f64> {jit.arg_type = 1 : ui32, jit.literal_val = 0 : i64}, %arg3: !fir.ref<!fir.array<?xf64>> {jit.arg_type = 1 : ui32}, %arg4: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}) {
    %c1024 = arith.constant 1024 : index
    %c1 = arith.constant 1 : index
    %0:2 = hlfir.declare %arg0 {uniq_name = "_QFFrun_benchmarkEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
    %1:2 = hlfir.declare %arg1 {uniq_name = "_QFFrun_benchmarkEn"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
    %2:2 = hlfir.declare %arg2 {uniq_name = "_QFFrun_benchmarkEsum_x"} : (!fir.ref<f64>) -> (!fir.ref<f64>, !fir.ref<f64>)
    %3 = fir.shape %c1024 : (index) -> !fir.shape<1>
    %4:2 = hlfir.declare %arg3(%3) {uniq_name = "_QFFrun_benchmarkEx"} : (!fir.ref<!fir.array<?xf64>>, !fir.shape<1>) -> (!fir.box<!fir.array<?xf64>>, !fir.ref<!fir.array<?xf64>>)
    %5 = fir.convert %c1 : (index) -> i32
    %6 = fir.do_loop %arg5 = %c1 to %c1024 step %c1 iter_args(%arg6 = %5) -> (i32) {
      fir.store %arg6 to %0#0 : !fir.ref<i32>
      %7 = fir.load %2#0 : !fir.ref<f64>
      %8 = fir.load %0#0 : !fir.ref<i32>
      %9 = fir.convert %8 : (i32) -> i64
      %10 = hlfir.designate %4#0 (%9)  : (!fir.box<!fir.array<?xf64>>, i64) -> !fir.ref<f64>
      %11 = fir.load %10 : !fir.ref<f64>
      %12 = arith.addf %7, %11 fastmath<contract> : f64
      hlfir.assign %12 to %2#0 : f64, !fir.ref<f64>
      %13 = fir.convert %c1 : (index) -> i32
      %14 = fir.load %0#0 : !fir.ref<i32>
      %15 = arith.addi %14, %13 overflow<nsw> : i32
      fir.result %15 : i32
    }
    fir.store %6 to %0#0 : !fir.ref<i32>
    omp.terminator
  }
}


// -----// IR Dump After CSEPass (cse) //----- //
module {
  func.func @kernel(%arg0: !fir.ref<i32> {jit.arg_type = 1 : ui32, jit.literal_val = 0 : i64}, %arg1: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg2: !fir.ref<f64> {jit.arg_type = 1 : ui32, jit.literal_val = 0 : i64}, %arg3: !fir.ref<!fir.array<?xf64>> {jit.arg_type = 1 : ui32}, %arg4: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}) {
    %c1024 = arith.constant 1024 : index
    %c1 = arith.constant 1 : index
    %0:2 = hlfir.declare %arg0 {uniq_name = "_QFFrun_benchmarkEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
    %1:2 = hlfir.declare %arg1 {uniq_name = "_QFFrun_benchmarkEn"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
    %2:2 = hlfir.declare %arg2 {uniq_name = "_QFFrun_benchmarkEsum_x"} : (!fir.ref<f64>) -> (!fir.ref<f64>, !fir.ref<f64>)
    %3 = fir.shape %c1024 : (index) -> !fir.shape<1>
    %4:2 = hlfir.declare %arg3(%3) {uniq_name = "_QFFrun_benchmarkEx"} : (!fir.ref<!fir.array<?xf64>>, !fir.shape<1>) -> (!fir.box<!fir.array<?xf64>>, !fir.ref<!fir.array<?xf64>>)
    %5 = fir.convert %c1 : (index) -> i32
    %6 = fir.do_loop %arg5 = %c1 to %c1024 step %c1 iter_args(%arg6 = %5) -> (i32) {
      fir.store %arg6 to %0#0 : !fir.ref<i32>
      %7 = fir.load %2#0 : !fir.ref<f64>
      %8 = fir.load %0#0 : !fir.ref<i32>
      %9 = fir.convert %8 : (i32) -> i64
      %10 = hlfir.designate %4#0 (%9)  : (!fir.box<!fir.array<?xf64>>, i64) -> !fir.ref<f64>
      %11 = fir.load %10 : !fir.ref<f64>
      %12 = arith.addf %7, %11 fastmath<contract> : f64
      hlfir.assign %12 to %2#0 : f64, !fir.ref<f64>
      %13 = fir.load %0#0 : !fir.ref<i32>
      %14 = arith.addi %13, %5 overflow<nsw> : i32
      fir.result %14 : i32
    }
    fir.store %6 to %0#0 : !fir.ref<i32>
    omp.terminator
  }
}


// -----// IR Dump After CanonicalizerPass (canonicalize) //----- //
module {
  func.func @kernel(%arg0: !fir.ref<i32> {jit.arg_type = 1 : ui32, jit.literal_val = 0 : i64}, %arg1: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg2: !fir.ref<f64> {jit.arg_type = 1 : ui32, jit.literal_val = 0 : i64}, %arg3: !fir.ref<!fir.array<?xf64>> {jit.arg_type = 1 : ui32}, %arg4: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}) {
    %c1024 = arith.constant 1024 : index
    %c1 = arith.constant 1 : index
    %0:2 = hlfir.declare %arg0 {uniq_name = "_QFFrun_benchmarkEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
    %1:2 = hlfir.declare %arg1 {uniq_name = "_QFFrun_benchmarkEn"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
    %2:2 = hlfir.declare %arg2 {uniq_name = "_QFFrun_benchmarkEsum_x"} : (!fir.ref<f64>) -> (!fir.ref<f64>, !fir.ref<f64>)
    %3 = fir.shape %c1024 : (index) -> !fir.shape<1>
    %4:2 = hlfir.declare %arg3(%3) {uniq_name = "_QFFrun_benchmarkEx"} : (!fir.ref<!fir.array<?xf64>>, !fir.shape<1>) -> (!fir.box<!fir.array<?xf64>>, !fir.ref<!fir.array<?xf64>>)
    %5 = fir.convert %c1 : (index) -> i32
    %6 = fir.do_loop %arg5 = %c1 to %c1024 step %c1 iter_args(%arg6 = %5) -> (i32) {
      fir.store %arg6 to %0#0 : !fir.ref<i32>
      %7 = fir.load %2#0 : !fir.ref<f64>
      %8 = fir.load %0#0 : !fir.ref<i32>
      %9 = fir.convert %8 : (i32) -> i64
      %10 = hlfir.designate %4#0 (%9)  : (!fir.box<!fir.array<?xf64>>, i64) -> !fir.ref<f64>
      %11 = fir.load %10 : !fir.ref<f64>
      %12 = arith.addf %7, %11 fastmath<contract> : f64
      hlfir.assign %12 to %2#0 : f64, !fir.ref<f64>
      %13 = fir.load %0#0 : !fir.ref<i32>
      %14 = arith.addi %13, %5 overflow<nsw> : i32
      fir.result %14 : i32
    }
    fir.store %6 to %0#0 : !fir.ref<i32>
    omp.terminator
  }
}


// -----// IR Dump After {anonymous}::ShapeInferPass (jforce-shape-infer) //----- //
func.func @kernel(%arg0: !fir.ref<i32> {jit.arg_type = 1 : ui32, jit.literal_val = 0 : i64}, %arg1: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg2: !fir.ref<f64> {jit.arg_type = 1 : ui32, jit.literal_val = 0 : i64}, %arg3: !fir.ref<!fir.array<1024xf64>> {jit.arg_type = 1 : ui32}, %arg4: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}) {
  %c1024 = arith.constant 1024 : index
  %c1 = arith.constant 1 : index
  %0:2 = hlfir.declare %arg0 {uniq_name = "_QFFrun_benchmarkEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %1:2 = hlfir.declare %arg1 {uniq_name = "_QFFrun_benchmarkEn"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %2:2 = hlfir.declare %arg2 {uniq_name = "_QFFrun_benchmarkEsum_x"} : (!fir.ref<f64>) -> (!fir.ref<f64>, !fir.ref<f64>)
  %3 = fir.shape %c1024 : (index) -> !fir.shape<1>
  %4:2 = hlfir.declare %arg3(%3) {uniq_name = "_QFFrun_benchmarkEx"} : (!fir.ref<!fir.array<1024xf64>>, !fir.shape<1>) -> (!fir.ref<!fir.array<1024xf64>>, !fir.ref<!fir.array<1024xf64>>)
  %5 = fir.convert %c1 : (index) -> i32
  %6 = fir.do_loop %arg5 = %c1 to %c1024 step %c1 iter_args(%arg6 = %5) -> (i32) {
    fir.store %arg6 to %0#0 : !fir.ref<i32>
    %7 = fir.load %2#0 : !fir.ref<f64>
    %8 = fir.load %0#0 : !fir.ref<i32>
    %9 = fir.convert %8 : (i32) -> i64
    %10 = hlfir.designate %4#0 (%9)  : (!fir.ref<!fir.array<1024xf64>>, i64) -> !fir.ref<f64>
    %11 = fir.load %10 : !fir.ref<f64>
    %12 = arith.addf %7, %11 fastmath<contract> : f64
    hlfir.assign %12 to %2#0 : f64, !fir.ref<f64>
    %13 = fir.load %0#0 : !fir.ref<i32>
    %14 = arith.addi %13, %5 overflow<nsw> : i32
    fir.result %14 : i32
  }
  fir.store %6 to %0#0 : !fir.ref<i32>
  omp.terminator
}

// -----// IR Dump After CanonicalizerPass (canonicalize) //----- //
module {
  func.func @kernel(%arg0: !fir.ref<i32> {jit.arg_type = 1 : ui32, jit.literal_val = 0 : i64}, %arg1: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg2: !fir.ref<f64> {jit.arg_type = 1 : ui32, jit.literal_val = 0 : i64}, %arg3: !fir.ref<!fir.array<1024xf64>> {jit.arg_type = 1 : ui32}, %arg4: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}) {
    %c1024 = arith.constant 1024 : index
    %c1 = arith.constant 1 : index
    %0:2 = hlfir.declare %arg0 {uniq_name = "_QFFrun_benchmarkEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
    %1:2 = hlfir.declare %arg1 {uniq_name = "_QFFrun_benchmarkEn"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
    %2:2 = hlfir.declare %arg2 {uniq_name = "_QFFrun_benchmarkEsum_x"} : (!fir.ref<f64>) -> (!fir.ref<f64>, !fir.ref<f64>)
    %3 = fir.shape %c1024 : (index) -> !fir.shape<1>
    %4:2 = hlfir.declare %arg3(%3) {uniq_name = "_QFFrun_benchmarkEx"} : (!fir.ref<!fir.array<1024xf64>>, !fir.shape<1>) -> (!fir.ref<!fir.array<1024xf64>>, !fir.ref<!fir.array<1024xf64>>)
    %5 = fir.convert %c1 : (index) -> i32
    %6 = fir.do_loop %arg5 = %c1 to %c1024 step %c1 iter_args(%arg6 = %5) -> (i32) {
      fir.store %arg6 to %0#0 : !fir.ref<i32>
      %7 = fir.load %2#0 : !fir.ref<f64>
      %8 = fir.load %0#0 : !fir.ref<i32>
      %9 = fir.convert %8 : (i32) -> i64
      %10 = hlfir.designate %4#0 (%9)  : (!fir.ref<!fir.array<1024xf64>>, i64) -> !fir.ref<f64>
      %11 = fir.load %10 : !fir.ref<f64>
      %12 = arith.addf %7, %11 fastmath<contract> : f64
      hlfir.assign %12 to %2#0 : f64, !fir.ref<f64>
      %13 = fir.load %0#0 : !fir.ref<i32>
      %14 = arith.addi %13, %5 overflow<nsw> : i32
      fir.result %14 : i32
    }
    fir.store %6 to %0#0 : !fir.ref<i32>
    omp.terminator
  }
}


// -----// IR Dump After ConvertHLFIRtoFIR (convert-hlfir-to-fir) //----- //
module {
  func.func @kernel(%arg0: !fir.ref<i32> {jit.arg_type = 1 : ui32, jit.literal_val = 0 : i64}, %arg1: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg2: !fir.ref<f64> {jit.arg_type = 1 : ui32, jit.literal_val = 0 : i64}, %arg3: !fir.ref<!fir.array<1024xf64>> {jit.arg_type = 1 : ui32}, %arg4: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}) {
    %c1024 = arith.constant 1024 : index
    %c1 = arith.constant 1 : index
    %0 = fir.declare %arg0 {uniq_name = "_QFFrun_benchmarkEi"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %1 = fir.declare %arg1 {uniq_name = "_QFFrun_benchmarkEn"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %2 = fir.declare %arg2 {uniq_name = "_QFFrun_benchmarkEsum_x"} : (!fir.ref<f64>) -> !fir.ref<f64>
    %3 = fir.shape %c1024 : (index) -> !fir.shape<1>
    %4 = fir.declare %arg3(%3) {uniq_name = "_QFFrun_benchmarkEx"} : (!fir.ref<!fir.array<1024xf64>>, !fir.shape<1>) -> !fir.ref<!fir.array<1024xf64>>
    %5 = fir.convert %c1 : (index) -> i32
    %6 = fir.do_loop %arg5 = %c1 to %c1024 step %c1 iter_args(%arg6 = %5) -> (i32) {
      fir.store %arg6 to %0 : !fir.ref<i32>
      %7 = fir.load %2 : !fir.ref<f64>
      %8 = fir.load %0 : !fir.ref<i32>
      %9 = fir.convert %8 : (i32) -> i64
      %10 = fir.array_coor %4(%3) %9 : (!fir.ref<!fir.array<1024xf64>>, !fir.shape<1>, i64) -> !fir.ref<f64>
      %11 = fir.load %10 : !fir.ref<f64>
      %12 = arith.addf %7, %11 fastmath<contract> : f64
      fir.store %12 to %2 : !fir.ref<f64>
      %13 = fir.load %0 : !fir.ref<i32>
      %14 = arith.addi %13, %5 overflow<nsw> : i32
      fir.result %14 : i32
    }
    fir.store %6 to %0 : !fir.ref<i32>
    omp.terminator
  }
}


// -----// IR Dump After FIRToMemRef (fir-to-memref) //----- //
func.func @kernel(%arg0: !fir.ref<i32> {jit.arg_type = 1 : ui32, jit.literal_val = 0 : i64}, %arg1: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg2: !fir.ref<f64> {jit.arg_type = 1 : ui32, jit.literal_val = 0 : i64}, %arg3: !fir.ref<!fir.array<1024xf64>> {jit.arg_type = 1 : ui32}, %arg4: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}) {
  %c1024 = arith.constant 1024 : index
  %c1 = arith.constant 1 : index
  %0 = fir.declare %arg0 {uniq_name = "_QFFrun_benchmarkEi"} : (!fir.ref<i32>) -> !fir.ref<i32>
  %1 = fir.declare %arg1 {uniq_name = "_QFFrun_benchmarkEn"} : (!fir.ref<i32>) -> !fir.ref<i32>
  %2 = fir.declare %arg2 {uniq_name = "_QFFrun_benchmarkEsum_x"} : (!fir.ref<f64>) -> !fir.ref<f64>
  %3 = fir.shape %c1024 : (index) -> !fir.shape<1>
  %4 = fir.declare %arg3(%3) {uniq_name = "_QFFrun_benchmarkEx"} : (!fir.ref<!fir.array<1024xf64>>, !fir.shape<1>) -> !fir.ref<!fir.array<1024xf64>>
  %5 = fir.convert %c1 : (index) -> i32
  %6 = fir.do_loop %arg5 = %c1 to %c1024 step %c1 iter_args(%arg6 = %5) -> (i32) {
    %8 = fir.convert %0 : (!fir.ref<i32>) -> memref<i32>
    memref.store %arg6, %8[] : memref<i32>
    %9 = fir.convert %2 : (!fir.ref<f64>) -> memref<f64>
    %10 = memref.load %9[] : memref<f64>
    %11 = fir.convert %0 : (!fir.ref<i32>) -> memref<i32>
    %12 = memref.load %11[] : memref<i32>
    %13 = fir.convert %12 : (i32) -> i64
    %14 = fir.convert %4 : (!fir.ref<!fir.array<1024xf64>>) -> memref<1024xf64>
    %c1_0 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %15 = arith.index_cast %13 : i64 to index
    %16 = arith.subi %15, %c1_0 : index
    %17 = arith.muli %16, %c1_0 : index
    %18 = arith.subi %c1_0, %c1_0 : index
    %19 = arith.addi %17, %18 : index
    %20 = memref.load %14[%19] : memref<1024xf64>
    %21 = arith.addf %10, %20 fastmath<contract> : f64
    %22 = fir.convert %2 : (!fir.ref<f64>) -> memref<f64>
    memref.store %21, %22[] : memref<f64>
    %23 = fir.convert %0 : (!fir.ref<i32>) -> memref<i32>
    %24 = memref.load %23[] : memref<i32>
    %25 = arith.addi %24, %5 overflow<nsw> : i32
    fir.result %25 : i32
  }
  %7 = fir.convert %0 : (!fir.ref<i32>) -> memref<i32>
  memref.store %6, %7[] : memref<i32>
  omp.terminator
}

// -----// IR Dump After CanonicalizerPass (canonicalize) //----- //
module {
  func.func @kernel(%arg0: !fir.ref<i32> {jit.arg_type = 1 : ui32, jit.literal_val = 0 : i64}, %arg1: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg2: !fir.ref<f64> {jit.arg_type = 1 : ui32, jit.literal_val = 0 : i64}, %arg3: !fir.ref<!fir.array<1024xf64>> {jit.arg_type = 1 : ui32}, %arg4: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}) {
    %c1024 = arith.constant 1024 : index
    %c1 = arith.constant 1 : index
    %0 = fir.declare %arg0 {uniq_name = "_QFFrun_benchmarkEi"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %1 = fir.declare %arg1 {uniq_name = "_QFFrun_benchmarkEn"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %2 = fir.declare %arg2 {uniq_name = "_QFFrun_benchmarkEsum_x"} : (!fir.ref<f64>) -> !fir.ref<f64>
    %3 = fir.shape %c1024 : (index) -> !fir.shape<1>
    %4 = fir.declare %arg3(%3) {uniq_name = "_QFFrun_benchmarkEx"} : (!fir.ref<!fir.array<1024xf64>>, !fir.shape<1>) -> !fir.ref<!fir.array<1024xf64>>
    %5 = fir.convert %c1 : (index) -> i32
    %6 = fir.do_loop %arg5 = %c1 to %c1024 step %c1 iter_args(%arg6 = %5) -> (i32) {
      %8 = fir.convert %0 : (!fir.ref<i32>) -> memref<i32>
      memref.store %arg6, %8[] : memref<i32>
      %9 = fir.convert %2 : (!fir.ref<f64>) -> memref<f64>
      %10 = memref.load %9[] : memref<f64>
      %11 = fir.convert %0 : (!fir.ref<i32>) -> memref<i32>
      %12 = memref.load %11[] : memref<i32>
      %13 = fir.convert %12 : (i32) -> i64
      %14 = fir.convert %4 : (!fir.ref<!fir.array<1024xf64>>) -> memref<1024xf64>
      %15 = arith.index_cast %13 : i64 to index
      %16 = arith.subi %15, %c1 : index
      %17 = memref.load %14[%16] : memref<1024xf64>
      %18 = arith.addf %10, %17 fastmath<contract> : f64
      %19 = fir.convert %2 : (!fir.ref<f64>) -> memref<f64>
      memref.store %18, %19[] : memref<f64>
      %20 = fir.convert %0 : (!fir.ref<i32>) -> memref<i32>
      %21 = memref.load %20[] : memref<i32>
      %22 = arith.addi %21, %5 overflow<nsw> : i32
      fir.result %22 : i32
    }
    %7 = fir.convert %0 : (!fir.ref<i32>) -> memref<i32>
    memref.store %6, %7[] : memref<i32>
    omp.terminator
  }
}


// -----// IR Dump After LoopInvariantCodeMotionPass (loop-invariant-code-motion) //----- //
module {
  func.func @kernel(%arg0: !fir.ref<i32> {jit.arg_type = 1 : ui32, jit.literal_val = 0 : i64}, %arg1: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg2: !fir.ref<f64> {jit.arg_type = 1 : ui32, jit.literal_val = 0 : i64}, %arg3: !fir.ref<!fir.array<1024xf64>> {jit.arg_type = 1 : ui32}, %arg4: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}) {
    %c1024 = arith.constant 1024 : index
    %c1 = arith.constant 1 : index
    %0 = fir.declare %arg0 {uniq_name = "_QFFrun_benchmarkEi"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %1 = fir.declare %arg1 {uniq_name = "_QFFrun_benchmarkEn"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %2 = fir.declare %arg2 {uniq_name = "_QFFrun_benchmarkEsum_x"} : (!fir.ref<f64>) -> !fir.ref<f64>
    %3 = fir.shape %c1024 : (index) -> !fir.shape<1>
    %4 = fir.declare %arg3(%3) {uniq_name = "_QFFrun_benchmarkEx"} : (!fir.ref<!fir.array<1024xf64>>, !fir.shape<1>) -> !fir.ref<!fir.array<1024xf64>>
    %5 = fir.convert %c1 : (index) -> i32
    %6 = fir.convert %0 : (!fir.ref<i32>) -> memref<i32>
    %7 = fir.convert %2 : (!fir.ref<f64>) -> memref<f64>
    %8 = fir.convert %0 : (!fir.ref<i32>) -> memref<i32>
    %9 = fir.convert %4 : (!fir.ref<!fir.array<1024xf64>>) -> memref<1024xf64>
    %10 = fir.convert %2 : (!fir.ref<f64>) -> memref<f64>
    %11 = fir.convert %0 : (!fir.ref<i32>) -> memref<i32>
    %12 = fir.do_loop %arg5 = %c1 to %c1024 step %c1 iter_args(%arg6 = %5) -> (i32) {
      memref.store %arg6, %6[] : memref<i32>
      %14 = memref.load %7[] : memref<f64>
      %15 = memref.load %8[] : memref<i32>
      %16 = fir.convert %15 : (i32) -> i64
      %17 = arith.index_cast %16 : i64 to index
      %18 = arith.subi %17, %c1 : index
      %19 = memref.load %9[%18] : memref<1024xf64>
      %20 = arith.addf %14, %19 fastmath<contract> : f64
      memref.store %20, %10[] : memref<f64>
      %21 = memref.load %11[] : memref<i32>
      %22 = arith.addi %21, %5 overflow<nsw> : i32
      fir.result %22 : i32
    }
    %13 = fir.convert %0 : (!fir.ref<i32>) -> memref<i32>
    memref.store %12, %13[] : memref<i32>
    omp.terminator
  }
}


// -----// IR Dump After CSEPass (cse) //----- //
module {
  func.func @kernel(%arg0: !fir.ref<i32> {jit.arg_type = 1 : ui32, jit.literal_val = 0 : i64}, %arg1: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg2: !fir.ref<f64> {jit.arg_type = 1 : ui32, jit.literal_val = 0 : i64}, %arg3: !fir.ref<!fir.array<1024xf64>> {jit.arg_type = 1 : ui32}, %arg4: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}) {
    %c1024 = arith.constant 1024 : index
    %c1 = arith.constant 1 : index
    %0 = fir.declare %arg0 {uniq_name = "_QFFrun_benchmarkEi"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %1 = fir.declare %arg1 {uniq_name = "_QFFrun_benchmarkEn"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %2 = fir.declare %arg2 {uniq_name = "_QFFrun_benchmarkEsum_x"} : (!fir.ref<f64>) -> !fir.ref<f64>
    %3 = fir.shape %c1024 : (index) -> !fir.shape<1>
    %4 = fir.declare %arg3(%3) {uniq_name = "_QFFrun_benchmarkEx"} : (!fir.ref<!fir.array<1024xf64>>, !fir.shape<1>) -> !fir.ref<!fir.array<1024xf64>>
    %5 = fir.convert %c1 : (index) -> i32
    %6 = fir.convert %0 : (!fir.ref<i32>) -> memref<i32>
    %7 = fir.convert %2 : (!fir.ref<f64>) -> memref<f64>
    %8 = fir.convert %4 : (!fir.ref<!fir.array<1024xf64>>) -> memref<1024xf64>
    %9 = fir.do_loop %arg5 = %c1 to %c1024 step %c1 iter_args(%arg6 = %5) -> (i32) {
      memref.store %arg6, %6[] : memref<i32>
      %10 = memref.load %7[] : memref<f64>
      %11 = memref.load %6[] : memref<i32>
      %12 = fir.convert %11 : (i32) -> i64
      %13 = arith.index_cast %12 : i64 to index
      %14 = arith.subi %13, %c1 : index
      %15 = memref.load %8[%14] : memref<1024xf64>
      %16 = arith.addf %10, %15 fastmath<contract> : f64
      memref.store %16, %7[] : memref<f64>
      %17 = memref.load %6[] : memref<i32>
      %18 = arith.addi %17, %5 overflow<nsw> : i32
      fir.result %18 : i32
    }
    memref.store %9, %6[] : memref<i32>
    omp.terminator
  }
}


// -----// IR Dump After CanonicalizerPass (canonicalize) //----- //
module {
  func.func @kernel(%arg0: !fir.ref<i32> {jit.arg_type = 1 : ui32, jit.literal_val = 0 : i64}, %arg1: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg2: !fir.ref<f64> {jit.arg_type = 1 : ui32, jit.literal_val = 0 : i64}, %arg3: !fir.ref<!fir.array<1024xf64>> {jit.arg_type = 1 : ui32}, %arg4: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}) {
    %c1024 = arith.constant 1024 : index
    %c1 = arith.constant 1 : index
    %0 = fir.declare %arg0 {uniq_name = "_QFFrun_benchmarkEi"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %1 = fir.declare %arg1 {uniq_name = "_QFFrun_benchmarkEn"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %2 = fir.declare %arg2 {uniq_name = "_QFFrun_benchmarkEsum_x"} : (!fir.ref<f64>) -> !fir.ref<f64>
    %3 = fir.shape %c1024 : (index) -> !fir.shape<1>
    %4 = fir.declare %arg3(%3) {uniq_name = "_QFFrun_benchmarkEx"} : (!fir.ref<!fir.array<1024xf64>>, !fir.shape<1>) -> !fir.ref<!fir.array<1024xf64>>
    %5 = fir.convert %c1 : (index) -> i32
    %6 = fir.convert %0 : (!fir.ref<i32>) -> memref<i32>
    %7 = fir.convert %2 : (!fir.ref<f64>) -> memref<f64>
    %8 = fir.convert %4 : (!fir.ref<!fir.array<1024xf64>>) -> memref<1024xf64>
    %9 = fir.do_loop %arg5 = %c1 to %c1024 step %c1 iter_args(%arg6 = %5) -> (i32) {
      memref.store %arg6, %6[] : memref<i32>
      %10 = memref.load %7[] : memref<f64>
      %11 = memref.load %6[] : memref<i32>
      %12 = fir.convert %11 : (i32) -> i64
      %13 = arith.index_cast %12 : i64 to index
      %14 = arith.subi %13, %c1 : index
      %15 = memref.load %8[%14] : memref<1024xf64>
      %16 = arith.addf %10, %15 fastmath<contract> : f64
      memref.store %16, %7[] : memref<f64>
      %17 = memref.load %6[] : memref<i32>
      %18 = arith.addi %17, %5 overflow<nsw> : i32
      fir.result %18 : i32
    }
    memref.store %9, %6[] : memref<i32>
    omp.terminator
  }
}


// -----// IR Dump After {anonymous}::CleanFIRLoopPass (jforce-clean-fir-loop) //----- //
func.func @kernel(%arg0: !fir.ref<i32> {jit.arg_type = 1 : ui32, jit.literal_val = 0 : i64}, %arg1: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg2: !fir.ref<f64> {jit.arg_type = 1 : ui32, jit.literal_val = 0 : i64}, %arg3: !fir.ref<!fir.array<1024xf64>> {jit.arg_type = 1 : ui32}, %arg4: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}) {
  %c1024 = arith.constant 1024 : index
  %c1 = arith.constant 1 : index
  %0 = fir.declare %arg0 {uniq_name = "_QFFrun_benchmarkEi"} : (!fir.ref<i32>) -> !fir.ref<i32>
  %1 = fir.declare %arg1 {uniq_name = "_QFFrun_benchmarkEn"} : (!fir.ref<i32>) -> !fir.ref<i32>
  %2 = fir.declare %arg2 {uniq_name = "_QFFrun_benchmarkEsum_x"} : (!fir.ref<f64>) -> !fir.ref<f64>
  %3 = fir.shape %c1024 : (index) -> !fir.shape<1>
  %4 = fir.declare %arg3(%3) {uniq_name = "_QFFrun_benchmarkEx"} : (!fir.ref<!fir.array<1024xf64>>, !fir.shape<1>) -> !fir.ref<!fir.array<1024xf64>>
  %5 = fir.convert %c1 : (index) -> i32
  %6 = fir.convert %0 : (!fir.ref<i32>) -> memref<i32>
  %7 = fir.convert %2 : (!fir.ref<f64>) -> memref<f64>
  %8 = fir.convert %4 : (!fir.ref<!fir.array<1024xf64>>) -> memref<1024xf64>
  fir.do_loop %arg5 = %c1 to %c1024 step %c1 {
    %10 = fir.convert %arg5 : (index) -> i32
    memref.store %10, %6[] : memref<i32>
    %11 = memref.load %7[] : memref<f64>
    %12 = memref.load %6[] : memref<i32>
    %13 = fir.convert %12 : (i32) -> i64
    %14 = arith.index_cast %13 : i64 to index
    %15 = arith.subi %14, %c1 : index
    %16 = memref.load %8[%15] : memref<1024xf64>
    %17 = arith.addf %11, %16 fastmath<contract> : f64
    memref.store %17, %7[] : memref<f64>
    %18 = memref.load %6[] : memref<i32>
    %19 = arith.addi %18, %5 overflow<nsw> : i32
    memref.store %19, %6[] : memref<i32>
  }
  %9 = memref.load %6[] : memref<i32>
  memref.store %9, %6[] : memref<i32>
  omp.terminator
}

// -----// IR Dump After CanonicalizerPass (canonicalize) //----- //
module {
  func.func @kernel(%arg0: !fir.ref<i32> {jit.arg_type = 1 : ui32, jit.literal_val = 0 : i64}, %arg1: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg2: !fir.ref<f64> {jit.arg_type = 1 : ui32, jit.literal_val = 0 : i64}, %arg3: !fir.ref<!fir.array<1024xf64>> {jit.arg_type = 1 : ui32}, %arg4: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}) {
    %c1024 = arith.constant 1024 : index
    %c1 = arith.constant 1 : index
    %0 = fir.declare %arg0 {uniq_name = "_QFFrun_benchmarkEi"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %1 = fir.declare %arg1 {uniq_name = "_QFFrun_benchmarkEn"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %2 = fir.declare %arg2 {uniq_name = "_QFFrun_benchmarkEsum_x"} : (!fir.ref<f64>) -> !fir.ref<f64>
    %3 = fir.shape %c1024 : (index) -> !fir.shape<1>
    %4 = fir.declare %arg3(%3) {uniq_name = "_QFFrun_benchmarkEx"} : (!fir.ref<!fir.array<1024xf64>>, !fir.shape<1>) -> !fir.ref<!fir.array<1024xf64>>
    %5 = fir.convert %c1 : (index) -> i32
    %6 = fir.convert %0 : (!fir.ref<i32>) -> memref<i32>
    %7 = fir.convert %2 : (!fir.ref<f64>) -> memref<f64>
    %8 = fir.convert %4 : (!fir.ref<!fir.array<1024xf64>>) -> memref<1024xf64>
    fir.do_loop %arg5 = %c1 to %c1024 step %c1 {
      %10 = fir.convert %arg5 : (index) -> i32
      memref.store %10, %6[] : memref<i32>
      %11 = memref.load %7[] : memref<f64>
      %12 = memref.load %6[] : memref<i32>
      %13 = fir.convert %12 : (i32) -> i64
      %14 = arith.index_cast %13 : i64 to index
      %15 = arith.subi %14, %c1 : index
      %16 = memref.load %8[%15] : memref<1024xf64>
      %17 = arith.addf %11, %16 fastmath<contract> : f64
      memref.store %17, %7[] : memref<f64>
      %18 = memref.load %6[] : memref<i32>
      %19 = arith.addi %18, %5 overflow<nsw> : i32
      memref.store %19, %6[] : memref<i32>
    }
    %9 = memref.load %6[] : memref<i32>
    memref.store %9, %6[] : memref<i32>
    omp.terminator
  }
}


// -----// IR Dump After {anonymous}::CleanFIROpsPass (jforce-clean-fir-op) //----- //
func.func @kernel(%arg0: !fir.ref<i32> {jit.arg_type = 1 : ui32, jit.literal_val = 0 : i64}, %arg1: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg2: !fir.ref<f64> {jit.arg_type = 1 : ui32, jit.literal_val = 0 : i64}, %arg3: !fir.ref<!fir.array<1024xf64>> {jit.arg_type = 1 : ui32}, %arg4: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}) {
  %c1024 = arith.constant 1024 : index
  %c1 = arith.constant 1 : index
  %0 = fir.declare %arg0 {uniq_name = "_QFFrun_benchmarkEi"} : (!fir.ref<i32>) -> !fir.ref<i32>
  %1 = fir.declare %arg1 {uniq_name = "_QFFrun_benchmarkEn"} : (!fir.ref<i32>) -> !fir.ref<i32>
  %2 = fir.declare %arg2 {uniq_name = "_QFFrun_benchmarkEsum_x"} : (!fir.ref<f64>) -> !fir.ref<f64>
  %3 = fir.shape %c1024 : (index) -> !fir.shape<1>
  %4 = fir.declare %arg3(%3) {uniq_name = "_QFFrun_benchmarkEx"} : (!fir.ref<!fir.array<1024xf64>>, !fir.shape<1>) -> !fir.ref<!fir.array<1024xf64>>
  %5 = fir.convert %c1 : (index) -> i32
  %6 = fir.convert %0 : (!fir.ref<i32>) -> memref<i32>
  %7 = fir.convert %2 : (!fir.ref<f64>) -> memref<f64>
  %8 = fir.convert %4 : (!fir.ref<!fir.array<1024xf64>>) -> memref<1024xf64>
  fir.do_loop %arg5 = %c1 to %c1024 step %c1 {
    %10 = arith.index_cast %arg5 : index to i32
    memref.store %10, %6[] : memref<i32>
    %11 = memref.load %7[] : memref<f64>
    %12 = memref.load %6[] : memref<i32>
    %13 = arith.extsi %12 : i32 to i64
    %14 = arith.index_cast %13 : i64 to index
    %15 = arith.subi %14, %c1 : index
    %16 = memref.load %8[%15] : memref<1024xf64>
    %17 = arith.addf %11, %16 fastmath<contract> : f64
    memref.store %17, %7[] : memref<f64>
    %18 = memref.load %6[] : memref<i32>
    %19 = arith.addi %18, %5 overflow<nsw> : i32
    memref.store %19, %6[] : memref<i32>
  }
  %9 = memref.load %6[] : memref<i32>
  memref.store %9, %6[] : memref<i32>
  omp.terminator
}

// -----// IR Dump After AffineDialectPromotion (promote-to-affine) //----- //
func.func @kernel(%arg0: !fir.ref<i32> {jit.arg_type = 1 : ui32, jit.literal_val = 0 : i64}, %arg1: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg2: !fir.ref<f64> {jit.arg_type = 1 : ui32, jit.literal_val = 0 : i64}, %arg3: !fir.ref<!fir.array<1024xf64>> {jit.arg_type = 1 : ui32}, %arg4: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}) {
  %c1024 = arith.constant 1024 : index
  %c1 = arith.constant 1 : index
  %0 = fir.declare %arg0 {uniq_name = "_QFFrun_benchmarkEi"} : (!fir.ref<i32>) -> !fir.ref<i32>
  %1 = fir.declare %arg1 {uniq_name = "_QFFrun_benchmarkEn"} : (!fir.ref<i32>) -> !fir.ref<i32>
  %2 = fir.declare %arg2 {uniq_name = "_QFFrun_benchmarkEsum_x"} : (!fir.ref<f64>) -> !fir.ref<f64>
  %3 = fir.shape %c1024 : (index) -> !fir.shape<1>
  %4 = fir.declare %arg3(%3) {uniq_name = "_QFFrun_benchmarkEx"} : (!fir.ref<!fir.array<1024xf64>>, !fir.shape<1>) -> !fir.ref<!fir.array<1024xf64>>
  %5 = fir.convert %c1 : (index) -> i32
  %6 = fir.convert %0 : (!fir.ref<i32>) -> memref<i32>
  %7 = fir.convert %2 : (!fir.ref<f64>) -> memref<f64>
  %8 = fir.convert %4 : (!fir.ref<!fir.array<1024xf64>>) -> memref<1024xf64>
  affine.for %arg5 = %c1 to affine_map<()[s0] -> (s0 + 1)>()[%c1024] {
    %10 = arith.index_cast %arg5 : index to i32
    memref.store %10, %6[] : memref<i32>
    %11 = memref.load %7[] : memref<f64>
    %12 = memref.load %6[] : memref<i32>
    %13 = arith.extsi %12 : i32 to i64
    %14 = arith.index_cast %13 : i64 to index
    %15 = arith.subi %14, %c1 : index
    %16 = memref.load %8[%15] : memref<1024xf64>
    %17 = arith.addf %11, %16 fastmath<contract> : f64
    memref.store %17, %7[] : memref<f64>
    %18 = memref.load %6[] : memref<i32>
    %19 = arith.addi %18, %5 overflow<nsw> : i32
    memref.store %19, %6[] : memref<i32>
  }
  %9 = memref.load %6[] : memref<i32>
  memref.store %9, %6[] : memref<i32>
  omp.terminator
}

// -----// IR Dump After AffineLoopNormalize (affine-loop-normalize) //----- //
func.func @kernel(%arg0: !fir.ref<i32> {jit.arg_type = 1 : ui32, jit.literal_val = 0 : i64}, %arg1: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg2: !fir.ref<f64> {jit.arg_type = 1 : ui32, jit.literal_val = 0 : i64}, %arg3: !fir.ref<!fir.array<1024xf64>> {jit.arg_type = 1 : ui32}, %arg4: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}) {
  %c1024 = arith.constant 1024 : index
  %c1 = arith.constant 1 : index
  %0 = fir.declare %arg0 {uniq_name = "_QFFrun_benchmarkEi"} : (!fir.ref<i32>) -> !fir.ref<i32>
  %1 = fir.declare %arg1 {uniq_name = "_QFFrun_benchmarkEn"} : (!fir.ref<i32>) -> !fir.ref<i32>
  %2 = fir.declare %arg2 {uniq_name = "_QFFrun_benchmarkEsum_x"} : (!fir.ref<f64>) -> !fir.ref<f64>
  %3 = fir.shape %c1024 : (index) -> !fir.shape<1>
  %4 = fir.declare %arg3(%3) {uniq_name = "_QFFrun_benchmarkEx"} : (!fir.ref<!fir.array<1024xf64>>, !fir.shape<1>) -> !fir.ref<!fir.array<1024xf64>>
  %5 = fir.convert %c1 : (index) -> i32
  %6 = fir.convert %0 : (!fir.ref<i32>) -> memref<i32>
  %7 = fir.convert %2 : (!fir.ref<f64>) -> memref<f64>
  %8 = fir.convert %4 : (!fir.ref<!fir.array<1024xf64>>) -> memref<1024xf64>
  affine.for %arg5 = 0 to 1024 {
    %10 = affine.apply affine_map<(d0) -> (d0 + 1)>(%arg5)
    %11 = arith.index_cast %10 : index to i32
    memref.store %11, %6[] : memref<i32>
    %12 = memref.load %7[] : memref<f64>
    %13 = memref.load %6[] : memref<i32>
    %14 = arith.extsi %13 : i32 to i64
    %15 = arith.index_cast %14 : i64 to index
    %16 = arith.subi %15, %c1 : index
    %17 = memref.load %8[%16] : memref<1024xf64>
    %18 = arith.addf %12, %17 fastmath<contract> : f64
    memref.store %18, %7[] : memref<f64>
    %19 = memref.load %6[] : memref<i32>
    %20 = arith.addi %19, %5 overflow<nsw> : i32
    memref.store %20, %6[] : memref<i32>
  }
  %9 = memref.load %6[] : memref<i32>
  memref.store %9, %6[] : memref<i32>
  omp.terminator
}

// -----// IR Dump After CanonicalizerPass (canonicalize) //----- //
#map = affine_map<(d0) -> (d0 + 1)>
module {
  func.func @kernel(%arg0: !fir.ref<i32> {jit.arg_type = 1 : ui32, jit.literal_val = 0 : i64}, %arg1: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg2: !fir.ref<f64> {jit.arg_type = 1 : ui32, jit.literal_val = 0 : i64}, %arg3: !fir.ref<!fir.array<1024xf64>> {jit.arg_type = 1 : ui32}, %arg4: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}) {
    %c1024 = arith.constant 1024 : index
    %c1 = arith.constant 1 : index
    %0 = fir.declare %arg0 {uniq_name = "_QFFrun_benchmarkEi"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %1 = fir.declare %arg1 {uniq_name = "_QFFrun_benchmarkEn"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %2 = fir.declare %arg2 {uniq_name = "_QFFrun_benchmarkEsum_x"} : (!fir.ref<f64>) -> !fir.ref<f64>
    %3 = fir.shape %c1024 : (index) -> !fir.shape<1>
    %4 = fir.declare %arg3(%3) {uniq_name = "_QFFrun_benchmarkEx"} : (!fir.ref<!fir.array<1024xf64>>, !fir.shape<1>) -> !fir.ref<!fir.array<1024xf64>>
    %5 = fir.convert %c1 : (index) -> i32
    %6 = fir.convert %0 : (!fir.ref<i32>) -> memref<i32>
    %7 = fir.convert %2 : (!fir.ref<f64>) -> memref<f64>
    %8 = fir.convert %4 : (!fir.ref<!fir.array<1024xf64>>) -> memref<1024xf64>
    affine.for %arg5 = 0 to 1024 {
      %10 = affine.apply #map(%arg5)
      %11 = arith.index_cast %10 : index to i32
      memref.store %11, %6[] : memref<i32>
      %12 = memref.load %7[] : memref<f64>
      %13 = memref.load %6[] : memref<i32>
      %14 = arith.index_cast %13 : i32 to index
      %15 = arith.subi %14, %c1 : index
      %16 = memref.load %8[%15] : memref<1024xf64>
      %17 = arith.addf %12, %16 fastmath<contract> : f64
      memref.store %17, %7[] : memref<f64>
      %18 = memref.load %6[] : memref<i32>
      %19 = arith.addi %18, %5 overflow<nsw> : i32
      memref.store %19, %6[] : memref<i32>
    }
    %9 = memref.load %6[] : memref<i32>
    memref.store %9, %6[] : memref<i32>
    omp.terminator
  }
}


[DEBUG]compare reading and storeOp
[DEBUG] memref.store %9, %6[] : memref<i32> 
[DEBUG] %9 = memref.load %6[] : memref<i32> 
[DEBUG]compare reading and storeOp
[DEBUG] memref.store %18, %6[] : memref<i32> 
[DEBUG] %18 = arith.addi %17, %5 overflow<nsw> : i32 
[DEBUG]Unexpected readOp: 
[DEBUG] %18 = arith.addi %17, %5 overflow<nsw> : i32 
[DEBUG]compare reading and storeOp
[DEBUG] memref.store %16, %7[] : memref<f64> 
[DEBUG] %16 = arith.addf %11, %15 fastmath<contract> : f64 
[DEBUG]Unexpected readOp: 
[DEBUG] %16 = arith.addf %11, %15 fastmath<contract> : f64 
[DEBUG]compare reading and storeOp
[DEBUG] memref.store %10, %6[] : memref<i32> 
[DEBUG] %10 = arith.index_cast %9 : index to i32 
[DEBUG]Unexpected readOp: 
[DEBUG] %10 = arith.index_cast %9 : index to i32 
[DEBUG]compare reading and storeOp
[DEBUG] memref.store %16, %6[] : memref<i32> 
[DEBUG] %16 = arith.addi %10, %5 overflow<nsw> : i32 
[DEBUG]Unexpected readOp: 
[DEBUG] %16 = arith.addi %10, %5 overflow<nsw> : i32 
[DEBUG]compare reading and storeOp
[DEBUG] memref.store %16, %6[] : memref<i32> 
[DEBUG] %16 = arith.addi %15, %5 overflow<nsw> : i32 
[DEBUG]Unexpected readOp: 
[DEBUG] %16 = arith.addi %15, %5 overflow<nsw> : i32 
[DEBUG]compare reading and storeOp
[DEBUG] memref.store %16, %7[] : memref<f64> 
[DEBUG] %16 = arith.addf %12, %15 fastmath<contract> : f64 
[DEBUG]Unexpected readOp: 
[DEBUG] %16 = arith.addf %12, %15 fastmath<contract> : f64 
[DEBUG]compare reading and storeOp
[DEBUG] memref.store %9, %6[] : memref<i32> 
[DEBUG] %9 = arith.addi %5, %c1024_i32 overflow<nsw> : i32 
[DEBUG]Unexpected readOp: 
[DEBUG] %9 = arith.addi %5, %c1024_i32 overflow<nsw> : i32 
[DEBUG]compare reading and storeOp
[DEBUG] memref.store %16, %7[] : memref<f64> 
[DEBUG] %16 = arith.addf %12, %15 fastmath<contract> : f64 
[DEBUG]Unexpected readOp: 
[DEBUG] %16 = arith.addf %12, %15 fastmath<contract> : f64 
// -----// IR Dump After {anonymous}::OptimizeMemOpsPass (jforce-optimize-mem-ops) //----- //
func.func @kernel(%arg0: !fir.ref<i32> {jit.arg_type = 1 : ui32, jit.literal_val = 0 : i64}, %arg1: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg2: !fir.ref<f64> {jit.arg_type = 1 : ui32, jit.literal_val = 0 : i64}, %arg3: !fir.ref<!fir.array<1024xf64>> {jit.arg_type = 1 : ui32}, %arg4: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}) {
  %c1024_i32 = arith.constant 1024 : i32
  %c1024 = arith.constant 1024 : index
  %c1 = arith.constant 1 : index
  %0 = fir.declare %arg0 {uniq_name = "_QFFrun_benchmarkEi"} : (!fir.ref<i32>) -> !fir.ref<i32>
  %1 = fir.declare %arg1 {uniq_name = "_QFFrun_benchmarkEn"} : (!fir.ref<i32>) -> !fir.ref<i32>
  %2 = fir.declare %arg2 {uniq_name = "_QFFrun_benchmarkEsum_x"} : (!fir.ref<f64>) -> !fir.ref<f64>
  %3 = fir.shape %c1024 : (index) -> !fir.shape<1>
  %4 = fir.declare %arg3(%3) {uniq_name = "_QFFrun_benchmarkEx"} : (!fir.ref<!fir.array<1024xf64>>, !fir.shape<1>) -> !fir.ref<!fir.array<1024xf64>>
  %5 = fir.convert %c1 : (index) -> i32
  %6 = fir.convert %0 : (!fir.ref<i32>) -> memref<i32>
  %7 = fir.convert %2 : (!fir.ref<f64>) -> memref<f64>
  %8 = fir.convert %4 : (!fir.ref<!fir.array<1024xf64>>) -> memref<1024xf64>
  affine.for %arg5 = 0 to 1024 {
    %10 = affine.apply affine_map<(d0) -> (d0 + 1)>(%arg5)
    %11 = arith.index_cast %10 : index to i32
    %12 = memref.load %7[] : memref<f64>
    %13 = arith.index_cast %11 : i32 to index
    %14 = arith.subi %13, %c1 : index
    %15 = memref.load %8[%14] : memref<1024xf64>
    %16 = arith.addf %12, %15 fastmath<contract> : f64
    memref.store %16, %7[] : memref<f64>
  }
  %9 = arith.addi %5, %c1024_i32 overflow<nsw> : i32
  memref.store %9, %6[] : memref<i32>
  omp.terminator
}

// -----// IR Dump After CanonicalizerPass (canonicalize) //----- //
#map = affine_map<(d0) -> (d0 + 1)>
module {
  func.func @kernel(%arg0: !fir.ref<i32> {jit.arg_type = 1 : ui32, jit.literal_val = 0 : i64}, %arg1: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg2: !fir.ref<f64> {jit.arg_type = 1 : ui32, jit.literal_val = 0 : i64}, %arg3: !fir.ref<!fir.array<1024xf64>> {jit.arg_type = 1 : ui32}, %arg4: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}) {
    %c1024_i32 = arith.constant 1024 : i32
    %c1024 = arith.constant 1024 : index
    %c1 = arith.constant 1 : index
    %0 = fir.declare %arg0 {uniq_name = "_QFFrun_benchmarkEi"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %1 = fir.declare %arg1 {uniq_name = "_QFFrun_benchmarkEn"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %2 = fir.declare %arg2 {uniq_name = "_QFFrun_benchmarkEsum_x"} : (!fir.ref<f64>) -> !fir.ref<f64>
    %3 = fir.shape %c1024 : (index) -> !fir.shape<1>
    %4 = fir.declare %arg3(%3) {uniq_name = "_QFFrun_benchmarkEx"} : (!fir.ref<!fir.array<1024xf64>>, !fir.shape<1>) -> !fir.ref<!fir.array<1024xf64>>
    %5 = fir.convert %c1 : (index) -> i32
    %6 = fir.convert %0 : (!fir.ref<i32>) -> memref<i32>
    %7 = fir.convert %2 : (!fir.ref<f64>) -> memref<f64>
    %8 = fir.convert %4 : (!fir.ref<!fir.array<1024xf64>>) -> memref<1024xf64>
    affine.for %arg5 = 0 to 1024 {
      %10 = affine.apply #map(%arg5)
      %11 = arith.index_cast %10 : index to i32
      %12 = memref.load %7[] : memref<f64>
      %13 = arith.index_cast %11 : i32 to index
      %14 = arith.subi %13, %c1 : index
      %15 = memref.load %8[%14] : memref<1024xf64>
      %16 = arith.addf %12, %15 fastmath<contract> : f64
      memref.store %16, %7[] : memref<f64>
    }
    %9 = arith.addi %5, %c1024_i32 overflow<nsw> : i32
    memref.store %9, %6[] : memref<i32>
    omp.terminator
  }
}


// -----// IR Dump After {anonymous}::LoopSinkingPass (jforce-loop-sink) //----- //
func.func @kernel(%arg0: !fir.ref<i32> {jit.arg_type = 1 : ui32, jit.literal_val = 0 : i64}, %arg1: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg2: !fir.ref<f64> {jit.arg_type = 1 : ui32, jit.literal_val = 0 : i64}, %arg3: !fir.ref<!fir.array<1024xf64>> {jit.arg_type = 1 : ui32}, %arg4: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}) {
  %c1024_i32 = arith.constant 1024 : i32
  %c1024 = arith.constant 1024 : index
  %c1 = arith.constant 1 : index
  %0 = fir.declare %arg0 {uniq_name = "_QFFrun_benchmarkEi"} : (!fir.ref<i32>) -> !fir.ref<i32>
  %1 = fir.declare %arg1 {uniq_name = "_QFFrun_benchmarkEn"} : (!fir.ref<i32>) -> !fir.ref<i32>
  %2 = fir.declare %arg2 {uniq_name = "_QFFrun_benchmarkEsum_x"} : (!fir.ref<f64>) -> !fir.ref<f64>
  %3 = fir.shape %c1024 : (index) -> !fir.shape<1>
  %4 = fir.declare %arg3(%3) {uniq_name = "_QFFrun_benchmarkEx"} : (!fir.ref<!fir.array<1024xf64>>, !fir.shape<1>) -> !fir.ref<!fir.array<1024xf64>>
  %5 = fir.convert %c1 : (index) -> i32
  %6 = fir.convert %0 : (!fir.ref<i32>) -> memref<i32>
  %7 = fir.convert %2 : (!fir.ref<f64>) -> memref<f64>
  %8 = fir.convert %4 : (!fir.ref<!fir.array<1024xf64>>) -> memref<1024xf64>
  affine.for %arg5 = 0 to 1024 {
    %10 = affine.apply affine_map<(d0) -> (d0 + 1)>(%arg5)
    %11 = arith.index_cast %10 : index to i32
    %12 = memref.load %7[] : memref<f64>
    %13 = arith.index_cast %11 : i32 to index
    %14 = arith.subi %13, %c1 : index
    %15 = memref.load %8[%14] : memref<1024xf64>
    %16 = arith.addf %12, %15 fastmath<contract> : f64
    memref.store %16, %7[] : memref<f64>
  }
  %9 = arith.addi %5, %c1024_i32 overflow<nsw> : i32
  memref.store %9, %6[] : memref<i32>
  omp.terminator
}

// -----// IR Dump After {anonymous}::AffineCFGPass (enzyme-affinecfg) //----- //
module {
  func.func @kernel(%arg0: !fir.ref<i32> {jit.arg_type = 1 : ui32, jit.literal_val = 0 : i64}, %arg1: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg2: !fir.ref<f64> {jit.arg_type = 1 : ui32, jit.literal_val = 0 : i64}, %arg3: !fir.ref<!fir.array<1024xf64>> {jit.arg_type = 1 : ui32}, %arg4: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}) {
    %c1024_i32 = arith.constant 1024 : i32
    %c1024 = arith.constant 1024 : index
    %c1 = arith.constant 1 : index
    %0 = fir.declare %arg0 {uniq_name = "_QFFrun_benchmarkEi"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %1 = fir.declare %arg1 {uniq_name = "_QFFrun_benchmarkEn"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %2 = fir.declare %arg2 {uniq_name = "_QFFrun_benchmarkEsum_x"} : (!fir.ref<f64>) -> !fir.ref<f64>
    %3 = fir.shape %c1024 : (index) -> !fir.shape<1>
    %4 = fir.declare %arg3(%3) {uniq_name = "_QFFrun_benchmarkEx"} : (!fir.ref<!fir.array<1024xf64>>, !fir.shape<1>) -> !fir.ref<!fir.array<1024xf64>>
    %5 = fir.convert %c1 : (index) -> i32
    %6 = fir.convert %0 : (!fir.ref<i32>) -> memref<i32>
    %7 = fir.convert %2 : (!fir.ref<f64>) -> memref<f64>
    %8 = fir.convert %4 : (!fir.ref<!fir.array<1024xf64>>) -> memref<1024xf64>
    %9 = affine.load %7[] : memref<f64>
    %10 = affine.parallel (%arg5) = (0) to (1024) reduce ("addf") -> (f64) {
      %13 = affine.load %8[%arg5] : memref<1024xf64>
      affine.yield %13 : f64
    }
    %11 = arith.addf %9, %10 fastmath<contract> : f64
    affine.store %11, %7[] : memref<f64>
    %12 = arith.addi %5, %c1024_i32 overflow<nsw> : i32
    affine.store %12, %6[] : memref<i32>
    omp.terminator
  }
}


// -----// IR Dump After {anonymous}::OutlineAffinePass (jforce-outline-affine) //----- //
module {
  func.func @outlined_affinefor_103112802683984(%arg0: memref<1024xf64>, %arg1: memref<f64>) {
    %0 = affine.parallel (%arg2) = (0) to (1024) reduce ("addf") -> (f64) {
      %1 = affine.load %arg0[%arg2] : memref<1024xf64>
      affine.yield %1 : f64
    }
    affine.store %0, %arg1[] : memref<f64>
    return
  }
  func.func @kernel(%arg0: !fir.ref<i32> {jit.arg_type = 1 : ui32, jit.literal_val = 0 : i64}, %arg1: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg2: !fir.ref<f64> {jit.arg_type = 1 : ui32, jit.literal_val = 0 : i64}, %arg3: !fir.ref<!fir.array<1024xf64>> {jit.arg_type = 1 : ui32}, %arg4: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}) {
    %c1024_i32 = arith.constant 1024 : i32
    %c1024 = arith.constant 1024 : index
    %c1 = arith.constant 1 : index
    %0 = fir.declare %arg0 {uniq_name = "_QFFrun_benchmarkEi"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %1 = fir.declare %arg1 {uniq_name = "_QFFrun_benchmarkEn"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %2 = fir.declare %arg2 {uniq_name = "_QFFrun_benchmarkEsum_x"} : (!fir.ref<f64>) -> !fir.ref<f64>
    %3 = fir.shape %c1024 : (index) -> !fir.shape<1>
    %4 = fir.declare %arg3(%3) {uniq_name = "_QFFrun_benchmarkEx"} : (!fir.ref<!fir.array<1024xf64>>, !fir.shape<1>) -> !fir.ref<!fir.array<1024xf64>>
    %5 = fir.convert %c1 : (index) -> i32
    %6 = fir.convert %0 : (!fir.ref<i32>) -> memref<i32>
    %7 = fir.convert %2 : (!fir.ref<f64>) -> memref<f64>
    %8 = fir.convert %4 : (!fir.ref<!fir.array<1024xf64>>) -> memref<1024xf64>
    %9 = affine.load %7[] : memref<f64>
    %alloca = memref.alloca() : memref<f64>
    call @outlined_affinefor_103112802683984(%8, %alloca) : (memref<1024xf64>, memref<f64>) -> ()
    %10 = affine.load %alloca[] : memref<f64>
    %11 = arith.addf %9, %10 fastmath<contract> : f64
    affine.store %11, %7[] : memref<f64>
    %12 = arith.addi %5, %c1024_i32 overflow<nsw> : i32
    affine.store %12, %6[] : memref<i32>
    omp.terminator
  }
}


// -----// IR Dump After {anonymous}::AffineToStableHLORaisingPass (enzyme-affine-to-stablehlo) //----- //
func.func @kernel(%arg0: !fir.ref<i32> {jit.arg_type = 1 : ui32, jit.literal_val = 0 : i64}, %arg1: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg2: !fir.ref<f64> {jit.arg_type = 1 : ui32, jit.literal_val = 0 : i64}, %arg3: !fir.ref<!fir.array<1024xf64>> {jit.arg_type = 1 : ui32}, %arg4: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}) {
  %c1024_i32 = arith.constant 1024 : i32
  %c1024 = arith.constant 1024 : index
  %c1 = arith.constant 1 : index
  %0 = fir.declare %arg0 {uniq_name = "_QFFrun_benchmarkEi"} : (!fir.ref<i32>) -> !fir.ref<i32>
  %1 = fir.declare %arg1 {uniq_name = "_QFFrun_benchmarkEn"} : (!fir.ref<i32>) -> !fir.ref<i32>
  %2 = fir.declare %arg2 {uniq_name = "_QFFrun_benchmarkEsum_x"} : (!fir.ref<f64>) -> !fir.ref<f64>
  %3 = fir.shape %c1024 : (index) -> !fir.shape<1>
  %4 = fir.declare %arg3(%3) {uniq_name = "_QFFrun_benchmarkEx"} : (!fir.ref<!fir.array<1024xf64>>, !fir.shape<1>) -> !fir.ref<!fir.array<1024xf64>>
  %5 = fir.convert %c1 : (index) -> i32
  %6 = fir.convert %0 : (!fir.ref<i32>) -> memref<i32>
  %7 = fir.convert %2 : (!fir.ref<f64>) -> memref<f64>
  %8 = fir.convert %4 : (!fir.ref<!fir.array<1024xf64>>) -> memref<1024xf64>
  %9 = affine.load %7[] : memref<f64>
  %alloca = memref.alloca() : memref<f64>
  call @outlined_affinefor_103112802683984(%8, %alloca) : (memref<1024xf64>, memref<f64>) -> ()
  %10 = affine.load %alloca[] : memref<f64>
  %11 = arith.addf %9, %10 fastmath<contract> : f64
  affine.store %11, %7[] : memref<f64>
  %12 = arith.addi %5, %c1024_i32 overflow<nsw> : i32
  affine.store %12, %6[] : memref<i32>
  omp.terminator
}

// -----// IR Dump After {anonymous}::AffineToStableHLORaisingPass (enzyme-affine-to-stablehlo) //----- //
func.func @outlined_affinefor_103112802683984(%arg0: memref<1024xf64>, %arg1: memref<f64>) {
  %0 = affine.parallel (%arg2) = (0) to (1024) reduce ("addf") -> (f64) {
    %1 = affine.load %arg0[%arg2] : memref<1024xf64>
    affine.yield %1 : f64
  }
  affine.store %0, %arg1[] : memref<f64>
  return
}

// -----// IR Dump After CanonicalizerPass (canonicalize) //----- //
module {
  func.func @outlined_affinefor_103112802683984(%arg0: memref<1024xf64>, %arg1: memref<f64>) {
    %0 = affine.parallel (%arg2) = (0) to (1024) reduce ("addf") -> (f64) {
      %1 = affine.load %arg0[%arg2] : memref<1024xf64>
      affine.yield %1 : f64
    }
    affine.store %0, %arg1[] : memref<f64>
    return
  }
  func.func @kernel(%arg0: !fir.ref<i32> {jit.arg_type = 1 : ui32, jit.literal_val = 0 : i64}, %arg1: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg2: !fir.ref<f64> {jit.arg_type = 1 : ui32, jit.literal_val = 0 : i64}, %arg3: !fir.ref<!fir.array<1024xf64>> {jit.arg_type = 1 : ui32}, %arg4: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}) {
    %c1024_i32 = arith.constant 1024 : i32
    %c1024 = arith.constant 1024 : index
    %c1 = arith.constant 1 : index
    %0 = fir.declare %arg0 {uniq_name = "_QFFrun_benchmarkEi"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %1 = fir.declare %arg1 {uniq_name = "_QFFrun_benchmarkEn"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %2 = fir.declare %arg2 {uniq_name = "_QFFrun_benchmarkEsum_x"} : (!fir.ref<f64>) -> !fir.ref<f64>
    %3 = fir.shape %c1024 : (index) -> !fir.shape<1>
    %4 = fir.declare %arg3(%3) {uniq_name = "_QFFrun_benchmarkEx"} : (!fir.ref<!fir.array<1024xf64>>, !fir.shape<1>) -> !fir.ref<!fir.array<1024xf64>>
    %5 = fir.convert %c1 : (index) -> i32
    %6 = fir.convert %0 : (!fir.ref<i32>) -> memref<i32>
    %7 = fir.convert %2 : (!fir.ref<f64>) -> memref<f64>
    %8 = fir.convert %4 : (!fir.ref<!fir.array<1024xf64>>) -> memref<1024xf64>
    %9 = affine.load %7[] : memref<f64>
    %alloca = memref.alloca() : memref<f64>
    call @outlined_affinefor_103112802683984(%8, %alloca) : (memref<1024xf64>, memref<f64>) -> ()
    %10 = affine.load %alloca[] : memref<f64>
    %11 = arith.addf %9, %10 fastmath<contract> : f64
    affine.store %11, %7[] : memref<f64>
    %12 = arith.addi %5, %c1024_i32 overflow<nsw> : i32
    affine.store %12, %6[] : memref<i32>
    omp.terminator
  }
  func.func private @outlined_affinefor_103112802683984_raised(%arg0: tensor<1024xf64>, %arg1: tensor<f64>) -> (tensor<1024xf64>, tensor<f64>) {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %0 = stablehlo.reshape %arg0 : (tensor<1024xf64>) -> tensor<1024xf64>
    %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<1024xf64>, tensor<f64>) -> tensor<f64>
    %2 = stablehlo.broadcast_in_dim %1, dims = [] : (tensor<f64>) -> tensor<f64>
    %3 = stablehlo.dynamic_update_slice %arg1, %2 : (tensor<f64>, tensor<f64>) -> tensor<f64>
    return %arg0, %3 : tensor<1024xf64>, tensor<f64>
  }
}


