// ./build/src/tool/jforce-opt ./tests/axpy-dl-2d.mlir \
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
//    --mlir-print-ir-after-all


// -----// IR Dump After {anonymous}::AnnotatePass (jforce-annotate) //----- //
func.func @kernel(%arg0: !fir.ref<i32> {jit.arg_type = 1 : ui32, jit.literal_val = 0 : i64}, %arg1: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 3 : i64}, %arg2: !fir.ref<i32> {jit.arg_type = 1 : ui32, jit.literal_val = 0 : i64}, %arg3: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 4 : i64}, %arg4: !fir.ref<!fir.array<?x?xf64>> {jit.arg_type = 1 : ui32}, %arg5: !fir.ref<!fir.array<?x?xf64>> {jit.arg_type = 1 : ui32}, %arg6: !fir.ref<!fir.array<?x?xf64>> {jit.arg_type = 1 : ui32}, %arg7: !fir.ref<f64> {jit.arg_type = 1 : ui32, jit.literal_val = 4619567317775286272 : i64}, %arg8: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 3 : i64}, %arg9: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 4 : i64}, %arg10: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 3 : i64}, %arg11: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 4 : i64}, %arg12: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 3 : i64}, %arg13: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 4 : i64}) {
  %0 = fir.load %arg13 : !fir.ref<i32>
  %1 = fir.load %arg12 : !fir.ref<i32>
  %2 = fir.load %arg11 : !fir.ref<i32>
  %3 = fir.load %arg10 : !fir.ref<i32>
  %4 = fir.load %arg9 : !fir.ref<i32>
  %5 = fir.load %arg8 : !fir.ref<i32>
  %6 = fir.convert %5 : (i32) -> i64
  %7 = fir.convert %4 : (i32) -> i64
  %8 = fir.convert %3 : (i32) -> i64
  %9 = fir.convert %2 : (i32) -> i64
  %10 = fir.convert %1 : (i32) -> i64
  %11 = fir.convert %0 : (i32) -> i64
  %c0 = arith.constant 0 : index
  %12 = fir.convert %11 : (i64) -> index
  %13 = arith.cmpi sgt, %12, %c0 : index
  %c0_0 = arith.constant 0 : index
  %14 = fir.convert %10 : (i64) -> index
  %15 = arith.cmpi sgt, %14, %c0_0 : index
  %c0_1 = arith.constant 0 : index
  %16 = fir.convert %9 : (i64) -> index
  %17 = arith.cmpi sgt, %16, %c0_1 : index
  %c0_2 = arith.constant 0 : index
  %18 = fir.convert %8 : (i64) -> index
  %19 = arith.cmpi sgt, %18, %c0_2 : index
  %c0_3 = arith.constant 0 : index
  %20 = fir.convert %7 : (i64) -> index
  %21 = arith.cmpi sgt, %20, %c0_3 : index
  %c0_4 = arith.constant 0 : index
  %22 = fir.convert %6 : (i64) -> index
  %23 = arith.cmpi sgt, %22, %c0_4 : index
  %24 = arith.select %23, %22, %c0_4 : index
  %25 = arith.select %21, %20, %c0_3 : index
  %26 = arith.select %19, %18, %c0_2 : index
  %27 = arith.select %17, %16, %c0_1 : index
  %28 = arith.select %15, %14, %c0_0 : index
  %29 = arith.select %13, %12, %c0 : index
  %30:2 = hlfir.declare %arg0 {uniq_name = "_QFFrun_benchmarkEj"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %31:2 = hlfir.declare %arg1 {uniq_name = "_QFFrun_benchmarkEm"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %32:2 = hlfir.declare %arg2 {uniq_name = "_QFFrun_benchmarkEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %33:2 = hlfir.declare %arg3 {uniq_name = "_QFFrun_benchmarkEn"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %34 = fir.shape %29, %28 : (index, index) -> !fir.shape<2>
  %35:2 = hlfir.declare %arg4(%34) {uniq_name = "_QFFrun_benchmarkEz"} : (!fir.ref<!fir.array<?x?xf64>>, !fir.shape<2>) -> (!fir.box<!fir.array<?x?xf64>>, !fir.ref<!fir.array<?x?xf64>>)
  %36 = fir.shape %27, %26 : (index, index) -> !fir.shape<2>
  %37:2 = hlfir.declare %arg5(%36) {uniq_name = "_QFFrun_benchmarkEx"} : (!fir.ref<!fir.array<?x?xf64>>, !fir.shape<2>) -> (!fir.box<!fir.array<?x?xf64>>, !fir.ref<!fir.array<?x?xf64>>)
  %38 = fir.shape %25, %24 : (index, index) -> !fir.shape<2>
  %39:2 = hlfir.declare %arg6(%38) {uniq_name = "_QFFrun_benchmarkEy"} : (!fir.ref<!fir.array<?x?xf64>>, !fir.shape<2>) -> (!fir.box<!fir.array<?x?xf64>>, !fir.ref<!fir.array<?x?xf64>>)
  %40:2 = hlfir.declare %arg7 {uniq_name = "_QFFrun_benchmarkEa"} : (!fir.ref<f64>) -> (!fir.ref<f64>, !fir.ref<f64>)
  %c1_i32 = arith.constant 1 : i32
  %41 = fir.convert %c1_i32 : (i32) -> index
  %42 = fir.load %31#0 : !fir.ref<i32>
  %43 = fir.convert %42 : (i32) -> index
  %c1 = arith.constant 1 : index
  %44 = fir.convert %41 : (index) -> i32
  %45 = fir.do_loop %arg14 = %41 to %43 step %c1 iter_args(%arg15 = %44) -> (i32) {
    fir.store %arg15 to %30#0 : !fir.ref<i32>
    %c1_i32_5 = arith.constant 1 : i32
    %46 = fir.convert %c1_i32_5 : (i32) -> index
    %47 = fir.load %33#0 : !fir.ref<i32>
    %48 = fir.convert %47 : (i32) -> index
    %c1_6 = arith.constant 1 : index
    %49 = fir.convert %46 : (index) -> i32
    %50 = fir.do_loop %arg16 = %46 to %48 step %c1_6 iter_args(%arg17 = %49) -> (i32) {
      fir.store %arg17 to %32#0 : !fir.ref<i32>
      %54 = fir.load %40#0 : !fir.ref<f64>
      %55 = fir.load %32#0 : !fir.ref<i32>
      %56 = fir.convert %55 : (i32) -> i64
      %57 = fir.load %30#0 : !fir.ref<i32>
      %58 = fir.convert %57 : (i32) -> i64
      %59 = hlfir.designate %37#0 (%56, %58)  : (!fir.box<!fir.array<?x?xf64>>, i64, i64) -> !fir.ref<f64>
      %60 = fir.load %59 : !fir.ref<f64>
      %61 = arith.mulf %54, %60 fastmath<contract> : f64
      %62 = fir.load %32#0 : !fir.ref<i32>
      %63 = fir.convert %62 : (i32) -> i64
      %64 = fir.load %30#0 : !fir.ref<i32>
      %65 = fir.convert %64 : (i32) -> i64
      %66 = hlfir.designate %39#0 (%63, %65)  : (!fir.box<!fir.array<?x?xf64>>, i64, i64) -> !fir.ref<f64>
      %67 = fir.load %66 : !fir.ref<f64>
      %68 = arith.addf %61, %67 fastmath<contract> : f64
      %69 = fir.load %32#0 : !fir.ref<i32>
      %70 = fir.convert %69 : (i32) -> i64
      %71 = fir.load %30#0 : !fir.ref<i32>
      %72 = fir.convert %71 : (i32) -> i64
      %73 = hlfir.designate %35#0 (%70, %72)  : (!fir.box<!fir.array<?x?xf64>>, i64, i64) -> !fir.ref<f64>
      hlfir.assign %68 to %73 : f64, !fir.ref<f64>
      %74 = fir.convert %c1_6 : (index) -> i32
      %75 = fir.load %32#0 : !fir.ref<i32>
      %76 = arith.addi %75, %74 overflow<nsw> : i32
      fir.result %76 : i32
    }
    fir.store %50 to %32#0 : !fir.ref<i32>
    %51 = fir.convert %c1 : (index) -> i32
    %52 = fir.load %30#0 : !fir.ref<i32>
    %53 = arith.addi %52, %51 overflow<nsw> : i32
    fir.result %53 : i32
  }
  fir.store %45 to %30#0 : !fir.ref<i32>
  omp.terminator
}

// -----// IR Dump After {anonymous}::PropagateConstantsPass (jforce-propagate-constants) //----- //
func.func @kernel(%arg0: !fir.ref<i32> {jit.arg_type = 1 : ui32, jit.literal_val = 0 : i64}, %arg1: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 3 : i64}, %arg2: !fir.ref<i32> {jit.arg_type = 1 : ui32, jit.literal_val = 0 : i64}, %arg3: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 4 : i64}, %arg4: !fir.ref<!fir.array<?x?xf64>> {jit.arg_type = 1 : ui32}, %arg5: !fir.ref<!fir.array<?x?xf64>> {jit.arg_type = 1 : ui32}, %arg6: !fir.ref<!fir.array<?x?xf64>> {jit.arg_type = 1 : ui32}, %arg7: !fir.ref<f64> {jit.arg_type = 1 : ui32, jit.literal_val = 4619567317775286272 : i64}, %arg8: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 3 : i64}, %arg9: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 4 : i64}, %arg10: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 3 : i64}, %arg11: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 4 : i64}, %arg12: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 3 : i64}, %arg13: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 4 : i64}) {
  %c4_i32 = arith.constant 4 : i32
  %c3_i32 = arith.constant 3 : i32
  %c4_i32_0 = arith.constant 4 : i32
  %c3_i32_1 = arith.constant 3 : i32
  %c4_i32_2 = arith.constant 4 : i32
  %c3_i32_3 = arith.constant 3 : i32
  %0 = fir.convert %c3_i32_3 : (i32) -> i64
  %1 = fir.convert %c4_i32_2 : (i32) -> i64
  %2 = fir.convert %c3_i32_1 : (i32) -> i64
  %3 = fir.convert %c4_i32_0 : (i32) -> i64
  %4 = fir.convert %c3_i32 : (i32) -> i64
  %5 = fir.convert %c4_i32 : (i32) -> i64
  %c0 = arith.constant 0 : index
  %6 = fir.convert %5 : (i64) -> index
  %7 = arith.cmpi sgt, %6, %c0 : index
  %c0_4 = arith.constant 0 : index
  %8 = fir.convert %4 : (i64) -> index
  %9 = arith.cmpi sgt, %8, %c0_4 : index
  %c0_5 = arith.constant 0 : index
  %10 = fir.convert %3 : (i64) -> index
  %11 = arith.cmpi sgt, %10, %c0_5 : index
  %c0_6 = arith.constant 0 : index
  %12 = fir.convert %2 : (i64) -> index
  %13 = arith.cmpi sgt, %12, %c0_6 : index
  %c0_7 = arith.constant 0 : index
  %14 = fir.convert %1 : (i64) -> index
  %15 = arith.cmpi sgt, %14, %c0_7 : index
  %c0_8 = arith.constant 0 : index
  %16 = fir.convert %0 : (i64) -> index
  %17 = arith.cmpi sgt, %16, %c0_8 : index
  %18 = arith.select %17, %16, %c0_8 : index
  %19 = arith.select %15, %14, %c0_7 : index
  %20 = arith.select %13, %12, %c0_6 : index
  %21 = arith.select %11, %10, %c0_5 : index
  %22 = arith.select %9, %8, %c0_4 : index
  %23 = arith.select %7, %6, %c0 : index
  %24:2 = hlfir.declare %arg0 {uniq_name = "_QFFrun_benchmarkEj"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %25:2 = hlfir.declare %arg1 {uniq_name = "_QFFrun_benchmarkEm"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %26:2 = hlfir.declare %arg2 {uniq_name = "_QFFrun_benchmarkEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %27:2 = hlfir.declare %arg3 {uniq_name = "_QFFrun_benchmarkEn"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %28 = fir.shape %23, %22 : (index, index) -> !fir.shape<2>
  %29:2 = hlfir.declare %arg4(%28) {uniq_name = "_QFFrun_benchmarkEz"} : (!fir.ref<!fir.array<?x?xf64>>, !fir.shape<2>) -> (!fir.box<!fir.array<?x?xf64>>, !fir.ref<!fir.array<?x?xf64>>)
  %30 = fir.shape %21, %20 : (index, index) -> !fir.shape<2>
  %31:2 = hlfir.declare %arg5(%30) {uniq_name = "_QFFrun_benchmarkEx"} : (!fir.ref<!fir.array<?x?xf64>>, !fir.shape<2>) -> (!fir.box<!fir.array<?x?xf64>>, !fir.ref<!fir.array<?x?xf64>>)
  %32 = fir.shape %19, %18 : (index, index) -> !fir.shape<2>
  %33:2 = hlfir.declare %arg6(%32) {uniq_name = "_QFFrun_benchmarkEy"} : (!fir.ref<!fir.array<?x?xf64>>, !fir.shape<2>) -> (!fir.box<!fir.array<?x?xf64>>, !fir.ref<!fir.array<?x?xf64>>)
  %34:2 = hlfir.declare %arg7 {uniq_name = "_QFFrun_benchmarkEa"} : (!fir.ref<f64>) -> (!fir.ref<f64>, !fir.ref<f64>)
  %c1_i32 = arith.constant 1 : i32
  %35 = fir.convert %c1_i32 : (i32) -> index
  %c3_i32_9 = arith.constant 3 : i32
  %36 = fir.convert %c3_i32_9 : (i32) -> index
  %c1 = arith.constant 1 : index
  %37 = fir.convert %35 : (index) -> i32
  %38 = fir.do_loop %arg14 = %35 to %36 step %c1 iter_args(%arg15 = %37) -> (i32) {
    fir.store %arg15 to %24#0 : !fir.ref<i32>
    %c1_i32_10 = arith.constant 1 : i32
    %39 = fir.convert %c1_i32_10 : (i32) -> index
    %c4_i32_11 = arith.constant 4 : i32
    %40 = fir.convert %c4_i32_11 : (i32) -> index
    %c1_12 = arith.constant 1 : index
    %41 = fir.convert %39 : (index) -> i32
    %42 = fir.do_loop %arg16 = %39 to %40 step %c1_12 iter_args(%arg17 = %41) -> (i32) {
      fir.store %arg17 to %26#0 : !fir.ref<i32>
      %46 = fir.load %34#0 : !fir.ref<f64>
      %47 = fir.load %26#0 : !fir.ref<i32>
      %48 = fir.convert %47 : (i32) -> i64
      %49 = fir.load %24#0 : !fir.ref<i32>
      %50 = fir.convert %49 : (i32) -> i64
      %51 = hlfir.designate %31#0 (%48, %50)  : (!fir.box<!fir.array<?x?xf64>>, i64, i64) -> !fir.ref<f64>
      %52 = fir.load %51 : !fir.ref<f64>
      %53 = arith.mulf %46, %52 fastmath<contract> : f64
      %54 = fir.load %26#0 : !fir.ref<i32>
      %55 = fir.convert %54 : (i32) -> i64
      %56 = fir.load %24#0 : !fir.ref<i32>
      %57 = fir.convert %56 : (i32) -> i64
      %58 = hlfir.designate %33#0 (%55, %57)  : (!fir.box<!fir.array<?x?xf64>>, i64, i64) -> !fir.ref<f64>
      %59 = fir.load %58 : !fir.ref<f64>
      %60 = arith.addf %53, %59 fastmath<contract> : f64
      %61 = fir.load %26#0 : !fir.ref<i32>
      %62 = fir.convert %61 : (i32) -> i64
      %63 = fir.load %24#0 : !fir.ref<i32>
      %64 = fir.convert %63 : (i32) -> i64
      %65 = hlfir.designate %29#0 (%62, %64)  : (!fir.box<!fir.array<?x?xf64>>, i64, i64) -> !fir.ref<f64>
      hlfir.assign %60 to %65 : f64, !fir.ref<f64>
      %66 = fir.convert %c1_12 : (index) -> i32
      %67 = fir.load %26#0 : !fir.ref<i32>
      %68 = arith.addi %67, %66 overflow<nsw> : i32
      fir.result %68 : i32
    }
    fir.store %42 to %26#0 : !fir.ref<i32>
    %43 = fir.convert %c1 : (index) -> i32
    %44 = fir.load %24#0 : !fir.ref<i32>
    %45 = arith.addi %44, %43 overflow<nsw> : i32
    fir.result %45 : i32
  }
  fir.store %38 to %24#0 : !fir.ref<i32>
  omp.terminator
}

// -----// IR Dump After CanonicalizerPass (canonicalize) //----- //
module {
  func.func @kernel(%arg0: !fir.ref<i32> {jit.arg_type = 1 : ui32, jit.literal_val = 0 : i64}, %arg1: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 3 : i64}, %arg2: !fir.ref<i32> {jit.arg_type = 1 : ui32, jit.literal_val = 0 : i64}, %arg3: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 4 : i64}, %arg4: !fir.ref<!fir.array<?x?xf64>> {jit.arg_type = 1 : ui32}, %arg5: !fir.ref<!fir.array<?x?xf64>> {jit.arg_type = 1 : ui32}, %arg6: !fir.ref<!fir.array<?x?xf64>> {jit.arg_type = 1 : ui32}, %arg7: !fir.ref<f64> {jit.arg_type = 1 : ui32, jit.literal_val = 4619567317775286272 : i64}, %arg8: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 3 : i64}, %arg9: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 4 : i64}, %arg10: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 3 : i64}, %arg11: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 4 : i64}, %arg12: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 3 : i64}, %arg13: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 4 : i64}) {
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c3 = arith.constant 3 : index
    %0:2 = hlfir.declare %arg0 {uniq_name = "_QFFrun_benchmarkEj"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
    %1:2 = hlfir.declare %arg1 {uniq_name = "_QFFrun_benchmarkEm"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
    %2:2 = hlfir.declare %arg2 {uniq_name = "_QFFrun_benchmarkEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
    %3:2 = hlfir.declare %arg3 {uniq_name = "_QFFrun_benchmarkEn"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
    %4 = fir.shape %c4, %c3 : (index, index) -> !fir.shape<2>
    %5:2 = hlfir.declare %arg4(%4) {uniq_name = "_QFFrun_benchmarkEz"} : (!fir.ref<!fir.array<?x?xf64>>, !fir.shape<2>) -> (!fir.box<!fir.array<?x?xf64>>, !fir.ref<!fir.array<?x?xf64>>)
    %6 = fir.shape %c4, %c3 : (index, index) -> !fir.shape<2>
    %7:2 = hlfir.declare %arg5(%6) {uniq_name = "_QFFrun_benchmarkEx"} : (!fir.ref<!fir.array<?x?xf64>>, !fir.shape<2>) -> (!fir.box<!fir.array<?x?xf64>>, !fir.ref<!fir.array<?x?xf64>>)
    %8 = fir.shape %c4, %c3 : (index, index) -> !fir.shape<2>
    %9:2 = hlfir.declare %arg6(%8) {uniq_name = "_QFFrun_benchmarkEy"} : (!fir.ref<!fir.array<?x?xf64>>, !fir.shape<2>) -> (!fir.box<!fir.array<?x?xf64>>, !fir.ref<!fir.array<?x?xf64>>)
    %10:2 = hlfir.declare %arg7 {uniq_name = "_QFFrun_benchmarkEa"} : (!fir.ref<f64>) -> (!fir.ref<f64>, !fir.ref<f64>)
    %11 = fir.convert %c1 : (index) -> i32
    %12 = fir.do_loop %arg14 = %c1 to %c3 step %c1 iter_args(%arg15 = %11) -> (i32) {
      fir.store %arg15 to %0#0 : !fir.ref<i32>
      %13 = fir.convert %c1 : (index) -> i32
      %14 = fir.do_loop %arg16 = %c1 to %c4 step %c1 iter_args(%arg17 = %13) -> (i32) {
        fir.store %arg17 to %2#0 : !fir.ref<i32>
        %18 = fir.load %10#0 : !fir.ref<f64>
        %19 = fir.load %2#0 : !fir.ref<i32>
        %20 = fir.convert %19 : (i32) -> i64
        %21 = fir.load %0#0 : !fir.ref<i32>
        %22 = fir.convert %21 : (i32) -> i64
        %23 = hlfir.designate %7#0 (%20, %22)  : (!fir.box<!fir.array<?x?xf64>>, i64, i64) -> !fir.ref<f64>
        %24 = fir.load %23 : !fir.ref<f64>
        %25 = arith.mulf %18, %24 fastmath<contract> : f64
        %26 = fir.load %2#0 : !fir.ref<i32>
        %27 = fir.convert %26 : (i32) -> i64
        %28 = fir.load %0#0 : !fir.ref<i32>
        %29 = fir.convert %28 : (i32) -> i64
        %30 = hlfir.designate %9#0 (%27, %29)  : (!fir.box<!fir.array<?x?xf64>>, i64, i64) -> !fir.ref<f64>
        %31 = fir.load %30 : !fir.ref<f64>
        %32 = arith.addf %25, %31 fastmath<contract> : f64
        %33 = fir.load %2#0 : !fir.ref<i32>
        %34 = fir.convert %33 : (i32) -> i64
        %35 = fir.load %0#0 : !fir.ref<i32>
        %36 = fir.convert %35 : (i32) -> i64
        %37 = hlfir.designate %5#0 (%34, %36)  : (!fir.box<!fir.array<?x?xf64>>, i64, i64) -> !fir.ref<f64>
        hlfir.assign %32 to %37 : f64, !fir.ref<f64>
        %38 = fir.convert %c1 : (index) -> i32
        %39 = fir.load %2#0 : !fir.ref<i32>
        %40 = arith.addi %39, %38 overflow<nsw> : i32
        fir.result %40 : i32
      }
      fir.store %14 to %2#0 : !fir.ref<i32>
      %15 = fir.convert %c1 : (index) -> i32
      %16 = fir.load %0#0 : !fir.ref<i32>
      %17 = arith.addi %16, %15 overflow<nsw> : i32
      fir.result %17 : i32
    }
    fir.store %12 to %0#0 : !fir.ref<i32>
    omp.terminator
  }
}


// -----// IR Dump After SCCPPass (sccp) //----- //
module {
  func.func @kernel(%arg0: !fir.ref<i32> {jit.arg_type = 1 : ui32, jit.literal_val = 0 : i64}, %arg1: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 3 : i64}, %arg2: !fir.ref<i32> {jit.arg_type = 1 : ui32, jit.literal_val = 0 : i64}, %arg3: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 4 : i64}, %arg4: !fir.ref<!fir.array<?x?xf64>> {jit.arg_type = 1 : ui32}, %arg5: !fir.ref<!fir.array<?x?xf64>> {jit.arg_type = 1 : ui32}, %arg6: !fir.ref<!fir.array<?x?xf64>> {jit.arg_type = 1 : ui32}, %arg7: !fir.ref<f64> {jit.arg_type = 1 : ui32, jit.literal_val = 4619567317775286272 : i64}, %arg8: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 3 : i64}, %arg9: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 4 : i64}, %arg10: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 3 : i64}, %arg11: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 4 : i64}, %arg12: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 3 : i64}, %arg13: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 4 : i64}) {
    %c3 = arith.constant 3 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %0:2 = hlfir.declare %arg0 {uniq_name = "_QFFrun_benchmarkEj"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
    %1:2 = hlfir.declare %arg1 {uniq_name = "_QFFrun_benchmarkEm"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
    %2:2 = hlfir.declare %arg2 {uniq_name = "_QFFrun_benchmarkEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
    %3:2 = hlfir.declare %arg3 {uniq_name = "_QFFrun_benchmarkEn"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
    %4 = fir.shape %c4, %c3 : (index, index) -> !fir.shape<2>
    %5:2 = hlfir.declare %arg4(%4) {uniq_name = "_QFFrun_benchmarkEz"} : (!fir.ref<!fir.array<?x?xf64>>, !fir.shape<2>) -> (!fir.box<!fir.array<?x?xf64>>, !fir.ref<!fir.array<?x?xf64>>)
    %6 = fir.shape %c4, %c3 : (index, index) -> !fir.shape<2>
    %7:2 = hlfir.declare %arg5(%6) {uniq_name = "_QFFrun_benchmarkEx"} : (!fir.ref<!fir.array<?x?xf64>>, !fir.shape<2>) -> (!fir.box<!fir.array<?x?xf64>>, !fir.ref<!fir.array<?x?xf64>>)
    %8 = fir.shape %c4, %c3 : (index, index) -> !fir.shape<2>
    %9:2 = hlfir.declare %arg6(%8) {uniq_name = "_QFFrun_benchmarkEy"} : (!fir.ref<!fir.array<?x?xf64>>, !fir.shape<2>) -> (!fir.box<!fir.array<?x?xf64>>, !fir.ref<!fir.array<?x?xf64>>)
    %10:2 = hlfir.declare %arg7 {uniq_name = "_QFFrun_benchmarkEa"} : (!fir.ref<f64>) -> (!fir.ref<f64>, !fir.ref<f64>)
    %11 = fir.convert %c1 : (index) -> i32
    %12 = fir.do_loop %arg14 = %c1 to %c3 step %c1 iter_args(%arg15 = %11) -> (i32) {
      fir.store %arg15 to %0#0 : !fir.ref<i32>
      %13 = fir.convert %c1 : (index) -> i32
      %14 = fir.do_loop %arg16 = %c1 to %c4 step %c1 iter_args(%arg17 = %13) -> (i32) {
        fir.store %arg17 to %2#0 : !fir.ref<i32>
        %18 = fir.load %10#0 : !fir.ref<f64>
        %19 = fir.load %2#0 : !fir.ref<i32>
        %20 = fir.convert %19 : (i32) -> i64
        %21 = fir.load %0#0 : !fir.ref<i32>
        %22 = fir.convert %21 : (i32) -> i64
        %23 = hlfir.designate %7#0 (%20, %22)  : (!fir.box<!fir.array<?x?xf64>>, i64, i64) -> !fir.ref<f64>
        %24 = fir.load %23 : !fir.ref<f64>
        %25 = arith.mulf %18, %24 fastmath<contract> : f64
        %26 = fir.load %2#0 : !fir.ref<i32>
        %27 = fir.convert %26 : (i32) -> i64
        %28 = fir.load %0#0 : !fir.ref<i32>
        %29 = fir.convert %28 : (i32) -> i64
        %30 = hlfir.designate %9#0 (%27, %29)  : (!fir.box<!fir.array<?x?xf64>>, i64, i64) -> !fir.ref<f64>
        %31 = fir.load %30 : !fir.ref<f64>
        %32 = arith.addf %25, %31 fastmath<contract> : f64
        %33 = fir.load %2#0 : !fir.ref<i32>
        %34 = fir.convert %33 : (i32) -> i64
        %35 = fir.load %0#0 : !fir.ref<i32>
        %36 = fir.convert %35 : (i32) -> i64
        %37 = hlfir.designate %5#0 (%34, %36)  : (!fir.box<!fir.array<?x?xf64>>, i64, i64) -> !fir.ref<f64>
        hlfir.assign %32 to %37 : f64, !fir.ref<f64>
        %38 = fir.convert %c1 : (index) -> i32
        %39 = fir.load %2#0 : !fir.ref<i32>
        %40 = arith.addi %39, %38 overflow<nsw> : i32
        fir.result %40 : i32
      }
      fir.store %14 to %2#0 : !fir.ref<i32>
      %15 = fir.convert %c1 : (index) -> i32
      %16 = fir.load %0#0 : !fir.ref<i32>
      %17 = arith.addi %16, %15 overflow<nsw> : i32
      fir.result %17 : i32
    }
    fir.store %12 to %0#0 : !fir.ref<i32>
    omp.terminator
  }
}


// -----// IR Dump After CSEPass (cse) //----- //
module {
  func.func @kernel(%arg0: !fir.ref<i32> {jit.arg_type = 1 : ui32, jit.literal_val = 0 : i64}, %arg1: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 3 : i64}, %arg2: !fir.ref<i32> {jit.arg_type = 1 : ui32, jit.literal_val = 0 : i64}, %arg3: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 4 : i64}, %arg4: !fir.ref<!fir.array<?x?xf64>> {jit.arg_type = 1 : ui32}, %arg5: !fir.ref<!fir.array<?x?xf64>> {jit.arg_type = 1 : ui32}, %arg6: !fir.ref<!fir.array<?x?xf64>> {jit.arg_type = 1 : ui32}, %arg7: !fir.ref<f64> {jit.arg_type = 1 : ui32, jit.literal_val = 4619567317775286272 : i64}, %arg8: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 3 : i64}, %arg9: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 4 : i64}, %arg10: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 3 : i64}, %arg11: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 4 : i64}, %arg12: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 3 : i64}, %arg13: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 4 : i64}) {
    %c3 = arith.constant 3 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %0:2 = hlfir.declare %arg0 {uniq_name = "_QFFrun_benchmarkEj"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
    %1:2 = hlfir.declare %arg1 {uniq_name = "_QFFrun_benchmarkEm"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
    %2:2 = hlfir.declare %arg2 {uniq_name = "_QFFrun_benchmarkEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
    %3:2 = hlfir.declare %arg3 {uniq_name = "_QFFrun_benchmarkEn"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
    %4 = fir.shape %c4, %c3 : (index, index) -> !fir.shape<2>
    %5:2 = hlfir.declare %arg4(%4) {uniq_name = "_QFFrun_benchmarkEz"} : (!fir.ref<!fir.array<?x?xf64>>, !fir.shape<2>) -> (!fir.box<!fir.array<?x?xf64>>, !fir.ref<!fir.array<?x?xf64>>)
    %6:2 = hlfir.declare %arg5(%4) {uniq_name = "_QFFrun_benchmarkEx"} : (!fir.ref<!fir.array<?x?xf64>>, !fir.shape<2>) -> (!fir.box<!fir.array<?x?xf64>>, !fir.ref<!fir.array<?x?xf64>>)
    %7:2 = hlfir.declare %arg6(%4) {uniq_name = "_QFFrun_benchmarkEy"} : (!fir.ref<!fir.array<?x?xf64>>, !fir.shape<2>) -> (!fir.box<!fir.array<?x?xf64>>, !fir.ref<!fir.array<?x?xf64>>)
    %8:2 = hlfir.declare %arg7 {uniq_name = "_QFFrun_benchmarkEa"} : (!fir.ref<f64>) -> (!fir.ref<f64>, !fir.ref<f64>)
    %9 = fir.convert %c1 : (index) -> i32
    %10 = fir.do_loop %arg14 = %c1 to %c3 step %c1 iter_args(%arg15 = %9) -> (i32) {
      fir.store %arg15 to %0#0 : !fir.ref<i32>
      %11 = fir.do_loop %arg16 = %c1 to %c4 step %c1 iter_args(%arg17 = %9) -> (i32) {
        fir.store %arg17 to %2#0 : !fir.ref<i32>
        %14 = fir.load %8#0 : !fir.ref<f64>
        %15 = fir.load %2#0 : !fir.ref<i32>
        %16 = fir.convert %15 : (i32) -> i64
        %17 = fir.load %0#0 : !fir.ref<i32>
        %18 = fir.convert %17 : (i32) -> i64
        %19 = hlfir.designate %6#0 (%16, %18)  : (!fir.box<!fir.array<?x?xf64>>, i64, i64) -> !fir.ref<f64>
        %20 = fir.load %19 : !fir.ref<f64>
        %21 = arith.mulf %14, %20 fastmath<contract> : f64
        %22 = hlfir.designate %7#0 (%16, %18)  : (!fir.box<!fir.array<?x?xf64>>, i64, i64) -> !fir.ref<f64>
        %23 = fir.load %22 : !fir.ref<f64>
        %24 = arith.addf %21, %23 fastmath<contract> : f64
        %25 = hlfir.designate %5#0 (%16, %18)  : (!fir.box<!fir.array<?x?xf64>>, i64, i64) -> !fir.ref<f64>
        hlfir.assign %24 to %25 : f64, !fir.ref<f64>
        %26 = fir.load %2#0 : !fir.ref<i32>
        %27 = arith.addi %26, %9 overflow<nsw> : i32
        fir.result %27 : i32
      }
      fir.store %11 to %2#0 : !fir.ref<i32>
      %12 = fir.load %0#0 : !fir.ref<i32>
      %13 = arith.addi %12, %9 overflow<nsw> : i32
      fir.result %13 : i32
    }
    fir.store %10 to %0#0 : !fir.ref<i32>
    omp.terminator
  }
}


// -----// IR Dump After CanonicalizerPass (canonicalize) //----- //
module {
  func.func @kernel(%arg0: !fir.ref<i32> {jit.arg_type = 1 : ui32, jit.literal_val = 0 : i64}, %arg1: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 3 : i64}, %arg2: !fir.ref<i32> {jit.arg_type = 1 : ui32, jit.literal_val = 0 : i64}, %arg3: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 4 : i64}, %arg4: !fir.ref<!fir.array<?x?xf64>> {jit.arg_type = 1 : ui32}, %arg5: !fir.ref<!fir.array<?x?xf64>> {jit.arg_type = 1 : ui32}, %arg6: !fir.ref<!fir.array<?x?xf64>> {jit.arg_type = 1 : ui32}, %arg7: !fir.ref<f64> {jit.arg_type = 1 : ui32, jit.literal_val = 4619567317775286272 : i64}, %arg8: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 3 : i64}, %arg9: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 4 : i64}, %arg10: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 3 : i64}, %arg11: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 4 : i64}, %arg12: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 3 : i64}, %arg13: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 4 : i64}) {
    %c3 = arith.constant 3 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %0:2 = hlfir.declare %arg0 {uniq_name = "_QFFrun_benchmarkEj"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
    %1:2 = hlfir.declare %arg1 {uniq_name = "_QFFrun_benchmarkEm"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
    %2:2 = hlfir.declare %arg2 {uniq_name = "_QFFrun_benchmarkEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
    %3:2 = hlfir.declare %arg3 {uniq_name = "_QFFrun_benchmarkEn"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
    %4 = fir.shape %c4, %c3 : (index, index) -> !fir.shape<2>
    %5:2 = hlfir.declare %arg4(%4) {uniq_name = "_QFFrun_benchmarkEz"} : (!fir.ref<!fir.array<?x?xf64>>, !fir.shape<2>) -> (!fir.box<!fir.array<?x?xf64>>, !fir.ref<!fir.array<?x?xf64>>)
    %6:2 = hlfir.declare %arg5(%4) {uniq_name = "_QFFrun_benchmarkEx"} : (!fir.ref<!fir.array<?x?xf64>>, !fir.shape<2>) -> (!fir.box<!fir.array<?x?xf64>>, !fir.ref<!fir.array<?x?xf64>>)
    %7:2 = hlfir.declare %arg6(%4) {uniq_name = "_QFFrun_benchmarkEy"} : (!fir.ref<!fir.array<?x?xf64>>, !fir.shape<2>) -> (!fir.box<!fir.array<?x?xf64>>, !fir.ref<!fir.array<?x?xf64>>)
    %8:2 = hlfir.declare %arg7 {uniq_name = "_QFFrun_benchmarkEa"} : (!fir.ref<f64>) -> (!fir.ref<f64>, !fir.ref<f64>)
    %9 = fir.convert %c1 : (index) -> i32
    %10 = fir.do_loop %arg14 = %c1 to %c3 step %c1 iter_args(%arg15 = %9) -> (i32) {
      fir.store %arg15 to %0#0 : !fir.ref<i32>
      %11 = fir.do_loop %arg16 = %c1 to %c4 step %c1 iter_args(%arg17 = %9) -> (i32) {
        fir.store %arg17 to %2#0 : !fir.ref<i32>
        %14 = fir.load %8#0 : !fir.ref<f64>
        %15 = fir.load %2#0 : !fir.ref<i32>
        %16 = fir.convert %15 : (i32) -> i64
        %17 = fir.load %0#0 : !fir.ref<i32>
        %18 = fir.convert %17 : (i32) -> i64
        %19 = hlfir.designate %6#0 (%16, %18)  : (!fir.box<!fir.array<?x?xf64>>, i64, i64) -> !fir.ref<f64>
        %20 = fir.load %19 : !fir.ref<f64>
        %21 = arith.mulf %14, %20 fastmath<contract> : f64
        %22 = hlfir.designate %7#0 (%16, %18)  : (!fir.box<!fir.array<?x?xf64>>, i64, i64) -> !fir.ref<f64>
        %23 = fir.load %22 : !fir.ref<f64>
        %24 = arith.addf %21, %23 fastmath<contract> : f64
        %25 = hlfir.designate %5#0 (%16, %18)  : (!fir.box<!fir.array<?x?xf64>>, i64, i64) -> !fir.ref<f64>
        hlfir.assign %24 to %25 : f64, !fir.ref<f64>
        %26 = fir.load %2#0 : !fir.ref<i32>
        %27 = arith.addi %26, %9 overflow<nsw> : i32
        fir.result %27 : i32
      }
      fir.store %11 to %2#0 : !fir.ref<i32>
      %12 = fir.load %0#0 : !fir.ref<i32>
      %13 = arith.addi %12, %9 overflow<nsw> : i32
      fir.result %13 : i32
    }
    fir.store %10 to %0#0 : !fir.ref<i32>
    omp.terminator
  }
}


// -----// IR Dump After {anonymous}::ShapeInferPass (jforce-shape-infer) //----- //
func.func @kernel(%arg0: !fir.ref<i32> {jit.arg_type = 1 : ui32, jit.literal_val = 0 : i64}, %arg1: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 3 : i64}, %arg2: !fir.ref<i32> {jit.arg_type = 1 : ui32, jit.literal_val = 0 : i64}, %arg3: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 4 : i64}, %arg4: !fir.ref<!fir.array<4x3xf64>> {jit.arg_type = 1 : ui32}, %arg5: !fir.ref<!fir.array<4x3xf64>> {jit.arg_type = 1 : ui32}, %arg6: !fir.ref<!fir.array<4x3xf64>> {jit.arg_type = 1 : ui32}, %arg7: !fir.ref<f64> {jit.arg_type = 1 : ui32, jit.literal_val = 4619567317775286272 : i64}, %arg8: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 3 : i64}, %arg9: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 4 : i64}, %arg10: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 3 : i64}, %arg11: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 4 : i64}, %arg12: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 3 : i64}, %arg13: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 4 : i64}) {
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %c1 = arith.constant 1 : index
  %0:2 = hlfir.declare %arg0 {uniq_name = "_QFFrun_benchmarkEj"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %1:2 = hlfir.declare %arg1 {uniq_name = "_QFFrun_benchmarkEm"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %2:2 = hlfir.declare %arg2 {uniq_name = "_QFFrun_benchmarkEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %3:2 = hlfir.declare %arg3 {uniq_name = "_QFFrun_benchmarkEn"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %4 = fir.shape %c4, %c3 : (index, index) -> !fir.shape<2>
  %5:2 = hlfir.declare %arg4(%4) {uniq_name = "_QFFrun_benchmarkEz"} : (!fir.ref<!fir.array<4x3xf64>>, !fir.shape<2>) -> (!fir.ref<!fir.array<4x3xf64>>, !fir.ref<!fir.array<4x3xf64>>)
  %6:2 = hlfir.declare %arg5(%4) {uniq_name = "_QFFrun_benchmarkEx"} : (!fir.ref<!fir.array<4x3xf64>>, !fir.shape<2>) -> (!fir.ref<!fir.array<4x3xf64>>, !fir.ref<!fir.array<4x3xf64>>)
  %7:2 = hlfir.declare %arg6(%4) {uniq_name = "_QFFrun_benchmarkEy"} : (!fir.ref<!fir.array<4x3xf64>>, !fir.shape<2>) -> (!fir.ref<!fir.array<4x3xf64>>, !fir.ref<!fir.array<4x3xf64>>)
  %8:2 = hlfir.declare %arg7 {uniq_name = "_QFFrun_benchmarkEa"} : (!fir.ref<f64>) -> (!fir.ref<f64>, !fir.ref<f64>)
  %9 = fir.convert %c1 : (index) -> i32
  %10 = fir.do_loop %arg14 = %c1 to %c3 step %c1 iter_args(%arg15 = %9) -> (i32) {
    fir.store %arg15 to %0#0 : !fir.ref<i32>
    %11 = fir.do_loop %arg16 = %c1 to %c4 step %c1 iter_args(%arg17 = %9) -> (i32) {
      fir.store %arg17 to %2#0 : !fir.ref<i32>
      %14 = fir.load %8#0 : !fir.ref<f64>
      %15 = fir.load %2#0 : !fir.ref<i32>
      %16 = fir.convert %15 : (i32) -> i64
      %17 = fir.load %0#0 : !fir.ref<i32>
      %18 = fir.convert %17 : (i32) -> i64
      %19 = hlfir.designate %6#0 (%16, %18)  : (!fir.ref<!fir.array<4x3xf64>>, i64, i64) -> !fir.ref<f64>
      %20 = fir.load %19 : !fir.ref<f64>
      %21 = arith.mulf %14, %20 fastmath<contract> : f64
      %22 = hlfir.designate %7#0 (%16, %18)  : (!fir.ref<!fir.array<4x3xf64>>, i64, i64) -> !fir.ref<f64>
      %23 = fir.load %22 : !fir.ref<f64>
      %24 = arith.addf %21, %23 fastmath<contract> : f64
      %25 = hlfir.designate %5#0 (%16, %18)  : (!fir.ref<!fir.array<4x3xf64>>, i64, i64) -> !fir.ref<f64>
      hlfir.assign %24 to %25 : f64, !fir.ref<f64>
      %26 = fir.load %2#0 : !fir.ref<i32>
      %27 = arith.addi %26, %9 overflow<nsw> : i32
      fir.result %27 : i32
    }
    fir.store %11 to %2#0 : !fir.ref<i32>
    %12 = fir.load %0#0 : !fir.ref<i32>
    %13 = arith.addi %12, %9 overflow<nsw> : i32
    fir.result %13 : i32
  }
  fir.store %10 to %0#0 : !fir.ref<i32>
  omp.terminator
}

// -----// IR Dump After CanonicalizerPass (canonicalize) //----- //
module {
  func.func @kernel(%arg0: !fir.ref<i32> {jit.arg_type = 1 : ui32, jit.literal_val = 0 : i64}, %arg1: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 3 : i64}, %arg2: !fir.ref<i32> {jit.arg_type = 1 : ui32, jit.literal_val = 0 : i64}, %arg3: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 4 : i64}, %arg4: !fir.ref<!fir.array<4x3xf64>> {jit.arg_type = 1 : ui32}, %arg5: !fir.ref<!fir.array<4x3xf64>> {jit.arg_type = 1 : ui32}, %arg6: !fir.ref<!fir.array<4x3xf64>> {jit.arg_type = 1 : ui32}, %arg7: !fir.ref<f64> {jit.arg_type = 1 : ui32, jit.literal_val = 4619567317775286272 : i64}, %arg8: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 3 : i64}, %arg9: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 4 : i64}, %arg10: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 3 : i64}, %arg11: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 4 : i64}, %arg12: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 3 : i64}, %arg13: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 4 : i64}) {
    %c3 = arith.constant 3 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %0:2 = hlfir.declare %arg0 {uniq_name = "_QFFrun_benchmarkEj"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
    %1:2 = hlfir.declare %arg1 {uniq_name = "_QFFrun_benchmarkEm"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
    %2:2 = hlfir.declare %arg2 {uniq_name = "_QFFrun_benchmarkEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
    %3:2 = hlfir.declare %arg3 {uniq_name = "_QFFrun_benchmarkEn"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
    %4 = fir.shape %c4, %c3 : (index, index) -> !fir.shape<2>
    %5:2 = hlfir.declare %arg4(%4) {uniq_name = "_QFFrun_benchmarkEz"} : (!fir.ref<!fir.array<4x3xf64>>, !fir.shape<2>) -> (!fir.ref<!fir.array<4x3xf64>>, !fir.ref<!fir.array<4x3xf64>>)
    %6:2 = hlfir.declare %arg5(%4) {uniq_name = "_QFFrun_benchmarkEx"} : (!fir.ref<!fir.array<4x3xf64>>, !fir.shape<2>) -> (!fir.ref<!fir.array<4x3xf64>>, !fir.ref<!fir.array<4x3xf64>>)
    %7:2 = hlfir.declare %arg6(%4) {uniq_name = "_QFFrun_benchmarkEy"} : (!fir.ref<!fir.array<4x3xf64>>, !fir.shape<2>) -> (!fir.ref<!fir.array<4x3xf64>>, !fir.ref<!fir.array<4x3xf64>>)
    %8:2 = hlfir.declare %arg7 {uniq_name = "_QFFrun_benchmarkEa"} : (!fir.ref<f64>) -> (!fir.ref<f64>, !fir.ref<f64>)
    %9 = fir.convert %c1 : (index) -> i32
    %10 = fir.do_loop %arg14 = %c1 to %c3 step %c1 iter_args(%arg15 = %9) -> (i32) {
      fir.store %arg15 to %0#0 : !fir.ref<i32>
      %11 = fir.do_loop %arg16 = %c1 to %c4 step %c1 iter_args(%arg17 = %9) -> (i32) {
        fir.store %arg17 to %2#0 : !fir.ref<i32>
        %14 = fir.load %8#0 : !fir.ref<f64>
        %15 = fir.load %2#0 : !fir.ref<i32>
        %16 = fir.convert %15 : (i32) -> i64
        %17 = fir.load %0#0 : !fir.ref<i32>
        %18 = fir.convert %17 : (i32) -> i64
        %19 = hlfir.designate %6#0 (%16, %18)  : (!fir.ref<!fir.array<4x3xf64>>, i64, i64) -> !fir.ref<f64>
        %20 = fir.load %19 : !fir.ref<f64>
        %21 = arith.mulf %14, %20 fastmath<contract> : f64
        %22 = hlfir.designate %7#0 (%16, %18)  : (!fir.ref<!fir.array<4x3xf64>>, i64, i64) -> !fir.ref<f64>
        %23 = fir.load %22 : !fir.ref<f64>
        %24 = arith.addf %21, %23 fastmath<contract> : f64
        %25 = hlfir.designate %5#0 (%16, %18)  : (!fir.ref<!fir.array<4x3xf64>>, i64, i64) -> !fir.ref<f64>
        hlfir.assign %24 to %25 : f64, !fir.ref<f64>
        %26 = fir.load %2#0 : !fir.ref<i32>
        %27 = arith.addi %26, %9 overflow<nsw> : i32
        fir.result %27 : i32
      }
      fir.store %11 to %2#0 : !fir.ref<i32>
      %12 = fir.load %0#0 : !fir.ref<i32>
      %13 = arith.addi %12, %9 overflow<nsw> : i32
      fir.result %13 : i32
    }
    fir.store %10 to %0#0 : !fir.ref<i32>
    omp.terminator
  }
}


// -----// IR Dump After ConvertHLFIRtoFIR (convert-hlfir-to-fir) //----- //
module {
  func.func @kernel(%arg0: !fir.ref<i32> {jit.arg_type = 1 : ui32, jit.literal_val = 0 : i64}, %arg1: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 3 : i64}, %arg2: !fir.ref<i32> {jit.arg_type = 1 : ui32, jit.literal_val = 0 : i64}, %arg3: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 4 : i64}, %arg4: !fir.ref<!fir.array<4x3xf64>> {jit.arg_type = 1 : ui32}, %arg5: !fir.ref<!fir.array<4x3xf64>> {jit.arg_type = 1 : ui32}, %arg6: !fir.ref<!fir.array<4x3xf64>> {jit.arg_type = 1 : ui32}, %arg7: !fir.ref<f64> {jit.arg_type = 1 : ui32, jit.literal_val = 4619567317775286272 : i64}, %arg8: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 3 : i64}, %arg9: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 4 : i64}, %arg10: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 3 : i64}, %arg11: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 4 : i64}, %arg12: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 3 : i64}, %arg13: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 4 : i64}) {
    %c3 = arith.constant 3 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %0 = fir.declare %arg0 {uniq_name = "_QFFrun_benchmarkEj"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %1 = fir.declare %arg1 {uniq_name = "_QFFrun_benchmarkEm"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %2 = fir.declare %arg2 {uniq_name = "_QFFrun_benchmarkEi"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %3 = fir.declare %arg3 {uniq_name = "_QFFrun_benchmarkEn"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %4 = fir.shape %c4, %c3 : (index, index) -> !fir.shape<2>
    %5 = fir.declare %arg4(%4) {uniq_name = "_QFFrun_benchmarkEz"} : (!fir.ref<!fir.array<4x3xf64>>, !fir.shape<2>) -> !fir.ref<!fir.array<4x3xf64>>
    %6 = fir.declare %arg5(%4) {uniq_name = "_QFFrun_benchmarkEx"} : (!fir.ref<!fir.array<4x3xf64>>, !fir.shape<2>) -> !fir.ref<!fir.array<4x3xf64>>
    %7 = fir.declare %arg6(%4) {uniq_name = "_QFFrun_benchmarkEy"} : (!fir.ref<!fir.array<4x3xf64>>, !fir.shape<2>) -> !fir.ref<!fir.array<4x3xf64>>
    %8 = fir.declare %arg7 {uniq_name = "_QFFrun_benchmarkEa"} : (!fir.ref<f64>) -> !fir.ref<f64>
    %9 = fir.convert %c1 : (index) -> i32
    %10 = fir.do_loop %arg14 = %c1 to %c3 step %c1 iter_args(%arg15 = %9) -> (i32) {
      fir.store %arg15 to %0 : !fir.ref<i32>
      %11 = fir.do_loop %arg16 = %c1 to %c4 step %c1 iter_args(%arg17 = %9) -> (i32) {
        fir.store %arg17 to %2 : !fir.ref<i32>
        %14 = fir.load %8 : !fir.ref<f64>
        %15 = fir.load %2 : !fir.ref<i32>
        %16 = fir.convert %15 : (i32) -> i64
        %17 = fir.load %0 : !fir.ref<i32>
        %18 = fir.convert %17 : (i32) -> i64
        %19 = fir.array_coor %6(%4) %16, %18 : (!fir.ref<!fir.array<4x3xf64>>, !fir.shape<2>, i64, i64) -> !fir.ref<f64>
        %20 = fir.load %19 : !fir.ref<f64>
        %21 = arith.mulf %14, %20 fastmath<contract> : f64
        %22 = fir.array_coor %7(%4) %16, %18 : (!fir.ref<!fir.array<4x3xf64>>, !fir.shape<2>, i64, i64) -> !fir.ref<f64>
        %23 = fir.load %22 : !fir.ref<f64>
        %24 = arith.addf %21, %23 fastmath<contract> : f64
        %25 = fir.array_coor %5(%4) %16, %18 : (!fir.ref<!fir.array<4x3xf64>>, !fir.shape<2>, i64, i64) -> !fir.ref<f64>
        fir.store %24 to %25 : !fir.ref<f64>
        %26 = fir.load %2 : !fir.ref<i32>
        %27 = arith.addi %26, %9 overflow<nsw> : i32
        fir.result %27 : i32
      }
      fir.store %11 to %2 : !fir.ref<i32>
      %12 = fir.load %0 : !fir.ref<i32>
      %13 = arith.addi %12, %9 overflow<nsw> : i32
      fir.result %13 : i32
    }
    fir.store %10 to %0 : !fir.ref<i32>
    omp.terminator
  }
}


// -----// IR Dump After FIRToMemRef (fir-to-memref) //----- //
func.func @kernel(%arg0: !fir.ref<i32> {jit.arg_type = 1 : ui32, jit.literal_val = 0 : i64}, %arg1: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 3 : i64}, %arg2: !fir.ref<i32> {jit.arg_type = 1 : ui32, jit.literal_val = 0 : i64}, %arg3: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 4 : i64}, %arg4: !fir.ref<!fir.array<4x3xf64>> {jit.arg_type = 1 : ui32}, %arg5: !fir.ref<!fir.array<4x3xf64>> {jit.arg_type = 1 : ui32}, %arg6: !fir.ref<!fir.array<4x3xf64>> {jit.arg_type = 1 : ui32}, %arg7: !fir.ref<f64> {jit.arg_type = 1 : ui32, jit.literal_val = 4619567317775286272 : i64}, %arg8: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 3 : i64}, %arg9: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 4 : i64}, %arg10: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 3 : i64}, %arg11: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 4 : i64}, %arg12: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 3 : i64}, %arg13: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 4 : i64}) {
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %c1 = arith.constant 1 : index
  %0 = fir.declare %arg0 {uniq_name = "_QFFrun_benchmarkEj"} : (!fir.ref<i32>) -> !fir.ref<i32>
  %1 = fir.declare %arg1 {uniq_name = "_QFFrun_benchmarkEm"} : (!fir.ref<i32>) -> !fir.ref<i32>
  %2 = fir.declare %arg2 {uniq_name = "_QFFrun_benchmarkEi"} : (!fir.ref<i32>) -> !fir.ref<i32>
  %3 = fir.declare %arg3 {uniq_name = "_QFFrun_benchmarkEn"} : (!fir.ref<i32>) -> !fir.ref<i32>
  %4 = fir.shape %c4, %c3 : (index, index) -> !fir.shape<2>
  %5 = fir.declare %arg4(%4) {uniq_name = "_QFFrun_benchmarkEz"} : (!fir.ref<!fir.array<4x3xf64>>, !fir.shape<2>) -> !fir.ref<!fir.array<4x3xf64>>
  %6 = fir.declare %arg5(%4) {uniq_name = "_QFFrun_benchmarkEx"} : (!fir.ref<!fir.array<4x3xf64>>, !fir.shape<2>) -> !fir.ref<!fir.array<4x3xf64>>
  %7 = fir.declare %arg6(%4) {uniq_name = "_QFFrun_benchmarkEy"} : (!fir.ref<!fir.array<4x3xf64>>, !fir.shape<2>) -> !fir.ref<!fir.array<4x3xf64>>
  %8 = fir.declare %arg7 {uniq_name = "_QFFrun_benchmarkEa"} : (!fir.ref<f64>) -> !fir.ref<f64>
  %9 = fir.convert %c1 : (index) -> i32
  %10 = fir.do_loop %arg14 = %c1 to %c3 step %c1 iter_args(%arg15 = %9) -> (i32) {
    %12 = fir.convert %0 : (!fir.ref<i32>) -> memref<i32>
    memref.store %arg15, %12[] : memref<i32>
    %13 = fir.do_loop %arg16 = %c1 to %c4 step %c1 iter_args(%arg17 = %9) -> (i32) {
      %18 = fir.convert %2 : (!fir.ref<i32>) -> memref<i32>
      memref.store %arg17, %18[] : memref<i32>
      %19 = fir.convert %8 : (!fir.ref<f64>) -> memref<f64>
      %20 = memref.load %19[] : memref<f64>
      %21 = fir.convert %2 : (!fir.ref<i32>) -> memref<i32>
      %22 = memref.load %21[] : memref<i32>
      %23 = fir.convert %22 : (i32) -> i64
      %24 = fir.convert %0 : (!fir.ref<i32>) -> memref<i32>
      %25 = memref.load %24[] : memref<i32>
      %26 = fir.convert %25 : (i32) -> i64
      %27 = fir.convert %6 : (!fir.ref<!fir.array<4x3xf64>>) -> memref<3x4xf64>
      %c1_0 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %28 = arith.index_cast %23 : i64 to index
      %29 = arith.subi %28, %c1_0 : index
      %30 = arith.muli %29, %c1_0 : index
      %31 = arith.subi %c1_0, %c1_0 : index
      %32 = arith.addi %30, %31 : index
      %33 = arith.index_cast %26 : i64 to index
      %34 = arith.subi %33, %c1_0 : index
      %35 = arith.muli %34, %c1_0 : index
      %36 = arith.subi %c1_0, %c1_0 : index
      %37 = arith.addi %35, %36 : index
      %38 = memref.load %27[%37, %32] : memref<3x4xf64>
      %39 = arith.mulf %20, %38 fastmath<contract> : f64
      %40 = fir.convert %7 : (!fir.ref<!fir.array<4x3xf64>>) -> memref<3x4xf64>
      %c1_1 = arith.constant 1 : index
      %c0_2 = arith.constant 0 : index
      %41 = arith.index_cast %23 : i64 to index
      %42 = arith.subi %41, %c1_1 : index
      %43 = arith.muli %42, %c1_1 : index
      %44 = arith.subi %c1_1, %c1_1 : index
      %45 = arith.addi %43, %44 : index
      %46 = arith.index_cast %26 : i64 to index
      %47 = arith.subi %46, %c1_1 : index
      %48 = arith.muli %47, %c1_1 : index
      %49 = arith.subi %c1_1, %c1_1 : index
      %50 = arith.addi %48, %49 : index
      %51 = memref.load %40[%50, %45] : memref<3x4xf64>
      %52 = arith.addf %39, %51 fastmath<contract> : f64
      %53 = fir.convert %5 : (!fir.ref<!fir.array<4x3xf64>>) -> memref<3x4xf64>
      %c1_3 = arith.constant 1 : index
      %c0_4 = arith.constant 0 : index
      %54 = arith.index_cast %23 : i64 to index
      %55 = arith.subi %54, %c1_3 : index
      %56 = arith.muli %55, %c1_3 : index
      %57 = arith.subi %c1_3, %c1_3 : index
      %58 = arith.addi %56, %57 : index
      %59 = arith.index_cast %26 : i64 to index
      %60 = arith.subi %59, %c1_3 : index
      %61 = arith.muli %60, %c1_3 : index
      %62 = arith.subi %c1_3, %c1_3 : index
      %63 = arith.addi %61, %62 : index
      memref.store %52, %53[%63, %58] : memref<3x4xf64>
      %64 = fir.convert %2 : (!fir.ref<i32>) -> memref<i32>
      %65 = memref.load %64[] : memref<i32>
      %66 = arith.addi %65, %9 overflow<nsw> : i32
      fir.result %66 : i32
    }
    %14 = fir.convert %2 : (!fir.ref<i32>) -> memref<i32>
    memref.store %13, %14[] : memref<i32>
    %15 = fir.convert %0 : (!fir.ref<i32>) -> memref<i32>
    %16 = memref.load %15[] : memref<i32>
    %17 = arith.addi %16, %9 overflow<nsw> : i32
    fir.result %17 : i32
  }
  %11 = fir.convert %0 : (!fir.ref<i32>) -> memref<i32>
  memref.store %10, %11[] : memref<i32>
  omp.terminator
}

// -----// IR Dump After CanonicalizerPass (canonicalize) //----- //
module {
  func.func @kernel(%arg0: !fir.ref<i32> {jit.arg_type = 1 : ui32, jit.literal_val = 0 : i64}, %arg1: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 3 : i64}, %arg2: !fir.ref<i32> {jit.arg_type = 1 : ui32, jit.literal_val = 0 : i64}, %arg3: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 4 : i64}, %arg4: !fir.ref<!fir.array<4x3xf64>> {jit.arg_type = 1 : ui32}, %arg5: !fir.ref<!fir.array<4x3xf64>> {jit.arg_type = 1 : ui32}, %arg6: !fir.ref<!fir.array<4x3xf64>> {jit.arg_type = 1 : ui32}, %arg7: !fir.ref<f64> {jit.arg_type = 1 : ui32, jit.literal_val = 4619567317775286272 : i64}, %arg8: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 3 : i64}, %arg9: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 4 : i64}, %arg10: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 3 : i64}, %arg11: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 4 : i64}, %arg12: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 3 : i64}, %arg13: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 4 : i64}) {
    %c3 = arith.constant 3 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %0 = fir.declare %arg0 {uniq_name = "_QFFrun_benchmarkEj"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %1 = fir.declare %arg1 {uniq_name = "_QFFrun_benchmarkEm"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %2 = fir.declare %arg2 {uniq_name = "_QFFrun_benchmarkEi"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %3 = fir.declare %arg3 {uniq_name = "_QFFrun_benchmarkEn"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %4 = fir.shape %c4, %c3 : (index, index) -> !fir.shape<2>
    %5 = fir.declare %arg4(%4) {uniq_name = "_QFFrun_benchmarkEz"} : (!fir.ref<!fir.array<4x3xf64>>, !fir.shape<2>) -> !fir.ref<!fir.array<4x3xf64>>
    %6 = fir.declare %arg5(%4) {uniq_name = "_QFFrun_benchmarkEx"} : (!fir.ref<!fir.array<4x3xf64>>, !fir.shape<2>) -> !fir.ref<!fir.array<4x3xf64>>
    %7 = fir.declare %arg6(%4) {uniq_name = "_QFFrun_benchmarkEy"} : (!fir.ref<!fir.array<4x3xf64>>, !fir.shape<2>) -> !fir.ref<!fir.array<4x3xf64>>
    %8 = fir.declare %arg7 {uniq_name = "_QFFrun_benchmarkEa"} : (!fir.ref<f64>) -> !fir.ref<f64>
    %9 = fir.convert %c1 : (index) -> i32
    %10 = fir.do_loop %arg14 = %c1 to %c3 step %c1 iter_args(%arg15 = %9) -> (i32) {
      %12 = fir.convert %0 : (!fir.ref<i32>) -> memref<i32>
      memref.store %arg15, %12[] : memref<i32>
      %13 = fir.do_loop %arg16 = %c1 to %c4 step %c1 iter_args(%arg17 = %9) -> (i32) {
        %18 = fir.convert %2 : (!fir.ref<i32>) -> memref<i32>
        memref.store %arg17, %18[] : memref<i32>
        %19 = fir.convert %8 : (!fir.ref<f64>) -> memref<f64>
        %20 = memref.load %19[] : memref<f64>
        %21 = fir.convert %2 : (!fir.ref<i32>) -> memref<i32>
        %22 = memref.load %21[] : memref<i32>
        %23 = fir.convert %22 : (i32) -> i64
        %24 = fir.convert %0 : (!fir.ref<i32>) -> memref<i32>
        %25 = memref.load %24[] : memref<i32>
        %26 = fir.convert %25 : (i32) -> i64
        %27 = fir.convert %6 : (!fir.ref<!fir.array<4x3xf64>>) -> memref<3x4xf64>
        %28 = arith.index_cast %23 : i64 to index
        %29 = arith.subi %28, %c1 : index
        %30 = arith.index_cast %26 : i64 to index
        %31 = arith.subi %30, %c1 : index
        %32 = memref.load %27[%31, %29] : memref<3x4xf64>
        %33 = arith.mulf %20, %32 fastmath<contract> : f64
        %34 = fir.convert %7 : (!fir.ref<!fir.array<4x3xf64>>) -> memref<3x4xf64>
        %35 = arith.index_cast %23 : i64 to index
        %36 = arith.subi %35, %c1 : index
        %37 = arith.index_cast %26 : i64 to index
        %38 = arith.subi %37, %c1 : index
        %39 = memref.load %34[%38, %36] : memref<3x4xf64>
        %40 = arith.addf %33, %39 fastmath<contract> : f64
        %41 = fir.convert %5 : (!fir.ref<!fir.array<4x3xf64>>) -> memref<3x4xf64>
        %42 = arith.index_cast %23 : i64 to index
        %43 = arith.subi %42, %c1 : index
        %44 = arith.index_cast %26 : i64 to index
        %45 = arith.subi %44, %c1 : index
        memref.store %40, %41[%45, %43] : memref<3x4xf64>
        %46 = fir.convert %2 : (!fir.ref<i32>) -> memref<i32>
        %47 = memref.load %46[] : memref<i32>
        %48 = arith.addi %47, %9 overflow<nsw> : i32
        fir.result %48 : i32
      }
      %14 = fir.convert %2 : (!fir.ref<i32>) -> memref<i32>
      memref.store %13, %14[] : memref<i32>
      %15 = fir.convert %0 : (!fir.ref<i32>) -> memref<i32>
      %16 = memref.load %15[] : memref<i32>
      %17 = arith.addi %16, %9 overflow<nsw> : i32
      fir.result %17 : i32
    }
    %11 = fir.convert %0 : (!fir.ref<i32>) -> memref<i32>
    memref.store %10, %11[] : memref<i32>
    omp.terminator
  }
}


// -----// IR Dump After LoopInvariantCodeMotionPass (loop-invariant-code-motion) //----- //
module {
  func.func @kernel(%arg0: !fir.ref<i32> {jit.arg_type = 1 : ui32, jit.literal_val = 0 : i64}, %arg1: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 3 : i64}, %arg2: !fir.ref<i32> {jit.arg_type = 1 : ui32, jit.literal_val = 0 : i64}, %arg3: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 4 : i64}, %arg4: !fir.ref<!fir.array<4x3xf64>> {jit.arg_type = 1 : ui32}, %arg5: !fir.ref<!fir.array<4x3xf64>> {jit.arg_type = 1 : ui32}, %arg6: !fir.ref<!fir.array<4x3xf64>> {jit.arg_type = 1 : ui32}, %arg7: !fir.ref<f64> {jit.arg_type = 1 : ui32, jit.literal_val = 4619567317775286272 : i64}, %arg8: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 3 : i64}, %arg9: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 4 : i64}, %arg10: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 3 : i64}, %arg11: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 4 : i64}, %arg12: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 3 : i64}, %arg13: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 4 : i64}) {
    %c3 = arith.constant 3 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %0 = fir.declare %arg0 {uniq_name = "_QFFrun_benchmarkEj"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %1 = fir.declare %arg1 {uniq_name = "_QFFrun_benchmarkEm"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %2 = fir.declare %arg2 {uniq_name = "_QFFrun_benchmarkEi"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %3 = fir.declare %arg3 {uniq_name = "_QFFrun_benchmarkEn"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %4 = fir.shape %c4, %c3 : (index, index) -> !fir.shape<2>
    %5 = fir.declare %arg4(%4) {uniq_name = "_QFFrun_benchmarkEz"} : (!fir.ref<!fir.array<4x3xf64>>, !fir.shape<2>) -> !fir.ref<!fir.array<4x3xf64>>
    %6 = fir.declare %arg5(%4) {uniq_name = "_QFFrun_benchmarkEx"} : (!fir.ref<!fir.array<4x3xf64>>, !fir.shape<2>) -> !fir.ref<!fir.array<4x3xf64>>
    %7 = fir.declare %arg6(%4) {uniq_name = "_QFFrun_benchmarkEy"} : (!fir.ref<!fir.array<4x3xf64>>, !fir.shape<2>) -> !fir.ref<!fir.array<4x3xf64>>
    %8 = fir.declare %arg7 {uniq_name = "_QFFrun_benchmarkEa"} : (!fir.ref<f64>) -> !fir.ref<f64>
    %9 = fir.convert %c1 : (index) -> i32
    %10 = fir.convert %0 : (!fir.ref<i32>) -> memref<i32>
    %11 = fir.convert %2 : (!fir.ref<i32>) -> memref<i32>
    %12 = fir.convert %8 : (!fir.ref<f64>) -> memref<f64>
    %13 = fir.convert %2 : (!fir.ref<i32>) -> memref<i32>
    %14 = fir.convert %0 : (!fir.ref<i32>) -> memref<i32>
    %15 = fir.convert %6 : (!fir.ref<!fir.array<4x3xf64>>) -> memref<3x4xf64>
    %16 = fir.convert %7 : (!fir.ref<!fir.array<4x3xf64>>) -> memref<3x4xf64>
    %17 = fir.convert %5 : (!fir.ref<!fir.array<4x3xf64>>) -> memref<3x4xf64>
    %18 = fir.convert %2 : (!fir.ref<i32>) -> memref<i32>
    %19 = fir.convert %2 : (!fir.ref<i32>) -> memref<i32>
    %20 = fir.convert %0 : (!fir.ref<i32>) -> memref<i32>
    %21 = fir.do_loop %arg14 = %c1 to %c3 step %c1 iter_args(%arg15 = %9) -> (i32) {
      memref.store %arg15, %10[] : memref<i32>
      %23 = fir.do_loop %arg16 = %c1 to %c4 step %c1 iter_args(%arg17 = %9) -> (i32) {
        memref.store %arg17, %11[] : memref<i32>
        %26 = memref.load %12[] : memref<f64>
        %27 = memref.load %13[] : memref<i32>
        %28 = fir.convert %27 : (i32) -> i64
        %29 = memref.load %14[] : memref<i32>
        %30 = fir.convert %29 : (i32) -> i64
        %31 = arith.index_cast %28 : i64 to index
        %32 = arith.subi %31, %c1 : index
        %33 = arith.index_cast %30 : i64 to index
        %34 = arith.subi %33, %c1 : index
        %35 = memref.load %15[%34, %32] : memref<3x4xf64>
        %36 = arith.mulf %26, %35 fastmath<contract> : f64
        %37 = arith.index_cast %28 : i64 to index
        %38 = arith.subi %37, %c1 : index
        %39 = arith.index_cast %30 : i64 to index
        %40 = arith.subi %39, %c1 : index
        %41 = memref.load %16[%40, %38] : memref<3x4xf64>
        %42 = arith.addf %36, %41 fastmath<contract> : f64
        %43 = arith.index_cast %28 : i64 to index
        %44 = arith.subi %43, %c1 : index
        %45 = arith.index_cast %30 : i64 to index
        %46 = arith.subi %45, %c1 : index
        memref.store %42, %17[%46, %44] : memref<3x4xf64>
        %47 = memref.load %18[] : memref<i32>
        %48 = arith.addi %47, %9 overflow<nsw> : i32
        fir.result %48 : i32
      }
      memref.store %23, %19[] : memref<i32>
      %24 = memref.load %20[] : memref<i32>
      %25 = arith.addi %24, %9 overflow<nsw> : i32
      fir.result %25 : i32
    }
    %22 = fir.convert %0 : (!fir.ref<i32>) -> memref<i32>
    memref.store %21, %22[] : memref<i32>
    omp.terminator
  }
}


// -----// IR Dump After CSEPass (cse) //----- //
module {
  func.func @kernel(%arg0: !fir.ref<i32> {jit.arg_type = 1 : ui32, jit.literal_val = 0 : i64}, %arg1: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 3 : i64}, %arg2: !fir.ref<i32> {jit.arg_type = 1 : ui32, jit.literal_val = 0 : i64}, %arg3: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 4 : i64}, %arg4: !fir.ref<!fir.array<4x3xf64>> {jit.arg_type = 1 : ui32}, %arg5: !fir.ref<!fir.array<4x3xf64>> {jit.arg_type = 1 : ui32}, %arg6: !fir.ref<!fir.array<4x3xf64>> {jit.arg_type = 1 : ui32}, %arg7: !fir.ref<f64> {jit.arg_type = 1 : ui32, jit.literal_val = 4619567317775286272 : i64}, %arg8: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 3 : i64}, %arg9: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 4 : i64}, %arg10: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 3 : i64}, %arg11: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 4 : i64}, %arg12: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 3 : i64}, %arg13: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 4 : i64}) {
    %c3 = arith.constant 3 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %0 = fir.declare %arg0 {uniq_name = "_QFFrun_benchmarkEj"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %1 = fir.declare %arg1 {uniq_name = "_QFFrun_benchmarkEm"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %2 = fir.declare %arg2 {uniq_name = "_QFFrun_benchmarkEi"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %3 = fir.declare %arg3 {uniq_name = "_QFFrun_benchmarkEn"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %4 = fir.shape %c4, %c3 : (index, index) -> !fir.shape<2>
    %5 = fir.declare %arg4(%4) {uniq_name = "_QFFrun_benchmarkEz"} : (!fir.ref<!fir.array<4x3xf64>>, !fir.shape<2>) -> !fir.ref<!fir.array<4x3xf64>>
    %6 = fir.declare %arg5(%4) {uniq_name = "_QFFrun_benchmarkEx"} : (!fir.ref<!fir.array<4x3xf64>>, !fir.shape<2>) -> !fir.ref<!fir.array<4x3xf64>>
    %7 = fir.declare %arg6(%4) {uniq_name = "_QFFrun_benchmarkEy"} : (!fir.ref<!fir.array<4x3xf64>>, !fir.shape<2>) -> !fir.ref<!fir.array<4x3xf64>>
    %8 = fir.declare %arg7 {uniq_name = "_QFFrun_benchmarkEa"} : (!fir.ref<f64>) -> !fir.ref<f64>
    %9 = fir.convert %c1 : (index) -> i32
    %10 = fir.convert %0 : (!fir.ref<i32>) -> memref<i32>
    %11 = fir.convert %2 : (!fir.ref<i32>) -> memref<i32>
    %12 = fir.convert %8 : (!fir.ref<f64>) -> memref<f64>
    %13 = fir.convert %6 : (!fir.ref<!fir.array<4x3xf64>>) -> memref<3x4xf64>
    %14 = fir.convert %7 : (!fir.ref<!fir.array<4x3xf64>>) -> memref<3x4xf64>
    %15 = fir.convert %5 : (!fir.ref<!fir.array<4x3xf64>>) -> memref<3x4xf64>
    %16 = fir.do_loop %arg14 = %c1 to %c3 step %c1 iter_args(%arg15 = %9) -> (i32) {
      memref.store %arg15, %10[] : memref<i32>
      %17 = fir.do_loop %arg16 = %c1 to %c4 step %c1 iter_args(%arg17 = %9) -> (i32) {
        memref.store %arg17, %11[] : memref<i32>
        %20 = memref.load %12[] : memref<f64>
        %21 = memref.load %11[] : memref<i32>
        %22 = fir.convert %21 : (i32) -> i64
        %23 = memref.load %10[] : memref<i32>
        %24 = fir.convert %23 : (i32) -> i64
        %25 = arith.index_cast %22 : i64 to index
        %26 = arith.subi %25, %c1 : index
        %27 = arith.index_cast %24 : i64 to index
        %28 = arith.subi %27, %c1 : index
        %29 = memref.load %13[%28, %26] : memref<3x4xf64>
        %30 = arith.mulf %20, %29 fastmath<contract> : f64
        %31 = memref.load %14[%28, %26] : memref<3x4xf64>
        %32 = arith.addf %30, %31 fastmath<contract> : f64
        memref.store %32, %15[%28, %26] : memref<3x4xf64>
        %33 = memref.load %11[] : memref<i32>
        %34 = arith.addi %33, %9 overflow<nsw> : i32
        fir.result %34 : i32
      }
      memref.store %17, %11[] : memref<i32>
      %18 = memref.load %10[] : memref<i32>
      %19 = arith.addi %18, %9 overflow<nsw> : i32
      fir.result %19 : i32
    }
    memref.store %16, %10[] : memref<i32>
    omp.terminator
  }
}


// -----// IR Dump After CanonicalizerPass (canonicalize) //----- //
module {
  func.func @kernel(%arg0: !fir.ref<i32> {jit.arg_type = 1 : ui32, jit.literal_val = 0 : i64}, %arg1: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 3 : i64}, %arg2: !fir.ref<i32> {jit.arg_type = 1 : ui32, jit.literal_val = 0 : i64}, %arg3: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 4 : i64}, %arg4: !fir.ref<!fir.array<4x3xf64>> {jit.arg_type = 1 : ui32}, %arg5: !fir.ref<!fir.array<4x3xf64>> {jit.arg_type = 1 : ui32}, %arg6: !fir.ref<!fir.array<4x3xf64>> {jit.arg_type = 1 : ui32}, %arg7: !fir.ref<f64> {jit.arg_type = 1 : ui32, jit.literal_val = 4619567317775286272 : i64}, %arg8: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 3 : i64}, %arg9: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 4 : i64}, %arg10: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 3 : i64}, %arg11: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 4 : i64}, %arg12: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 3 : i64}, %arg13: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 4 : i64}) {
    %c3 = arith.constant 3 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %0 = fir.declare %arg0 {uniq_name = "_QFFrun_benchmarkEj"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %1 = fir.declare %arg1 {uniq_name = "_QFFrun_benchmarkEm"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %2 = fir.declare %arg2 {uniq_name = "_QFFrun_benchmarkEi"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %3 = fir.declare %arg3 {uniq_name = "_QFFrun_benchmarkEn"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %4 = fir.shape %c4, %c3 : (index, index) -> !fir.shape<2>
    %5 = fir.declare %arg4(%4) {uniq_name = "_QFFrun_benchmarkEz"} : (!fir.ref<!fir.array<4x3xf64>>, !fir.shape<2>) -> !fir.ref<!fir.array<4x3xf64>>
    %6 = fir.declare %arg5(%4) {uniq_name = "_QFFrun_benchmarkEx"} : (!fir.ref<!fir.array<4x3xf64>>, !fir.shape<2>) -> !fir.ref<!fir.array<4x3xf64>>
    %7 = fir.declare %arg6(%4) {uniq_name = "_QFFrun_benchmarkEy"} : (!fir.ref<!fir.array<4x3xf64>>, !fir.shape<2>) -> !fir.ref<!fir.array<4x3xf64>>
    %8 = fir.declare %arg7 {uniq_name = "_QFFrun_benchmarkEa"} : (!fir.ref<f64>) -> !fir.ref<f64>
    %9 = fir.convert %c1 : (index) -> i32
    %10 = fir.convert %0 : (!fir.ref<i32>) -> memref<i32>
    %11 = fir.convert %2 : (!fir.ref<i32>) -> memref<i32>
    %12 = fir.convert %8 : (!fir.ref<f64>) -> memref<f64>
    %13 = fir.convert %6 : (!fir.ref<!fir.array<4x3xf64>>) -> memref<3x4xf64>
    %14 = fir.convert %7 : (!fir.ref<!fir.array<4x3xf64>>) -> memref<3x4xf64>
    %15 = fir.convert %5 : (!fir.ref<!fir.array<4x3xf64>>) -> memref<3x4xf64>
    %16 = fir.do_loop %arg14 = %c1 to %c3 step %c1 iter_args(%arg15 = %9) -> (i32) {
      memref.store %arg15, %10[] : memref<i32>
      %17 = fir.do_loop %arg16 = %c1 to %c4 step %c1 iter_args(%arg17 = %9) -> (i32) {
        memref.store %arg17, %11[] : memref<i32>
        %20 = memref.load %12[] : memref<f64>
        %21 = memref.load %11[] : memref<i32>
        %22 = fir.convert %21 : (i32) -> i64
        %23 = memref.load %10[] : memref<i32>
        %24 = fir.convert %23 : (i32) -> i64
        %25 = arith.index_cast %22 : i64 to index
        %26 = arith.subi %25, %c1 : index
        %27 = arith.index_cast %24 : i64 to index
        %28 = arith.subi %27, %c1 : index
        %29 = memref.load %13[%28, %26] : memref<3x4xf64>
        %30 = arith.mulf %20, %29 fastmath<contract> : f64
        %31 = memref.load %14[%28, %26] : memref<3x4xf64>
        %32 = arith.addf %30, %31 fastmath<contract> : f64
        memref.store %32, %15[%28, %26] : memref<3x4xf64>
        %33 = memref.load %11[] : memref<i32>
        %34 = arith.addi %33, %9 overflow<nsw> : i32
        fir.result %34 : i32
      }
      memref.store %17, %11[] : memref<i32>
      %18 = memref.load %10[] : memref<i32>
      %19 = arith.addi %18, %9 overflow<nsw> : i32
      fir.result %19 : i32
    }
    memref.store %16, %10[] : memref<i32>
    omp.terminator
  }
}


// -----// IR Dump After {anonymous}::CleanFIRLoopPass (jforce-clean-fir-loop) //----- //
func.func @kernel(%arg0: !fir.ref<i32> {jit.arg_type = 1 : ui32, jit.literal_val = 0 : i64}, %arg1: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 3 : i64}, %arg2: !fir.ref<i32> {jit.arg_type = 1 : ui32, jit.literal_val = 0 : i64}, %arg3: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 4 : i64}, %arg4: !fir.ref<!fir.array<4x3xf64>> {jit.arg_type = 1 : ui32}, %arg5: !fir.ref<!fir.array<4x3xf64>> {jit.arg_type = 1 : ui32}, %arg6: !fir.ref<!fir.array<4x3xf64>> {jit.arg_type = 1 : ui32}, %arg7: !fir.ref<f64> {jit.arg_type = 1 : ui32, jit.literal_val = 4619567317775286272 : i64}, %arg8: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 3 : i64}, %arg9: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 4 : i64}, %arg10: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 3 : i64}, %arg11: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 4 : i64}, %arg12: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 3 : i64}, %arg13: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 4 : i64}) {
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %c1 = arith.constant 1 : index
  %0 = fir.declare %arg0 {uniq_name = "_QFFrun_benchmarkEj"} : (!fir.ref<i32>) -> !fir.ref<i32>
  %1 = fir.declare %arg1 {uniq_name = "_QFFrun_benchmarkEm"} : (!fir.ref<i32>) -> !fir.ref<i32>
  %2 = fir.declare %arg2 {uniq_name = "_QFFrun_benchmarkEi"} : (!fir.ref<i32>) -> !fir.ref<i32>
  %3 = fir.declare %arg3 {uniq_name = "_QFFrun_benchmarkEn"} : (!fir.ref<i32>) -> !fir.ref<i32>
  %4 = fir.shape %c4, %c3 : (index, index) -> !fir.shape<2>
  %5 = fir.declare %arg4(%4) {uniq_name = "_QFFrun_benchmarkEz"} : (!fir.ref<!fir.array<4x3xf64>>, !fir.shape<2>) -> !fir.ref<!fir.array<4x3xf64>>
  %6 = fir.declare %arg5(%4) {uniq_name = "_QFFrun_benchmarkEx"} : (!fir.ref<!fir.array<4x3xf64>>, !fir.shape<2>) -> !fir.ref<!fir.array<4x3xf64>>
  %7 = fir.declare %arg6(%4) {uniq_name = "_QFFrun_benchmarkEy"} : (!fir.ref<!fir.array<4x3xf64>>, !fir.shape<2>) -> !fir.ref<!fir.array<4x3xf64>>
  %8 = fir.declare %arg7 {uniq_name = "_QFFrun_benchmarkEa"} : (!fir.ref<f64>) -> !fir.ref<f64>
  %9 = fir.convert %c1 : (index) -> i32
  %10 = fir.convert %0 : (!fir.ref<i32>) -> memref<i32>
  %11 = fir.convert %2 : (!fir.ref<i32>) -> memref<i32>
  %12 = fir.convert %8 : (!fir.ref<f64>) -> memref<f64>
  %13 = fir.convert %6 : (!fir.ref<!fir.array<4x3xf64>>) -> memref<3x4xf64>
  %14 = fir.convert %7 : (!fir.ref<!fir.array<4x3xf64>>) -> memref<3x4xf64>
  %15 = fir.convert %5 : (!fir.ref<!fir.array<4x3xf64>>) -> memref<3x4xf64>
  fir.do_loop %arg14 = %c1 to %c3 step %c1 {
    %17 = fir.convert %arg14 : (index) -> i32
    memref.store %17, %10[] : memref<i32>
    fir.do_loop %arg15 = %c1 to %c4 step %c1 {
      %21 = fir.convert %arg15 : (index) -> i32
      memref.store %21, %11[] : memref<i32>
      %22 = memref.load %12[] : memref<f64>
      %23 = memref.load %11[] : memref<i32>
      %24 = fir.convert %23 : (i32) -> i64
      %25 = memref.load %10[] : memref<i32>
      %26 = fir.convert %25 : (i32) -> i64
      %27 = arith.index_cast %24 : i64 to index
      %28 = arith.subi %27, %c1 : index
      %29 = arith.index_cast %26 : i64 to index
      %30 = arith.subi %29, %c1 : index
      %31 = memref.load %13[%30, %28] : memref<3x4xf64>
      %32 = arith.mulf %22, %31 fastmath<contract> : f64
      %33 = memref.load %14[%30, %28] : memref<3x4xf64>
      %34 = arith.addf %32, %33 fastmath<contract> : f64
      memref.store %34, %15[%30, %28] : memref<3x4xf64>
      %35 = memref.load %11[] : memref<i32>
      %36 = arith.addi %35, %9 overflow<nsw> : i32
      memref.store %36, %11[] : memref<i32>
    }
    %18 = memref.load %11[] : memref<i32>
    memref.store %18, %11[] : memref<i32>
    %19 = memref.load %10[] : memref<i32>
    %20 = arith.addi %19, %9 overflow<nsw> : i32
    memref.store %20, %10[] : memref<i32>
  }
  %16 = memref.load %10[] : memref<i32>
  memref.store %16, %10[] : memref<i32>
  omp.terminator
}

// -----// IR Dump After CanonicalizerPass (canonicalize) //----- //
module {
  func.func @kernel(%arg0: !fir.ref<i32> {jit.arg_type = 1 : ui32, jit.literal_val = 0 : i64}, %arg1: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 3 : i64}, %arg2: !fir.ref<i32> {jit.arg_type = 1 : ui32, jit.literal_val = 0 : i64}, %arg3: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 4 : i64}, %arg4: !fir.ref<!fir.array<4x3xf64>> {jit.arg_type = 1 : ui32}, %arg5: !fir.ref<!fir.array<4x3xf64>> {jit.arg_type = 1 : ui32}, %arg6: !fir.ref<!fir.array<4x3xf64>> {jit.arg_type = 1 : ui32}, %arg7: !fir.ref<f64> {jit.arg_type = 1 : ui32, jit.literal_val = 4619567317775286272 : i64}, %arg8: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 3 : i64}, %arg9: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 4 : i64}, %arg10: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 3 : i64}, %arg11: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 4 : i64}, %arg12: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 3 : i64}, %arg13: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 4 : i64}) {
    %c3 = arith.constant 3 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %0 = fir.declare %arg0 {uniq_name = "_QFFrun_benchmarkEj"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %1 = fir.declare %arg1 {uniq_name = "_QFFrun_benchmarkEm"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %2 = fir.declare %arg2 {uniq_name = "_QFFrun_benchmarkEi"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %3 = fir.declare %arg3 {uniq_name = "_QFFrun_benchmarkEn"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %4 = fir.shape %c4, %c3 : (index, index) -> !fir.shape<2>
    %5 = fir.declare %arg4(%4) {uniq_name = "_QFFrun_benchmarkEz"} : (!fir.ref<!fir.array<4x3xf64>>, !fir.shape<2>) -> !fir.ref<!fir.array<4x3xf64>>
    %6 = fir.declare %arg5(%4) {uniq_name = "_QFFrun_benchmarkEx"} : (!fir.ref<!fir.array<4x3xf64>>, !fir.shape<2>) -> !fir.ref<!fir.array<4x3xf64>>
    %7 = fir.declare %arg6(%4) {uniq_name = "_QFFrun_benchmarkEy"} : (!fir.ref<!fir.array<4x3xf64>>, !fir.shape<2>) -> !fir.ref<!fir.array<4x3xf64>>
    %8 = fir.declare %arg7 {uniq_name = "_QFFrun_benchmarkEa"} : (!fir.ref<f64>) -> !fir.ref<f64>
    %9 = fir.convert %c1 : (index) -> i32
    %10 = fir.convert %0 : (!fir.ref<i32>) -> memref<i32>
    %11 = fir.convert %2 : (!fir.ref<i32>) -> memref<i32>
    %12 = fir.convert %8 : (!fir.ref<f64>) -> memref<f64>
    %13 = fir.convert %6 : (!fir.ref<!fir.array<4x3xf64>>) -> memref<3x4xf64>
    %14 = fir.convert %7 : (!fir.ref<!fir.array<4x3xf64>>) -> memref<3x4xf64>
    %15 = fir.convert %5 : (!fir.ref<!fir.array<4x3xf64>>) -> memref<3x4xf64>
    fir.do_loop %arg14 = %c1 to %c3 step %c1 {
      %17 = fir.convert %arg14 : (index) -> i32
      memref.store %17, %10[] : memref<i32>
      fir.do_loop %arg15 = %c1 to %c4 step %c1 {
        %21 = fir.convert %arg15 : (index) -> i32
        memref.store %21, %11[] : memref<i32>
        %22 = memref.load %12[] : memref<f64>
        %23 = memref.load %11[] : memref<i32>
        %24 = fir.convert %23 : (i32) -> i64
        %25 = memref.load %10[] : memref<i32>
        %26 = fir.convert %25 : (i32) -> i64
        %27 = arith.index_cast %24 : i64 to index
        %28 = arith.subi %27, %c1 : index
        %29 = arith.index_cast %26 : i64 to index
        %30 = arith.subi %29, %c1 : index
        %31 = memref.load %13[%30, %28] : memref<3x4xf64>
        %32 = arith.mulf %22, %31 fastmath<contract> : f64
        %33 = memref.load %14[%30, %28] : memref<3x4xf64>
        %34 = arith.addf %32, %33 fastmath<contract> : f64
        memref.store %34, %15[%30, %28] : memref<3x4xf64>
        %35 = memref.load %11[] : memref<i32>
        %36 = arith.addi %35, %9 overflow<nsw> : i32
        memref.store %36, %11[] : memref<i32>
      }
      %18 = memref.load %11[] : memref<i32>
      memref.store %18, %11[] : memref<i32>
      %19 = memref.load %10[] : memref<i32>
      %20 = arith.addi %19, %9 overflow<nsw> : i32
      memref.store %20, %10[] : memref<i32>
    }
    %16 = memref.load %10[] : memref<i32>
    memref.store %16, %10[] : memref<i32>
    omp.terminator
  }
}


// -----// IR Dump After {anonymous}::CleanFIROpsPass (jforce-clean-fir-op) //----- //
func.func @kernel(%arg0: !fir.ref<i32> {jit.arg_type = 1 : ui32, jit.literal_val = 0 : i64}, %arg1: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 3 : i64}, %arg2: !fir.ref<i32> {jit.arg_type = 1 : ui32, jit.literal_val = 0 : i64}, %arg3: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 4 : i64}, %arg4: !fir.ref<!fir.array<4x3xf64>> {jit.arg_type = 1 : ui32}, %arg5: !fir.ref<!fir.array<4x3xf64>> {jit.arg_type = 1 : ui32}, %arg6: !fir.ref<!fir.array<4x3xf64>> {jit.arg_type = 1 : ui32}, %arg7: !fir.ref<f64> {jit.arg_type = 1 : ui32, jit.literal_val = 4619567317775286272 : i64}, %arg8: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 3 : i64}, %arg9: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 4 : i64}, %arg10: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 3 : i64}, %arg11: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 4 : i64}, %arg12: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 3 : i64}, %arg13: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 4 : i64}) {
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %c1 = arith.constant 1 : index
  %0 = fir.declare %arg0 {uniq_name = "_QFFrun_benchmarkEj"} : (!fir.ref<i32>) -> !fir.ref<i32>
  %1 = fir.declare %arg1 {uniq_name = "_QFFrun_benchmarkEm"} : (!fir.ref<i32>) -> !fir.ref<i32>
  %2 = fir.declare %arg2 {uniq_name = "_QFFrun_benchmarkEi"} : (!fir.ref<i32>) -> !fir.ref<i32>
  %3 = fir.declare %arg3 {uniq_name = "_QFFrun_benchmarkEn"} : (!fir.ref<i32>) -> !fir.ref<i32>
  %4 = fir.shape %c4, %c3 : (index, index) -> !fir.shape<2>
  %5 = fir.declare %arg4(%4) {uniq_name = "_QFFrun_benchmarkEz"} : (!fir.ref<!fir.array<4x3xf64>>, !fir.shape<2>) -> !fir.ref<!fir.array<4x3xf64>>
  %6 = fir.declare %arg5(%4) {uniq_name = "_QFFrun_benchmarkEx"} : (!fir.ref<!fir.array<4x3xf64>>, !fir.shape<2>) -> !fir.ref<!fir.array<4x3xf64>>
  %7 = fir.declare %arg6(%4) {uniq_name = "_QFFrun_benchmarkEy"} : (!fir.ref<!fir.array<4x3xf64>>, !fir.shape<2>) -> !fir.ref<!fir.array<4x3xf64>>
  %8 = fir.declare %arg7 {uniq_name = "_QFFrun_benchmarkEa"} : (!fir.ref<f64>) -> !fir.ref<f64>
  %9 = fir.convert %c1 : (index) -> i32
  %10 = fir.convert %0 : (!fir.ref<i32>) -> memref<i32>
  %11 = fir.convert %2 : (!fir.ref<i32>) -> memref<i32>
  %12 = fir.convert %8 : (!fir.ref<f64>) -> memref<f64>
  %13 = fir.convert %6 : (!fir.ref<!fir.array<4x3xf64>>) -> memref<3x4xf64>
  %14 = fir.convert %7 : (!fir.ref<!fir.array<4x3xf64>>) -> memref<3x4xf64>
  %15 = fir.convert %5 : (!fir.ref<!fir.array<4x3xf64>>) -> memref<3x4xf64>
  fir.do_loop %arg14 = %c1 to %c3 step %c1 {
    %17 = arith.index_cast %arg14 : index to i32
    memref.store %17, %10[] : memref<i32>
    fir.do_loop %arg15 = %c1 to %c4 step %c1 {
      %21 = arith.index_cast %arg15 : index to i32
      memref.store %21, %11[] : memref<i32>
      %22 = memref.load %12[] : memref<f64>
      %23 = memref.load %11[] : memref<i32>
      %24 = arith.extsi %23 : i32 to i64
      %25 = memref.load %10[] : memref<i32>
      %26 = arith.extsi %25 : i32 to i64
      %27 = arith.index_cast %24 : i64 to index
      %28 = arith.subi %27, %c1 : index
      %29 = arith.index_cast %26 : i64 to index
      %30 = arith.subi %29, %c1 : index
      %31 = memref.load %13[%30, %28] : memref<3x4xf64>
      %32 = arith.mulf %22, %31 fastmath<contract> : f64
      %33 = memref.load %14[%30, %28] : memref<3x4xf64>
      %34 = arith.addf %32, %33 fastmath<contract> : f64
      memref.store %34, %15[%30, %28] : memref<3x4xf64>
      %35 = memref.load %11[] : memref<i32>
      %36 = arith.addi %35, %9 overflow<nsw> : i32
      memref.store %36, %11[] : memref<i32>
    }
    %18 = memref.load %11[] : memref<i32>
    memref.store %18, %11[] : memref<i32>
    %19 = memref.load %10[] : memref<i32>
    %20 = arith.addi %19, %9 overflow<nsw> : i32
    memref.store %20, %10[] : memref<i32>
  }
  %16 = memref.load %10[] : memref<i32>
  memref.store %16, %10[] : memref<i32>
  omp.terminator
}

// -----// IR Dump After AffineDialectPromotion (promote-to-affine) //----- //
func.func @kernel(%arg0: !fir.ref<i32> {jit.arg_type = 1 : ui32, jit.literal_val = 0 : i64}, %arg1: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 3 : i64}, %arg2: !fir.ref<i32> {jit.arg_type = 1 : ui32, jit.literal_val = 0 : i64}, %arg3: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 4 : i64}, %arg4: !fir.ref<!fir.array<4x3xf64>> {jit.arg_type = 1 : ui32}, %arg5: !fir.ref<!fir.array<4x3xf64>> {jit.arg_type = 1 : ui32}, %arg6: !fir.ref<!fir.array<4x3xf64>> {jit.arg_type = 1 : ui32}, %arg7: !fir.ref<f64> {jit.arg_type = 1 : ui32, jit.literal_val = 4619567317775286272 : i64}, %arg8: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 3 : i64}, %arg9: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 4 : i64}, %arg10: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 3 : i64}, %arg11: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 4 : i64}, %arg12: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 3 : i64}, %arg13: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 4 : i64}) {
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %c1 = arith.constant 1 : index
  %0 = fir.declare %arg0 {uniq_name = "_QFFrun_benchmarkEj"} : (!fir.ref<i32>) -> !fir.ref<i32>
  %1 = fir.declare %arg1 {uniq_name = "_QFFrun_benchmarkEm"} : (!fir.ref<i32>) -> !fir.ref<i32>
  %2 = fir.declare %arg2 {uniq_name = "_QFFrun_benchmarkEi"} : (!fir.ref<i32>) -> !fir.ref<i32>
  %3 = fir.declare %arg3 {uniq_name = "_QFFrun_benchmarkEn"} : (!fir.ref<i32>) -> !fir.ref<i32>
  %4 = fir.shape %c4, %c3 : (index, index) -> !fir.shape<2>
  %5 = fir.declare %arg4(%4) {uniq_name = "_QFFrun_benchmarkEz"} : (!fir.ref<!fir.array<4x3xf64>>, !fir.shape<2>) -> !fir.ref<!fir.array<4x3xf64>>
  %6 = fir.declare %arg5(%4) {uniq_name = "_QFFrun_benchmarkEx"} : (!fir.ref<!fir.array<4x3xf64>>, !fir.shape<2>) -> !fir.ref<!fir.array<4x3xf64>>
  %7 = fir.declare %arg6(%4) {uniq_name = "_QFFrun_benchmarkEy"} : (!fir.ref<!fir.array<4x3xf64>>, !fir.shape<2>) -> !fir.ref<!fir.array<4x3xf64>>
  %8 = fir.declare %arg7 {uniq_name = "_QFFrun_benchmarkEa"} : (!fir.ref<f64>) -> !fir.ref<f64>
  %9 = fir.convert %c1 : (index) -> i32
  %10 = fir.convert %0 : (!fir.ref<i32>) -> memref<i32>
  %11 = fir.convert %2 : (!fir.ref<i32>) -> memref<i32>
  %12 = fir.convert %8 : (!fir.ref<f64>) -> memref<f64>
  %13 = fir.convert %6 : (!fir.ref<!fir.array<4x3xf64>>) -> memref<3x4xf64>
  %14 = fir.convert %7 : (!fir.ref<!fir.array<4x3xf64>>) -> memref<3x4xf64>
  %15 = fir.convert %5 : (!fir.ref<!fir.array<4x3xf64>>) -> memref<3x4xf64>
  affine.for %arg14 = %c1 to affine_map<()[s0] -> (s0 + 1)>()[%c3] {
    %17 = arith.index_cast %arg14 : index to i32
    memref.store %17, %10[] : memref<i32>
    affine.for %arg15 = %c1 to affine_map<()[s0] -> (s0 + 1)>()[%c4] {
      %21 = arith.index_cast %arg15 : index to i32
      memref.store %21, %11[] : memref<i32>
      %22 = memref.load %12[] : memref<f64>
      %23 = memref.load %11[] : memref<i32>
      %24 = arith.extsi %23 : i32 to i64
      %25 = memref.load %10[] : memref<i32>
      %26 = arith.extsi %25 : i32 to i64
      %27 = arith.index_cast %24 : i64 to index
      %28 = arith.subi %27, %c1 : index
      %29 = arith.index_cast %26 : i64 to index
      %30 = arith.subi %29, %c1 : index
      %31 = memref.load %13[%30, %28] : memref<3x4xf64>
      %32 = arith.mulf %22, %31 fastmath<contract> : f64
      %33 = memref.load %14[%30, %28] : memref<3x4xf64>
      %34 = arith.addf %32, %33 fastmath<contract> : f64
      memref.store %34, %15[%30, %28] : memref<3x4xf64>
      %35 = memref.load %11[] : memref<i32>
      %36 = arith.addi %35, %9 overflow<nsw> : i32
      memref.store %36, %11[] : memref<i32>
    }
    %18 = memref.load %11[] : memref<i32>
    memref.store %18, %11[] : memref<i32>
    %19 = memref.load %10[] : memref<i32>
    %20 = arith.addi %19, %9 overflow<nsw> : i32
    memref.store %20, %10[] : memref<i32>
  }
  %16 = memref.load %10[] : memref<i32>
  memref.store %16, %10[] : memref<i32>
  omp.terminator
}

// -----// IR Dump After CanonicalizerPass (canonicalize) //----- //
module {
  func.func @kernel(%arg0: !fir.ref<i32> {jit.arg_type = 1 : ui32, jit.literal_val = 0 : i64}, %arg1: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 3 : i64}, %arg2: !fir.ref<i32> {jit.arg_type = 1 : ui32, jit.literal_val = 0 : i64}, %arg3: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 4 : i64}, %arg4: !fir.ref<!fir.array<4x3xf64>> {jit.arg_type = 1 : ui32}, %arg5: !fir.ref<!fir.array<4x3xf64>> {jit.arg_type = 1 : ui32}, %arg6: !fir.ref<!fir.array<4x3xf64>> {jit.arg_type = 1 : ui32}, %arg7: !fir.ref<f64> {jit.arg_type = 1 : ui32, jit.literal_val = 4619567317775286272 : i64}, %arg8: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 3 : i64}, %arg9: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 4 : i64}, %arg10: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 3 : i64}, %arg11: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 4 : i64}, %arg12: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 3 : i64}, %arg13: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 4 : i64}) {
    %c3 = arith.constant 3 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %0 = fir.declare %arg0 {uniq_name = "_QFFrun_benchmarkEj"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %1 = fir.declare %arg1 {uniq_name = "_QFFrun_benchmarkEm"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %2 = fir.declare %arg2 {uniq_name = "_QFFrun_benchmarkEi"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %3 = fir.declare %arg3 {uniq_name = "_QFFrun_benchmarkEn"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %4 = fir.shape %c4, %c3 : (index, index) -> !fir.shape<2>
    %5 = fir.declare %arg4(%4) {uniq_name = "_QFFrun_benchmarkEz"} : (!fir.ref<!fir.array<4x3xf64>>, !fir.shape<2>) -> !fir.ref<!fir.array<4x3xf64>>
    %6 = fir.declare %arg5(%4) {uniq_name = "_QFFrun_benchmarkEx"} : (!fir.ref<!fir.array<4x3xf64>>, !fir.shape<2>) -> !fir.ref<!fir.array<4x3xf64>>
    %7 = fir.declare %arg6(%4) {uniq_name = "_QFFrun_benchmarkEy"} : (!fir.ref<!fir.array<4x3xf64>>, !fir.shape<2>) -> !fir.ref<!fir.array<4x3xf64>>
    %8 = fir.declare %arg7 {uniq_name = "_QFFrun_benchmarkEa"} : (!fir.ref<f64>) -> !fir.ref<f64>
    %9 = fir.convert %c1 : (index) -> i32
    %10 = fir.convert %0 : (!fir.ref<i32>) -> memref<i32>
    %11 = fir.convert %2 : (!fir.ref<i32>) -> memref<i32>
    %12 = fir.convert %8 : (!fir.ref<f64>) -> memref<f64>
    %13 = fir.convert %6 : (!fir.ref<!fir.array<4x3xf64>>) -> memref<3x4xf64>
    %14 = fir.convert %7 : (!fir.ref<!fir.array<4x3xf64>>) -> memref<3x4xf64>
    %15 = fir.convert %5 : (!fir.ref<!fir.array<4x3xf64>>) -> memref<3x4xf64>
    affine.for %arg14 = 1 to 4 {
      %17 = arith.index_cast %arg14 : index to i32
      memref.store %17, %10[] : memref<i32>
      affine.for %arg15 = 1 to 5 {
        %21 = arith.index_cast %arg15 : index to i32
        memref.store %21, %11[] : memref<i32>
        %22 = memref.load %12[] : memref<f64>
        %23 = memref.load %11[] : memref<i32>
        %24 = memref.load %10[] : memref<i32>
        %25 = arith.index_cast %23 : i32 to index
        %26 = arith.subi %25, %c1 : index
        %27 = arith.index_cast %24 : i32 to index
        %28 = arith.subi %27, %c1 : index
        %29 = memref.load %13[%28, %26] : memref<3x4xf64>
        %30 = arith.mulf %22, %29 fastmath<contract> : f64
        %31 = memref.load %14[%28, %26] : memref<3x4xf64>
        %32 = arith.addf %30, %31 fastmath<contract> : f64
        memref.store %32, %15[%28, %26] : memref<3x4xf64>
        %33 = memref.load %11[] : memref<i32>
        %34 = arith.addi %33, %9 overflow<nsw> : i32
        memref.store %34, %11[] : memref<i32>
      }
      %18 = memref.load %11[] : memref<i32>
      memref.store %18, %11[] : memref<i32>
      %19 = memref.load %10[] : memref<i32>
      %20 = arith.addi %19, %9 overflow<nsw> : i32
      memref.store %20, %10[] : memref<i32>
    }
    %16 = memref.load %10[] : memref<i32>
    memref.store %16, %10[] : memref<i32>
    omp.terminator
  }
}


[DEBUG]compare reading and storeOp
[DEBUG] memref.store %16, %10[] : memref<i32> 
[DEBUG] %16 = memref.load %10[] : memref<i32> 
[DEBUG]compare reading and storeOp
[DEBUG] memref.store %19, %10[] : memref<i32> 
[DEBUG] %19 = arith.addi %18, %9 overflow<nsw> : i32 
[DEBUG]Unexpected readOp: 
[DEBUG] %19 = arith.addi %18, %9 overflow<nsw> : i32 
[DEBUG]compare reading and storeOp
[DEBUG] memref.store %17, %11[] : memref<i32> 
[DEBUG] %17 = memref.load %11[] : memref<i32> 
[DEBUG]compare reading and storeOp
[DEBUG] memref.store %31, %11[] : memref<i32> 
[DEBUG] %31 = arith.addi %30, %9 overflow<nsw> : i32 
[DEBUG]Unexpected readOp: 
[DEBUG] %31 = arith.addi %30, %9 overflow<nsw> : i32 
[DEBUG]compare reading and storeOp
[DEBUG] memref.store %29, %15[%25, %23] : memref<3x4xf64> 
[DEBUG] %29 = arith.addf %27, %28 fastmath<contract> : f64 
[DEBUG]Unexpected readOp: 
[DEBUG] %29 = arith.addf %27, %28 fastmath<contract> : f64 
[DEBUG]Dependency chain relies on LoadOp!
[DEBUG]compare reading and storeOp
[DEBUG] memref.store %19, %11[] : memref<i32> 
[DEBUG] %19 = arith.index_cast %arg15 : index to i32 
[DEBUG]Unexpected readOp: 
[DEBUG] %19 = arith.index_cast %arg15 : index to i32 
[DEBUG]compare reading and storeOp
[DEBUG] memref.store %17, %10[] : memref<i32> 
[DEBUG] %17 = arith.index_cast %arg14 : index to i32 
[DEBUG]Unexpected readOp: 
[DEBUG] %17 = arith.index_cast %arg14 : index to i32 
[DEBUG]compare reading and storeOp
[DEBUG] memref.store %18, %10[] : memref<i32> 
[DEBUG] %18 = arith.addi %17, %9 overflow<nsw> : i32 
[DEBUG]Unexpected readOp: 
[DEBUG] %18 = arith.addi %17, %9 overflow<nsw> : i32 
[DEBUG]compare reading and storeOp
[DEBUG] memref.store %23, %10[] : memref<i32> 
[DEBUG] %23 = arith.addi %22, %9 overflow<nsw> : i32 
[DEBUG]Unexpected readOp: 
[DEBUG] %23 = arith.addi %22, %9 overflow<nsw> : i32 
[DEBUG]compare reading and storeOp
[DEBUG] memref.store %28, %11[] : memref<i32> 
[DEBUG] %28 = arith.addi %19, %9 overflow<nsw> : i32 
[DEBUG]Unexpected readOp: 
[DEBUG] %28 = arith.addi %19, %9 overflow<nsw> : i32 
[DEBUG]compare reading and storeOp
[DEBUG] memref.store %25, %11[] : memref<i32> 
[DEBUG] %25 = arith.addi %24, %9 overflow<nsw> : i32 
[DEBUG]Unexpected readOp: 
[DEBUG] %25 = arith.addi %24, %9 overflow<nsw> : i32 
[DEBUG]compare reading and storeOp
[DEBUG] memref.store %28, %11[] : memref<i32> 
[DEBUG] %28 = arith.addi %27, %9 overflow<nsw> : i32 
[DEBUG]Unexpected readOp: 
[DEBUG] %28 = arith.addi %27, %9 overflow<nsw> : i32 
[DEBUG]compare reading and storeOp
[DEBUG] memref.store %28, %15[%24, %22] : memref<3x4xf64> 
[DEBUG] %28 = arith.addf %26, %27 fastmath<contract> : f64 
[DEBUG]Unexpected readOp: 
[DEBUG] %28 = arith.addf %26, %27 fastmath<contract> : f64 
[DEBUG]Dependency chain relies on LoadOp!
[DEBUG]compare reading and storeOp
[DEBUG] memref.store %18, %10[] : memref<i32> 
[DEBUG] %18 = arith.addi %9, %c3_i32 overflow<nsw> : i32 
[DEBUG]Unexpected readOp: 
[DEBUG] %18 = arith.addi %9, %c3_i32 overflow<nsw> : i32 
[DEBUG]compare reading and storeOp
[DEBUG] memref.store %17, %11[] : memref<i32> 
[DEBUG] %17 = arith.addi %9, %c4_i32 overflow<nsw> : i32 
[DEBUG]Unexpected readOp: 
[DEBUG] %17 = arith.addi %9, %c4_i32 overflow<nsw> : i32 
[DEBUG]compare reading and storeOp
[DEBUG] memref.store %28, %15[%24, %22] : memref<3x4xf64> 
[DEBUG] %28 = arith.addf %26, %27 fastmath<contract> : f64 
[DEBUG]Unexpected readOp: 
[DEBUG] %28 = arith.addf %26, %27 fastmath<contract> : f64 
[DEBUG]Dependency chain relies on LoadOp!
// -----// IR Dump After {anonymous}::OptimizeMemOpsPass (jforce-optimize-mem-ops) //----- //
func.func @kernel(%arg0: !fir.ref<i32> {jit.arg_type = 1 : ui32, jit.literal_val = 0 : i64}, %arg1: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 3 : i64}, %arg2: !fir.ref<i32> {jit.arg_type = 1 : ui32, jit.literal_val = 0 : i64}, %arg3: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 4 : i64}, %arg4: !fir.ref<!fir.array<4x3xf64>> {jit.arg_type = 1 : ui32}, %arg5: !fir.ref<!fir.array<4x3xf64>> {jit.arg_type = 1 : ui32}, %arg6: !fir.ref<!fir.array<4x3xf64>> {jit.arg_type = 1 : ui32}, %arg7: !fir.ref<f64> {jit.arg_type = 1 : ui32, jit.literal_val = 4619567317775286272 : i64}, %arg8: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 3 : i64}, %arg9: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 4 : i64}, %arg10: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 3 : i64}, %arg11: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 4 : i64}, %arg12: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 3 : i64}, %arg13: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 4 : i64}) {
  %c3_i32 = arith.constant 3 : i32
  %c4_i32 = arith.constant 4 : i32
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %c1 = arith.constant 1 : index
  %0 = fir.declare %arg0 {uniq_name = "_QFFrun_benchmarkEj"} : (!fir.ref<i32>) -> !fir.ref<i32>
  %1 = fir.declare %arg1 {uniq_name = "_QFFrun_benchmarkEm"} : (!fir.ref<i32>) -> !fir.ref<i32>
  %2 = fir.declare %arg2 {uniq_name = "_QFFrun_benchmarkEi"} : (!fir.ref<i32>) -> !fir.ref<i32>
  %3 = fir.declare %arg3 {uniq_name = "_QFFrun_benchmarkEn"} : (!fir.ref<i32>) -> !fir.ref<i32>
  %4 = fir.shape %c4, %c3 : (index, index) -> !fir.shape<2>
  %5 = fir.declare %arg4(%4) {uniq_name = "_QFFrun_benchmarkEz"} : (!fir.ref<!fir.array<4x3xf64>>, !fir.shape<2>) -> !fir.ref<!fir.array<4x3xf64>>
  %6 = fir.declare %arg5(%4) {uniq_name = "_QFFrun_benchmarkEx"} : (!fir.ref<!fir.array<4x3xf64>>, !fir.shape<2>) -> !fir.ref<!fir.array<4x3xf64>>
  %7 = fir.declare %arg6(%4) {uniq_name = "_QFFrun_benchmarkEy"} : (!fir.ref<!fir.array<4x3xf64>>, !fir.shape<2>) -> !fir.ref<!fir.array<4x3xf64>>
  %8 = fir.declare %arg7 {uniq_name = "_QFFrun_benchmarkEa"} : (!fir.ref<f64>) -> !fir.ref<f64>
  %9 = fir.convert %c1 : (index) -> i32
  %10 = fir.convert %0 : (!fir.ref<i32>) -> memref<i32>
  %11 = fir.convert %2 : (!fir.ref<i32>) -> memref<i32>
  %12 = fir.convert %8 : (!fir.ref<f64>) -> memref<f64>
  %13 = fir.convert %6 : (!fir.ref<!fir.array<4x3xf64>>) -> memref<3x4xf64>
  %14 = fir.convert %7 : (!fir.ref<!fir.array<4x3xf64>>) -> memref<3x4xf64>
  %15 = fir.convert %5 : (!fir.ref<!fir.array<4x3xf64>>) -> memref<3x4xf64>
  %16 = memref.load %12[] : memref<f64>
  affine.for %arg14 = 1 to 4 {
    %19 = arith.index_cast %arg14 : index to i32
    affine.for %arg15 = 1 to 5 {
      %20 = arith.index_cast %arg15 : index to i32
      %21 = arith.index_cast %20 : i32 to index
      %22 = arith.subi %21, %c1 : index
      %23 = arith.index_cast %19 : i32 to index
      %24 = arith.subi %23, %c1 : index
      %25 = memref.load %13[%24, %22] : memref<3x4xf64>
      %26 = arith.mulf %16, %25 fastmath<contract> : f64
      %27 = memref.load %14[%24, %22] : memref<3x4xf64>
      %28 = arith.addf %26, %27 fastmath<contract> : f64
      memref.store %28, %15[%24, %22] : memref<3x4xf64>
    }
  }
  %17 = arith.addi %9, %c4_i32 overflow<nsw> : i32
  memref.store %17, %11[] : memref<i32>
  %18 = arith.addi %9, %c3_i32 overflow<nsw> : i32
  memref.store %18, %10[] : memref<i32>
  omp.terminator
}

// -----// IR Dump After CanonicalizerPass (canonicalize) //----- //
module {
  func.func @kernel(%arg0: !fir.ref<i32> {jit.arg_type = 1 : ui32, jit.literal_val = 0 : i64}, %arg1: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 3 : i64}, %arg2: !fir.ref<i32> {jit.arg_type = 1 : ui32, jit.literal_val = 0 : i64}, %arg3: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 4 : i64}, %arg4: !fir.ref<!fir.array<4x3xf64>> {jit.arg_type = 1 : ui32}, %arg5: !fir.ref<!fir.array<4x3xf64>> {jit.arg_type = 1 : ui32}, %arg6: !fir.ref<!fir.array<4x3xf64>> {jit.arg_type = 1 : ui32}, %arg7: !fir.ref<f64> {jit.arg_type = 1 : ui32, jit.literal_val = 4619567317775286272 : i64}, %arg8: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 3 : i64}, %arg9: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 4 : i64}, %arg10: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 3 : i64}, %arg11: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 4 : i64}, %arg12: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 3 : i64}, %arg13: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 4 : i64}) {
    %c3_i32 = arith.constant 3 : i32
    %c4_i32 = arith.constant 4 : i32
    %c3 = arith.constant 3 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %0 = fir.declare %arg0 {uniq_name = "_QFFrun_benchmarkEj"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %1 = fir.declare %arg1 {uniq_name = "_QFFrun_benchmarkEm"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %2 = fir.declare %arg2 {uniq_name = "_QFFrun_benchmarkEi"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %3 = fir.declare %arg3 {uniq_name = "_QFFrun_benchmarkEn"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %4 = fir.shape %c4, %c3 : (index, index) -> !fir.shape<2>
    %5 = fir.declare %arg4(%4) {uniq_name = "_QFFrun_benchmarkEz"} : (!fir.ref<!fir.array<4x3xf64>>, !fir.shape<2>) -> !fir.ref<!fir.array<4x3xf64>>
    %6 = fir.declare %arg5(%4) {uniq_name = "_QFFrun_benchmarkEx"} : (!fir.ref<!fir.array<4x3xf64>>, !fir.shape<2>) -> !fir.ref<!fir.array<4x3xf64>>
    %7 = fir.declare %arg6(%4) {uniq_name = "_QFFrun_benchmarkEy"} : (!fir.ref<!fir.array<4x3xf64>>, !fir.shape<2>) -> !fir.ref<!fir.array<4x3xf64>>
    %8 = fir.declare %arg7 {uniq_name = "_QFFrun_benchmarkEa"} : (!fir.ref<f64>) -> !fir.ref<f64>
    %9 = fir.convert %c1 : (index) -> i32
    %10 = fir.convert %0 : (!fir.ref<i32>) -> memref<i32>
    %11 = fir.convert %2 : (!fir.ref<i32>) -> memref<i32>
    %12 = fir.convert %8 : (!fir.ref<f64>) -> memref<f64>
    %13 = fir.convert %6 : (!fir.ref<!fir.array<4x3xf64>>) -> memref<3x4xf64>
    %14 = fir.convert %7 : (!fir.ref<!fir.array<4x3xf64>>) -> memref<3x4xf64>
    %15 = fir.convert %5 : (!fir.ref<!fir.array<4x3xf64>>) -> memref<3x4xf64>
    %16 = memref.load %12[] : memref<f64>
    affine.for %arg14 = 1 to 4 {
      %19 = arith.index_cast %arg14 : index to i32
      affine.for %arg15 = 1 to 5 {
        %20 = arith.index_cast %arg15 : index to i32
        %21 = arith.index_cast %20 : i32 to index
        %22 = arith.subi %21, %c1 : index
        %23 = arith.index_cast %19 : i32 to index
        %24 = arith.subi %23, %c1 : index
        %25 = memref.load %13[%24, %22] : memref<3x4xf64>
        %26 = arith.mulf %16, %25 fastmath<contract> : f64
        %27 = memref.load %14[%24, %22] : memref<3x4xf64>
        %28 = arith.addf %26, %27 fastmath<contract> : f64
        memref.store %28, %15[%24, %22] : memref<3x4xf64>
      }
    }
    %17 = arith.addi %9, %c4_i32 overflow<nsw> : i32
    memref.store %17, %11[] : memref<i32>
    %18 = arith.addi %9, %c3_i32 overflow<nsw> : i32
    memref.store %18, %10[] : memref<i32>
    omp.terminator
  }
}


// -----// IR Dump After {anonymous}::LoopSinkingPass (jforce-loop-sink) //----- //
func.func @kernel(%arg0: !fir.ref<i32> {jit.arg_type = 1 : ui32, jit.literal_val = 0 : i64}, %arg1: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 3 : i64}, %arg2: !fir.ref<i32> {jit.arg_type = 1 : ui32, jit.literal_val = 0 : i64}, %arg3: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 4 : i64}, %arg4: !fir.ref<!fir.array<4x3xf64>> {jit.arg_type = 1 : ui32}, %arg5: !fir.ref<!fir.array<4x3xf64>> {jit.arg_type = 1 : ui32}, %arg6: !fir.ref<!fir.array<4x3xf64>> {jit.arg_type = 1 : ui32}, %arg7: !fir.ref<f64> {jit.arg_type = 1 : ui32, jit.literal_val = 4619567317775286272 : i64}, %arg8: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 3 : i64}, %arg9: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 4 : i64}, %arg10: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 3 : i64}, %arg11: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 4 : i64}, %arg12: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 3 : i64}, %arg13: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 4 : i64}) {
  %c3_i32 = arith.constant 3 : i32
  %c4_i32 = arith.constant 4 : i32
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %c1 = arith.constant 1 : index
  %0 = fir.declare %arg0 {uniq_name = "_QFFrun_benchmarkEj"} : (!fir.ref<i32>) -> !fir.ref<i32>
  %1 = fir.declare %arg1 {uniq_name = "_QFFrun_benchmarkEm"} : (!fir.ref<i32>) -> !fir.ref<i32>
  %2 = fir.declare %arg2 {uniq_name = "_QFFrun_benchmarkEi"} : (!fir.ref<i32>) -> !fir.ref<i32>
  %3 = fir.declare %arg3 {uniq_name = "_QFFrun_benchmarkEn"} : (!fir.ref<i32>) -> !fir.ref<i32>
  %4 = fir.shape %c4, %c3 : (index, index) -> !fir.shape<2>
  %5 = fir.declare %arg4(%4) {uniq_name = "_QFFrun_benchmarkEz"} : (!fir.ref<!fir.array<4x3xf64>>, !fir.shape<2>) -> !fir.ref<!fir.array<4x3xf64>>
  %6 = fir.declare %arg5(%4) {uniq_name = "_QFFrun_benchmarkEx"} : (!fir.ref<!fir.array<4x3xf64>>, !fir.shape<2>) -> !fir.ref<!fir.array<4x3xf64>>
  %7 = fir.declare %arg6(%4) {uniq_name = "_QFFrun_benchmarkEy"} : (!fir.ref<!fir.array<4x3xf64>>, !fir.shape<2>) -> !fir.ref<!fir.array<4x3xf64>>
  %8 = fir.declare %arg7 {uniq_name = "_QFFrun_benchmarkEa"} : (!fir.ref<f64>) -> !fir.ref<f64>
  %9 = fir.convert %c1 : (index) -> i32
  %10 = fir.convert %0 : (!fir.ref<i32>) -> memref<i32>
  %11 = fir.convert %2 : (!fir.ref<i32>) -> memref<i32>
  %12 = fir.convert %8 : (!fir.ref<f64>) -> memref<f64>
  %13 = fir.convert %6 : (!fir.ref<!fir.array<4x3xf64>>) -> memref<3x4xf64>
  %14 = fir.convert %7 : (!fir.ref<!fir.array<4x3xf64>>) -> memref<3x4xf64>
  %15 = fir.convert %5 : (!fir.ref<!fir.array<4x3xf64>>) -> memref<3x4xf64>
  %16 = memref.load %12[] : memref<f64>
  affine.for %arg14 = 1 to 4 {
    affine.for %arg15 = 1 to 5 {
      %19 = arith.index_cast %arg14 : index to i32
      %20 = arith.index_cast %arg15 : index to i32
      %21 = arith.index_cast %20 : i32 to index
      %22 = arith.subi %21, %c1 : index
      %23 = arith.index_cast %19 : i32 to index
      %24 = arith.subi %23, %c1 : index
      %25 = memref.load %13[%24, %22] : memref<3x4xf64>
      %26 = arith.mulf %16, %25 fastmath<contract> : f64
      %27 = memref.load %14[%24, %22] : memref<3x4xf64>
      %28 = arith.addf %26, %27 fastmath<contract> : f64
      memref.store %28, %15[%24, %22] : memref<3x4xf64>
    }
  }
  %17 = arith.addi %9, %c4_i32 overflow<nsw> : i32
  memref.store %17, %11[] : memref<i32>
  %18 = arith.addi %9, %c3_i32 overflow<nsw> : i32
  memref.store %18, %10[] : memref<i32>
  omp.terminator
}

// -----// IR Dump After {anonymous}::AffineCFGPass (enzyme-affinecfg) //----- //
module {
  func.func @kernel(%arg0: !fir.ref<i32> {jit.arg_type = 1 : ui32, jit.literal_val = 0 : i64}, %arg1: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 3 : i64}, %arg2: !fir.ref<i32> {jit.arg_type = 1 : ui32, jit.literal_val = 0 : i64}, %arg3: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 4 : i64}, %arg4: !fir.ref<!fir.array<4x3xf64>> {jit.arg_type = 1 : ui32}, %arg5: !fir.ref<!fir.array<4x3xf64>> {jit.arg_type = 1 : ui32}, %arg6: !fir.ref<!fir.array<4x3xf64>> {jit.arg_type = 1 : ui32}, %arg7: !fir.ref<f64> {jit.arg_type = 1 : ui32, jit.literal_val = 4619567317775286272 : i64}, %arg8: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 3 : i64}, %arg9: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 4 : i64}, %arg10: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 3 : i64}, %arg11: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 4 : i64}, %arg12: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 3 : i64}, %arg13: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 4 : i64}) {
    %c3_i32 = arith.constant 3 : i32
    %c4_i32 = arith.constant 4 : i32
    %c3 = arith.constant 3 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %0 = fir.declare %arg0 {uniq_name = "_QFFrun_benchmarkEj"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %1 = fir.declare %arg1 {uniq_name = "_QFFrun_benchmarkEm"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %2 = fir.declare %arg2 {uniq_name = "_QFFrun_benchmarkEi"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %3 = fir.declare %arg3 {uniq_name = "_QFFrun_benchmarkEn"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %4 = fir.shape %c4, %c3 : (index, index) -> !fir.shape<2>
    %5 = fir.declare %arg4(%4) {uniq_name = "_QFFrun_benchmarkEz"} : (!fir.ref<!fir.array<4x3xf64>>, !fir.shape<2>) -> !fir.ref<!fir.array<4x3xf64>>
    %6 = fir.declare %arg5(%4) {uniq_name = "_QFFrun_benchmarkEx"} : (!fir.ref<!fir.array<4x3xf64>>, !fir.shape<2>) -> !fir.ref<!fir.array<4x3xf64>>
    %7 = fir.declare %arg6(%4) {uniq_name = "_QFFrun_benchmarkEy"} : (!fir.ref<!fir.array<4x3xf64>>, !fir.shape<2>) -> !fir.ref<!fir.array<4x3xf64>>
    %8 = fir.declare %arg7 {uniq_name = "_QFFrun_benchmarkEa"} : (!fir.ref<f64>) -> !fir.ref<f64>
    %9 = fir.convert %c1 : (index) -> i32
    %10 = fir.convert %0 : (!fir.ref<i32>) -> memref<i32>
    %11 = fir.convert %2 : (!fir.ref<i32>) -> memref<i32>
    %12 = fir.convert %8 : (!fir.ref<f64>) -> memref<f64>
    %13 = fir.convert %6 : (!fir.ref<!fir.array<4x3xf64>>) -> memref<3x4xf64>
    %14 = fir.convert %7 : (!fir.ref<!fir.array<4x3xf64>>) -> memref<3x4xf64>
    %15 = fir.convert %5 : (!fir.ref<!fir.array<4x3xf64>>) -> memref<3x4xf64>
    %16 = affine.load %12[] : memref<f64>
    affine.parallel (%arg14, %arg15) = (1, 1) to (4, 5) {
      %19 = affine.load %13[%arg14 - 1, %arg15 - 1] : memref<3x4xf64>
      %20 = arith.mulf %16, %19 fastmath<contract> : f64
      %21 = affine.load %14[%arg14 - 1, %arg15 - 1] : memref<3x4xf64>
      %22 = arith.addf %20, %21 fastmath<contract> : f64
      affine.store %22, %15[%arg14 - 1, %arg15 - 1] : memref<3x4xf64>
    }
    %17 = arith.addi %9, %c4_i32 overflow<nsw> : i32
    affine.store %17, %11[] : memref<i32>
    %18 = arith.addi %9, %c3_i32 overflow<nsw> : i32
    affine.store %18, %10[] : memref<i32>
    omp.terminator
  }
}


module {
  func.func @outlined_affinefor_102159583063760(%arg0: memref<3x4xf64>, %arg1: memref<f64>, %arg2: memref<3x4xf64>, %arg3: memref<3x4xf64>) {
    %0 = affine.load %arg1[] : memref<f64>
    affine.parallel (%arg4, %arg5) = (1, 1) to (4, 5) {
      %1 = affine.load %arg0[%arg4 - 1, %arg5 - 1] : memref<3x4xf64>
      %2 = arith.mulf %0, %1 fastmath<contract> : f64
      %3 = affine.load %arg2[%arg4 - 1, %arg5 - 1] : memref<3x4xf64>
      %4 = arith.addf %2, %3 fastmath<contract> : f64
      affine.store %4, %arg3[%arg4 - 1, %arg5 - 1] : memref<3x4xf64>
    }
    return
  }
  func.func @kernel(%arg0: !fir.ref<i32> {jit.arg_type = 1 : ui32, jit.literal_val = 0 : i64}, %arg1: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 3 : i64}, %arg2: !fir.ref<i32> {jit.arg_type = 1 : ui32, jit.literal_val = 0 : i64}, %arg3: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 4 : i64}, %arg4: !fir.ref<!fir.array<4x3xf64>> {jit.arg_type = 1 : ui32}, %arg5: !fir.ref<!fir.array<4x3xf64>> {jit.arg_type = 1 : ui32}, %arg6: !fir.ref<!fir.array<4x3xf64>> {jit.arg_type = 1 : ui32}, %arg7: !fir.ref<f64> {jit.arg_type = 1 : ui32, jit.literal_val = 4619567317775286272 : i64}, %arg8: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 3 : i64}, %arg9: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 4 : i64}, %arg10: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 3 : i64}, %arg11: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 4 : i64}, %arg12: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 3 : i64}, %arg13: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 4 : i64}) {
    %c3_i32 = arith.constant 3 : i32
    %c4_i32 = arith.constant 4 : i32
    %c3 = arith.constant 3 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %0 = fir.declare %arg0 {uniq_name = "_QFFrun_benchmarkEj"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %1 = fir.declare %arg1 {uniq_name = "_QFFrun_benchmarkEm"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %2 = fir.declare %arg2 {uniq_name = "_QFFrun_benchmarkEi"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %3 = fir.declare %arg3 {uniq_name = "_QFFrun_benchmarkEn"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %4 = fir.shape %c4, %c3 : (index, index) -> !fir.shape<2>
    %5 = fir.declare %arg4(%4) {uniq_name = "_QFFrun_benchmarkEz"} : (!fir.ref<!fir.array<4x3xf64>>, !fir.shape<2>) -> !fir.ref<!fir.array<4x3xf64>>
    %6 = fir.declare %arg5(%4) {uniq_name = "_QFFrun_benchmarkEx"} : (!fir.ref<!fir.array<4x3xf64>>, !fir.shape<2>) -> !fir.ref<!fir.array<4x3xf64>>
    %7 = fir.declare %arg6(%4) {uniq_name = "_QFFrun_benchmarkEy"} : (!fir.ref<!fir.array<4x3xf64>>, !fir.shape<2>) -> !fir.ref<!fir.array<4x3xf64>>
    %8 = fir.declare %arg7 {uniq_name = "_QFFrun_benchmarkEa"} : (!fir.ref<f64>) -> !fir.ref<f64>
    %9 = fir.convert %c1 : (index) -> i32
    %10 = fir.convert %0 : (!fir.ref<i32>) -> memref<i32>
    %11 = fir.convert %2 : (!fir.ref<i32>) -> memref<i32>
    %12 = fir.convert %8 : (!fir.ref<f64>) -> memref<f64>
    %13 = fir.convert %6 : (!fir.ref<!fir.array<4x3xf64>>) -> memref<3x4xf64>
    %14 = fir.convert %7 : (!fir.ref<!fir.array<4x3xf64>>) -> memref<3x4xf64>
    %15 = fir.convert %5 : (!fir.ref<!fir.array<4x3xf64>>) -> memref<3x4xf64>
    %16 = affine.load %12[] : memref<f64>
    %alloca = memref.alloca() : memref<f64>
    memref.store %16, %alloca[] : memref<f64>
    call @outlined_affinefor_102159583063760(%13, %alloca, %14, %15) : (memref<3x4xf64>, memref<f64>, memref<3x4xf64>, memref<3x4xf64>) -> ()
    %17 = arith.addi %9, %c4_i32 overflow<nsw> : i32
    affine.store %17, %11[] : memref<i32>
    %18 = arith.addi %9, %c3_i32 overflow<nsw> : i32
    affine.store %18, %10[] : memref<i32>
    omp.terminator
  }
}
// -----// IR Dump After {anonymous}::OutlineAffinePass (jforce-outline-affine) //----- //
module {
  func.func @outlined_affinefor_102159583063760(%arg0: memref<3x4xf64>, %arg1: memref<f64>, %arg2: memref<3x4xf64>, %arg3: memref<3x4xf64>) {
    %0 = affine.load %arg1[] : memref<f64>
    affine.parallel (%arg4, %arg5) = (1, 1) to (4, 5) {
      %1 = affine.load %arg0[%arg4 - 1, %arg5 - 1] : memref<3x4xf64>
      %2 = arith.mulf %0, %1 fastmath<contract> : f64
      %3 = affine.load %arg2[%arg4 - 1, %arg5 - 1] : memref<3x4xf64>
      %4 = arith.addf %2, %3 fastmath<contract> : f64
      affine.store %4, %arg3[%arg4 - 1, %arg5 - 1] : memref<3x4xf64>
    }
    return
  }
  func.func @kernel(%arg0: !fir.ref<i32> {jit.arg_type = 1 : ui32, jit.literal_val = 0 : i64}, %arg1: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 3 : i64}, %arg2: !fir.ref<i32> {jit.arg_type = 1 : ui32, jit.literal_val = 0 : i64}, %arg3: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 4 : i64}, %arg4: !fir.ref<!fir.array<4x3xf64>> {jit.arg_type = 1 : ui32}, %arg5: !fir.ref<!fir.array<4x3xf64>> {jit.arg_type = 1 : ui32}, %arg6: !fir.ref<!fir.array<4x3xf64>> {jit.arg_type = 1 : ui32}, %arg7: !fir.ref<f64> {jit.arg_type = 1 : ui32, jit.literal_val = 4619567317775286272 : i64}, %arg8: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 3 : i64}, %arg9: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 4 : i64}, %arg10: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 3 : i64}, %arg11: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 4 : i64}, %arg12: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 3 : i64}, %arg13: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 4 : i64}) {
    %c3_i32 = arith.constant 3 : i32
    %c4_i32 = arith.constant 4 : i32
    %c3 = arith.constant 3 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %0 = fir.declare %arg0 {uniq_name = "_QFFrun_benchmarkEj"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %1 = fir.declare %arg1 {uniq_name = "_QFFrun_benchmarkEm"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %2 = fir.declare %arg2 {uniq_name = "_QFFrun_benchmarkEi"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %3 = fir.declare %arg3 {uniq_name = "_QFFrun_benchmarkEn"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %4 = fir.shape %c4, %c3 : (index, index) -> !fir.shape<2>
    %5 = fir.declare %arg4(%4) {uniq_name = "_QFFrun_benchmarkEz"} : (!fir.ref<!fir.array<4x3xf64>>, !fir.shape<2>) -> !fir.ref<!fir.array<4x3xf64>>
    %6 = fir.declare %arg5(%4) {uniq_name = "_QFFrun_benchmarkEx"} : (!fir.ref<!fir.array<4x3xf64>>, !fir.shape<2>) -> !fir.ref<!fir.array<4x3xf64>>
    %7 = fir.declare %arg6(%4) {uniq_name = "_QFFrun_benchmarkEy"} : (!fir.ref<!fir.array<4x3xf64>>, !fir.shape<2>) -> !fir.ref<!fir.array<4x3xf64>>
    %8 = fir.declare %arg7 {uniq_name = "_QFFrun_benchmarkEa"} : (!fir.ref<f64>) -> !fir.ref<f64>
    %9 = fir.convert %c1 : (index) -> i32
    %10 = fir.convert %0 : (!fir.ref<i32>) -> memref<i32>
    %11 = fir.convert %2 : (!fir.ref<i32>) -> memref<i32>
    %12 = fir.convert %8 : (!fir.ref<f64>) -> memref<f64>
    %13 = fir.convert %6 : (!fir.ref<!fir.array<4x3xf64>>) -> memref<3x4xf64>
    %14 = fir.convert %7 : (!fir.ref<!fir.array<4x3xf64>>) -> memref<3x4xf64>
    %15 = fir.convert %5 : (!fir.ref<!fir.array<4x3xf64>>) -> memref<3x4xf64>
    %16 = affine.load %12[] : memref<f64>
    %alloca = memref.alloca() : memref<f64>
    memref.store %16, %alloca[] : memref<f64>
    call @outlined_affinefor_102159583063760(%13, %alloca, %14, %15) : (memref<3x4xf64>, memref<f64>, memref<3x4xf64>, memref<3x4xf64>) -> ()
    %17 = arith.addi %9, %c4_i32 overflow<nsw> : i32
    affine.store %17, %11[] : memref<i32>
    %18 = arith.addi %9, %c3_i32 overflow<nsw> : i32
    affine.store %18, %10[] : memref<i32>
    omp.terminator
  }
}


// -----// IR Dump After {anonymous}::AffineToStableHLORaisingPass (enzyme-affine-to-stablehlo) //----- //
func.func @kernel(%arg0: !fir.ref<i32> {jit.arg_type = 1 : ui32, jit.literal_val = 0 : i64}, %arg1: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 3 : i64}, %arg2: !fir.ref<i32> {jit.arg_type = 1 : ui32, jit.literal_val = 0 : i64}, %arg3: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 4 : i64}, %arg4: !fir.ref<!fir.array<4x3xf64>> {jit.arg_type = 1 : ui32}, %arg5: !fir.ref<!fir.array<4x3xf64>> {jit.arg_type = 1 : ui32}, %arg6: !fir.ref<!fir.array<4x3xf64>> {jit.arg_type = 1 : ui32}, %arg7: !fir.ref<f64> {jit.arg_type = 1 : ui32, jit.literal_val = 4619567317775286272 : i64}, %arg8: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 3 : i64}, %arg9: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 4 : i64}, %arg10: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 3 : i64}, %arg11: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 4 : i64}, %arg12: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 3 : i64}, %arg13: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 4 : i64}) {
  %c3_i32 = arith.constant 3 : i32
  %c4_i32 = arith.constant 4 : i32
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %c1 = arith.constant 1 : index
  %0 = fir.declare %arg0 {uniq_name = "_QFFrun_benchmarkEj"} : (!fir.ref<i32>) -> !fir.ref<i32>
  %1 = fir.declare %arg1 {uniq_name = "_QFFrun_benchmarkEm"} : (!fir.ref<i32>) -> !fir.ref<i32>
  %2 = fir.declare %arg2 {uniq_name = "_QFFrun_benchmarkEi"} : (!fir.ref<i32>) -> !fir.ref<i32>
  %3 = fir.declare %arg3 {uniq_name = "_QFFrun_benchmarkEn"} : (!fir.ref<i32>) -> !fir.ref<i32>
  %4 = fir.shape %c4, %c3 : (index, index) -> !fir.shape<2>
  %5 = fir.declare %arg4(%4) {uniq_name = "_QFFrun_benchmarkEz"} : (!fir.ref<!fir.array<4x3xf64>>, !fir.shape<2>) -> !fir.ref<!fir.array<4x3xf64>>
  %6 = fir.declare %arg5(%4) {uniq_name = "_QFFrun_benchmarkEx"} : (!fir.ref<!fir.array<4x3xf64>>, !fir.shape<2>) -> !fir.ref<!fir.array<4x3xf64>>
  %7 = fir.declare %arg6(%4) {uniq_name = "_QFFrun_benchmarkEy"} : (!fir.ref<!fir.array<4x3xf64>>, !fir.shape<2>) -> !fir.ref<!fir.array<4x3xf64>>
  %8 = fir.declare %arg7 {uniq_name = "_QFFrun_benchmarkEa"} : (!fir.ref<f64>) -> !fir.ref<f64>
  %9 = fir.convert %c1 : (index) -> i32
  %10 = fir.convert %0 : (!fir.ref<i32>) -> memref<i32>
  %11 = fir.convert %2 : (!fir.ref<i32>) -> memref<i32>
  %12 = fir.convert %8 : (!fir.ref<f64>) -> memref<f64>
  %13 = fir.convert %6 : (!fir.ref<!fir.array<4x3xf64>>) -> memref<3x4xf64>
  %14 = fir.convert %7 : (!fir.ref<!fir.array<4x3xf64>>) -> memref<3x4xf64>
  %15 = fir.convert %5 : (!fir.ref<!fir.array<4x3xf64>>) -> memref<3x4xf64>
  %16 = affine.load %12[] : memref<f64>
  %alloca = memref.alloca() : memref<f64>
  memref.store %16, %alloca[] : memref<f64>
  call @outlined_affinefor_102159583063760(%13, %alloca, %14, %15) : (memref<3x4xf64>, memref<f64>, memref<3x4xf64>, memref<3x4xf64>) -> ()
  %17 = arith.addi %9, %c4_i32 overflow<nsw> : i32
  affine.store %17, %11[] : memref<i32>
  %18 = arith.addi %9, %c3_i32 overflow<nsw> : i32
  affine.store %18, %10[] : memref<i32>
  omp.terminator
}

// -----// IR Dump After {anonymous}::AffineToStableHLORaisingPass (enzyme-affine-to-stablehlo) //----- //
func.func @outlined_affinefor_102159583063760(%arg0: memref<3x4xf64>, %arg1: memref<f64>, %arg2: memref<3x4xf64>, %arg3: memref<3x4xf64>) {
  %0 = affine.load %arg1[] : memref<f64>
  affine.parallel (%arg4, %arg5) = (1, 1) to (4, 5) {
    %1 = affine.load %arg0[%arg4 - 1, %arg5 - 1] : memref<3x4xf64>
    %2 = arith.mulf %0, %1 fastmath<contract> : f64
    %3 = affine.load %arg2[%arg4 - 1, %arg5 - 1] : memref<3x4xf64>
    %4 = arith.addf %2, %3 fastmath<contract> : f64
    affine.store %4, %arg3[%arg4 - 1, %arg5 - 1] : memref<3x4xf64>
  }
  return
}

// -----// IR Dump After CanonicalizerPass (canonicalize) //----- //
module {
  func.func @outlined_affinefor_102159583063760(%arg0: memref<3x4xf64>, %arg1: memref<f64>, %arg2: memref<3x4xf64>, %arg3: memref<3x4xf64>) {
    %0 = affine.load %arg1[] : memref<f64>
    affine.parallel (%arg4, %arg5) = (1, 1) to (4, 5) {
      %1 = affine.load %arg0[%arg4 - 1, %arg5 - 1] : memref<3x4xf64>
      %2 = arith.mulf %0, %1 fastmath<contract> : f64
      %3 = affine.load %arg2[%arg4 - 1, %arg5 - 1] : memref<3x4xf64>
      %4 = arith.addf %2, %3 fastmath<contract> : f64
      affine.store %4, %arg3[%arg4 - 1, %arg5 - 1] : memref<3x4xf64>
    }
    return
  }
  func.func @kernel(%arg0: !fir.ref<i32> {jit.arg_type = 1 : ui32, jit.literal_val = 0 : i64}, %arg1: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 3 : i64}, %arg2: !fir.ref<i32> {jit.arg_type = 1 : ui32, jit.literal_val = 0 : i64}, %arg3: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 4 : i64}, %arg4: !fir.ref<!fir.array<4x3xf64>> {jit.arg_type = 1 : ui32}, %arg5: !fir.ref<!fir.array<4x3xf64>> {jit.arg_type = 1 : ui32}, %arg6: !fir.ref<!fir.array<4x3xf64>> {jit.arg_type = 1 : ui32}, %arg7: !fir.ref<f64> {jit.arg_type = 1 : ui32, jit.literal_val = 4619567317775286272 : i64}, %arg8: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 3 : i64}, %arg9: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 4 : i64}, %arg10: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 3 : i64}, %arg11: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 4 : i64}, %arg12: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 3 : i64}, %arg13: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 4 : i64}) {
    %c3_i32 = arith.constant 3 : i32
    %c4_i32 = arith.constant 4 : i32
    %c3 = arith.constant 3 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %0 = fir.declare %arg0 {uniq_name = "_QFFrun_benchmarkEj"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %1 = fir.declare %arg1 {uniq_name = "_QFFrun_benchmarkEm"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %2 = fir.declare %arg2 {uniq_name = "_QFFrun_benchmarkEi"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %3 = fir.declare %arg3 {uniq_name = "_QFFrun_benchmarkEn"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %4 = fir.shape %c4, %c3 : (index, index) -> !fir.shape<2>
    %5 = fir.declare %arg4(%4) {uniq_name = "_QFFrun_benchmarkEz"} : (!fir.ref<!fir.array<4x3xf64>>, !fir.shape<2>) -> !fir.ref<!fir.array<4x3xf64>>
    %6 = fir.declare %arg5(%4) {uniq_name = "_QFFrun_benchmarkEx"} : (!fir.ref<!fir.array<4x3xf64>>, !fir.shape<2>) -> !fir.ref<!fir.array<4x3xf64>>
    %7 = fir.declare %arg6(%4) {uniq_name = "_QFFrun_benchmarkEy"} : (!fir.ref<!fir.array<4x3xf64>>, !fir.shape<2>) -> !fir.ref<!fir.array<4x3xf64>>
    %8 = fir.declare %arg7 {uniq_name = "_QFFrun_benchmarkEa"} : (!fir.ref<f64>) -> !fir.ref<f64>
    %9 = fir.convert %c1 : (index) -> i32
    %10 = fir.convert %0 : (!fir.ref<i32>) -> memref<i32>
    %11 = fir.convert %2 : (!fir.ref<i32>) -> memref<i32>
    %12 = fir.convert %8 : (!fir.ref<f64>) -> memref<f64>
    %13 = fir.convert %6 : (!fir.ref<!fir.array<4x3xf64>>) -> memref<3x4xf64>
    %14 = fir.convert %7 : (!fir.ref<!fir.array<4x3xf64>>) -> memref<3x4xf64>
    %15 = fir.convert %5 : (!fir.ref<!fir.array<4x3xf64>>) -> memref<3x4xf64>
    %16 = affine.load %12[] : memref<f64>
    %alloca = memref.alloca() : memref<f64>
    memref.store %16, %alloca[] : memref<f64>
    call @outlined_affinefor_102159583063760(%13, %alloca, %14, %15) : (memref<3x4xf64>, memref<f64>, memref<3x4xf64>, memref<3x4xf64>) -> ()
    %17 = arith.addi %9, %c4_i32 overflow<nsw> : i32
    affine.store %17, %11[] : memref<i32>
    %18 = arith.addi %9, %c3_i32 overflow<nsw> : i32
    affine.store %18, %10[] : memref<i32>
    omp.terminator
  }
  func.func private @outlined_affinefor_102159583063760_raised(%arg0: tensor<3x4xf64>, %arg1: tensor<f64>, %arg2: tensor<3x4xf64>, %arg3: tensor<3x4xf64>) -> (tensor<3x4xf64>, tensor<f64>, tensor<3x4xf64>, tensor<3x4xf64>) {
    %c = stablehlo.constant dense<0> : tensor<i64>
    %0 = stablehlo.reshape %arg1 : (tensor<f64>) -> tensor<f64>
    %1 = stablehlo.reshape %arg0 : (tensor<3x4xf64>) -> tensor<3x4xf64>
    %2 = stablehlo.broadcast_in_dim %0, dims = [] : (tensor<f64>) -> tensor<3x4xf64>
    %3 = arith.mulf %2, %1 fastmath<contract> : tensor<3x4xf64>
    %4 = stablehlo.reshape %arg2 : (tensor<3x4xf64>) -> tensor<3x4xf64>
    %5 = arith.addf %3, %4 fastmath<contract> : tensor<3x4xf64>
    %6 = stablehlo.broadcast_in_dim %5, dims = [0, 1] : (tensor<3x4xf64>) -> tensor<3x4xf64>
    %7 = stablehlo.dynamic_update_slice %arg3, %6, %c, %c : (tensor<3x4xf64>, tensor<3x4xf64>, tensor<i64>, tensor<i64>) -> tensor<3x4xf64>
    return %arg0, %arg1, %arg2, %7 : tensor<3x4xf64>, tensor<f64>, tensor<3x4xf64>, tensor<3x4xf64>
  }
}


