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
func.func @kernel(%arg0: !fir.ref<i32> {jit.arg_type = 1 : ui32, jit.literal_val = 0 : i64}, %arg1: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg2: !fir.ref<i32> {jit.arg_type = 1 : ui32, jit.literal_val = 0 : i64}, %arg3: !fir.ref<!fir.array<?x?xf64>> {jit.arg_type = 1 : ui32}, %arg4: !fir.ref<!fir.array<?x?xf64>> {jit.arg_type = 1 : ui32}, %arg5: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg6: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg7: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg8: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}) {
  %0 = fir.load %arg8 : !fir.ref<i32>
  %1 = fir.load %arg7 : !fir.ref<i32>
  %2 = fir.load %arg6 : !fir.ref<i32>
  %3 = fir.load %arg5 : !fir.ref<i32>
  %4 = fir.convert %3 : (i32) -> i64
  %5 = fir.convert %2 : (i32) -> i64
  %6 = fir.convert %1 : (i32) -> i64
  %7 = fir.convert %0 : (i32) -> i64
  %c0 = arith.constant 0 : index
  %8 = fir.convert %7 : (i64) -> index
  %9 = arith.cmpi sgt, %8, %c0 : index
  %c0_0 = arith.constant 0 : index
  %10 = fir.convert %6 : (i64) -> index
  %11 = arith.cmpi sgt, %10, %c0_0 : index
  %c0_1 = arith.constant 0 : index
  %12 = fir.convert %5 : (i64) -> index
  %13 = arith.cmpi sgt, %12, %c0_1 : index
  %c0_2 = arith.constant 0 : index
  %14 = fir.convert %4 : (i64) -> index
  %15 = arith.cmpi sgt, %14, %c0_2 : index
  %16 = arith.select %15, %14, %c0_2 : index
  %17 = arith.select %13, %12, %c0_1 : index
  %18 = arith.select %11, %10, %c0_0 : index
  %19 = arith.select %9, %8, %c0 : index
  %20:2 = hlfir.declare %arg0 {uniq_name = "_QFFrun_benchmarkEj"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %21:2 = hlfir.declare %arg1 {uniq_name = "_QFFrun_benchmarkEn"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %22:2 = hlfir.declare %arg2 {uniq_name = "_QFFrun_benchmarkEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %23 = fir.shape %19, %18 : (index, index) -> !fir.shape<2>
  %24:2 = hlfir.declare %arg3(%23) {uniq_name = "_QFFrun_benchmarkEy"} : (!fir.ref<!fir.array<?x?xf64>>, !fir.shape<2>) -> (!fir.box<!fir.array<?x?xf64>>, !fir.ref<!fir.array<?x?xf64>>)
  %25 = fir.shape %17, %16 : (index, index) -> !fir.shape<2>
  %26:2 = hlfir.declare %arg4(%25) {uniq_name = "_QFFrun_benchmarkEx"} : (!fir.ref<!fir.array<?x?xf64>>, !fir.shape<2>) -> (!fir.box<!fir.array<?x?xf64>>, !fir.ref<!fir.array<?x?xf64>>)
  %c2_i32 = arith.constant 2 : i32
  %27 = fir.convert %c2_i32 : (i32) -> index
  %28 = fir.load %21#0 : !fir.ref<i32>
  %c1_i32 = arith.constant 1 : i32
  %29 = arith.subi %28, %c1_i32 overflow<nsw> : i32
  %30 = fir.convert %29 : (i32) -> index
  %c1 = arith.constant 1 : index
  %31 = fir.convert %27 : (index) -> i32
  %32 = fir.do_loop %arg9 = %27 to %30 step %c1 iter_args(%arg10 = %31) -> (i32) {
    fir.store %arg10 to %20#0 : !fir.ref<i32>
    %c2_i32_3 = arith.constant 2 : i32
    %33 = fir.convert %c2_i32_3 : (i32) -> index
    %34 = fir.load %21#0 : !fir.ref<i32>
    %c1_i32_4 = arith.constant 1 : i32
    %35 = arith.subi %34, %c1_i32_4 overflow<nsw> : i32
    %36 = fir.convert %35 : (i32) -> index
    %c1_5 = arith.constant 1 : index
    %37 = fir.convert %33 : (index) -> i32
    %38 = fir.do_loop %arg11 = %33 to %36 step %c1_5 iter_args(%arg12 = %37) -> (i32) {
      fir.store %arg12 to %22#0 : !fir.ref<i32>
      %42 = fir.load %22#0 : !fir.ref<i32>
      %c1_i32_6 = arith.constant 1 : i32
      %43 = arith.subi %42, %c1_i32_6 overflow<nsw> : i32
      %44 = fir.convert %43 : (i32) -> i64
      %45 = fir.load %20#0 : !fir.ref<i32>
      %46 = fir.convert %45 : (i32) -> i64
      %47 = hlfir.designate %26#0 (%44, %46)  : (!fir.box<!fir.array<?x?xf64>>, i64, i64) -> !fir.ref<f64>
      %48 = fir.load %47 : !fir.ref<f64>
      %49 = fir.load %22#0 : !fir.ref<i32>
      %c1_i32_7 = arith.constant 1 : i32
      %50 = arith.addi %49, %c1_i32_7 overflow<nsw> : i32
      %51 = fir.convert %50 : (i32) -> i64
      %52 = fir.load %20#0 : !fir.ref<i32>
      %53 = fir.convert %52 : (i32) -> i64
      %54 = hlfir.designate %26#0 (%51, %53)  : (!fir.box<!fir.array<?x?xf64>>, i64, i64) -> !fir.ref<f64>
      %55 = fir.load %54 : !fir.ref<f64>
      %56 = arith.addf %48, %55 fastmath<contract> : f64
      %57 = fir.load %22#0 : !fir.ref<i32>
      %58 = fir.convert %57 : (i32) -> i64
      %59 = fir.load %20#0 : !fir.ref<i32>
      %c1_i32_8 = arith.constant 1 : i32
      %60 = arith.subi %59, %c1_i32_8 overflow<nsw> : i32
      %61 = fir.convert %60 : (i32) -> i64
      %62 = hlfir.designate %26#0 (%58, %61)  : (!fir.box<!fir.array<?x?xf64>>, i64, i64) -> !fir.ref<f64>
      %63 = fir.load %62 : !fir.ref<f64>
      %64 = arith.addf %56, %63 fastmath<contract> : f64
      %65 = fir.load %22#0 : !fir.ref<i32>
      %66 = fir.convert %65 : (i32) -> i64
      %67 = fir.load %20#0 : !fir.ref<i32>
      %c1_i32_9 = arith.constant 1 : i32
      %68 = arith.addi %67, %c1_i32_9 overflow<nsw> : i32
      %69 = fir.convert %68 : (i32) -> i64
      %70 = hlfir.designate %26#0 (%66, %69)  : (!fir.box<!fir.array<?x?xf64>>, i64, i64) -> !fir.ref<f64>
      %71 = fir.load %70 : !fir.ref<f64>
      %72 = arith.addf %64, %71 fastmath<contract> : f64
      %73 = fir.load %22#0 : !fir.ref<i32>
      %74 = fir.convert %73 : (i32) -> i64
      %75 = fir.load %20#0 : !fir.ref<i32>
      %76 = fir.convert %75 : (i32) -> i64
      %77 = hlfir.designate %24#0 (%74, %76)  : (!fir.box<!fir.array<?x?xf64>>, i64, i64) -> !fir.ref<f64>
      hlfir.assign %72 to %77 : f64, !fir.ref<f64>
      %78 = fir.convert %c1_5 : (index) -> i32
      %79 = fir.load %22#0 : !fir.ref<i32>
      %80 = arith.addi %79, %78 overflow<nsw> : i32
      fir.result %80 : i32
    }
    fir.store %38 to %22#0 : !fir.ref<i32>
    %39 = fir.convert %c1 : (index) -> i32
    %40 = fir.load %20#0 : !fir.ref<i32>
    %41 = arith.addi %40, %39 overflow<nsw> : i32
    fir.result %41 : i32
  }
  fir.store %32 to %20#0 : !fir.ref<i32>
  omp.terminator
}

// -----// IR Dump After {anonymous}::PropagateConstantsPass (jforce-propagate-constants) //----- //
func.func @kernel(%arg0: !fir.ref<i32> {jit.arg_type = 1 : ui32, jit.literal_val = 0 : i64}, %arg1: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg2: !fir.ref<i32> {jit.arg_type = 1 : ui32, jit.literal_val = 0 : i64}, %arg3: !fir.ref<!fir.array<?x?xf64>> {jit.arg_type = 1 : ui32}, %arg4: !fir.ref<!fir.array<?x?xf64>> {jit.arg_type = 1 : ui32}, %arg5: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg6: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg7: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg8: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}) {
  %c1024_i32 = arith.constant 1024 : i32
  %c1024_i32_0 = arith.constant 1024 : i32
  %c1024_i32_1 = arith.constant 1024 : i32
  %c1024_i32_2 = arith.constant 1024 : i32
  %0 = fir.convert %c1024_i32_2 : (i32) -> i64
  %1 = fir.convert %c1024_i32_1 : (i32) -> i64
  %2 = fir.convert %c1024_i32_0 : (i32) -> i64
  %3 = fir.convert %c1024_i32 : (i32) -> i64
  %c0 = arith.constant 0 : index
  %4 = fir.convert %3 : (i64) -> index
  %5 = arith.cmpi sgt, %4, %c0 : index
  %c0_3 = arith.constant 0 : index
  %6 = fir.convert %2 : (i64) -> index
  %7 = arith.cmpi sgt, %6, %c0_3 : index
  %c0_4 = arith.constant 0 : index
  %8 = fir.convert %1 : (i64) -> index
  %9 = arith.cmpi sgt, %8, %c0_4 : index
  %c0_5 = arith.constant 0 : index
  %10 = fir.convert %0 : (i64) -> index
  %11 = arith.cmpi sgt, %10, %c0_5 : index
  %12 = arith.select %11, %10, %c0_5 : index
  %13 = arith.select %9, %8, %c0_4 : index
  %14 = arith.select %7, %6, %c0_3 : index
  %15 = arith.select %5, %4, %c0 : index
  %16:2 = hlfir.declare %arg0 {uniq_name = "_QFFrun_benchmarkEj"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %17:2 = hlfir.declare %arg1 {uniq_name = "_QFFrun_benchmarkEn"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %18:2 = hlfir.declare %arg2 {uniq_name = "_QFFrun_benchmarkEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %19 = fir.shape %15, %14 : (index, index) -> !fir.shape<2>
  %20:2 = hlfir.declare %arg3(%19) {uniq_name = "_QFFrun_benchmarkEy"} : (!fir.ref<!fir.array<?x?xf64>>, !fir.shape<2>) -> (!fir.box<!fir.array<?x?xf64>>, !fir.ref<!fir.array<?x?xf64>>)
  %21 = fir.shape %13, %12 : (index, index) -> !fir.shape<2>
  %22:2 = hlfir.declare %arg4(%21) {uniq_name = "_QFFrun_benchmarkEx"} : (!fir.ref<!fir.array<?x?xf64>>, !fir.shape<2>) -> (!fir.box<!fir.array<?x?xf64>>, !fir.ref<!fir.array<?x?xf64>>)
  %c2_i32 = arith.constant 2 : i32
  %23 = fir.convert %c2_i32 : (i32) -> index
  %c1024_i32_6 = arith.constant 1024 : i32
  %c1_i32 = arith.constant 1 : i32
  %24 = arith.subi %c1024_i32_6, %c1_i32 overflow<nsw> : i32
  %25 = fir.convert %24 : (i32) -> index
  %c1 = arith.constant 1 : index
  %26 = fir.convert %23 : (index) -> i32
  %27 = fir.do_loop %arg9 = %23 to %25 step %c1 iter_args(%arg10 = %26) -> (i32) {
    fir.store %arg10 to %16#0 : !fir.ref<i32>
    %c2_i32_7 = arith.constant 2 : i32
    %28 = fir.convert %c2_i32_7 : (i32) -> index
    %c1024_i32_8 = arith.constant 1024 : i32
    %c1_i32_9 = arith.constant 1 : i32
    %29 = arith.subi %c1024_i32_8, %c1_i32_9 overflow<nsw> : i32
    %30 = fir.convert %29 : (i32) -> index
    %c1_10 = arith.constant 1 : index
    %31 = fir.convert %28 : (index) -> i32
    %32 = fir.do_loop %arg11 = %28 to %30 step %c1_10 iter_args(%arg12 = %31) -> (i32) {
      fir.store %arg12 to %18#0 : !fir.ref<i32>
      %36 = fir.load %18#0 : !fir.ref<i32>
      %c1_i32_11 = arith.constant 1 : i32
      %37 = arith.subi %36, %c1_i32_11 overflow<nsw> : i32
      %38 = fir.convert %37 : (i32) -> i64
      %39 = fir.load %16#0 : !fir.ref<i32>
      %40 = fir.convert %39 : (i32) -> i64
      %41 = hlfir.designate %22#0 (%38, %40)  : (!fir.box<!fir.array<?x?xf64>>, i64, i64) -> !fir.ref<f64>
      %42 = fir.load %41 : !fir.ref<f64>
      %43 = fir.load %18#0 : !fir.ref<i32>
      %c1_i32_12 = arith.constant 1 : i32
      %44 = arith.addi %43, %c1_i32_12 overflow<nsw> : i32
      %45 = fir.convert %44 : (i32) -> i64
      %46 = fir.load %16#0 : !fir.ref<i32>
      %47 = fir.convert %46 : (i32) -> i64
      %48 = hlfir.designate %22#0 (%45, %47)  : (!fir.box<!fir.array<?x?xf64>>, i64, i64) -> !fir.ref<f64>
      %49 = fir.load %48 : !fir.ref<f64>
      %50 = arith.addf %42, %49 fastmath<contract> : f64
      %51 = fir.load %18#0 : !fir.ref<i32>
      %52 = fir.convert %51 : (i32) -> i64
      %53 = fir.load %16#0 : !fir.ref<i32>
      %c1_i32_13 = arith.constant 1 : i32
      %54 = arith.subi %53, %c1_i32_13 overflow<nsw> : i32
      %55 = fir.convert %54 : (i32) -> i64
      %56 = hlfir.designate %22#0 (%52, %55)  : (!fir.box<!fir.array<?x?xf64>>, i64, i64) -> !fir.ref<f64>
      %57 = fir.load %56 : !fir.ref<f64>
      %58 = arith.addf %50, %57 fastmath<contract> : f64
      %59 = fir.load %18#0 : !fir.ref<i32>
      %60 = fir.convert %59 : (i32) -> i64
      %61 = fir.load %16#0 : !fir.ref<i32>
      %c1_i32_14 = arith.constant 1 : i32
      %62 = arith.addi %61, %c1_i32_14 overflow<nsw> : i32
      %63 = fir.convert %62 : (i32) -> i64
      %64 = hlfir.designate %22#0 (%60, %63)  : (!fir.box<!fir.array<?x?xf64>>, i64, i64) -> !fir.ref<f64>
      %65 = fir.load %64 : !fir.ref<f64>
      %66 = arith.addf %58, %65 fastmath<contract> : f64
      %67 = fir.load %18#0 : !fir.ref<i32>
      %68 = fir.convert %67 : (i32) -> i64
      %69 = fir.load %16#0 : !fir.ref<i32>
      %70 = fir.convert %69 : (i32) -> i64
      %71 = hlfir.designate %20#0 (%68, %70)  : (!fir.box<!fir.array<?x?xf64>>, i64, i64) -> !fir.ref<f64>
      hlfir.assign %66 to %71 : f64, !fir.ref<f64>
      %72 = fir.convert %c1_10 : (index) -> i32
      %73 = fir.load %18#0 : !fir.ref<i32>
      %74 = arith.addi %73, %72 overflow<nsw> : i32
      fir.result %74 : i32
    }
    fir.store %32 to %18#0 : !fir.ref<i32>
    %33 = fir.convert %c1 : (index) -> i32
    %34 = fir.load %16#0 : !fir.ref<i32>
    %35 = arith.addi %34, %33 overflow<nsw> : i32
    fir.result %35 : i32
  }
  fir.store %27 to %16#0 : !fir.ref<i32>
  omp.terminator
}

// -----// IR Dump After CanonicalizerPass (canonicalize) //----- //
module {
  func.func @kernel(%arg0: !fir.ref<i32> {jit.arg_type = 1 : ui32, jit.literal_val = 0 : i64}, %arg1: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg2: !fir.ref<i32> {jit.arg_type = 1 : ui32, jit.literal_val = 0 : i64}, %arg3: !fir.ref<!fir.array<?x?xf64>> {jit.arg_type = 1 : ui32}, %arg4: !fir.ref<!fir.array<?x?xf64>> {jit.arg_type = 1 : ui32}, %arg5: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg6: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg7: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg8: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}) {
    %c1023 = arith.constant 1023 : index
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %c1_i32 = arith.constant 1 : i32
    %c1024 = arith.constant 1024 : index
    %0:2 = hlfir.declare %arg0 {uniq_name = "_QFFrun_benchmarkEj"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
    %1:2 = hlfir.declare %arg1 {uniq_name = "_QFFrun_benchmarkEn"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
    %2:2 = hlfir.declare %arg2 {uniq_name = "_QFFrun_benchmarkEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
    %3 = fir.shape %c1024, %c1024 : (index, index) -> !fir.shape<2>
    %4:2 = hlfir.declare %arg3(%3) {uniq_name = "_QFFrun_benchmarkEy"} : (!fir.ref<!fir.array<?x?xf64>>, !fir.shape<2>) -> (!fir.box<!fir.array<?x?xf64>>, !fir.ref<!fir.array<?x?xf64>>)
    %5 = fir.shape %c1024, %c1024 : (index, index) -> !fir.shape<2>
    %6:2 = hlfir.declare %arg4(%5) {uniq_name = "_QFFrun_benchmarkEx"} : (!fir.ref<!fir.array<?x?xf64>>, !fir.shape<2>) -> (!fir.box<!fir.array<?x?xf64>>, !fir.ref<!fir.array<?x?xf64>>)
    %7 = fir.convert %c2 : (index) -> i32
    %8 = fir.do_loop %arg9 = %c2 to %c1023 step %c1 iter_args(%arg10 = %7) -> (i32) {
      fir.store %arg10 to %0#0 : !fir.ref<i32>
      %9 = fir.convert %c2 : (index) -> i32
      %10 = fir.do_loop %arg11 = %c2 to %c1023 step %c1 iter_args(%arg12 = %9) -> (i32) {
        fir.store %arg12 to %2#0 : !fir.ref<i32>
        %14 = fir.load %2#0 : !fir.ref<i32>
        %15 = arith.subi %14, %c1_i32 overflow<nsw> : i32
        %16 = fir.convert %15 : (i32) -> i64
        %17 = fir.load %0#0 : !fir.ref<i32>
        %18 = fir.convert %17 : (i32) -> i64
        %19 = hlfir.designate %6#0 (%16, %18)  : (!fir.box<!fir.array<?x?xf64>>, i64, i64) -> !fir.ref<f64>
        %20 = fir.load %19 : !fir.ref<f64>
        %21 = fir.load %2#0 : !fir.ref<i32>
        %22 = arith.addi %21, %c1_i32 overflow<nsw> : i32
        %23 = fir.convert %22 : (i32) -> i64
        %24 = fir.load %0#0 : !fir.ref<i32>
        %25 = fir.convert %24 : (i32) -> i64
        %26 = hlfir.designate %6#0 (%23, %25)  : (!fir.box<!fir.array<?x?xf64>>, i64, i64) -> !fir.ref<f64>
        %27 = fir.load %26 : !fir.ref<f64>
        %28 = arith.addf %20, %27 fastmath<contract> : f64
        %29 = fir.load %2#0 : !fir.ref<i32>
        %30 = fir.convert %29 : (i32) -> i64
        %31 = fir.load %0#0 : !fir.ref<i32>
        %32 = arith.subi %31, %c1_i32 overflow<nsw> : i32
        %33 = fir.convert %32 : (i32) -> i64
        %34 = hlfir.designate %6#0 (%30, %33)  : (!fir.box<!fir.array<?x?xf64>>, i64, i64) -> !fir.ref<f64>
        %35 = fir.load %34 : !fir.ref<f64>
        %36 = arith.addf %28, %35 fastmath<contract> : f64
        %37 = fir.load %2#0 : !fir.ref<i32>
        %38 = fir.convert %37 : (i32) -> i64
        %39 = fir.load %0#0 : !fir.ref<i32>
        %40 = arith.addi %39, %c1_i32 overflow<nsw> : i32
        %41 = fir.convert %40 : (i32) -> i64
        %42 = hlfir.designate %6#0 (%38, %41)  : (!fir.box<!fir.array<?x?xf64>>, i64, i64) -> !fir.ref<f64>
        %43 = fir.load %42 : !fir.ref<f64>
        %44 = arith.addf %36, %43 fastmath<contract> : f64
        %45 = fir.load %2#0 : !fir.ref<i32>
        %46 = fir.convert %45 : (i32) -> i64
        %47 = fir.load %0#0 : !fir.ref<i32>
        %48 = fir.convert %47 : (i32) -> i64
        %49 = hlfir.designate %4#0 (%46, %48)  : (!fir.box<!fir.array<?x?xf64>>, i64, i64) -> !fir.ref<f64>
        hlfir.assign %44 to %49 : f64, !fir.ref<f64>
        %50 = fir.convert %c1 : (index) -> i32
        %51 = fir.load %2#0 : !fir.ref<i32>
        %52 = arith.addi %51, %50 overflow<nsw> : i32
        fir.result %52 : i32
      }
      fir.store %10 to %2#0 : !fir.ref<i32>
      %11 = fir.convert %c1 : (index) -> i32
      %12 = fir.load %0#0 : !fir.ref<i32>
      %13 = arith.addi %12, %11 overflow<nsw> : i32
      fir.result %13 : i32
    }
    fir.store %8 to %0#0 : !fir.ref<i32>
    omp.terminator
  }
}


// -----// IR Dump After SCCPPass (sccp) //----- //
module {
  func.func @kernel(%arg0: !fir.ref<i32> {jit.arg_type = 1 : ui32, jit.literal_val = 0 : i64}, %arg1: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg2: !fir.ref<i32> {jit.arg_type = 1 : ui32, jit.literal_val = 0 : i64}, %arg3: !fir.ref<!fir.array<?x?xf64>> {jit.arg_type = 1 : ui32}, %arg4: !fir.ref<!fir.array<?x?xf64>> {jit.arg_type = 1 : ui32}, %arg5: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg6: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg7: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg8: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}) {
    %c1024 = arith.constant 1024 : index
    %c1_i32 = arith.constant 1 : i32
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c1023 = arith.constant 1023 : index
    %0:2 = hlfir.declare %arg0 {uniq_name = "_QFFrun_benchmarkEj"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
    %1:2 = hlfir.declare %arg1 {uniq_name = "_QFFrun_benchmarkEn"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
    %2:2 = hlfir.declare %arg2 {uniq_name = "_QFFrun_benchmarkEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
    %3 = fir.shape %c1024, %c1024 : (index, index) -> !fir.shape<2>
    %4:2 = hlfir.declare %arg3(%3) {uniq_name = "_QFFrun_benchmarkEy"} : (!fir.ref<!fir.array<?x?xf64>>, !fir.shape<2>) -> (!fir.box<!fir.array<?x?xf64>>, !fir.ref<!fir.array<?x?xf64>>)
    %5 = fir.shape %c1024, %c1024 : (index, index) -> !fir.shape<2>
    %6:2 = hlfir.declare %arg4(%5) {uniq_name = "_QFFrun_benchmarkEx"} : (!fir.ref<!fir.array<?x?xf64>>, !fir.shape<2>) -> (!fir.box<!fir.array<?x?xf64>>, !fir.ref<!fir.array<?x?xf64>>)
    %7 = fir.convert %c2 : (index) -> i32
    %8 = fir.do_loop %arg9 = %c2 to %c1023 step %c1 iter_args(%arg10 = %7) -> (i32) {
      fir.store %arg10 to %0#0 : !fir.ref<i32>
      %9 = fir.convert %c2 : (index) -> i32
      %10 = fir.do_loop %arg11 = %c2 to %c1023 step %c1 iter_args(%arg12 = %9) -> (i32) {
        fir.store %arg12 to %2#0 : !fir.ref<i32>
        %14 = fir.load %2#0 : !fir.ref<i32>
        %15 = arith.subi %14, %c1_i32 overflow<nsw> : i32
        %16 = fir.convert %15 : (i32) -> i64
        %17 = fir.load %0#0 : !fir.ref<i32>
        %18 = fir.convert %17 : (i32) -> i64
        %19 = hlfir.designate %6#0 (%16, %18)  : (!fir.box<!fir.array<?x?xf64>>, i64, i64) -> !fir.ref<f64>
        %20 = fir.load %19 : !fir.ref<f64>
        %21 = fir.load %2#0 : !fir.ref<i32>
        %22 = arith.addi %21, %c1_i32 overflow<nsw> : i32
        %23 = fir.convert %22 : (i32) -> i64
        %24 = fir.load %0#0 : !fir.ref<i32>
        %25 = fir.convert %24 : (i32) -> i64
        %26 = hlfir.designate %6#0 (%23, %25)  : (!fir.box<!fir.array<?x?xf64>>, i64, i64) -> !fir.ref<f64>
        %27 = fir.load %26 : !fir.ref<f64>
        %28 = arith.addf %20, %27 fastmath<contract> : f64
        %29 = fir.load %2#0 : !fir.ref<i32>
        %30 = fir.convert %29 : (i32) -> i64
        %31 = fir.load %0#0 : !fir.ref<i32>
        %32 = arith.subi %31, %c1_i32 overflow<nsw> : i32
        %33 = fir.convert %32 : (i32) -> i64
        %34 = hlfir.designate %6#0 (%30, %33)  : (!fir.box<!fir.array<?x?xf64>>, i64, i64) -> !fir.ref<f64>
        %35 = fir.load %34 : !fir.ref<f64>
        %36 = arith.addf %28, %35 fastmath<contract> : f64
        %37 = fir.load %2#0 : !fir.ref<i32>
        %38 = fir.convert %37 : (i32) -> i64
        %39 = fir.load %0#0 : !fir.ref<i32>
        %40 = arith.addi %39, %c1_i32 overflow<nsw> : i32
        %41 = fir.convert %40 : (i32) -> i64
        %42 = hlfir.designate %6#0 (%38, %41)  : (!fir.box<!fir.array<?x?xf64>>, i64, i64) -> !fir.ref<f64>
        %43 = fir.load %42 : !fir.ref<f64>
        %44 = arith.addf %36, %43 fastmath<contract> : f64
        %45 = fir.load %2#0 : !fir.ref<i32>
        %46 = fir.convert %45 : (i32) -> i64
        %47 = fir.load %0#0 : !fir.ref<i32>
        %48 = fir.convert %47 : (i32) -> i64
        %49 = hlfir.designate %4#0 (%46, %48)  : (!fir.box<!fir.array<?x?xf64>>, i64, i64) -> !fir.ref<f64>
        hlfir.assign %44 to %49 : f64, !fir.ref<f64>
        %50 = fir.convert %c1 : (index) -> i32
        %51 = fir.load %2#0 : !fir.ref<i32>
        %52 = arith.addi %51, %50 overflow<nsw> : i32
        fir.result %52 : i32
      }
      fir.store %10 to %2#0 : !fir.ref<i32>
      %11 = fir.convert %c1 : (index) -> i32
      %12 = fir.load %0#0 : !fir.ref<i32>
      %13 = arith.addi %12, %11 overflow<nsw> : i32
      fir.result %13 : i32
    }
    fir.store %8 to %0#0 : !fir.ref<i32>
    omp.terminator
  }
}


// -----// IR Dump After CSEPass (cse) //----- //
module {
  func.func @kernel(%arg0: !fir.ref<i32> {jit.arg_type = 1 : ui32, jit.literal_val = 0 : i64}, %arg1: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg2: !fir.ref<i32> {jit.arg_type = 1 : ui32, jit.literal_val = 0 : i64}, %arg3: !fir.ref<!fir.array<?x?xf64>> {jit.arg_type = 1 : ui32}, %arg4: !fir.ref<!fir.array<?x?xf64>> {jit.arg_type = 1 : ui32}, %arg5: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg6: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg7: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg8: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}) {
    %c1024 = arith.constant 1024 : index
    %c1_i32 = arith.constant 1 : i32
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c1023 = arith.constant 1023 : index
    %0:2 = hlfir.declare %arg0 {uniq_name = "_QFFrun_benchmarkEj"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
    %1:2 = hlfir.declare %arg1 {uniq_name = "_QFFrun_benchmarkEn"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
    %2:2 = hlfir.declare %arg2 {uniq_name = "_QFFrun_benchmarkEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
    %3 = fir.shape %c1024, %c1024 : (index, index) -> !fir.shape<2>
    %4:2 = hlfir.declare %arg3(%3) {uniq_name = "_QFFrun_benchmarkEy"} : (!fir.ref<!fir.array<?x?xf64>>, !fir.shape<2>) -> (!fir.box<!fir.array<?x?xf64>>, !fir.ref<!fir.array<?x?xf64>>)
    %5:2 = hlfir.declare %arg4(%3) {uniq_name = "_QFFrun_benchmarkEx"} : (!fir.ref<!fir.array<?x?xf64>>, !fir.shape<2>) -> (!fir.box<!fir.array<?x?xf64>>, !fir.ref<!fir.array<?x?xf64>>)
    %6 = fir.convert %c2 : (index) -> i32
    %7 = fir.do_loop %arg9 = %c2 to %c1023 step %c1 iter_args(%arg10 = %6) -> (i32) {
      fir.store %arg10 to %0#0 : !fir.ref<i32>
      %8 = fir.do_loop %arg11 = %c2 to %c1023 step %c1 iter_args(%arg12 = %6) -> (i32) {
        fir.store %arg12 to %2#0 : !fir.ref<i32>
        %12 = fir.load %2#0 : !fir.ref<i32>
        %13 = arith.subi %12, %c1_i32 overflow<nsw> : i32
        %14 = fir.convert %13 : (i32) -> i64
        %15 = fir.load %0#0 : !fir.ref<i32>
        %16 = fir.convert %15 : (i32) -> i64
        %17 = hlfir.designate %5#0 (%14, %16)  : (!fir.box<!fir.array<?x?xf64>>, i64, i64) -> !fir.ref<f64>
        %18 = fir.load %17 : !fir.ref<f64>
        %19 = arith.addi %12, %c1_i32 overflow<nsw> : i32
        %20 = fir.convert %19 : (i32) -> i64
        %21 = hlfir.designate %5#0 (%20, %16)  : (!fir.box<!fir.array<?x?xf64>>, i64, i64) -> !fir.ref<f64>
        %22 = fir.load %21 : !fir.ref<f64>
        %23 = arith.addf %18, %22 fastmath<contract> : f64
        %24 = fir.convert %12 : (i32) -> i64
        %25 = arith.subi %15, %c1_i32 overflow<nsw> : i32
        %26 = fir.convert %25 : (i32) -> i64
        %27 = hlfir.designate %5#0 (%24, %26)  : (!fir.box<!fir.array<?x?xf64>>, i64, i64) -> !fir.ref<f64>
        %28 = fir.load %27 : !fir.ref<f64>
        %29 = arith.addf %23, %28 fastmath<contract> : f64
        %30 = arith.addi %15, %c1_i32 overflow<nsw> : i32
        %31 = fir.convert %30 : (i32) -> i64
        %32 = hlfir.designate %5#0 (%24, %31)  : (!fir.box<!fir.array<?x?xf64>>, i64, i64) -> !fir.ref<f64>
        %33 = fir.load %32 : !fir.ref<f64>
        %34 = arith.addf %29, %33 fastmath<contract> : f64
        %35 = hlfir.designate %4#0 (%24, %16)  : (!fir.box<!fir.array<?x?xf64>>, i64, i64) -> !fir.ref<f64>
        hlfir.assign %34 to %35 : f64, !fir.ref<f64>
        %36 = fir.convert %c1 : (index) -> i32
        %37 = fir.load %2#0 : !fir.ref<i32>
        %38 = arith.addi %37, %36 overflow<nsw> : i32
        fir.result %38 : i32
      }
      fir.store %8 to %2#0 : !fir.ref<i32>
      %9 = fir.convert %c1 : (index) -> i32
      %10 = fir.load %0#0 : !fir.ref<i32>
      %11 = arith.addi %10, %9 overflow<nsw> : i32
      fir.result %11 : i32
    }
    fir.store %7 to %0#0 : !fir.ref<i32>
    omp.terminator
  }
}


// -----// IR Dump After CanonicalizerPass (canonicalize) //----- //
module {
  func.func @kernel(%arg0: !fir.ref<i32> {jit.arg_type = 1 : ui32, jit.literal_val = 0 : i64}, %arg1: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg2: !fir.ref<i32> {jit.arg_type = 1 : ui32, jit.literal_val = 0 : i64}, %arg3: !fir.ref<!fir.array<?x?xf64>> {jit.arg_type = 1 : ui32}, %arg4: !fir.ref<!fir.array<?x?xf64>> {jit.arg_type = 1 : ui32}, %arg5: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg6: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg7: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg8: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}) {
    %c1024 = arith.constant 1024 : index
    %c1_i32 = arith.constant 1 : i32
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c1023 = arith.constant 1023 : index
    %0:2 = hlfir.declare %arg0 {uniq_name = "_QFFrun_benchmarkEj"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
    %1:2 = hlfir.declare %arg1 {uniq_name = "_QFFrun_benchmarkEn"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
    %2:2 = hlfir.declare %arg2 {uniq_name = "_QFFrun_benchmarkEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
    %3 = fir.shape %c1024, %c1024 : (index, index) -> !fir.shape<2>
    %4:2 = hlfir.declare %arg3(%3) {uniq_name = "_QFFrun_benchmarkEy"} : (!fir.ref<!fir.array<?x?xf64>>, !fir.shape<2>) -> (!fir.box<!fir.array<?x?xf64>>, !fir.ref<!fir.array<?x?xf64>>)
    %5:2 = hlfir.declare %arg4(%3) {uniq_name = "_QFFrun_benchmarkEx"} : (!fir.ref<!fir.array<?x?xf64>>, !fir.shape<2>) -> (!fir.box<!fir.array<?x?xf64>>, !fir.ref<!fir.array<?x?xf64>>)
    %6 = fir.convert %c2 : (index) -> i32
    %7 = fir.do_loop %arg9 = %c2 to %c1023 step %c1 iter_args(%arg10 = %6) -> (i32) {
      fir.store %arg10 to %0#0 : !fir.ref<i32>
      %8 = fir.do_loop %arg11 = %c2 to %c1023 step %c1 iter_args(%arg12 = %6) -> (i32) {
        fir.store %arg12 to %2#0 : !fir.ref<i32>
        %12 = fir.load %2#0 : !fir.ref<i32>
        %13 = arith.subi %12, %c1_i32 overflow<nsw> : i32
        %14 = fir.convert %13 : (i32) -> i64
        %15 = fir.load %0#0 : !fir.ref<i32>
        %16 = fir.convert %15 : (i32) -> i64
        %17 = hlfir.designate %5#0 (%14, %16)  : (!fir.box<!fir.array<?x?xf64>>, i64, i64) -> !fir.ref<f64>
        %18 = fir.load %17 : !fir.ref<f64>
        %19 = arith.addi %12, %c1_i32 overflow<nsw> : i32
        %20 = fir.convert %19 : (i32) -> i64
        %21 = hlfir.designate %5#0 (%20, %16)  : (!fir.box<!fir.array<?x?xf64>>, i64, i64) -> !fir.ref<f64>
        %22 = fir.load %21 : !fir.ref<f64>
        %23 = arith.addf %18, %22 fastmath<contract> : f64
        %24 = fir.convert %12 : (i32) -> i64
        %25 = arith.subi %15, %c1_i32 overflow<nsw> : i32
        %26 = fir.convert %25 : (i32) -> i64
        %27 = hlfir.designate %5#0 (%24, %26)  : (!fir.box<!fir.array<?x?xf64>>, i64, i64) -> !fir.ref<f64>
        %28 = fir.load %27 : !fir.ref<f64>
        %29 = arith.addf %23, %28 fastmath<contract> : f64
        %30 = arith.addi %15, %c1_i32 overflow<nsw> : i32
        %31 = fir.convert %30 : (i32) -> i64
        %32 = hlfir.designate %5#0 (%24, %31)  : (!fir.box<!fir.array<?x?xf64>>, i64, i64) -> !fir.ref<f64>
        %33 = fir.load %32 : !fir.ref<f64>
        %34 = arith.addf %29, %33 fastmath<contract> : f64
        %35 = hlfir.designate %4#0 (%24, %16)  : (!fir.box<!fir.array<?x?xf64>>, i64, i64) -> !fir.ref<f64>
        hlfir.assign %34 to %35 : f64, !fir.ref<f64>
        %36 = fir.convert %c1 : (index) -> i32
        %37 = fir.load %2#0 : !fir.ref<i32>
        %38 = arith.addi %37, %36 overflow<nsw> : i32
        fir.result %38 : i32
      }
      fir.store %8 to %2#0 : !fir.ref<i32>
      %9 = fir.convert %c1 : (index) -> i32
      %10 = fir.load %0#0 : !fir.ref<i32>
      %11 = arith.addi %10, %9 overflow<nsw> : i32
      fir.result %11 : i32
    }
    fir.store %7 to %0#0 : !fir.ref<i32>
    omp.terminator
  }
}


// -----// IR Dump After {anonymous}::ShapeInferPass (jforce-shape-infer) //----- //
func.func @kernel(%arg0: !fir.ref<i32> {jit.arg_type = 1 : ui32, jit.literal_val = 0 : i64}, %arg1: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg2: !fir.ref<i32> {jit.arg_type = 1 : ui32, jit.literal_val = 0 : i64}, %arg3: !fir.ref<!fir.array<1024x1024xf64>> {jit.arg_type = 1 : ui32}, %arg4: !fir.ref<!fir.array<1024x1024xf64>> {jit.arg_type = 1 : ui32}, %arg5: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg6: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg7: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg8: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}) {
  %c1024 = arith.constant 1024 : index
  %c1_i32 = arith.constant 1 : i32
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c1023 = arith.constant 1023 : index
  %0:2 = hlfir.declare %arg0 {uniq_name = "_QFFrun_benchmarkEj"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %1:2 = hlfir.declare %arg1 {uniq_name = "_QFFrun_benchmarkEn"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %2:2 = hlfir.declare %arg2 {uniq_name = "_QFFrun_benchmarkEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %3 = fir.shape %c1024, %c1024 : (index, index) -> !fir.shape<2>
  %4:2 = hlfir.declare %arg3(%3) {uniq_name = "_QFFrun_benchmarkEy"} : (!fir.ref<!fir.array<1024x1024xf64>>, !fir.shape<2>) -> (!fir.ref<!fir.array<1024x1024xf64>>, !fir.ref<!fir.array<1024x1024xf64>>)
  %5:2 = hlfir.declare %arg4(%3) {uniq_name = "_QFFrun_benchmarkEx"} : (!fir.ref<!fir.array<1024x1024xf64>>, !fir.shape<2>) -> (!fir.ref<!fir.array<1024x1024xf64>>, !fir.ref<!fir.array<1024x1024xf64>>)
  %6 = fir.convert %c2 : (index) -> i32
  %7 = fir.do_loop %arg9 = %c2 to %c1023 step %c1 iter_args(%arg10 = %6) -> (i32) {
    fir.store %arg10 to %0#0 : !fir.ref<i32>
    %8 = fir.do_loop %arg11 = %c2 to %c1023 step %c1 iter_args(%arg12 = %6) -> (i32) {
      fir.store %arg12 to %2#0 : !fir.ref<i32>
      %12 = fir.load %2#0 : !fir.ref<i32>
      %13 = arith.subi %12, %c1_i32 overflow<nsw> : i32
      %14 = fir.convert %13 : (i32) -> i64
      %15 = fir.load %0#0 : !fir.ref<i32>
      %16 = fir.convert %15 : (i32) -> i64
      %17 = hlfir.designate %5#0 (%14, %16)  : (!fir.ref<!fir.array<1024x1024xf64>>, i64, i64) -> !fir.ref<f64>
      %18 = fir.load %17 : !fir.ref<f64>
      %19 = arith.addi %12, %c1_i32 overflow<nsw> : i32
      %20 = fir.convert %19 : (i32) -> i64
      %21 = hlfir.designate %5#0 (%20, %16)  : (!fir.ref<!fir.array<1024x1024xf64>>, i64, i64) -> !fir.ref<f64>
      %22 = fir.load %21 : !fir.ref<f64>
      %23 = arith.addf %18, %22 fastmath<contract> : f64
      %24 = fir.convert %12 : (i32) -> i64
      %25 = arith.subi %15, %c1_i32 overflow<nsw> : i32
      %26 = fir.convert %25 : (i32) -> i64
      %27 = hlfir.designate %5#0 (%24, %26)  : (!fir.ref<!fir.array<1024x1024xf64>>, i64, i64) -> !fir.ref<f64>
      %28 = fir.load %27 : !fir.ref<f64>
      %29 = arith.addf %23, %28 fastmath<contract> : f64
      %30 = arith.addi %15, %c1_i32 overflow<nsw> : i32
      %31 = fir.convert %30 : (i32) -> i64
      %32 = hlfir.designate %5#0 (%24, %31)  : (!fir.ref<!fir.array<1024x1024xf64>>, i64, i64) -> !fir.ref<f64>
      %33 = fir.load %32 : !fir.ref<f64>
      %34 = arith.addf %29, %33 fastmath<contract> : f64
      %35 = hlfir.designate %4#0 (%24, %16)  : (!fir.ref<!fir.array<1024x1024xf64>>, i64, i64) -> !fir.ref<f64>
      hlfir.assign %34 to %35 : f64, !fir.ref<f64>
      %36 = fir.convert %c1 : (index) -> i32
      %37 = fir.load %2#0 : !fir.ref<i32>
      %38 = arith.addi %37, %36 overflow<nsw> : i32
      fir.result %38 : i32
    }
    fir.store %8 to %2#0 : !fir.ref<i32>
    %9 = fir.convert %c1 : (index) -> i32
    %10 = fir.load %0#0 : !fir.ref<i32>
    %11 = arith.addi %10, %9 overflow<nsw> : i32
    fir.result %11 : i32
  }
  fir.store %7 to %0#0 : !fir.ref<i32>
  omp.terminator
}

// -----// IR Dump After CanonicalizerPass (canonicalize) //----- //
module {
  func.func @kernel(%arg0: !fir.ref<i32> {jit.arg_type = 1 : ui32, jit.literal_val = 0 : i64}, %arg1: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg2: !fir.ref<i32> {jit.arg_type = 1 : ui32, jit.literal_val = 0 : i64}, %arg3: !fir.ref<!fir.array<1024x1024xf64>> {jit.arg_type = 1 : ui32}, %arg4: !fir.ref<!fir.array<1024x1024xf64>> {jit.arg_type = 1 : ui32}, %arg5: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg6: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg7: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg8: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}) {
    %c1024 = arith.constant 1024 : index
    %c1_i32 = arith.constant 1 : i32
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c1023 = arith.constant 1023 : index
    %0:2 = hlfir.declare %arg0 {uniq_name = "_QFFrun_benchmarkEj"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
    %1:2 = hlfir.declare %arg1 {uniq_name = "_QFFrun_benchmarkEn"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
    %2:2 = hlfir.declare %arg2 {uniq_name = "_QFFrun_benchmarkEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
    %3 = fir.shape %c1024, %c1024 : (index, index) -> !fir.shape<2>
    %4:2 = hlfir.declare %arg3(%3) {uniq_name = "_QFFrun_benchmarkEy"} : (!fir.ref<!fir.array<1024x1024xf64>>, !fir.shape<2>) -> (!fir.ref<!fir.array<1024x1024xf64>>, !fir.ref<!fir.array<1024x1024xf64>>)
    %5:2 = hlfir.declare %arg4(%3) {uniq_name = "_QFFrun_benchmarkEx"} : (!fir.ref<!fir.array<1024x1024xf64>>, !fir.shape<2>) -> (!fir.ref<!fir.array<1024x1024xf64>>, !fir.ref<!fir.array<1024x1024xf64>>)
    %6 = fir.convert %c2 : (index) -> i32
    %7 = fir.do_loop %arg9 = %c2 to %c1023 step %c1 iter_args(%arg10 = %6) -> (i32) {
      fir.store %arg10 to %0#0 : !fir.ref<i32>
      %8 = fir.do_loop %arg11 = %c2 to %c1023 step %c1 iter_args(%arg12 = %6) -> (i32) {
        fir.store %arg12 to %2#0 : !fir.ref<i32>
        %12 = fir.load %2#0 : !fir.ref<i32>
        %13 = arith.subi %12, %c1_i32 overflow<nsw> : i32
        %14 = fir.convert %13 : (i32) -> i64
        %15 = fir.load %0#0 : !fir.ref<i32>
        %16 = fir.convert %15 : (i32) -> i64
        %17 = hlfir.designate %5#0 (%14, %16)  : (!fir.ref<!fir.array<1024x1024xf64>>, i64, i64) -> !fir.ref<f64>
        %18 = fir.load %17 : !fir.ref<f64>
        %19 = arith.addi %12, %c1_i32 overflow<nsw> : i32
        %20 = fir.convert %19 : (i32) -> i64
        %21 = hlfir.designate %5#0 (%20, %16)  : (!fir.ref<!fir.array<1024x1024xf64>>, i64, i64) -> !fir.ref<f64>
        %22 = fir.load %21 : !fir.ref<f64>
        %23 = arith.addf %18, %22 fastmath<contract> : f64
        %24 = fir.convert %12 : (i32) -> i64
        %25 = arith.subi %15, %c1_i32 overflow<nsw> : i32
        %26 = fir.convert %25 : (i32) -> i64
        %27 = hlfir.designate %5#0 (%24, %26)  : (!fir.ref<!fir.array<1024x1024xf64>>, i64, i64) -> !fir.ref<f64>
        %28 = fir.load %27 : !fir.ref<f64>
        %29 = arith.addf %23, %28 fastmath<contract> : f64
        %30 = arith.addi %15, %c1_i32 overflow<nsw> : i32
        %31 = fir.convert %30 : (i32) -> i64
        %32 = hlfir.designate %5#0 (%24, %31)  : (!fir.ref<!fir.array<1024x1024xf64>>, i64, i64) -> !fir.ref<f64>
        %33 = fir.load %32 : !fir.ref<f64>
        %34 = arith.addf %29, %33 fastmath<contract> : f64
        %35 = hlfir.designate %4#0 (%24, %16)  : (!fir.ref<!fir.array<1024x1024xf64>>, i64, i64) -> !fir.ref<f64>
        hlfir.assign %34 to %35 : f64, !fir.ref<f64>
        %36 = fir.convert %c1 : (index) -> i32
        %37 = fir.load %2#0 : !fir.ref<i32>
        %38 = arith.addi %37, %36 overflow<nsw> : i32
        fir.result %38 : i32
      }
      fir.store %8 to %2#0 : !fir.ref<i32>
      %9 = fir.convert %c1 : (index) -> i32
      %10 = fir.load %0#0 : !fir.ref<i32>
      %11 = arith.addi %10, %9 overflow<nsw> : i32
      fir.result %11 : i32
    }
    fir.store %7 to %0#0 : !fir.ref<i32>
    omp.terminator
  }
}


// -----// IR Dump After ConvertHLFIRtoFIR (convert-hlfir-to-fir) //----- //
module {
  func.func @kernel(%arg0: !fir.ref<i32> {jit.arg_type = 1 : ui32, jit.literal_val = 0 : i64}, %arg1: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg2: !fir.ref<i32> {jit.arg_type = 1 : ui32, jit.literal_val = 0 : i64}, %arg3: !fir.ref<!fir.array<1024x1024xf64>> {jit.arg_type = 1 : ui32}, %arg4: !fir.ref<!fir.array<1024x1024xf64>> {jit.arg_type = 1 : ui32}, %arg5: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg6: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg7: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg8: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}) {
    %c1024 = arith.constant 1024 : index
    %c1_i32 = arith.constant 1 : i32
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c1023 = arith.constant 1023 : index
    %0 = fir.declare %arg0 {uniq_name = "_QFFrun_benchmarkEj"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %1 = fir.declare %arg1 {uniq_name = "_QFFrun_benchmarkEn"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %2 = fir.declare %arg2 {uniq_name = "_QFFrun_benchmarkEi"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %3 = fir.shape %c1024, %c1024 : (index, index) -> !fir.shape<2>
    %4 = fir.declare %arg3(%3) {uniq_name = "_QFFrun_benchmarkEy"} : (!fir.ref<!fir.array<1024x1024xf64>>, !fir.shape<2>) -> !fir.ref<!fir.array<1024x1024xf64>>
    %5 = fir.declare %arg4(%3) {uniq_name = "_QFFrun_benchmarkEx"} : (!fir.ref<!fir.array<1024x1024xf64>>, !fir.shape<2>) -> !fir.ref<!fir.array<1024x1024xf64>>
    %6 = fir.convert %c2 : (index) -> i32
    %7 = fir.do_loop %arg9 = %c2 to %c1023 step %c1 iter_args(%arg10 = %6) -> (i32) {
      fir.store %arg10 to %0 : !fir.ref<i32>
      %8 = fir.do_loop %arg11 = %c2 to %c1023 step %c1 iter_args(%arg12 = %6) -> (i32) {
        fir.store %arg12 to %2 : !fir.ref<i32>
        %12 = fir.load %2 : !fir.ref<i32>
        %13 = arith.subi %12, %c1_i32 overflow<nsw> : i32
        %14 = fir.convert %13 : (i32) -> i64
        %15 = fir.load %0 : !fir.ref<i32>
        %16 = fir.convert %15 : (i32) -> i64
        %17 = fir.array_coor %5(%3) %14, %16 : (!fir.ref<!fir.array<1024x1024xf64>>, !fir.shape<2>, i64, i64) -> !fir.ref<f64>
        %18 = fir.load %17 : !fir.ref<f64>
        %19 = arith.addi %12, %c1_i32 overflow<nsw> : i32
        %20 = fir.convert %19 : (i32) -> i64
        %21 = fir.array_coor %5(%3) %20, %16 : (!fir.ref<!fir.array<1024x1024xf64>>, !fir.shape<2>, i64, i64) -> !fir.ref<f64>
        %22 = fir.load %21 : !fir.ref<f64>
        %23 = arith.addf %18, %22 fastmath<contract> : f64
        %24 = fir.convert %12 : (i32) -> i64
        %25 = arith.subi %15, %c1_i32 overflow<nsw> : i32
        %26 = fir.convert %25 : (i32) -> i64
        %27 = fir.array_coor %5(%3) %24, %26 : (!fir.ref<!fir.array<1024x1024xf64>>, !fir.shape<2>, i64, i64) -> !fir.ref<f64>
        %28 = fir.load %27 : !fir.ref<f64>
        %29 = arith.addf %23, %28 fastmath<contract> : f64
        %30 = arith.addi %15, %c1_i32 overflow<nsw> : i32
        %31 = fir.convert %30 : (i32) -> i64
        %32 = fir.array_coor %5(%3) %24, %31 : (!fir.ref<!fir.array<1024x1024xf64>>, !fir.shape<2>, i64, i64) -> !fir.ref<f64>
        %33 = fir.load %32 : !fir.ref<f64>
        %34 = arith.addf %29, %33 fastmath<contract> : f64
        %35 = fir.array_coor %4(%3) %24, %16 : (!fir.ref<!fir.array<1024x1024xf64>>, !fir.shape<2>, i64, i64) -> !fir.ref<f64>
        fir.store %34 to %35 : !fir.ref<f64>
        %36 = fir.convert %c1 : (index) -> i32
        %37 = fir.load %2 : !fir.ref<i32>
        %38 = arith.addi %37, %36 overflow<nsw> : i32
        fir.result %38 : i32
      }
      fir.store %8 to %2 : !fir.ref<i32>
      %9 = fir.convert %c1 : (index) -> i32
      %10 = fir.load %0 : !fir.ref<i32>
      %11 = arith.addi %10, %9 overflow<nsw> : i32
      fir.result %11 : i32
    }
    fir.store %7 to %0 : !fir.ref<i32>
    omp.terminator
  }
}


// -----// IR Dump After FIRToMemRef (fir-to-memref) //----- //
func.func @kernel(%arg0: !fir.ref<i32> {jit.arg_type = 1 : ui32, jit.literal_val = 0 : i64}, %arg1: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg2: !fir.ref<i32> {jit.arg_type = 1 : ui32, jit.literal_val = 0 : i64}, %arg3: !fir.ref<!fir.array<1024x1024xf64>> {jit.arg_type = 1 : ui32}, %arg4: !fir.ref<!fir.array<1024x1024xf64>> {jit.arg_type = 1 : ui32}, %arg5: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg6: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg7: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg8: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}) {
  %c1024 = arith.constant 1024 : index
  %c1_i32 = arith.constant 1 : i32
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c1023 = arith.constant 1023 : index
  %0 = fir.declare %arg0 {uniq_name = "_QFFrun_benchmarkEj"} : (!fir.ref<i32>) -> !fir.ref<i32>
  %1 = fir.declare %arg1 {uniq_name = "_QFFrun_benchmarkEn"} : (!fir.ref<i32>) -> !fir.ref<i32>
  %2 = fir.declare %arg2 {uniq_name = "_QFFrun_benchmarkEi"} : (!fir.ref<i32>) -> !fir.ref<i32>
  %3 = fir.shape %c1024, %c1024 : (index, index) -> !fir.shape<2>
  %4 = fir.declare %arg3(%3) {uniq_name = "_QFFrun_benchmarkEy"} : (!fir.ref<!fir.array<1024x1024xf64>>, !fir.shape<2>) -> !fir.ref<!fir.array<1024x1024xf64>>
  %5 = fir.declare %arg4(%3) {uniq_name = "_QFFrun_benchmarkEx"} : (!fir.ref<!fir.array<1024x1024xf64>>, !fir.shape<2>) -> !fir.ref<!fir.array<1024x1024xf64>>
  %6 = fir.convert %c2 : (index) -> i32
  %7 = fir.do_loop %arg9 = %c2 to %c1023 step %c1 iter_args(%arg10 = %6) -> (i32) {
    %9 = fir.convert %0 : (!fir.ref<i32>) -> memref<i32>
    memref.store %arg10, %9[] : memref<i32>
    %10 = fir.do_loop %arg11 = %c2 to %c1023 step %c1 iter_args(%arg12 = %6) -> (i32) {
      %16 = fir.convert %2 : (!fir.ref<i32>) -> memref<i32>
      memref.store %arg12, %16[] : memref<i32>
      %17 = fir.convert %2 : (!fir.ref<i32>) -> memref<i32>
      %18 = memref.load %17[] : memref<i32>
      %19 = arith.subi %18, %c1_i32 overflow<nsw> : i32
      %20 = fir.convert %19 : (i32) -> i64
      %21 = fir.convert %0 : (!fir.ref<i32>) -> memref<i32>
      %22 = memref.load %21[] : memref<i32>
      %23 = fir.convert %22 : (i32) -> i64
      %24 = fir.convert %5 : (!fir.ref<!fir.array<1024x1024xf64>>) -> memref<1024x1024xf64>
      %c1_0 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %25 = arith.index_cast %20 : i64 to index
      %26 = arith.subi %25, %c1_0 : index
      %27 = arith.muli %26, %c1_0 : index
      %28 = arith.subi %c1_0, %c1_0 : index
      %29 = arith.addi %27, %28 : index
      %30 = arith.index_cast %23 : i64 to index
      %31 = arith.subi %30, %c1_0 : index
      %32 = arith.muli %31, %c1_0 : index
      %33 = arith.subi %c1_0, %c1_0 : index
      %34 = arith.addi %32, %33 : index
      %35 = memref.load %24[%34, %29] : memref<1024x1024xf64>
      %36 = arith.addi %18, %c1_i32 overflow<nsw> : i32
      %37 = fir.convert %36 : (i32) -> i64
      %38 = fir.convert %5 : (!fir.ref<!fir.array<1024x1024xf64>>) -> memref<1024x1024xf64>
      %c1_1 = arith.constant 1 : index
      %c0_2 = arith.constant 0 : index
      %39 = arith.index_cast %37 : i64 to index
      %40 = arith.subi %39, %c1_1 : index
      %41 = arith.muli %40, %c1_1 : index
      %42 = arith.subi %c1_1, %c1_1 : index
      %43 = arith.addi %41, %42 : index
      %44 = arith.index_cast %23 : i64 to index
      %45 = arith.subi %44, %c1_1 : index
      %46 = arith.muli %45, %c1_1 : index
      %47 = arith.subi %c1_1, %c1_1 : index
      %48 = arith.addi %46, %47 : index
      %49 = memref.load %38[%48, %43] : memref<1024x1024xf64>
      %50 = arith.addf %35, %49 fastmath<contract> : f64
      %51 = fir.convert %18 : (i32) -> i64
      %52 = arith.subi %22, %c1_i32 overflow<nsw> : i32
      %53 = fir.convert %52 : (i32) -> i64
      %54 = fir.convert %5 : (!fir.ref<!fir.array<1024x1024xf64>>) -> memref<1024x1024xf64>
      %c1_3 = arith.constant 1 : index
      %c0_4 = arith.constant 0 : index
      %55 = arith.index_cast %51 : i64 to index
      %56 = arith.subi %55, %c1_3 : index
      %57 = arith.muli %56, %c1_3 : index
      %58 = arith.subi %c1_3, %c1_3 : index
      %59 = arith.addi %57, %58 : index
      %60 = arith.index_cast %53 : i64 to index
      %61 = arith.subi %60, %c1_3 : index
      %62 = arith.muli %61, %c1_3 : index
      %63 = arith.subi %c1_3, %c1_3 : index
      %64 = arith.addi %62, %63 : index
      %65 = memref.load %54[%64, %59] : memref<1024x1024xf64>
      %66 = arith.addf %50, %65 fastmath<contract> : f64
      %67 = arith.addi %22, %c1_i32 overflow<nsw> : i32
      %68 = fir.convert %67 : (i32) -> i64
      %69 = fir.convert %5 : (!fir.ref<!fir.array<1024x1024xf64>>) -> memref<1024x1024xf64>
      %c1_5 = arith.constant 1 : index
      %c0_6 = arith.constant 0 : index
      %70 = arith.index_cast %51 : i64 to index
      %71 = arith.subi %70, %c1_5 : index
      %72 = arith.muli %71, %c1_5 : index
      %73 = arith.subi %c1_5, %c1_5 : index
      %74 = arith.addi %72, %73 : index
      %75 = arith.index_cast %68 : i64 to index
      %76 = arith.subi %75, %c1_5 : index
      %77 = arith.muli %76, %c1_5 : index
      %78 = arith.subi %c1_5, %c1_5 : index
      %79 = arith.addi %77, %78 : index
      %80 = memref.load %69[%79, %74] : memref<1024x1024xf64>
      %81 = arith.addf %66, %80 fastmath<contract> : f64
      %82 = fir.convert %4 : (!fir.ref<!fir.array<1024x1024xf64>>) -> memref<1024x1024xf64>
      %c1_7 = arith.constant 1 : index
      %c0_8 = arith.constant 0 : index
      %83 = arith.index_cast %51 : i64 to index
      %84 = arith.subi %83, %c1_7 : index
      %85 = arith.muli %84, %c1_7 : index
      %86 = arith.subi %c1_7, %c1_7 : index
      %87 = arith.addi %85, %86 : index
      %88 = arith.index_cast %23 : i64 to index
      %89 = arith.subi %88, %c1_7 : index
      %90 = arith.muli %89, %c1_7 : index
      %91 = arith.subi %c1_7, %c1_7 : index
      %92 = arith.addi %90, %91 : index
      memref.store %81, %82[%92, %87] : memref<1024x1024xf64>
      %93 = fir.convert %c1 : (index) -> i32
      %94 = fir.convert %2 : (!fir.ref<i32>) -> memref<i32>
      %95 = memref.load %94[] : memref<i32>
      %96 = arith.addi %95, %93 overflow<nsw> : i32
      fir.result %96 : i32
    }
    %11 = fir.convert %2 : (!fir.ref<i32>) -> memref<i32>
    memref.store %10, %11[] : memref<i32>
    %12 = fir.convert %c1 : (index) -> i32
    %13 = fir.convert %0 : (!fir.ref<i32>) -> memref<i32>
    %14 = memref.load %13[] : memref<i32>
    %15 = arith.addi %14, %12 overflow<nsw> : i32
    fir.result %15 : i32
  }
  %8 = fir.convert %0 : (!fir.ref<i32>) -> memref<i32>
  memref.store %7, %8[] : memref<i32>
  omp.terminator
}

// -----// IR Dump After CanonicalizerPass (canonicalize) //----- //
module {
  func.func @kernel(%arg0: !fir.ref<i32> {jit.arg_type = 1 : ui32, jit.literal_val = 0 : i64}, %arg1: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg2: !fir.ref<i32> {jit.arg_type = 1 : ui32, jit.literal_val = 0 : i64}, %arg3: !fir.ref<!fir.array<1024x1024xf64>> {jit.arg_type = 1 : ui32}, %arg4: !fir.ref<!fir.array<1024x1024xf64>> {jit.arg_type = 1 : ui32}, %arg5: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg6: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg7: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg8: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}) {
    %c1024 = arith.constant 1024 : index
    %c1_i32 = arith.constant 1 : i32
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c1023 = arith.constant 1023 : index
    %0 = fir.declare %arg0 {uniq_name = "_QFFrun_benchmarkEj"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %1 = fir.declare %arg1 {uniq_name = "_QFFrun_benchmarkEn"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %2 = fir.declare %arg2 {uniq_name = "_QFFrun_benchmarkEi"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %3 = fir.shape %c1024, %c1024 : (index, index) -> !fir.shape<2>
    %4 = fir.declare %arg3(%3) {uniq_name = "_QFFrun_benchmarkEy"} : (!fir.ref<!fir.array<1024x1024xf64>>, !fir.shape<2>) -> !fir.ref<!fir.array<1024x1024xf64>>
    %5 = fir.declare %arg4(%3) {uniq_name = "_QFFrun_benchmarkEx"} : (!fir.ref<!fir.array<1024x1024xf64>>, !fir.shape<2>) -> !fir.ref<!fir.array<1024x1024xf64>>
    %6 = fir.convert %c2 : (index) -> i32
    %7 = fir.do_loop %arg9 = %c2 to %c1023 step %c1 iter_args(%arg10 = %6) -> (i32) {
      %9 = fir.convert %0 : (!fir.ref<i32>) -> memref<i32>
      memref.store %arg10, %9[] : memref<i32>
      %10 = fir.do_loop %arg11 = %c2 to %c1023 step %c1 iter_args(%arg12 = %6) -> (i32) {
        %16 = fir.convert %2 : (!fir.ref<i32>) -> memref<i32>
        memref.store %arg12, %16[] : memref<i32>
        %17 = fir.convert %2 : (!fir.ref<i32>) -> memref<i32>
        %18 = memref.load %17[] : memref<i32>
        %19 = arith.subi %18, %c1_i32 overflow<nsw> : i32
        %20 = fir.convert %19 : (i32) -> i64
        %21 = fir.convert %0 : (!fir.ref<i32>) -> memref<i32>
        %22 = memref.load %21[] : memref<i32>
        %23 = fir.convert %22 : (i32) -> i64
        %24 = fir.convert %5 : (!fir.ref<!fir.array<1024x1024xf64>>) -> memref<1024x1024xf64>
        %25 = arith.index_cast %20 : i64 to index
        %26 = arith.subi %25, %c1 : index
        %27 = arith.index_cast %23 : i64 to index
        %28 = arith.subi %27, %c1 : index
        %29 = memref.load %24[%28, %26] : memref<1024x1024xf64>
        %30 = arith.addi %18, %c1_i32 overflow<nsw> : i32
        %31 = fir.convert %30 : (i32) -> i64
        %32 = fir.convert %5 : (!fir.ref<!fir.array<1024x1024xf64>>) -> memref<1024x1024xf64>
        %33 = arith.index_cast %31 : i64 to index
        %34 = arith.subi %33, %c1 : index
        %35 = arith.index_cast %23 : i64 to index
        %36 = arith.subi %35, %c1 : index
        %37 = memref.load %32[%36, %34] : memref<1024x1024xf64>
        %38 = arith.addf %29, %37 fastmath<contract> : f64
        %39 = fir.convert %18 : (i32) -> i64
        %40 = arith.subi %22, %c1_i32 overflow<nsw> : i32
        %41 = fir.convert %40 : (i32) -> i64
        %42 = fir.convert %5 : (!fir.ref<!fir.array<1024x1024xf64>>) -> memref<1024x1024xf64>
        %43 = arith.index_cast %39 : i64 to index
        %44 = arith.subi %43, %c1 : index
        %45 = arith.index_cast %41 : i64 to index
        %46 = arith.subi %45, %c1 : index
        %47 = memref.load %42[%46, %44] : memref<1024x1024xf64>
        %48 = arith.addf %38, %47 fastmath<contract> : f64
        %49 = arith.addi %22, %c1_i32 overflow<nsw> : i32
        %50 = fir.convert %49 : (i32) -> i64
        %51 = fir.convert %5 : (!fir.ref<!fir.array<1024x1024xf64>>) -> memref<1024x1024xf64>
        %52 = arith.index_cast %39 : i64 to index
        %53 = arith.subi %52, %c1 : index
        %54 = arith.index_cast %50 : i64 to index
        %55 = arith.subi %54, %c1 : index
        %56 = memref.load %51[%55, %53] : memref<1024x1024xf64>
        %57 = arith.addf %48, %56 fastmath<contract> : f64
        %58 = fir.convert %4 : (!fir.ref<!fir.array<1024x1024xf64>>) -> memref<1024x1024xf64>
        %59 = arith.index_cast %39 : i64 to index
        %60 = arith.subi %59, %c1 : index
        %61 = arith.index_cast %23 : i64 to index
        %62 = arith.subi %61, %c1 : index
        memref.store %57, %58[%62, %60] : memref<1024x1024xf64>
        %63 = fir.convert %c1 : (index) -> i32
        %64 = fir.convert %2 : (!fir.ref<i32>) -> memref<i32>
        %65 = memref.load %64[] : memref<i32>
        %66 = arith.addi %65, %63 overflow<nsw> : i32
        fir.result %66 : i32
      }
      %11 = fir.convert %2 : (!fir.ref<i32>) -> memref<i32>
      memref.store %10, %11[] : memref<i32>
      %12 = fir.convert %c1 : (index) -> i32
      %13 = fir.convert %0 : (!fir.ref<i32>) -> memref<i32>
      %14 = memref.load %13[] : memref<i32>
      %15 = arith.addi %14, %12 overflow<nsw> : i32
      fir.result %15 : i32
    }
    %8 = fir.convert %0 : (!fir.ref<i32>) -> memref<i32>
    memref.store %7, %8[] : memref<i32>
    omp.terminator
  }
}


// -----// IR Dump After LoopInvariantCodeMotionPass (loop-invariant-code-motion) //----- //
module {
  func.func @kernel(%arg0: !fir.ref<i32> {jit.arg_type = 1 : ui32, jit.literal_val = 0 : i64}, %arg1: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg2: !fir.ref<i32> {jit.arg_type = 1 : ui32, jit.literal_val = 0 : i64}, %arg3: !fir.ref<!fir.array<1024x1024xf64>> {jit.arg_type = 1 : ui32}, %arg4: !fir.ref<!fir.array<1024x1024xf64>> {jit.arg_type = 1 : ui32}, %arg5: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg6: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg7: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg8: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}) {
    %c1024 = arith.constant 1024 : index
    %c1_i32 = arith.constant 1 : i32
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c1023 = arith.constant 1023 : index
    %0 = fir.declare %arg0 {uniq_name = "_QFFrun_benchmarkEj"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %1 = fir.declare %arg1 {uniq_name = "_QFFrun_benchmarkEn"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %2 = fir.declare %arg2 {uniq_name = "_QFFrun_benchmarkEi"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %3 = fir.shape %c1024, %c1024 : (index, index) -> !fir.shape<2>
    %4 = fir.declare %arg3(%3) {uniq_name = "_QFFrun_benchmarkEy"} : (!fir.ref<!fir.array<1024x1024xf64>>, !fir.shape<2>) -> !fir.ref<!fir.array<1024x1024xf64>>
    %5 = fir.declare %arg4(%3) {uniq_name = "_QFFrun_benchmarkEx"} : (!fir.ref<!fir.array<1024x1024xf64>>, !fir.shape<2>) -> !fir.ref<!fir.array<1024x1024xf64>>
    %6 = fir.convert %c2 : (index) -> i32
    %7 = fir.convert %0 : (!fir.ref<i32>) -> memref<i32>
    %8 = fir.convert %2 : (!fir.ref<i32>) -> memref<i32>
    %9 = fir.convert %2 : (!fir.ref<i32>) -> memref<i32>
    %10 = fir.convert %0 : (!fir.ref<i32>) -> memref<i32>
    %11 = fir.convert %5 : (!fir.ref<!fir.array<1024x1024xf64>>) -> memref<1024x1024xf64>
    %12 = fir.convert %5 : (!fir.ref<!fir.array<1024x1024xf64>>) -> memref<1024x1024xf64>
    %13 = fir.convert %5 : (!fir.ref<!fir.array<1024x1024xf64>>) -> memref<1024x1024xf64>
    %14 = fir.convert %5 : (!fir.ref<!fir.array<1024x1024xf64>>) -> memref<1024x1024xf64>
    %15 = fir.convert %4 : (!fir.ref<!fir.array<1024x1024xf64>>) -> memref<1024x1024xf64>
    %16 = fir.convert %c1 : (index) -> i32
    %17 = fir.convert %2 : (!fir.ref<i32>) -> memref<i32>
    %18 = fir.convert %2 : (!fir.ref<i32>) -> memref<i32>
    %19 = fir.convert %c1 : (index) -> i32
    %20 = fir.convert %0 : (!fir.ref<i32>) -> memref<i32>
    %21 = fir.do_loop %arg9 = %c2 to %c1023 step %c1 iter_args(%arg10 = %6) -> (i32) {
      memref.store %arg10, %7[] : memref<i32>
      %23 = fir.do_loop %arg11 = %c2 to %c1023 step %c1 iter_args(%arg12 = %6) -> (i32) {
        memref.store %arg12, %8[] : memref<i32>
        %26 = memref.load %9[] : memref<i32>
        %27 = arith.subi %26, %c1_i32 overflow<nsw> : i32
        %28 = fir.convert %27 : (i32) -> i64
        %29 = memref.load %10[] : memref<i32>
        %30 = fir.convert %29 : (i32) -> i64
        %31 = arith.index_cast %28 : i64 to index
        %32 = arith.subi %31, %c1 : index
        %33 = arith.index_cast %30 : i64 to index
        %34 = arith.subi %33, %c1 : index
        %35 = memref.load %11[%34, %32] : memref<1024x1024xf64>
        %36 = arith.addi %26, %c1_i32 overflow<nsw> : i32
        %37 = fir.convert %36 : (i32) -> i64
        %38 = arith.index_cast %37 : i64 to index
        %39 = arith.subi %38, %c1 : index
        %40 = arith.index_cast %30 : i64 to index
        %41 = arith.subi %40, %c1 : index
        %42 = memref.load %12[%41, %39] : memref<1024x1024xf64>
        %43 = arith.addf %35, %42 fastmath<contract> : f64
        %44 = fir.convert %26 : (i32) -> i64
        %45 = arith.subi %29, %c1_i32 overflow<nsw> : i32
        %46 = fir.convert %45 : (i32) -> i64
        %47 = arith.index_cast %44 : i64 to index
        %48 = arith.subi %47, %c1 : index
        %49 = arith.index_cast %46 : i64 to index
        %50 = arith.subi %49, %c1 : index
        %51 = memref.load %13[%50, %48] : memref<1024x1024xf64>
        %52 = arith.addf %43, %51 fastmath<contract> : f64
        %53 = arith.addi %29, %c1_i32 overflow<nsw> : i32
        %54 = fir.convert %53 : (i32) -> i64
        %55 = arith.index_cast %44 : i64 to index
        %56 = arith.subi %55, %c1 : index
        %57 = arith.index_cast %54 : i64 to index
        %58 = arith.subi %57, %c1 : index
        %59 = memref.load %14[%58, %56] : memref<1024x1024xf64>
        %60 = arith.addf %52, %59 fastmath<contract> : f64
        %61 = arith.index_cast %44 : i64 to index
        %62 = arith.subi %61, %c1 : index
        %63 = arith.index_cast %30 : i64 to index
        %64 = arith.subi %63, %c1 : index
        memref.store %60, %15[%64, %62] : memref<1024x1024xf64>
        %65 = memref.load %17[] : memref<i32>
        %66 = arith.addi %65, %16 overflow<nsw> : i32
        fir.result %66 : i32
      }
      memref.store %23, %18[] : memref<i32>
      %24 = memref.load %20[] : memref<i32>
      %25 = arith.addi %24, %19 overflow<nsw> : i32
      fir.result %25 : i32
    }
    %22 = fir.convert %0 : (!fir.ref<i32>) -> memref<i32>
    memref.store %21, %22[] : memref<i32>
    omp.terminator
  }
}


// -----// IR Dump After CSEPass (cse) //----- //
module {
  func.func @kernel(%arg0: !fir.ref<i32> {jit.arg_type = 1 : ui32, jit.literal_val = 0 : i64}, %arg1: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg2: !fir.ref<i32> {jit.arg_type = 1 : ui32, jit.literal_val = 0 : i64}, %arg3: !fir.ref<!fir.array<1024x1024xf64>> {jit.arg_type = 1 : ui32}, %arg4: !fir.ref<!fir.array<1024x1024xf64>> {jit.arg_type = 1 : ui32}, %arg5: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg6: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg7: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg8: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}) {
    %c1024 = arith.constant 1024 : index
    %c1_i32 = arith.constant 1 : i32
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c1023 = arith.constant 1023 : index
    %0 = fir.declare %arg0 {uniq_name = "_QFFrun_benchmarkEj"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %1 = fir.declare %arg1 {uniq_name = "_QFFrun_benchmarkEn"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %2 = fir.declare %arg2 {uniq_name = "_QFFrun_benchmarkEi"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %3 = fir.shape %c1024, %c1024 : (index, index) -> !fir.shape<2>
    %4 = fir.declare %arg3(%3) {uniq_name = "_QFFrun_benchmarkEy"} : (!fir.ref<!fir.array<1024x1024xf64>>, !fir.shape<2>) -> !fir.ref<!fir.array<1024x1024xf64>>
    %5 = fir.declare %arg4(%3) {uniq_name = "_QFFrun_benchmarkEx"} : (!fir.ref<!fir.array<1024x1024xf64>>, !fir.shape<2>) -> !fir.ref<!fir.array<1024x1024xf64>>
    %6 = fir.convert %c2 : (index) -> i32
    %7 = fir.convert %0 : (!fir.ref<i32>) -> memref<i32>
    %8 = fir.convert %2 : (!fir.ref<i32>) -> memref<i32>
    %9 = fir.convert %5 : (!fir.ref<!fir.array<1024x1024xf64>>) -> memref<1024x1024xf64>
    %10 = fir.convert %4 : (!fir.ref<!fir.array<1024x1024xf64>>) -> memref<1024x1024xf64>
    %11 = fir.convert %c1 : (index) -> i32
    %12 = fir.do_loop %arg9 = %c2 to %c1023 step %c1 iter_args(%arg10 = %6) -> (i32) {
      memref.store %arg10, %7[] : memref<i32>
      %13 = fir.do_loop %arg11 = %c2 to %c1023 step %c1 iter_args(%arg12 = %6) -> (i32) {
        memref.store %arg12, %8[] : memref<i32>
        %16 = memref.load %8[] : memref<i32>
        %17 = arith.subi %16, %c1_i32 overflow<nsw> : i32
        %18 = fir.convert %17 : (i32) -> i64
        %19 = memref.load %7[] : memref<i32>
        %20 = fir.convert %19 : (i32) -> i64
        %21 = arith.index_cast %18 : i64 to index
        %22 = arith.subi %21, %c1 : index
        %23 = arith.index_cast %20 : i64 to index
        %24 = arith.subi %23, %c1 : index
        %25 = memref.load %9[%24, %22] : memref<1024x1024xf64>
        %26 = arith.addi %16, %c1_i32 overflow<nsw> : i32
        %27 = fir.convert %26 : (i32) -> i64
        %28 = arith.index_cast %27 : i64 to index
        %29 = arith.subi %28, %c1 : index
        %30 = memref.load %9[%24, %29] : memref<1024x1024xf64>
        %31 = arith.addf %25, %30 fastmath<contract> : f64
        %32 = fir.convert %16 : (i32) -> i64
        %33 = arith.subi %19, %c1_i32 overflow<nsw> : i32
        %34 = fir.convert %33 : (i32) -> i64
        %35 = arith.index_cast %32 : i64 to index
        %36 = arith.subi %35, %c1 : index
        %37 = arith.index_cast %34 : i64 to index
        %38 = arith.subi %37, %c1 : index
        %39 = memref.load %9[%38, %36] : memref<1024x1024xf64>
        %40 = arith.addf %31, %39 fastmath<contract> : f64
        %41 = arith.addi %19, %c1_i32 overflow<nsw> : i32
        %42 = fir.convert %41 : (i32) -> i64
        %43 = arith.index_cast %42 : i64 to index
        %44 = arith.subi %43, %c1 : index
        %45 = memref.load %9[%44, %36] : memref<1024x1024xf64>
        %46 = arith.addf %40, %45 fastmath<contract> : f64
        memref.store %46, %10[%24, %36] : memref<1024x1024xf64>
        %47 = memref.load %8[] : memref<i32>
        %48 = arith.addi %47, %11 overflow<nsw> : i32
        fir.result %48 : i32
      }
      memref.store %13, %8[] : memref<i32>
      %14 = memref.load %7[] : memref<i32>
      %15 = arith.addi %14, %11 overflow<nsw> : i32
      fir.result %15 : i32
    }
    memref.store %12, %7[] : memref<i32>
    omp.terminator
  }
}


// -----// IR Dump After CanonicalizerPass (canonicalize) //----- //
module {
  func.func @kernel(%arg0: !fir.ref<i32> {jit.arg_type = 1 : ui32, jit.literal_val = 0 : i64}, %arg1: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg2: !fir.ref<i32> {jit.arg_type = 1 : ui32, jit.literal_val = 0 : i64}, %arg3: !fir.ref<!fir.array<1024x1024xf64>> {jit.arg_type = 1 : ui32}, %arg4: !fir.ref<!fir.array<1024x1024xf64>> {jit.arg_type = 1 : ui32}, %arg5: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg6: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg7: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg8: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}) {
    %c1024 = arith.constant 1024 : index
    %c1_i32 = arith.constant 1 : i32
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c1023 = arith.constant 1023 : index
    %0 = fir.declare %arg0 {uniq_name = "_QFFrun_benchmarkEj"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %1 = fir.declare %arg1 {uniq_name = "_QFFrun_benchmarkEn"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %2 = fir.declare %arg2 {uniq_name = "_QFFrun_benchmarkEi"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %3 = fir.shape %c1024, %c1024 : (index, index) -> !fir.shape<2>
    %4 = fir.declare %arg3(%3) {uniq_name = "_QFFrun_benchmarkEy"} : (!fir.ref<!fir.array<1024x1024xf64>>, !fir.shape<2>) -> !fir.ref<!fir.array<1024x1024xf64>>
    %5 = fir.declare %arg4(%3) {uniq_name = "_QFFrun_benchmarkEx"} : (!fir.ref<!fir.array<1024x1024xf64>>, !fir.shape<2>) -> !fir.ref<!fir.array<1024x1024xf64>>
    %6 = fir.convert %c2 : (index) -> i32
    %7 = fir.convert %0 : (!fir.ref<i32>) -> memref<i32>
    %8 = fir.convert %2 : (!fir.ref<i32>) -> memref<i32>
    %9 = fir.convert %5 : (!fir.ref<!fir.array<1024x1024xf64>>) -> memref<1024x1024xf64>
    %10 = fir.convert %4 : (!fir.ref<!fir.array<1024x1024xf64>>) -> memref<1024x1024xf64>
    %11 = fir.convert %c1 : (index) -> i32
    %12 = fir.do_loop %arg9 = %c2 to %c1023 step %c1 iter_args(%arg10 = %6) -> (i32) {
      memref.store %arg10, %7[] : memref<i32>
      %13 = fir.do_loop %arg11 = %c2 to %c1023 step %c1 iter_args(%arg12 = %6) -> (i32) {
        memref.store %arg12, %8[] : memref<i32>
        %16 = memref.load %8[] : memref<i32>
        %17 = arith.subi %16, %c1_i32 overflow<nsw> : i32
        %18 = fir.convert %17 : (i32) -> i64
        %19 = memref.load %7[] : memref<i32>
        %20 = fir.convert %19 : (i32) -> i64
        %21 = arith.index_cast %18 : i64 to index
        %22 = arith.subi %21, %c1 : index
        %23 = arith.index_cast %20 : i64 to index
        %24 = arith.subi %23, %c1 : index
        %25 = memref.load %9[%24, %22] : memref<1024x1024xf64>
        %26 = arith.addi %16, %c1_i32 overflow<nsw> : i32
        %27 = fir.convert %26 : (i32) -> i64
        %28 = arith.index_cast %27 : i64 to index
        %29 = arith.subi %28, %c1 : index
        %30 = memref.load %9[%24, %29] : memref<1024x1024xf64>
        %31 = arith.addf %25, %30 fastmath<contract> : f64
        %32 = fir.convert %16 : (i32) -> i64
        %33 = arith.subi %19, %c1_i32 overflow<nsw> : i32
        %34 = fir.convert %33 : (i32) -> i64
        %35 = arith.index_cast %32 : i64 to index
        %36 = arith.subi %35, %c1 : index
        %37 = arith.index_cast %34 : i64 to index
        %38 = arith.subi %37, %c1 : index
        %39 = memref.load %9[%38, %36] : memref<1024x1024xf64>
        %40 = arith.addf %31, %39 fastmath<contract> : f64
        %41 = arith.addi %19, %c1_i32 overflow<nsw> : i32
        %42 = fir.convert %41 : (i32) -> i64
        %43 = arith.index_cast %42 : i64 to index
        %44 = arith.subi %43, %c1 : index
        %45 = memref.load %9[%44, %36] : memref<1024x1024xf64>
        %46 = arith.addf %40, %45 fastmath<contract> : f64
        memref.store %46, %10[%24, %36] : memref<1024x1024xf64>
        %47 = memref.load %8[] : memref<i32>
        %48 = arith.addi %47, %11 overflow<nsw> : i32
        fir.result %48 : i32
      }
      memref.store %13, %8[] : memref<i32>
      %14 = memref.load %7[] : memref<i32>
      %15 = arith.addi %14, %11 overflow<nsw> : i32
      fir.result %15 : i32
    }
    memref.store %12, %7[] : memref<i32>
    omp.terminator
  }
}


// -----// IR Dump After {anonymous}::CleanFIRLoopPass (jforce-clean-fir-loop) //----- //
func.func @kernel(%arg0: !fir.ref<i32> {jit.arg_type = 1 : ui32, jit.literal_val = 0 : i64}, %arg1: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg2: !fir.ref<i32> {jit.arg_type = 1 : ui32, jit.literal_val = 0 : i64}, %arg3: !fir.ref<!fir.array<1024x1024xf64>> {jit.arg_type = 1 : ui32}, %arg4: !fir.ref<!fir.array<1024x1024xf64>> {jit.arg_type = 1 : ui32}, %arg5: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg6: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg7: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg8: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}) {
  %c1024 = arith.constant 1024 : index
  %c1_i32 = arith.constant 1 : i32
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c1023 = arith.constant 1023 : index
  %0 = fir.declare %arg0 {uniq_name = "_QFFrun_benchmarkEj"} : (!fir.ref<i32>) -> !fir.ref<i32>
  %1 = fir.declare %arg1 {uniq_name = "_QFFrun_benchmarkEn"} : (!fir.ref<i32>) -> !fir.ref<i32>
  %2 = fir.declare %arg2 {uniq_name = "_QFFrun_benchmarkEi"} : (!fir.ref<i32>) -> !fir.ref<i32>
  %3 = fir.shape %c1024, %c1024 : (index, index) -> !fir.shape<2>
  %4 = fir.declare %arg3(%3) {uniq_name = "_QFFrun_benchmarkEy"} : (!fir.ref<!fir.array<1024x1024xf64>>, !fir.shape<2>) -> !fir.ref<!fir.array<1024x1024xf64>>
  %5 = fir.declare %arg4(%3) {uniq_name = "_QFFrun_benchmarkEx"} : (!fir.ref<!fir.array<1024x1024xf64>>, !fir.shape<2>) -> !fir.ref<!fir.array<1024x1024xf64>>
  %6 = fir.convert %c2 : (index) -> i32
  %7 = fir.convert %0 : (!fir.ref<i32>) -> memref<i32>
  %8 = fir.convert %2 : (!fir.ref<i32>) -> memref<i32>
  %9 = fir.convert %5 : (!fir.ref<!fir.array<1024x1024xf64>>) -> memref<1024x1024xf64>
  %10 = fir.convert %4 : (!fir.ref<!fir.array<1024x1024xf64>>) -> memref<1024x1024xf64>
  %11 = fir.convert %c1 : (index) -> i32
  fir.do_loop %arg9 = %c2 to %c1023 step %c1 {
    %13 = fir.convert %arg9 : (index) -> i32
    memref.store %13, %7[] : memref<i32>
    fir.do_loop %arg10 = %c2 to %c1023 step %c1 {
      %17 = fir.convert %arg10 : (index) -> i32
      memref.store %17, %8[] : memref<i32>
      %18 = memref.load %8[] : memref<i32>
      %19 = arith.subi %18, %c1_i32 overflow<nsw> : i32
      %20 = fir.convert %19 : (i32) -> i64
      %21 = memref.load %7[] : memref<i32>
      %22 = fir.convert %21 : (i32) -> i64
      %23 = arith.index_cast %20 : i64 to index
      %24 = arith.subi %23, %c1 : index
      %25 = arith.index_cast %22 : i64 to index
      %26 = arith.subi %25, %c1 : index
      %27 = memref.load %9[%26, %24] : memref<1024x1024xf64>
      %28 = arith.addi %18, %c1_i32 overflow<nsw> : i32
      %29 = fir.convert %28 : (i32) -> i64
      %30 = arith.index_cast %29 : i64 to index
      %31 = arith.subi %30, %c1 : index
      %32 = memref.load %9[%26, %31] : memref<1024x1024xf64>
      %33 = arith.addf %27, %32 fastmath<contract> : f64
      %34 = fir.convert %18 : (i32) -> i64
      %35 = arith.subi %21, %c1_i32 overflow<nsw> : i32
      %36 = fir.convert %35 : (i32) -> i64
      %37 = arith.index_cast %34 : i64 to index
      %38 = arith.subi %37, %c1 : index
      %39 = arith.index_cast %36 : i64 to index
      %40 = arith.subi %39, %c1 : index
      %41 = memref.load %9[%40, %38] : memref<1024x1024xf64>
      %42 = arith.addf %33, %41 fastmath<contract> : f64
      %43 = arith.addi %21, %c1_i32 overflow<nsw> : i32
      %44 = fir.convert %43 : (i32) -> i64
      %45 = arith.index_cast %44 : i64 to index
      %46 = arith.subi %45, %c1 : index
      %47 = memref.load %9[%46, %38] : memref<1024x1024xf64>
      %48 = arith.addf %42, %47 fastmath<contract> : f64
      memref.store %48, %10[%26, %38] : memref<1024x1024xf64>
      %49 = memref.load %8[] : memref<i32>
      %50 = arith.addi %49, %11 overflow<nsw> : i32
      memref.store %50, %8[] : memref<i32>
    }
    %14 = memref.load %8[] : memref<i32>
    memref.store %14, %8[] : memref<i32>
    %15 = memref.load %7[] : memref<i32>
    %16 = arith.addi %15, %11 overflow<nsw> : i32
    memref.store %16, %7[] : memref<i32>
  }
  %12 = memref.load %7[] : memref<i32>
  memref.store %12, %7[] : memref<i32>
  omp.terminator
}

// -----// IR Dump After CanonicalizerPass (canonicalize) //----- //
module {
  func.func @kernel(%arg0: !fir.ref<i32> {jit.arg_type = 1 : ui32, jit.literal_val = 0 : i64}, %arg1: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg2: !fir.ref<i32> {jit.arg_type = 1 : ui32, jit.literal_val = 0 : i64}, %arg3: !fir.ref<!fir.array<1024x1024xf64>> {jit.arg_type = 1 : ui32}, %arg4: !fir.ref<!fir.array<1024x1024xf64>> {jit.arg_type = 1 : ui32}, %arg5: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg6: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg7: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg8: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}) {
    %c1024 = arith.constant 1024 : index
    %c1_i32 = arith.constant 1 : i32
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c1023 = arith.constant 1023 : index
    %0 = fir.declare %arg0 {uniq_name = "_QFFrun_benchmarkEj"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %1 = fir.declare %arg1 {uniq_name = "_QFFrun_benchmarkEn"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %2 = fir.declare %arg2 {uniq_name = "_QFFrun_benchmarkEi"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %3 = fir.shape %c1024, %c1024 : (index, index) -> !fir.shape<2>
    %4 = fir.declare %arg3(%3) {uniq_name = "_QFFrun_benchmarkEy"} : (!fir.ref<!fir.array<1024x1024xf64>>, !fir.shape<2>) -> !fir.ref<!fir.array<1024x1024xf64>>
    %5 = fir.declare %arg4(%3) {uniq_name = "_QFFrun_benchmarkEx"} : (!fir.ref<!fir.array<1024x1024xf64>>, !fir.shape<2>) -> !fir.ref<!fir.array<1024x1024xf64>>
    %6 = fir.convert %0 : (!fir.ref<i32>) -> memref<i32>
    %7 = fir.convert %2 : (!fir.ref<i32>) -> memref<i32>
    %8 = fir.convert %5 : (!fir.ref<!fir.array<1024x1024xf64>>) -> memref<1024x1024xf64>
    %9 = fir.convert %4 : (!fir.ref<!fir.array<1024x1024xf64>>) -> memref<1024x1024xf64>
    %10 = fir.convert %c1 : (index) -> i32
    fir.do_loop %arg9 = %c2 to %c1023 step %c1 {
      %12 = fir.convert %arg9 : (index) -> i32
      memref.store %12, %6[] : memref<i32>
      fir.do_loop %arg10 = %c2 to %c1023 step %c1 {
        %16 = fir.convert %arg10 : (index) -> i32
        memref.store %16, %7[] : memref<i32>
        %17 = memref.load %7[] : memref<i32>
        %18 = arith.subi %17, %c1_i32 overflow<nsw> : i32
        %19 = fir.convert %18 : (i32) -> i64
        %20 = memref.load %6[] : memref<i32>
        %21 = fir.convert %20 : (i32) -> i64
        %22 = arith.index_cast %19 : i64 to index
        %23 = arith.subi %22, %c1 : index
        %24 = arith.index_cast %21 : i64 to index
        %25 = arith.subi %24, %c1 : index
        %26 = memref.load %8[%25, %23] : memref<1024x1024xf64>
        %27 = arith.addi %17, %c1_i32 overflow<nsw> : i32
        %28 = fir.convert %27 : (i32) -> i64
        %29 = arith.index_cast %28 : i64 to index
        %30 = arith.subi %29, %c1 : index
        %31 = memref.load %8[%25, %30] : memref<1024x1024xf64>
        %32 = arith.addf %26, %31 fastmath<contract> : f64
        %33 = fir.convert %17 : (i32) -> i64
        %34 = arith.subi %20, %c1_i32 overflow<nsw> : i32
        %35 = fir.convert %34 : (i32) -> i64
        %36 = arith.index_cast %33 : i64 to index
        %37 = arith.subi %36, %c1 : index
        %38 = arith.index_cast %35 : i64 to index
        %39 = arith.subi %38, %c1 : index
        %40 = memref.load %8[%39, %37] : memref<1024x1024xf64>
        %41 = arith.addf %32, %40 fastmath<contract> : f64
        %42 = arith.addi %20, %c1_i32 overflow<nsw> : i32
        %43 = fir.convert %42 : (i32) -> i64
        %44 = arith.index_cast %43 : i64 to index
        %45 = arith.subi %44, %c1 : index
        %46 = memref.load %8[%45, %37] : memref<1024x1024xf64>
        %47 = arith.addf %41, %46 fastmath<contract> : f64
        memref.store %47, %9[%25, %37] : memref<1024x1024xf64>
        %48 = memref.load %7[] : memref<i32>
        %49 = arith.addi %48, %10 overflow<nsw> : i32
        memref.store %49, %7[] : memref<i32>
      }
      %13 = memref.load %7[] : memref<i32>
      memref.store %13, %7[] : memref<i32>
      %14 = memref.load %6[] : memref<i32>
      %15 = arith.addi %14, %10 overflow<nsw> : i32
      memref.store %15, %6[] : memref<i32>
    }
    %11 = memref.load %6[] : memref<i32>
    memref.store %11, %6[] : memref<i32>
    omp.terminator
  }
}


// -----// IR Dump After {anonymous}::CleanFIROpsPass (jforce-clean-fir-op) //----- //
func.func @kernel(%arg0: !fir.ref<i32> {jit.arg_type = 1 : ui32, jit.literal_val = 0 : i64}, %arg1: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg2: !fir.ref<i32> {jit.arg_type = 1 : ui32, jit.literal_val = 0 : i64}, %arg3: !fir.ref<!fir.array<1024x1024xf64>> {jit.arg_type = 1 : ui32}, %arg4: !fir.ref<!fir.array<1024x1024xf64>> {jit.arg_type = 1 : ui32}, %arg5: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg6: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg7: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg8: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}) {
  %c1024 = arith.constant 1024 : index
  %c1_i32 = arith.constant 1 : i32
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c1023 = arith.constant 1023 : index
  %0 = fir.declare %arg0 {uniq_name = "_QFFrun_benchmarkEj"} : (!fir.ref<i32>) -> !fir.ref<i32>
  %1 = fir.declare %arg1 {uniq_name = "_QFFrun_benchmarkEn"} : (!fir.ref<i32>) -> !fir.ref<i32>
  %2 = fir.declare %arg2 {uniq_name = "_QFFrun_benchmarkEi"} : (!fir.ref<i32>) -> !fir.ref<i32>
  %3 = fir.shape %c1024, %c1024 : (index, index) -> !fir.shape<2>
  %4 = fir.declare %arg3(%3) {uniq_name = "_QFFrun_benchmarkEy"} : (!fir.ref<!fir.array<1024x1024xf64>>, !fir.shape<2>) -> !fir.ref<!fir.array<1024x1024xf64>>
  %5 = fir.declare %arg4(%3) {uniq_name = "_QFFrun_benchmarkEx"} : (!fir.ref<!fir.array<1024x1024xf64>>, !fir.shape<2>) -> !fir.ref<!fir.array<1024x1024xf64>>
  %6 = fir.convert %0 : (!fir.ref<i32>) -> memref<i32>
  %7 = fir.convert %2 : (!fir.ref<i32>) -> memref<i32>
  %8 = fir.convert %5 : (!fir.ref<!fir.array<1024x1024xf64>>) -> memref<1024x1024xf64>
  %9 = fir.convert %4 : (!fir.ref<!fir.array<1024x1024xf64>>) -> memref<1024x1024xf64>
  %10 = fir.convert %c1 : (index) -> i32
  fir.do_loop %arg9 = %c2 to %c1023 step %c1 {
    %12 = arith.index_cast %arg9 : index to i32
    memref.store %12, %6[] : memref<i32>
    fir.do_loop %arg10 = %c2 to %c1023 step %c1 {
      %16 = arith.index_cast %arg10 : index to i32
      memref.store %16, %7[] : memref<i32>
      %17 = memref.load %7[] : memref<i32>
      %18 = arith.subi %17, %c1_i32 overflow<nsw> : i32
      %19 = arith.extsi %18 : i32 to i64
      %20 = memref.load %6[] : memref<i32>
      %21 = arith.extsi %20 : i32 to i64
      %22 = arith.index_cast %19 : i64 to index
      %23 = arith.subi %22, %c1 : index
      %24 = arith.index_cast %21 : i64 to index
      %25 = arith.subi %24, %c1 : index
      %26 = memref.load %8[%25, %23] : memref<1024x1024xf64>
      %27 = arith.addi %17, %c1_i32 overflow<nsw> : i32
      %28 = arith.extsi %27 : i32 to i64
      %29 = arith.index_cast %28 : i64 to index
      %30 = arith.subi %29, %c1 : index
      %31 = memref.load %8[%25, %30] : memref<1024x1024xf64>
      %32 = arith.addf %26, %31 fastmath<contract> : f64
      %33 = arith.extsi %17 : i32 to i64
      %34 = arith.subi %20, %c1_i32 overflow<nsw> : i32
      %35 = arith.extsi %34 : i32 to i64
      %36 = arith.index_cast %33 : i64 to index
      %37 = arith.subi %36, %c1 : index
      %38 = arith.index_cast %35 : i64 to index
      %39 = arith.subi %38, %c1 : index
      %40 = memref.load %8[%39, %37] : memref<1024x1024xf64>
      %41 = arith.addf %32, %40 fastmath<contract> : f64
      %42 = arith.addi %20, %c1_i32 overflow<nsw> : i32
      %43 = arith.extsi %42 : i32 to i64
      %44 = arith.index_cast %43 : i64 to index
      %45 = arith.subi %44, %c1 : index
      %46 = memref.load %8[%45, %37] : memref<1024x1024xf64>
      %47 = arith.addf %41, %46 fastmath<contract> : f64
      memref.store %47, %9[%25, %37] : memref<1024x1024xf64>
      %48 = memref.load %7[] : memref<i32>
      %49 = arith.addi %48, %10 overflow<nsw> : i32
      memref.store %49, %7[] : memref<i32>
    }
    %13 = memref.load %7[] : memref<i32>
    memref.store %13, %7[] : memref<i32>
    %14 = memref.load %6[] : memref<i32>
    %15 = arith.addi %14, %10 overflow<nsw> : i32
    memref.store %15, %6[] : memref<i32>
  }
  %11 = memref.load %6[] : memref<i32>
  memref.store %11, %6[] : memref<i32>
  omp.terminator
}

// -----// IR Dump After AffineDialectPromotion (promote-to-affine) //----- //
func.func @kernel(%arg0: !fir.ref<i32> {jit.arg_type = 1 : ui32, jit.literal_val = 0 : i64}, %arg1: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg2: !fir.ref<i32> {jit.arg_type = 1 : ui32, jit.literal_val = 0 : i64}, %arg3: !fir.ref<!fir.array<1024x1024xf64>> {jit.arg_type = 1 : ui32}, %arg4: !fir.ref<!fir.array<1024x1024xf64>> {jit.arg_type = 1 : ui32}, %arg5: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg6: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg7: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg8: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}) {
  %c1024 = arith.constant 1024 : index
  %c1_i32 = arith.constant 1 : i32
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c1023 = arith.constant 1023 : index
  %0 = fir.declare %arg0 {uniq_name = "_QFFrun_benchmarkEj"} : (!fir.ref<i32>) -> !fir.ref<i32>
  %1 = fir.declare %arg1 {uniq_name = "_QFFrun_benchmarkEn"} : (!fir.ref<i32>) -> !fir.ref<i32>
  %2 = fir.declare %arg2 {uniq_name = "_QFFrun_benchmarkEi"} : (!fir.ref<i32>) -> !fir.ref<i32>
  %3 = fir.shape %c1024, %c1024 : (index, index) -> !fir.shape<2>
  %4 = fir.declare %arg3(%3) {uniq_name = "_QFFrun_benchmarkEy"} : (!fir.ref<!fir.array<1024x1024xf64>>, !fir.shape<2>) -> !fir.ref<!fir.array<1024x1024xf64>>
  %5 = fir.declare %arg4(%3) {uniq_name = "_QFFrun_benchmarkEx"} : (!fir.ref<!fir.array<1024x1024xf64>>, !fir.shape<2>) -> !fir.ref<!fir.array<1024x1024xf64>>
  %6 = fir.convert %0 : (!fir.ref<i32>) -> memref<i32>
  %7 = fir.convert %2 : (!fir.ref<i32>) -> memref<i32>
  %8 = fir.convert %5 : (!fir.ref<!fir.array<1024x1024xf64>>) -> memref<1024x1024xf64>
  %9 = fir.convert %4 : (!fir.ref<!fir.array<1024x1024xf64>>) -> memref<1024x1024xf64>
  %10 = fir.convert %c1 : (index) -> i32
  affine.for %arg9 = %c2 to affine_map<()[s0] -> (s0 + 1)>()[%c1023] {
    %12 = arith.index_cast %arg9 : index to i32
    memref.store %12, %6[] : memref<i32>
    affine.for %arg10 = %c2 to affine_map<()[s0] -> (s0 + 1)>()[%c1023] {
      %16 = arith.index_cast %arg10 : index to i32
      memref.store %16, %7[] : memref<i32>
      %17 = memref.load %7[] : memref<i32>
      %18 = arith.subi %17, %c1_i32 overflow<nsw> : i32
      %19 = arith.extsi %18 : i32 to i64
      %20 = memref.load %6[] : memref<i32>
      %21 = arith.extsi %20 : i32 to i64
      %22 = arith.index_cast %19 : i64 to index
      %23 = arith.subi %22, %c1 : index
      %24 = arith.index_cast %21 : i64 to index
      %25 = arith.subi %24, %c1 : index
      %26 = memref.load %8[%25, %23] : memref<1024x1024xf64>
      %27 = arith.addi %17, %c1_i32 overflow<nsw> : i32
      %28 = arith.extsi %27 : i32 to i64
      %29 = arith.index_cast %28 : i64 to index
      %30 = arith.subi %29, %c1 : index
      %31 = memref.load %8[%25, %30] : memref<1024x1024xf64>
      %32 = arith.addf %26, %31 fastmath<contract> : f64
      %33 = arith.extsi %17 : i32 to i64
      %34 = arith.subi %20, %c1_i32 overflow<nsw> : i32
      %35 = arith.extsi %34 : i32 to i64
      %36 = arith.index_cast %33 : i64 to index
      %37 = arith.subi %36, %c1 : index
      %38 = arith.index_cast %35 : i64 to index
      %39 = arith.subi %38, %c1 : index
      %40 = memref.load %8[%39, %37] : memref<1024x1024xf64>
      %41 = arith.addf %32, %40 fastmath<contract> : f64
      %42 = arith.addi %20, %c1_i32 overflow<nsw> : i32
      %43 = arith.extsi %42 : i32 to i64
      %44 = arith.index_cast %43 : i64 to index
      %45 = arith.subi %44, %c1 : index
      %46 = memref.load %8[%45, %37] : memref<1024x1024xf64>
      %47 = arith.addf %41, %46 fastmath<contract> : f64
      memref.store %47, %9[%25, %37] : memref<1024x1024xf64>
      %48 = memref.load %7[] : memref<i32>
      %49 = arith.addi %48, %10 overflow<nsw> : i32
      memref.store %49, %7[] : memref<i32>
    }
    %13 = memref.load %7[] : memref<i32>
    memref.store %13, %7[] : memref<i32>
    %14 = memref.load %6[] : memref<i32>
    %15 = arith.addi %14, %10 overflow<nsw> : i32
    memref.store %15, %6[] : memref<i32>
  }
  %11 = memref.load %6[] : memref<i32>
  memref.store %11, %6[] : memref<i32>
  omp.terminator
}

// -----// IR Dump After AffineLoopNormalize (affine-loop-normalize) //----- //
func.func @kernel(%arg0: !fir.ref<i32> {jit.arg_type = 1 : ui32, jit.literal_val = 0 : i64}, %arg1: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg2: !fir.ref<i32> {jit.arg_type = 1 : ui32, jit.literal_val = 0 : i64}, %arg3: !fir.ref<!fir.array<1024x1024xf64>> {jit.arg_type = 1 : ui32}, %arg4: !fir.ref<!fir.array<1024x1024xf64>> {jit.arg_type = 1 : ui32}, %arg5: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg6: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg7: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg8: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}) {
  %c1024 = arith.constant 1024 : index
  %c1_i32 = arith.constant 1 : i32
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c1023 = arith.constant 1023 : index
  %0 = fir.declare %arg0 {uniq_name = "_QFFrun_benchmarkEj"} : (!fir.ref<i32>) -> !fir.ref<i32>
  %1 = fir.declare %arg1 {uniq_name = "_QFFrun_benchmarkEn"} : (!fir.ref<i32>) -> !fir.ref<i32>
  %2 = fir.declare %arg2 {uniq_name = "_QFFrun_benchmarkEi"} : (!fir.ref<i32>) -> !fir.ref<i32>
  %3 = fir.shape %c1024, %c1024 : (index, index) -> !fir.shape<2>
  %4 = fir.declare %arg3(%3) {uniq_name = "_QFFrun_benchmarkEy"} : (!fir.ref<!fir.array<1024x1024xf64>>, !fir.shape<2>) -> !fir.ref<!fir.array<1024x1024xf64>>
  %5 = fir.declare %arg4(%3) {uniq_name = "_QFFrun_benchmarkEx"} : (!fir.ref<!fir.array<1024x1024xf64>>, !fir.shape<2>) -> !fir.ref<!fir.array<1024x1024xf64>>
  %6 = fir.convert %0 : (!fir.ref<i32>) -> memref<i32>
  %7 = fir.convert %2 : (!fir.ref<i32>) -> memref<i32>
  %8 = fir.convert %5 : (!fir.ref<!fir.array<1024x1024xf64>>) -> memref<1024x1024xf64>
  %9 = fir.convert %4 : (!fir.ref<!fir.array<1024x1024xf64>>) -> memref<1024x1024xf64>
  %10 = fir.convert %c1 : (index) -> i32
  affine.for %arg9 = 0 to 1022 {
    %12 = affine.apply affine_map<(d0) -> (d0 + 2)>(%arg9)
    %13 = arith.index_cast %12 : index to i32
    memref.store %13, %6[] : memref<i32>
    affine.for %arg10 = 0 to 1022 {
      %17 = affine.apply affine_map<(d0) -> (d0 + 2)>(%arg10)
      %18 = arith.index_cast %17 : index to i32
      memref.store %18, %7[] : memref<i32>
      %19 = memref.load %7[] : memref<i32>
      %20 = arith.subi %19, %c1_i32 overflow<nsw> : i32
      %21 = arith.extsi %20 : i32 to i64
      %22 = memref.load %6[] : memref<i32>
      %23 = arith.extsi %22 : i32 to i64
      %24 = arith.index_cast %21 : i64 to index
      %25 = arith.subi %24, %c1 : index
      %26 = arith.index_cast %23 : i64 to index
      %27 = arith.subi %26, %c1 : index
      %28 = memref.load %8[%27, %25] : memref<1024x1024xf64>
      %29 = arith.addi %19, %c1_i32 overflow<nsw> : i32
      %30 = arith.extsi %29 : i32 to i64
      %31 = arith.index_cast %30 : i64 to index
      %32 = arith.subi %31, %c1 : index
      %33 = memref.load %8[%27, %32] : memref<1024x1024xf64>
      %34 = arith.addf %28, %33 fastmath<contract> : f64
      %35 = arith.extsi %19 : i32 to i64
      %36 = arith.subi %22, %c1_i32 overflow<nsw> : i32
      %37 = arith.extsi %36 : i32 to i64
      %38 = arith.index_cast %35 : i64 to index
      %39 = arith.subi %38, %c1 : index
      %40 = arith.index_cast %37 : i64 to index
      %41 = arith.subi %40, %c1 : index
      %42 = memref.load %8[%41, %39] : memref<1024x1024xf64>
      %43 = arith.addf %34, %42 fastmath<contract> : f64
      %44 = arith.addi %22, %c1_i32 overflow<nsw> : i32
      %45 = arith.extsi %44 : i32 to i64
      %46 = arith.index_cast %45 : i64 to index
      %47 = arith.subi %46, %c1 : index
      %48 = memref.load %8[%47, %39] : memref<1024x1024xf64>
      %49 = arith.addf %43, %48 fastmath<contract> : f64
      memref.store %49, %9[%27, %39] : memref<1024x1024xf64>
      %50 = memref.load %7[] : memref<i32>
      %51 = arith.addi %50, %10 overflow<nsw> : i32
      memref.store %51, %7[] : memref<i32>
    }
    %14 = memref.load %7[] : memref<i32>
    memref.store %14, %7[] : memref<i32>
    %15 = memref.load %6[] : memref<i32>
    %16 = arith.addi %15, %10 overflow<nsw> : i32
    memref.store %16, %6[] : memref<i32>
  }
  %11 = memref.load %6[] : memref<i32>
  memref.store %11, %6[] : memref<i32>
  omp.terminator
}

// -----// IR Dump After CanonicalizerPass (canonicalize) //----- //
#map = affine_map<(d0) -> (d0 + 2)>
module {
  func.func @kernel(%arg0: !fir.ref<i32> {jit.arg_type = 1 : ui32, jit.literal_val = 0 : i64}, %arg1: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg2: !fir.ref<i32> {jit.arg_type = 1 : ui32, jit.literal_val = 0 : i64}, %arg3: !fir.ref<!fir.array<1024x1024xf64>> {jit.arg_type = 1 : ui32}, %arg4: !fir.ref<!fir.array<1024x1024xf64>> {jit.arg_type = 1 : ui32}, %arg5: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg6: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg7: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg8: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}) {
    %c1024 = arith.constant 1024 : index
    %c1_i32 = arith.constant 1 : i32
    %c1 = arith.constant 1 : index
    %0 = fir.declare %arg0 {uniq_name = "_QFFrun_benchmarkEj"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %1 = fir.declare %arg1 {uniq_name = "_QFFrun_benchmarkEn"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %2 = fir.declare %arg2 {uniq_name = "_QFFrun_benchmarkEi"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %3 = fir.shape %c1024, %c1024 : (index, index) -> !fir.shape<2>
    %4 = fir.declare %arg3(%3) {uniq_name = "_QFFrun_benchmarkEy"} : (!fir.ref<!fir.array<1024x1024xf64>>, !fir.shape<2>) -> !fir.ref<!fir.array<1024x1024xf64>>
    %5 = fir.declare %arg4(%3) {uniq_name = "_QFFrun_benchmarkEx"} : (!fir.ref<!fir.array<1024x1024xf64>>, !fir.shape<2>) -> !fir.ref<!fir.array<1024x1024xf64>>
    %6 = fir.convert %0 : (!fir.ref<i32>) -> memref<i32>
    %7 = fir.convert %2 : (!fir.ref<i32>) -> memref<i32>
    %8 = fir.convert %5 : (!fir.ref<!fir.array<1024x1024xf64>>) -> memref<1024x1024xf64>
    %9 = fir.convert %4 : (!fir.ref<!fir.array<1024x1024xf64>>) -> memref<1024x1024xf64>
    %10 = fir.convert %c1 : (index) -> i32
    affine.for %arg9 = 0 to 1022 {
      %12 = affine.apply #map(%arg9)
      %13 = arith.index_cast %12 : index to i32
      memref.store %13, %6[] : memref<i32>
      affine.for %arg10 = 0 to 1022 {
        %17 = affine.apply #map(%arg10)
        %18 = arith.index_cast %17 : index to i32
        memref.store %18, %7[] : memref<i32>
        %19 = memref.load %7[] : memref<i32>
        %20 = arith.subi %19, %c1_i32 overflow<nsw> : i32
        %21 = memref.load %6[] : memref<i32>
        %22 = arith.index_cast %20 : i32 to index
        %23 = arith.subi %22, %c1 : index
        %24 = arith.index_cast %21 : i32 to index
        %25 = arith.subi %24, %c1 : index
        %26 = memref.load %8[%25, %23] : memref<1024x1024xf64>
        %27 = arith.addi %19, %c1_i32 overflow<nsw> : i32
        %28 = arith.index_cast %27 : i32 to index
        %29 = arith.subi %28, %c1 : index
        %30 = memref.load %8[%25, %29] : memref<1024x1024xf64>
        %31 = arith.addf %26, %30 fastmath<contract> : f64
        %32 = arith.subi %21, %c1_i32 overflow<nsw> : i32
        %33 = arith.index_cast %19 : i32 to index
        %34 = arith.subi %33, %c1 : index
        %35 = arith.index_cast %32 : i32 to index
        %36 = arith.subi %35, %c1 : index
        %37 = memref.load %8[%36, %34] : memref<1024x1024xf64>
        %38 = arith.addf %31, %37 fastmath<contract> : f64
        %39 = arith.addi %21, %c1_i32 overflow<nsw> : i32
        %40 = arith.index_cast %39 : i32 to index
        %41 = arith.subi %40, %c1 : index
        %42 = memref.load %8[%41, %34] : memref<1024x1024xf64>
        %43 = arith.addf %38, %42 fastmath<contract> : f64
        memref.store %43, %9[%25, %34] : memref<1024x1024xf64>
        %44 = memref.load %7[] : memref<i32>
        %45 = arith.addi %44, %10 overflow<nsw> : i32
        memref.store %45, %7[] : memref<i32>
      }
      %14 = memref.load %7[] : memref<i32>
      memref.store %14, %7[] : memref<i32>
      %15 = memref.load %6[] : memref<i32>
      %16 = arith.addi %15, %10 overflow<nsw> : i32
      memref.store %16, %6[] : memref<i32>
    }
    %11 = memref.load %6[] : memref<i32>
    memref.store %11, %6[] : memref<i32>
    omp.terminator
  }
}


[DEBUG]compare reading and storeOp
[DEBUG] memref.store %11, %6[] : memref<i32> 
[DEBUG] %11 = memref.load %6[] : memref<i32> 
[DEBUG]compare reading and storeOp
[DEBUG] memref.store %15, %6[] : memref<i32> 
[DEBUG] %15 = arith.addi %14, %10 overflow<nsw> : i32 
[DEBUG]Unexpected readOp: 
[DEBUG] %15 = arith.addi %14, %10 overflow<nsw> : i32 
[DEBUG]compare reading and storeOp
[DEBUG] memref.store %13, %7[] : memref<i32> 
[DEBUG] %13 = memref.load %7[] : memref<i32> 
[DEBUG]compare reading and storeOp
[DEBUG] memref.store %42, %7[] : memref<i32> 
[DEBUG] %42 = arith.addi %41, %10 overflow<nsw> : i32 
[DEBUG]Unexpected readOp: 
[DEBUG] %42 = arith.addi %41, %10 overflow<nsw> : i32 
[DEBUG]compare reading and storeOp
[DEBUG] memref.store %40, %9[%22, %31] : memref<1024x1024xf64> 
[DEBUG] %40 = arith.addf %35, %39 fastmath<contract> : f64 
[DEBUG]Unexpected readOp: 
[DEBUG] %40 = arith.addf %35, %39 fastmath<contract> : f64 
[DEBUG]Dependency chain relies on LoadOp!
[DEBUG]compare reading and storeOp
[DEBUG] memref.store %15, %7[] : memref<i32> 
[DEBUG] %15 = arith.index_cast %14 : index to i32 
[DEBUG]Unexpected readOp: 
[DEBUG] %15 = arith.index_cast %14 : index to i32 
[DEBUG]compare reading and storeOp
[DEBUG] memref.store %12, %6[] : memref<i32> 
[DEBUG] %12 = arith.index_cast %11 : index to i32 
[DEBUG]Unexpected readOp: 
[DEBUG] %12 = arith.index_cast %11 : index to i32 
[DEBUG]compare reading and storeOp
[DEBUG] memref.store %13, %6[] : memref<i32> 
[DEBUG] %13 = arith.addi %12, %10 overflow<nsw> : i32 
[DEBUG]Unexpected readOp: 
[DEBUG] %13 = arith.addi %12, %10 overflow<nsw> : i32 
[DEBUG]compare reading and storeOp
[DEBUG] memref.store %18, %6[] : memref<i32> 
[DEBUG] %18 = arith.addi %17, %10 overflow<nsw> : i32 
[DEBUG]Unexpected readOp: 
[DEBUG] %18 = arith.addi %17, %10 overflow<nsw> : i32 
[DEBUG]compare reading and storeOp
[DEBUG] memref.store %39, %7[] : memref<i32> 
[DEBUG] %39 = arith.addi %15, %10 overflow<nsw> : i32 
[DEBUG]Unexpected readOp: 
[DEBUG] %39 = arith.addi %15, %10 overflow<nsw> : i32 
[DEBUG]compare reading and storeOp
[DEBUG] memref.store %21, %7[] : memref<i32> 
[DEBUG] %21 = arith.addi %20, %10 overflow<nsw> : i32 
[DEBUG]Unexpected readOp: 
[DEBUG] %21 = arith.addi %20, %10 overflow<nsw> : i32 
[DEBUG]compare reading and storeOp
[DEBUG] memref.store %23, %7[] : memref<i32> 
[DEBUG] %23 = arith.addi %22, %10 overflow<nsw> : i32 
[DEBUG]Unexpected readOp: 
[DEBUG] %23 = arith.addi %22, %10 overflow<nsw> : i32 
[DEBUG]compare reading and storeOp
[DEBUG] memref.store %39, %9[%21, %30] : memref<1024x1024xf64> 
[DEBUG] %39 = arith.addf %34, %38 fastmath<contract> : f64 
[DEBUG]Unexpected readOp: 
[DEBUG] %39 = arith.addf %34, %38 fastmath<contract> : f64 
[DEBUG]Dependency chain relies on LoadOp!
[DEBUG]compare reading and storeOp
[DEBUG] memref.store %12, %6[] : memref<i32> 
[DEBUG] %12 = arith.addi %10, %c1023_i32 overflow<nsw> : i32 
[DEBUG]Unexpected readOp: 
[DEBUG] %12 = arith.addi %10, %c1023_i32 overflow<nsw> : i32 
[DEBUG]compare reading and storeOp
[DEBUG] memref.store %11, %7[] : memref<i32> 
[DEBUG] %11 = arith.addi %10, %c1023_i32 overflow<nsw> : i32 
[DEBUG]Unexpected readOp: 
[DEBUG] %11 = arith.addi %10, %c1023_i32 overflow<nsw> : i32 
[DEBUG]compare reading and storeOp
[DEBUG] memref.store %39, %9[%21, %30] : memref<1024x1024xf64> 
[DEBUG] %39 = arith.addf %34, %38 fastmath<contract> : f64 
[DEBUG]Unexpected readOp: 
[DEBUG] %39 = arith.addf %34, %38 fastmath<contract> : f64 
[DEBUG]Dependency chain relies on LoadOp!
// -----// IR Dump After {anonymous}::OptimizeMemOpsPass (jforce-optimize-mem-ops) //----- //
func.func @kernel(%arg0: !fir.ref<i32> {jit.arg_type = 1 : ui32, jit.literal_val = 0 : i64}, %arg1: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg2: !fir.ref<i32> {jit.arg_type = 1 : ui32, jit.literal_val = 0 : i64}, %arg3: !fir.ref<!fir.array<1024x1024xf64>> {jit.arg_type = 1 : ui32}, %arg4: !fir.ref<!fir.array<1024x1024xf64>> {jit.arg_type = 1 : ui32}, %arg5: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg6: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg7: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg8: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}) {
  %c1023_i32 = arith.constant 1023 : i32
  %c1024 = arith.constant 1024 : index
  %c1_i32 = arith.constant 1 : i32
  %c1 = arith.constant 1 : index
  %0 = fir.declare %arg0 {uniq_name = "_QFFrun_benchmarkEj"} : (!fir.ref<i32>) -> !fir.ref<i32>
  %1 = fir.declare %arg1 {uniq_name = "_QFFrun_benchmarkEn"} : (!fir.ref<i32>) -> !fir.ref<i32>
  %2 = fir.declare %arg2 {uniq_name = "_QFFrun_benchmarkEi"} : (!fir.ref<i32>) -> !fir.ref<i32>
  %3 = fir.shape %c1024, %c1024 : (index, index) -> !fir.shape<2>
  %4 = fir.declare %arg3(%3) {uniq_name = "_QFFrun_benchmarkEy"} : (!fir.ref<!fir.array<1024x1024xf64>>, !fir.shape<2>) -> !fir.ref<!fir.array<1024x1024xf64>>
  %5 = fir.declare %arg4(%3) {uniq_name = "_QFFrun_benchmarkEx"} : (!fir.ref<!fir.array<1024x1024xf64>>, !fir.shape<2>) -> !fir.ref<!fir.array<1024x1024xf64>>
  %6 = fir.convert %0 : (!fir.ref<i32>) -> memref<i32>
  %7 = fir.convert %2 : (!fir.ref<i32>) -> memref<i32>
  %8 = fir.convert %5 : (!fir.ref<!fir.array<1024x1024xf64>>) -> memref<1024x1024xf64>
  %9 = fir.convert %4 : (!fir.ref<!fir.array<1024x1024xf64>>) -> memref<1024x1024xf64>
  %10 = fir.convert %c1 : (index) -> i32
  affine.for %arg9 = 0 to 1022 {
    %13 = affine.apply affine_map<(d0) -> (d0 + 2)>(%arg9)
    %14 = arith.index_cast %13 : index to i32
    affine.for %arg10 = 0 to 1022 {
      %15 = affine.apply affine_map<(d0) -> (d0 + 2)>(%arg10)
      %16 = arith.index_cast %15 : index to i32
      %17 = arith.subi %16, %c1_i32 overflow<nsw> : i32
      %18 = arith.index_cast %17 : i32 to index
      %19 = arith.subi %18, %c1 : index
      %20 = arith.index_cast %14 : i32 to index
      %21 = arith.subi %20, %c1 : index
      %22 = memref.load %8[%21, %19] : memref<1024x1024xf64>
      %23 = arith.addi %16, %c1_i32 overflow<nsw> : i32
      %24 = arith.index_cast %23 : i32 to index
      %25 = arith.subi %24, %c1 : index
      %26 = memref.load %8[%21, %25] : memref<1024x1024xf64>
      %27 = arith.addf %22, %26 fastmath<contract> : f64
      %28 = arith.subi %14, %c1_i32 overflow<nsw> : i32
      %29 = arith.index_cast %16 : i32 to index
      %30 = arith.subi %29, %c1 : index
      %31 = arith.index_cast %28 : i32 to index
      %32 = arith.subi %31, %c1 : index
      %33 = memref.load %8[%32, %30] : memref<1024x1024xf64>
      %34 = arith.addf %27, %33 fastmath<contract> : f64
      %35 = arith.addi %14, %c1_i32 overflow<nsw> : i32
      %36 = arith.index_cast %35 : i32 to index
      %37 = arith.subi %36, %c1 : index
      %38 = memref.load %8[%37, %30] : memref<1024x1024xf64>
      %39 = arith.addf %34, %38 fastmath<contract> : f64
      memref.store %39, %9[%21, %30] : memref<1024x1024xf64>
    }
  }
  %11 = arith.addi %10, %c1023_i32 overflow<nsw> : i32
  memref.store %11, %7[] : memref<i32>
  %12 = arith.addi %10, %c1023_i32 overflow<nsw> : i32
  memref.store %12, %6[] : memref<i32>
  omp.terminator
}

// -----// IR Dump After CanonicalizerPass (canonicalize) //----- //
#map = affine_map<(d0) -> (d0 + 2)>
module {
  func.func @kernel(%arg0: !fir.ref<i32> {jit.arg_type = 1 : ui32, jit.literal_val = 0 : i64}, %arg1: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg2: !fir.ref<i32> {jit.arg_type = 1 : ui32, jit.literal_val = 0 : i64}, %arg3: !fir.ref<!fir.array<1024x1024xf64>> {jit.arg_type = 1 : ui32}, %arg4: !fir.ref<!fir.array<1024x1024xf64>> {jit.arg_type = 1 : ui32}, %arg5: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg6: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg7: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg8: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}) {
    %c1023_i32 = arith.constant 1023 : i32
    %c1024 = arith.constant 1024 : index
    %c1_i32 = arith.constant 1 : i32
    %c1 = arith.constant 1 : index
    %0 = fir.declare %arg0 {uniq_name = "_QFFrun_benchmarkEj"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %1 = fir.declare %arg1 {uniq_name = "_QFFrun_benchmarkEn"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %2 = fir.declare %arg2 {uniq_name = "_QFFrun_benchmarkEi"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %3 = fir.shape %c1024, %c1024 : (index, index) -> !fir.shape<2>
    %4 = fir.declare %arg3(%3) {uniq_name = "_QFFrun_benchmarkEy"} : (!fir.ref<!fir.array<1024x1024xf64>>, !fir.shape<2>) -> !fir.ref<!fir.array<1024x1024xf64>>
    %5 = fir.declare %arg4(%3) {uniq_name = "_QFFrun_benchmarkEx"} : (!fir.ref<!fir.array<1024x1024xf64>>, !fir.shape<2>) -> !fir.ref<!fir.array<1024x1024xf64>>
    %6 = fir.convert %0 : (!fir.ref<i32>) -> memref<i32>
    %7 = fir.convert %2 : (!fir.ref<i32>) -> memref<i32>
    %8 = fir.convert %5 : (!fir.ref<!fir.array<1024x1024xf64>>) -> memref<1024x1024xf64>
    %9 = fir.convert %4 : (!fir.ref<!fir.array<1024x1024xf64>>) -> memref<1024x1024xf64>
    %10 = fir.convert %c1 : (index) -> i32
    affine.for %arg9 = 0 to 1022 {
      %13 = affine.apply #map(%arg9)
      %14 = arith.index_cast %13 : index to i32
      affine.for %arg10 = 0 to 1022 {
        %15 = affine.apply #map(%arg10)
        %16 = arith.index_cast %15 : index to i32
        %17 = arith.subi %16, %c1_i32 overflow<nsw> : i32
        %18 = arith.index_cast %17 : i32 to index
        %19 = arith.subi %18, %c1 : index
        %20 = arith.index_cast %14 : i32 to index
        %21 = arith.subi %20, %c1 : index
        %22 = memref.load %8[%21, %19] : memref<1024x1024xf64>
        %23 = arith.addi %16, %c1_i32 overflow<nsw> : i32
        %24 = arith.index_cast %23 : i32 to index
        %25 = arith.subi %24, %c1 : index
        %26 = memref.load %8[%21, %25] : memref<1024x1024xf64>
        %27 = arith.addf %22, %26 fastmath<contract> : f64
        %28 = arith.subi %14, %c1_i32 overflow<nsw> : i32
        %29 = arith.index_cast %16 : i32 to index
        %30 = arith.subi %29, %c1 : index
        %31 = arith.index_cast %28 : i32 to index
        %32 = arith.subi %31, %c1 : index
        %33 = memref.load %8[%32, %30] : memref<1024x1024xf64>
        %34 = arith.addf %27, %33 fastmath<contract> : f64
        %35 = arith.addi %14, %c1_i32 overflow<nsw> : i32
        %36 = arith.index_cast %35 : i32 to index
        %37 = arith.subi %36, %c1 : index
        %38 = memref.load %8[%37, %30] : memref<1024x1024xf64>
        %39 = arith.addf %34, %38 fastmath<contract> : f64
        memref.store %39, %9[%21, %30] : memref<1024x1024xf64>
      }
    }
    %11 = arith.addi %10, %c1023_i32 overflow<nsw> : i32
    memref.store %11, %7[] : memref<i32>
    %12 = arith.addi %10, %c1023_i32 overflow<nsw> : i32
    memref.store %12, %6[] : memref<i32>
    omp.terminator
  }
}


func.func @kernel(%arg0: !fir.ref<i32> {jit.arg_type = 1 : ui32, jit.literal_val = 0 : i64}, %arg1: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg2: !fir.ref<i32> {jit.arg_type = 1 : ui32, jit.literal_val = 0 : i64}, %arg3: !fir.ref<!fir.array<1024x1024xf64>> {jit.arg_type = 1 : ui32}, %arg4: !fir.ref<!fir.array<1024x1024xf64>> {jit.arg_type = 1 : ui32}, %arg5: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg6: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg7: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg8: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}) {
  %c1023_i32 = arith.constant 1023 : i32
  %c1024 = arith.constant 1024 : index
  %c1_i32 = arith.constant 1 : i32
  %c1 = arith.constant 1 : index
  %0 = fir.declare %arg0 {uniq_name = "_QFFrun_benchmarkEj"} : (!fir.ref<i32>) -> !fir.ref<i32>
  %1 = fir.declare %arg1 {uniq_name = "_QFFrun_benchmarkEn"} : (!fir.ref<i32>) -> !fir.ref<i32>
  %2 = fir.declare %arg2 {uniq_name = "_QFFrun_benchmarkEi"} : (!fir.ref<i32>) -> !fir.ref<i32>
  %3 = fir.shape %c1024, %c1024 : (index, index) -> !fir.shape<2>
  %4 = fir.declare %arg3(%3) {uniq_name = "_QFFrun_benchmarkEy"} : (!fir.ref<!fir.array<1024x1024xf64>>, !fir.shape<2>) -> !fir.ref<!fir.array<1024x1024xf64>>
  %5 = fir.declare %arg4(%3) {uniq_name = "_QFFrun_benchmarkEx"} : (!fir.ref<!fir.array<1024x1024xf64>>, !fir.shape<2>) -> !fir.ref<!fir.array<1024x1024xf64>>
  %6 = fir.convert %0 : (!fir.ref<i32>) -> memref<i32>
  %7 = fir.convert %2 : (!fir.ref<i32>) -> memref<i32>
  %8 = fir.convert %5 : (!fir.ref<!fir.array<1024x1024xf64>>) -> memref<1024x1024xf64>
  %9 = fir.convert %4 : (!fir.ref<!fir.array<1024x1024xf64>>) -> memref<1024x1024xf64>
  %10 = fir.convert %c1 : (index) -> i32
  affine.for %arg9 = 0 to 1022 {
    affine.for %arg10 = 0 to 1022 {
      %13 = affine.apply affine_map<(d0) -> (d0 + 2)>(%arg9)
      %14 = arith.index_cast %13 : index to i32
      %15 = affine.apply affine_map<(d0) -> (d0 + 2)>(%arg10)
      %16 = arith.index_cast %15 : index to i32
      %17 = arith.subi %16, %c1_i32 overflow<nsw> : i32
      %18 = arith.index_cast %17 : i32 to index
      %19 = arith.subi %18, %c1 : index
      %20 = arith.index_cast %14 : i32 to index
      %21 = arith.subi %20, %c1 : index
      %22 = memref.load %8[%21, %19] : memref<1024x1024xf64>
      %23 = arith.addi %16, %c1_i32 overflow<nsw> : i32
      %24 = arith.index_cast %23 : i32 to index
      %25 = arith.subi %24, %c1 : index
      %26 = memref.load %8[%21, %25] : memref<1024x1024xf64>
      %27 = arith.addf %22, %26 fastmath<contract> : f64
      %28 = arith.subi %14, %c1_i32 overflow<nsw> : i32
      %29 = arith.index_cast %16 : i32 to index
      %30 = arith.subi %29, %c1 : index
      %31 = arith.index_cast %28 : i32 to index
      %32 = arith.subi %31, %c1 : index
      %33 = memref.load %8[%32, %30] : memref<1024x1024xf64>
      %34 = arith.addf %27, %33 fastmath<contract> : f64
      %35 = arith.addi %14, %c1_i32 overflow<nsw> : i32
      %36 = arith.index_cast %35 : i32 to index
      %37 = arith.subi %36, %c1 : index
      %38 = memref.load %8[%37, %30] : memref<1024x1024xf64>
      %39 = arith.addf %34, %38 fastmath<contract> : f64
      memref.store %39, %9[%21, %30] : memref<1024x1024xf64>
    }
  }
  %11 = arith.addi %10, %c1023_i32 overflow<nsw> : i32
  memref.store %11, %7[] : memref<i32>
  %12 = arith.addi %10, %c1023_i32 overflow<nsw> : i32
  memref.store %12, %6[] : memref<i32>
  omp.terminator
}
// -----// IR Dump After {anonymous}::LoopSinkingPass (jforce-loop-sink) //----- //
func.func @kernel(%arg0: !fir.ref<i32> {jit.arg_type = 1 : ui32, jit.literal_val = 0 : i64}, %arg1: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg2: !fir.ref<i32> {jit.arg_type = 1 : ui32, jit.literal_val = 0 : i64}, %arg3: !fir.ref<!fir.array<1024x1024xf64>> {jit.arg_type = 1 : ui32}, %arg4: !fir.ref<!fir.array<1024x1024xf64>> {jit.arg_type = 1 : ui32}, %arg5: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg6: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg7: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg8: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}) {
  %c1023_i32 = arith.constant 1023 : i32
  %c1024 = arith.constant 1024 : index
  %c1_i32 = arith.constant 1 : i32
  %c1 = arith.constant 1 : index
  %0 = fir.declare %arg0 {uniq_name = "_QFFrun_benchmarkEj"} : (!fir.ref<i32>) -> !fir.ref<i32>
  %1 = fir.declare %arg1 {uniq_name = "_QFFrun_benchmarkEn"} : (!fir.ref<i32>) -> !fir.ref<i32>
  %2 = fir.declare %arg2 {uniq_name = "_QFFrun_benchmarkEi"} : (!fir.ref<i32>) -> !fir.ref<i32>
  %3 = fir.shape %c1024, %c1024 : (index, index) -> !fir.shape<2>
  %4 = fir.declare %arg3(%3) {uniq_name = "_QFFrun_benchmarkEy"} : (!fir.ref<!fir.array<1024x1024xf64>>, !fir.shape<2>) -> !fir.ref<!fir.array<1024x1024xf64>>
  %5 = fir.declare %arg4(%3) {uniq_name = "_QFFrun_benchmarkEx"} : (!fir.ref<!fir.array<1024x1024xf64>>, !fir.shape<2>) -> !fir.ref<!fir.array<1024x1024xf64>>
  %6 = fir.convert %0 : (!fir.ref<i32>) -> memref<i32>
  %7 = fir.convert %2 : (!fir.ref<i32>) -> memref<i32>
  %8 = fir.convert %5 : (!fir.ref<!fir.array<1024x1024xf64>>) -> memref<1024x1024xf64>
  %9 = fir.convert %4 : (!fir.ref<!fir.array<1024x1024xf64>>) -> memref<1024x1024xf64>
  %10 = fir.convert %c1 : (index) -> i32
  affine.for %arg9 = 0 to 1022 {
    affine.for %arg10 = 0 to 1022 {
      %13 = affine.apply affine_map<(d0) -> (d0 + 2)>(%arg9)
      %14 = arith.index_cast %13 : index to i32
      %15 = affine.apply affine_map<(d0) -> (d0 + 2)>(%arg10)
      %16 = arith.index_cast %15 : index to i32
      %17 = arith.subi %16, %c1_i32 overflow<nsw> : i32
      %18 = arith.index_cast %17 : i32 to index
      %19 = arith.subi %18, %c1 : index
      %20 = arith.index_cast %14 : i32 to index
      %21 = arith.subi %20, %c1 : index
      %22 = memref.load %8[%21, %19] : memref<1024x1024xf64>
      %23 = arith.addi %16, %c1_i32 overflow<nsw> : i32
      %24 = arith.index_cast %23 : i32 to index
      %25 = arith.subi %24, %c1 : index
      %26 = memref.load %8[%21, %25] : memref<1024x1024xf64>
      %27 = arith.addf %22, %26 fastmath<contract> : f64
      %28 = arith.subi %14, %c1_i32 overflow<nsw> : i32
      %29 = arith.index_cast %16 : i32 to index
      %30 = arith.subi %29, %c1 : index
      %31 = arith.index_cast %28 : i32 to index
      %32 = arith.subi %31, %c1 : index
      %33 = memref.load %8[%32, %30] : memref<1024x1024xf64>
      %34 = arith.addf %27, %33 fastmath<contract> : f64
      %35 = arith.addi %14, %c1_i32 overflow<nsw> : i32
      %36 = arith.index_cast %35 : i32 to index
      %37 = arith.subi %36, %c1 : index
      %38 = memref.load %8[%37, %30] : memref<1024x1024xf64>
      %39 = arith.addf %34, %38 fastmath<contract> : f64
      memref.store %39, %9[%21, %30] : memref<1024x1024xf64>
    }
  }
  %11 = arith.addi %10, %c1023_i32 overflow<nsw> : i32
  memref.store %11, %7[] : memref<i32>
  %12 = arith.addi %10, %c1023_i32 overflow<nsw> : i32
  memref.store %12, %6[] : memref<i32>
  omp.terminator
}

// -----// IR Dump After {anonymous}::AffineCFGPass (enzyme-affinecfg) //----- //
module {
  func.func @kernel(%arg0: !fir.ref<i32> {jit.arg_type = 1 : ui32, jit.literal_val = 0 : i64}, %arg1: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg2: !fir.ref<i32> {jit.arg_type = 1 : ui32, jit.literal_val = 0 : i64}, %arg3: !fir.ref<!fir.array<1024x1024xf64>> {jit.arg_type = 1 : ui32}, %arg4: !fir.ref<!fir.array<1024x1024xf64>> {jit.arg_type = 1 : ui32}, %arg5: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg6: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg7: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg8: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}) {
    %c1023_i32 = arith.constant 1023 : i32
    %c1024 = arith.constant 1024 : index
    %c1 = arith.constant 1 : index
    %0 = fir.declare %arg0 {uniq_name = "_QFFrun_benchmarkEj"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %1 = fir.declare %arg1 {uniq_name = "_QFFrun_benchmarkEn"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %2 = fir.declare %arg2 {uniq_name = "_QFFrun_benchmarkEi"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %3 = fir.shape %c1024, %c1024 : (index, index) -> !fir.shape<2>
    %4 = fir.declare %arg3(%3) {uniq_name = "_QFFrun_benchmarkEy"} : (!fir.ref<!fir.array<1024x1024xf64>>, !fir.shape<2>) -> !fir.ref<!fir.array<1024x1024xf64>>
    %5 = fir.declare %arg4(%3) {uniq_name = "_QFFrun_benchmarkEx"} : (!fir.ref<!fir.array<1024x1024xf64>>, !fir.shape<2>) -> !fir.ref<!fir.array<1024x1024xf64>>
    %6 = fir.convert %0 : (!fir.ref<i32>) -> memref<i32>
    %7 = fir.convert %2 : (!fir.ref<i32>) -> memref<i32>
    %8 = fir.convert %5 : (!fir.ref<!fir.array<1024x1024xf64>>) -> memref<1024x1024xf64>
    %9 = fir.convert %4 : (!fir.ref<!fir.array<1024x1024xf64>>) -> memref<1024x1024xf64>
    %10 = fir.convert %c1 : (index) -> i32
    affine.parallel (%arg9, %arg10) = (0, 0) to (1022, 1022) {
      %13 = affine.load %8[%arg9 + 1, %arg10] : memref<1024x1024xf64>
      %14 = affine.load %8[%arg9 + 1, %arg10 + 2] : memref<1024x1024xf64>
      %15 = arith.addf %13, %14 fastmath<contract> : f64
      %16 = affine.load %8[%arg9, %arg10 + 1] : memref<1024x1024xf64>
      %17 = arith.addf %15, %16 fastmath<contract> : f64
      %18 = affine.load %8[%arg9 + 2, %arg10 + 1] : memref<1024x1024xf64>
      %19 = arith.addf %17, %18 fastmath<contract> : f64
      affine.store %19, %9[%arg9 + 1, %arg10 + 1] : memref<1024x1024xf64>
    }
    %11 = arith.addi %10, %c1023_i32 overflow<nsw> : i32
    affine.store %11, %7[] : memref<i32>
    %12 = arith.addi %10, %c1023_i32 overflow<nsw> : i32
    affine.store %12, %6[] : memref<i32>
    omp.terminator
  }
}


module {
  func.func @outlined_affinefor_102169923486784(%arg0: memref<1024x1024xf64>, %arg1: memref<1024x1024xf64>) {
    affine.parallel (%arg2, %arg3) = (0, 0) to (1022, 1022) {
      %0 = affine.load %arg0[%arg2 + 1, %arg3] : memref<1024x1024xf64>
      %1 = affine.load %arg0[%arg2 + 1, %arg3 + 2] : memref<1024x1024xf64>
      %2 = arith.addf %0, %1 fastmath<contract> : f64
      %3 = affine.load %arg0[%arg2, %arg3 + 1] : memref<1024x1024xf64>
      %4 = arith.addf %2, %3 fastmath<contract> : f64
      %5 = affine.load %arg0[%arg2 + 2, %arg3 + 1] : memref<1024x1024xf64>
      %6 = arith.addf %4, %5 fastmath<contract> : f64
      affine.store %6, %arg1[%arg2 + 1, %arg3 + 1] : memref<1024x1024xf64>
    }
    return
  }
  func.func @kernel(%arg0: !fir.ref<i32> {jit.arg_type = 1 : ui32, jit.literal_val = 0 : i64}, %arg1: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg2: !fir.ref<i32> {jit.arg_type = 1 : ui32, jit.literal_val = 0 : i64}, %arg3: !fir.ref<!fir.array<1024x1024xf64>> {jit.arg_type = 1 : ui32}, %arg4: !fir.ref<!fir.array<1024x1024xf64>> {jit.arg_type = 1 : ui32}, %arg5: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg6: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg7: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg8: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}) {
    %c1023_i32 = arith.constant 1023 : i32
    %c1024 = arith.constant 1024 : index
    %c1 = arith.constant 1 : index
    %0 = fir.declare %arg0 {uniq_name = "_QFFrun_benchmarkEj"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %1 = fir.declare %arg1 {uniq_name = "_QFFrun_benchmarkEn"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %2 = fir.declare %arg2 {uniq_name = "_QFFrun_benchmarkEi"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %3 = fir.shape %c1024, %c1024 : (index, index) -> !fir.shape<2>
    %4 = fir.declare %arg3(%3) {uniq_name = "_QFFrun_benchmarkEy"} : (!fir.ref<!fir.array<1024x1024xf64>>, !fir.shape<2>) -> !fir.ref<!fir.array<1024x1024xf64>>
    %5 = fir.declare %arg4(%3) {uniq_name = "_QFFrun_benchmarkEx"} : (!fir.ref<!fir.array<1024x1024xf64>>, !fir.shape<2>) -> !fir.ref<!fir.array<1024x1024xf64>>
    %6 = fir.convert %0 : (!fir.ref<i32>) -> memref<i32>
    %7 = fir.convert %2 : (!fir.ref<i32>) -> memref<i32>
    %8 = fir.convert %5 : (!fir.ref<!fir.array<1024x1024xf64>>) -> memref<1024x1024xf64>
    %9 = fir.convert %4 : (!fir.ref<!fir.array<1024x1024xf64>>) -> memref<1024x1024xf64>
    %10 = fir.convert %c1 : (index) -> i32
    call @outlined_affinefor_102169923486784(%8, %9) : (memref<1024x1024xf64>, memref<1024x1024xf64>) -> ()
    %11 = arith.addi %10, %c1023_i32 overflow<nsw> : i32
    affine.store %11, %7[] : memref<i32>
    %12 = arith.addi %10, %c1023_i32 overflow<nsw> : i32
    affine.store %12, %6[] : memref<i32>
    omp.terminator
  }
}
// -----// IR Dump After {anonymous}::OutlineAffinePass (jforce-outline-affine) //----- //
module {
  func.func @outlined_affinefor_102169923486784(%arg0: memref<1024x1024xf64>, %arg1: memref<1024x1024xf64>) {
    affine.parallel (%arg2, %arg3) = (0, 0) to (1022, 1022) {
      %0 = affine.load %arg0[%arg2 + 1, %arg3] : memref<1024x1024xf64>
      %1 = affine.load %arg0[%arg2 + 1, %arg3 + 2] : memref<1024x1024xf64>
      %2 = arith.addf %0, %1 fastmath<contract> : f64
      %3 = affine.load %arg0[%arg2, %arg3 + 1] : memref<1024x1024xf64>
      %4 = arith.addf %2, %3 fastmath<contract> : f64
      %5 = affine.load %arg0[%arg2 + 2, %arg3 + 1] : memref<1024x1024xf64>
      %6 = arith.addf %4, %5 fastmath<contract> : f64
      affine.store %6, %arg1[%arg2 + 1, %arg3 + 1] : memref<1024x1024xf64>
    }
    return
  }
  func.func @kernel(%arg0: !fir.ref<i32> {jit.arg_type = 1 : ui32, jit.literal_val = 0 : i64}, %arg1: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg2: !fir.ref<i32> {jit.arg_type = 1 : ui32, jit.literal_val = 0 : i64}, %arg3: !fir.ref<!fir.array<1024x1024xf64>> {jit.arg_type = 1 : ui32}, %arg4: !fir.ref<!fir.array<1024x1024xf64>> {jit.arg_type = 1 : ui32}, %arg5: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg6: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg7: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg8: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}) {
    %c1023_i32 = arith.constant 1023 : i32
    %c1024 = arith.constant 1024 : index
    %c1 = arith.constant 1 : index
    %0 = fir.declare %arg0 {uniq_name = "_QFFrun_benchmarkEj"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %1 = fir.declare %arg1 {uniq_name = "_QFFrun_benchmarkEn"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %2 = fir.declare %arg2 {uniq_name = "_QFFrun_benchmarkEi"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %3 = fir.shape %c1024, %c1024 : (index, index) -> !fir.shape<2>
    %4 = fir.declare %arg3(%3) {uniq_name = "_QFFrun_benchmarkEy"} : (!fir.ref<!fir.array<1024x1024xf64>>, !fir.shape<2>) -> !fir.ref<!fir.array<1024x1024xf64>>
    %5 = fir.declare %arg4(%3) {uniq_name = "_QFFrun_benchmarkEx"} : (!fir.ref<!fir.array<1024x1024xf64>>, !fir.shape<2>) -> !fir.ref<!fir.array<1024x1024xf64>>
    %6 = fir.convert %0 : (!fir.ref<i32>) -> memref<i32>
    %7 = fir.convert %2 : (!fir.ref<i32>) -> memref<i32>
    %8 = fir.convert %5 : (!fir.ref<!fir.array<1024x1024xf64>>) -> memref<1024x1024xf64>
    %9 = fir.convert %4 : (!fir.ref<!fir.array<1024x1024xf64>>) -> memref<1024x1024xf64>
    %10 = fir.convert %c1 : (index) -> i32
    call @outlined_affinefor_102169923486784(%8, %9) : (memref<1024x1024xf64>, memref<1024x1024xf64>) -> ()
    %11 = arith.addi %10, %c1023_i32 overflow<nsw> : i32
    affine.store %11, %7[] : memref<i32>
    %12 = arith.addi %10, %c1023_i32 overflow<nsw> : i32
    affine.store %12, %6[] : memref<i32>
    omp.terminator
  }
}


// -----// IR Dump After {anonymous}::AffineToStableHLORaisingPass (enzyme-affine-to-stablehlo) //----- //
func.func @kernel(%arg0: !fir.ref<i32> {jit.arg_type = 1 : ui32, jit.literal_val = 0 : i64}, %arg1: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg2: !fir.ref<i32> {jit.arg_type = 1 : ui32, jit.literal_val = 0 : i64}, %arg3: !fir.ref<!fir.array<1024x1024xf64>> {jit.arg_type = 1 : ui32}, %arg4: !fir.ref<!fir.array<1024x1024xf64>> {jit.arg_type = 1 : ui32}, %arg5: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg6: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg7: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg8: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}) {
  %c1023_i32 = arith.constant 1023 : i32
  %c1024 = arith.constant 1024 : index
  %c1 = arith.constant 1 : index
  %0 = fir.declare %arg0 {uniq_name = "_QFFrun_benchmarkEj"} : (!fir.ref<i32>) -> !fir.ref<i32>
  %1 = fir.declare %arg1 {uniq_name = "_QFFrun_benchmarkEn"} : (!fir.ref<i32>) -> !fir.ref<i32>
  %2 = fir.declare %arg2 {uniq_name = "_QFFrun_benchmarkEi"} : (!fir.ref<i32>) -> !fir.ref<i32>
  %3 = fir.shape %c1024, %c1024 : (index, index) -> !fir.shape<2>
  %4 = fir.declare %arg3(%3) {uniq_name = "_QFFrun_benchmarkEy"} : (!fir.ref<!fir.array<1024x1024xf64>>, !fir.shape<2>) -> !fir.ref<!fir.array<1024x1024xf64>>
  %5 = fir.declare %arg4(%3) {uniq_name = "_QFFrun_benchmarkEx"} : (!fir.ref<!fir.array<1024x1024xf64>>, !fir.shape<2>) -> !fir.ref<!fir.array<1024x1024xf64>>
  %6 = fir.convert %0 : (!fir.ref<i32>) -> memref<i32>
  %7 = fir.convert %2 : (!fir.ref<i32>) -> memref<i32>
  %8 = fir.convert %5 : (!fir.ref<!fir.array<1024x1024xf64>>) -> memref<1024x1024xf64>
  %9 = fir.convert %4 : (!fir.ref<!fir.array<1024x1024xf64>>) -> memref<1024x1024xf64>
  %10 = fir.convert %c1 : (index) -> i32
  call @outlined_affinefor_102169923486784(%8, %9) : (memref<1024x1024xf64>, memref<1024x1024xf64>) -> ()
  %11 = arith.addi %10, %c1023_i32 overflow<nsw> : i32
  affine.store %11, %7[] : memref<i32>
  %12 = arith.addi %10, %c1023_i32 overflow<nsw> : i32
  affine.store %12, %6[] : memref<i32>
  omp.terminator
}

// -----// IR Dump After {anonymous}::AffineToStableHLORaisingPass (enzyme-affine-to-stablehlo) //----- //
func.func @outlined_affinefor_102169923486784(%arg0: memref<1024x1024xf64>, %arg1: memref<1024x1024xf64>) {
  affine.parallel (%arg2, %arg3) = (0, 0) to (1022, 1022) {
    %0 = affine.load %arg0[%arg2 + 1, %arg3] : memref<1024x1024xf64>
    %1 = affine.load %arg0[%arg2 + 1, %arg3 + 2] : memref<1024x1024xf64>
    %2 = arith.addf %0, %1 fastmath<contract> : f64
    %3 = affine.load %arg0[%arg2, %arg3 + 1] : memref<1024x1024xf64>
    %4 = arith.addf %2, %3 fastmath<contract> : f64
    %5 = affine.load %arg0[%arg2 + 2, %arg3 + 1] : memref<1024x1024xf64>
    %6 = arith.addf %4, %5 fastmath<contract> : f64
    affine.store %6, %arg1[%arg2 + 1, %arg3 + 1] : memref<1024x1024xf64>
  }
  return
}

// -----// IR Dump After CanonicalizerPass (canonicalize) //----- //
module {
  func.func @outlined_affinefor_102169923486784(%arg0: memref<1024x1024xf64>, %arg1: memref<1024x1024xf64>) {
    affine.parallel (%arg2, %arg3) = (0, 0) to (1022, 1022) {
      %0 = affine.load %arg0[%arg2 + 1, %arg3] : memref<1024x1024xf64>
      %1 = affine.load %arg0[%arg2 + 1, %arg3 + 2] : memref<1024x1024xf64>
      %2 = arith.addf %0, %1 fastmath<contract> : f64
      %3 = affine.load %arg0[%arg2, %arg3 + 1] : memref<1024x1024xf64>
      %4 = arith.addf %2, %3 fastmath<contract> : f64
      %5 = affine.load %arg0[%arg2 + 2, %arg3 + 1] : memref<1024x1024xf64>
      %6 = arith.addf %4, %5 fastmath<contract> : f64
      affine.store %6, %arg1[%arg2 + 1, %arg3 + 1] : memref<1024x1024xf64>
    }
    return
  }
  func.func @kernel(%arg0: !fir.ref<i32> {jit.arg_type = 1 : ui32, jit.literal_val = 0 : i64}, %arg1: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg2: !fir.ref<i32> {jit.arg_type = 1 : ui32, jit.literal_val = 0 : i64}, %arg3: !fir.ref<!fir.array<1024x1024xf64>> {jit.arg_type = 1 : ui32}, %arg4: !fir.ref<!fir.array<1024x1024xf64>> {jit.arg_type = 1 : ui32}, %arg5: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg6: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg7: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}, %arg8: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 1024 : i64}) {
    %c1023_i32 = arith.constant 1023 : i32
    %c1024 = arith.constant 1024 : index
    %c1 = arith.constant 1 : index
    %0 = fir.declare %arg0 {uniq_name = "_QFFrun_benchmarkEj"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %1 = fir.declare %arg1 {uniq_name = "_QFFrun_benchmarkEn"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %2 = fir.declare %arg2 {uniq_name = "_QFFrun_benchmarkEi"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %3 = fir.shape %c1024, %c1024 : (index, index) -> !fir.shape<2>
    %4 = fir.declare %arg3(%3) {uniq_name = "_QFFrun_benchmarkEy"} : (!fir.ref<!fir.array<1024x1024xf64>>, !fir.shape<2>) -> !fir.ref<!fir.array<1024x1024xf64>>
    %5 = fir.declare %arg4(%3) {uniq_name = "_QFFrun_benchmarkEx"} : (!fir.ref<!fir.array<1024x1024xf64>>, !fir.shape<2>) -> !fir.ref<!fir.array<1024x1024xf64>>
    %6 = fir.convert %0 : (!fir.ref<i32>) -> memref<i32>
    %7 = fir.convert %2 : (!fir.ref<i32>) -> memref<i32>
    %8 = fir.convert %5 : (!fir.ref<!fir.array<1024x1024xf64>>) -> memref<1024x1024xf64>
    %9 = fir.convert %4 : (!fir.ref<!fir.array<1024x1024xf64>>) -> memref<1024x1024xf64>
    %10 = fir.convert %c1 : (index) -> i32
    call @outlined_affinefor_102169923486784(%8, %9) : (memref<1024x1024xf64>, memref<1024x1024xf64>) -> ()
    %11 = arith.addi %10, %c1023_i32 overflow<nsw> : i32
    affine.store %11, %7[] : memref<i32>
    %12 = arith.addi %10, %c1023_i32 overflow<nsw> : i32
    affine.store %12, %6[] : memref<i32>
    omp.terminator
  }
  func.func private @outlined_affinefor_102169923486784_raised(%arg0: tensor<1024x1024xf64>, %arg1: tensor<1024x1024xf64>) -> (tensor<1024x1024xf64>, tensor<1024x1024xf64>) {
    %c = stablehlo.constant dense<1> : tensor<i64>
    %0 = stablehlo.slice %arg0 [1:1023, 0:1022] : (tensor<1024x1024xf64>) -> tensor<1022x1022xf64>
    %1 = stablehlo.reshape %0 : (tensor<1022x1022xf64>) -> tensor<1022x1022xf64>
    %2 = stablehlo.slice %arg0 [1:1023, 2:1024] : (tensor<1024x1024xf64>) -> tensor<1022x1022xf64>
    %3 = stablehlo.reshape %2 : (tensor<1022x1022xf64>) -> tensor<1022x1022xf64>
    %4 = arith.addf %1, %3 fastmath<contract> : tensor<1022x1022xf64>
    %5 = stablehlo.slice %arg0 [0:1022, 1:1023] : (tensor<1024x1024xf64>) -> tensor<1022x1022xf64>
    %6 = stablehlo.reshape %5 : (tensor<1022x1022xf64>) -> tensor<1022x1022xf64>
    %7 = arith.addf %4, %6 fastmath<contract> : tensor<1022x1022xf64>
    %8 = stablehlo.slice %arg0 [2:1024, 1:1023] : (tensor<1024x1024xf64>) -> tensor<1022x1022xf64>
    %9 = stablehlo.reshape %8 : (tensor<1022x1022xf64>) -> tensor<1022x1022xf64>
    %10 = arith.addf %7, %9 fastmath<contract> : tensor<1022x1022xf64>
    %11 = stablehlo.broadcast_in_dim %10, dims = [0, 1] : (tensor<1022x1022xf64>) -> tensor<1022x1022xf64>
    %12 = stablehlo.dynamic_update_slice %arg1, %11, %c, %c : (tensor<1024x1024xf64>, tensor<1022x1022xf64>, tensor<i64>, tensor<i64>) -> tensor<1024x1024xf64>
    return %arg0, %12 : tensor<1024x1024xf64>, tensor<1024x1024xf64>
  }
}


