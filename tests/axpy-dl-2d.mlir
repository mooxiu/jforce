func.func @kernel(%arg0: !fir.ref<i32> {jit.literal_val = 0 : i64}, %arg1: !fir.ref<i32> {jit.literal_val = 3 : i64}, %arg2: !fir.ref<i32> {jit.literal_val = 0 : i64}, %arg3: !fir.ref<i32> {jit.literal_val = 4 : i64}, %arg4: !fir.ref<!fir.array<?x?xf64>>, %arg5: !fir.ref<!fir.array<?x?xf64>>, %arg6: !fir.ref<!fir.array<?x?xf64>>, %arg7: !fir.ref<f64> {jit.literal_val = 4619567317775286272 : i64}, %arg8: !fir.ref<i32> {jit.literal_val = 3 : i64}, %arg9: !fir.ref<i32> {jit.literal_val = 4 : i64}, %arg10: !fir.ref<i32> {jit.literal_val = 3 : i64}, %arg11: !fir.ref<i32> {jit.literal_val = 4 : i64}, %arg12: !fir.ref<i32> {jit.literal_val = 3 : i64}, %arg13: !fir.ref<i32> {jit.literal_val = 4 : i64}) {
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
