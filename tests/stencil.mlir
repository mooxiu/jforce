func.func @kernel(%arg0: !fir.ref<i32>, %arg1: !fir.ref<i32>, %arg2: !fir.ref<i32>, %arg3: !fir.ref<!fir.array<?x?xf64>>, %arg4: !fir.ref<!fir.array<?x?xf64>>, %arg5: !fir.ref<i32>, %arg6: !fir.ref<i32>, %arg7: !fir.ref<i32>, %arg8: !fir.ref<i32>) {
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
