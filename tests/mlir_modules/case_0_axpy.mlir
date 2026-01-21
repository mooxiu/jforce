module {
  func.func @kernel(%arg0: !fir.ref<!fir.array<?x?xf64>>, %arg1: !fir.ref<!fir.array<?x?xf64>>, %arg2: !fir.ref<!fir.array<?x?xf64>>, %arg3: !fir.ref<f64>, %arg4: !fir.ref<i32>, %arg5: !fir.ref<i32>, %arg6: !fir.ref<i32>, %arg7: !fir.ref<i32>, %arg8: !fir.ref<i32>, %arg9: !fir.ref<i32>) {
    %c0 = arith.constant 0 : index
    %0 = arith.constant 10 : i32
    %1 = arith.constant 10 : i32
    %2 = arith.constant 10 : i32
    %3 = arith.constant 10 : i32
    %4 = arith.constant 10 : i32
    %5 = arith.constant 10 : i32
    %6 = fir.convert %0 : (i32) -> index
    %7 = arith.cmpi sgt, %6, %c0 : index
    %8 = fir.convert %1 : (i32) -> index
    %9 = arith.cmpi sgt, %8, %c0 : index
    %10 = fir.convert %2 : (i32) -> index
    %11 = arith.cmpi sgt, %10, %c0 : index
    %12 = fir.convert %3 : (i32) -> index
    %13 = arith.cmpi sgt, %12, %c0 : index
    %14 = fir.convert %4 : (i32) -> index
    %15 = arith.cmpi sgt, %14, %c0 : index
    %16 = fir.convert %5 : (i32) -> index
    %17 = arith.cmpi sgt, %16, %c0 : index
    %18 = arith.select %17, %16, %c0 : index
    %19 = arith.select %15, %14, %c0 : index
    %20 = arith.select %13, %12, %c0 : index
    %21 = arith.select %11, %10, %c0 : index
    %22 = arith.select %9, %8, %c0 : index
    %23 = arith.select %7, %6, %c0 : index
    %24 = fir.shape %23, %22 : (index, index) -> !fir.shape<2>
    %25:2 = hlfir.declare %arg0(%24) {uniq_name = "_QFFcoexecute_aEz"} : (!fir.ref<!fir.array<?x?xf64>>, !fir.shape<2>) -> (!fir.box<!fir.array<?x?xf64>>, !fir.ref<!fir.array<?x?xf64>>)
    %26 = fir.shape %21, %20 : (index, index) -> !fir.shape<2>
    %27:2 = hlfir.declare %arg1(%26) {uniq_name = "_QFFcoexecute_aEx"} : (!fir.ref<!fir.array<?x?xf64>>, !fir.shape<2>) -> (!fir.box<!fir.array<?x?xf64>>, !fir.ref<!fir.array<?x?xf64>>)
    %28 = fir.shape %19, %18 : (index, index) -> !fir.shape<2>
    %29:2 = hlfir.declare %arg2(%28) {uniq_name = "_QFFcoexecute_aEy"} : (!fir.ref<!fir.array<?x?xf64>>, !fir.shape<2>) -> (!fir.box<!fir.array<?x?xf64>>, !fir.ref<!fir.array<?x?xf64>>)
    %30:2 = hlfir.declare %arg3 {uniq_name = "_QFFcoexecute_aEa"} : (!fir.ref<f64>) -> (!fir.ref<f64>, !fir.ref<f64>)
    omp.teams {
      omp.workdistribute {
        %31 = fir.load %30#0 : !fir.ref<f64>
        %32 = hlfir.elemental %26 unordered : (!fir.shape<2>) -> !hlfir.expr<?x?xf64> {
        ^bb0(%arg10: index, %arg11: index):
          %34 = hlfir.designate %27#0 (%arg10, %arg11)  : (!fir.box<!fir.array<?x?xf64>>, index, index) -> !fir.ref<f64>
          %35 = fir.load %34 : !fir.ref<f64>
          %36 = arith.mulf %31, %35 fastmath<contract> : f64
          hlfir.yield_element %36 : f64
        }
        %33 = hlfir.elemental %26 unordered : (!fir.shape<2>) -> !hlfir.expr<?x?xf64> {
        ^bb0(%arg10: index, %arg11: index):
          %34 = hlfir.apply %32, %arg10, %arg11 : (!hlfir.expr<?x?xf64>, index, index) -> f64
          %35 = hlfir.designate %29#0 (%arg10, %arg11)  : (!fir.box<!fir.array<?x?xf64>>, index, index) -> !fir.ref<f64>
          %36 = fir.load %35 : !fir.ref<f64>
          %37 = arith.addf %34, %36 fastmath<contract> : f64
          hlfir.yield_element %37 : f64
        }
        hlfir.assign %33 to %25#0 : !hlfir.expr<?x?xf64>, !fir.box<!fir.array<?x?xf64>>
        hlfir.destroy %33 : !hlfir.expr<?x?xf64>
        hlfir.destroy %32 : !hlfir.expr<?x?xf64>
        omp.terminator
      }
      omp.terminator
    }
    omp.terminator
  }
}
