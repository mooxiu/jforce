module {
  func.func @kernel(%arg0: !fir.ref<!fir.array<?xf64>>, %arg1: !fir.ref<!fir.array<?xf64>>, %arg2: !fir.ref<!fir.array<?xf64>>, %arg3: !fir.ref<f64> {jit.literal_val = 7 : i64}, %arg4: !fir.ref<i32> {jit.literal_val = 4 : i64}, %arg5: !fir.ref<i32> {jit.literal_val = 4 : i64}, %arg6: !fir.ref<i32> {jit.literal_val = 4 : i64}) {
    %0 = fir.load %arg6 : !fir.ref<i32>
    %1 = fir.load %arg5 : !fir.ref<i32>
    %2 = fir.load %arg4 : !fir.ref<i32>
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
    %15 = fir.shape %14 : (index) -> !fir.shape<1>
    %16:2 = hlfir.declare %arg0(%15) {uniq_name = "_QFFrun_benchmarkEz"} : (!fir.ref<!fir.array<?xf64>>, !fir.shape<1>) -> (!fir.box<!fir.array<?xf64>>, !fir.ref<!fir.array<?xf64>>)
    %17 = fir.shape %13 : (index) -> !fir.shape<1>
    %18:2 = hlfir.declare %arg1(%17) {uniq_name = "_QFFrun_benchmarkEx"} : (!fir.ref<!fir.array<?xf64>>, !fir.shape<1>) -> (!fir.box<!fir.array<?xf64>>, !fir.ref<!fir.array<?xf64>>)
    %19 = fir.shape %12 : (index) -> !fir.shape<1>
    %20:2 = hlfir.declare %arg2(%19) {uniq_name = "_QFFrun_benchmarkEy"} : (!fir.ref<!fir.array<?xf64>>, !fir.shape<1>) -> (!fir.box<!fir.array<?xf64>>, !fir.ref<!fir.array<?xf64>>)
    %21:2 = hlfir.declare %arg3 {uniq_name = "_QFFrun_benchmarkEa"} : (!fir.ref<f64>) -> (!fir.ref<f64>, !fir.ref<f64>)
    omp.teams {
      omp.workdistribute {
        %22 = fir.load %21#0 : !fir.ref<f64>
        %23 = hlfir.elemental %17 unordered : (!fir.shape<1>) -> !hlfir.expr<?xf64> {
        ^bb0(%arg7: index):
          %25 = hlfir.designate %18#0 (%arg7)  : (!fir.box<!fir.array<?xf64>>, index) -> !fir.ref<f64>
          %26 = fir.load %25 : !fir.ref<f64>
          %27 = arith.mulf %22, %26 fastmath<contract> : f64
          hlfir.yield_element %27 : f64
        }
        %24 = hlfir.elemental %17 unordered : (!fir.shape<1>) -> !hlfir.expr<?xf64> {
        ^bb0(%arg7: index):
          %25 = hlfir.apply %23, %arg7 : (!hlfir.expr<?xf64>, index) -> f64
          %26 = hlfir.designate %20#0 (%arg7)  : (!fir.box<!fir.array<?xf64>>, index) -> !fir.ref<f64>
          %27 = fir.load %26 : !fir.ref<f64>
          %28 = arith.addf %25, %27 fastmath<contract> : f64
          hlfir.yield_element %28 : f64
        }
        hlfir.assign %24 to %16#0 : !hlfir.expr<?xf64>, !fir.box<!fir.array<?xf64>>
        hlfir.destroy %24 : !hlfir.expr<?xf64>
        hlfir.destroy %23 : !hlfir.expr<?xf64>
        omp.terminator
      }
      omp.terminator
    }
    omp.terminator
  }
}
