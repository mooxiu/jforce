module {
  func.func @kernel(%arg0: !fir.ref<!fir.array<4xf64>> {jit.compute_arg}, %arg1: !fir.ref<!fir.array<4xf64>> {jit.compute_arg}, %arg2: !fir.ref<!fir.array<4xf64>> {jit.compute_arg}, %arg3: !fir.ref<f64> {jit.compute_arg, jit.literal_val = 7 : i64}, %arg4: !fir.ref<i32> {jit.literal_val = 4 : i64}, %arg5: !fir.ref<i32> {jit.literal_val = 4 : i64}, %arg6: !fir.ref<i32> {jit.literal_val = 4 : i64}) {
    %c4 = arith.constant 4 : index
    %0 = fir.shape %c4 : (index) -> !fir.shape<1>
    %1:2 = hlfir.declare %arg0(%0) {uniq_name = "_QFFrun_benchmarkEz"} : (!fir.ref<!fir.array<4xf64>>, !fir.shape<1>) -> (!fir.ref<!fir.array<4xf64>>, !fir.ref<!fir.array<4xf64>>)
    %2:2 = hlfir.declare %arg1(%0) {uniq_name = "_QFFrun_benchmarkEx"} : (!fir.ref<!fir.array<4xf64>>, !fir.shape<1>) -> (!fir.ref<!fir.array<4xf64>>, !fir.ref<!fir.array<4xf64>>)
    %3:2 = hlfir.declare %arg2(%0) {uniq_name = "_QFFrun_benchmarkEy"} : (!fir.ref<!fir.array<4xf64>>, !fir.shape<1>) -> (!fir.ref<!fir.array<4xf64>>, !fir.ref<!fir.array<4xf64>>)
    %4:2 = hlfir.declare %arg3 {uniq_name = "_QFFrun_benchmarkEa"} : (!fir.ref<f64>) -> (!fir.ref<f64>, !fir.ref<f64>)
    omp.teams {
      omp.workdistribute {
        %5 = fir.load %4#0 : !fir.ref<f64>
        %6 = hlfir.elemental %0 : (!fir.shape<1>) -> !hlfir.expr<4xf64> {
        ^bb0(%arg7: index):
          %8 = hlfir.designate %2#0 (%arg7)  : (!fir.ref<!fir.array<4xf64>>, index) -> !fir.ref<f64>
          %9 = fir.load %8 : !fir.ref<f64>
          %10 = arith.mulf %5, %9 fastmath<contract> : f64
          hlfir.yield_element %10 : f64
        }
        %7 = hlfir.elemental %0 : (!fir.shape<1>) -> !hlfir.expr<4xf64> {
        ^bb0(%arg7: index):
          %8 = hlfir.apply %6, %arg7 : (!hlfir.expr<4xf64>, index) -> f64
          %9 = hlfir.designate %3#0 (%arg7)  : (!fir.ref<!fir.array<4xf64>>, index) -> !fir.ref<f64>
          %10 = fir.load %9 : !fir.ref<f64>
          %11 = arith.addf %8, %10 fastmath<contract> : f64
          hlfir.yield_element %11 : f64
        }
        hlfir.assign %7 to %1#0 : !hlfir.expr<4xf64>, !fir.ref<!fir.array<4xf64>>
        hlfir.destroy %7 : !hlfir.expr<4xf64>
        hlfir.destroy %6 : !hlfir.expr<4xf64>
        omp.terminator
      }
      omp.terminator
    }
    omp.terminator
  }
}

