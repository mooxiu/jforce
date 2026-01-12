func.func @kernel(%arg0: !fir.ref<f32>, %arg1: !fir.ref<!fir.array<10xf32>>, %arg2: !fir.ref<!fir.array<10xf32>>) {
  %c1 = arith.constant 1 : index
  %cst = arith.constant 0.000000e+00 : f32
  %c10 = arith.constant 10 : index
  %0:2 = hlfir.declare %arg0 {uniq_name = "_QFEout"} : (!fir.ref<f32>) -> (!fir.ref<f32>, !fir.ref<f32>)
  %1 = fir.shape %c10 : (index) -> !fir.shape<1>
  %2:2 = hlfir.declare %arg1(%1) {uniq_name = "_QFEaa"} : (!fir.ref<!fir.array<10xf32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<10xf32>>, !fir.ref<!fir.array<10xf32>>)
  %3 = fir.shape %c10 : (index) -> !fir.shape<1>
  %4:2 = hlfir.declare %arg2(%3) {uniq_name = "_QFEbb"} : (!fir.ref<!fir.array<10xf32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<10xf32>>, !fir.ref<!fir.array<10xf32>>)
  omp.teams {
    omp.workdistribute {
      %5 = fir.do_loop %arg3 = %c1 to %c10 step %c1 iter_args(%arg4 = %cst) -> (f32) {
        %6 = hlfir.designate %2#0 (%arg3)  : (!fir.ref<!fir.array<10xf32>>, index) -> !fir.ref<f32>
        %7 = fir.load %6 : !fir.ref<f32>
        %8 = hlfir.designate %4#0 (%arg3)  : (!fir.ref<!fir.array<10xf32>>, index) -> !fir.ref<f32>
        %9 = fir.load %8 : !fir.ref<f32>
        %10 = arith.mulf %7, %9 fastmath<contract> : f32
        %11 = arith.addf %arg4, %10 fastmath<contract> : f32
        fir.result %11 : f32
      }
      hlfir.assign %5 to %0#0 : f32, !fir.ref<f32>
      omp.terminator
    }
    omp.terminator
  }
  omp.terminator
}
