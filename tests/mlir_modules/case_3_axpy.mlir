func.func @kernel(%arg0: !fir.ref<!fir.array<10xf32>>, %arg1: !fir.ref<f32>, %arg2: !fir.ref<!fir.array<10xf32>>) {
  %c10 = arith.constant 10 : index
  %0 = fir.shape %c10 : (index) -> !fir.shape<1>
  %1:2 = hlfir.declare %arg0(%0) {uniq_name = "_QFEy"} : (!fir.ref<!fir.array<10xf32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<10xf32>>, !fir.ref<!fir.array<10xf32>>)
  %2:2 = hlfir.declare %arg1 {uniq_name = "_QFEa"} : (!fir.ref<f32>) -> (!fir.ref<f32>, !fir.ref<f32>)
  %3 = fir.shape %c10 : (index) -> !fir.shape<1>
  %4:2 = hlfir.declare %arg2(%3) {uniq_name = "_QFEx"} : (!fir.ref<!fir.array<10xf32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<10xf32>>, !fir.ref<!fir.array<10xf32>>)
  omp.teams {
    omp.workdistribute {
      %5 = fir.load %2#0 : !fir.ref<f32>
      %6 = hlfir.elemental %3 unordered : (!fir.shape<1>) -> !hlfir.expr<10xf32> {
      ^bb0(%arg3: index):
        %8 = hlfir.designate %4#0 (%arg3)  : (!fir.ref<!fir.array<10xf32>>, index) -> !fir.ref<f32>
        %9 = fir.load %8 : !fir.ref<f32>
        %10 = arith.mulf %5, %9 fastmath<contract> : f32
        hlfir.yield_element %10 : f32
      }
      %7 = hlfir.elemental %3 unordered : (!fir.shape<1>) -> !hlfir.expr<10xf32> {
      ^bb0(%arg3: index):
        %8 = hlfir.apply %6, %arg3 : (!hlfir.expr<10xf32>, index) -> f32
        %9 = hlfir.designate %1#0 (%arg3)  : (!fir.ref<!fir.array<10xf32>>, index) -> !fir.ref<f32>
        %10 = fir.load %9 : !fir.ref<f32>
        %11 = arith.addf %8, %10 fastmath<contract> : f32
        hlfir.yield_element %11 : f32
      }
      hlfir.assign %7 to %1#0 : !hlfir.expr<10xf32>, !fir.ref<!fir.array<10xf32>>
      hlfir.destroy %7 : !hlfir.expr<10xf32>
      hlfir.destroy %6 : !hlfir.expr<10xf32>
      omp.terminator
    }
    omp.terminator
  }
  omp.terminator
}
