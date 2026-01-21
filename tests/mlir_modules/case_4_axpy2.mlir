func.func @kernel(%arg0: !fir.ref<!fir.array<10xf32>>, %arg1: !fir.ref<f32>, %arg2: !fir.ref<!fir.array<10xf32>>, %arg3: !fir.ref<!fir.array<10xf32>>) {
  %c10 = arith.constant 10 : index
  %0 = fir.shape %c10 : (index) -> !fir.shape<1>
  %1:2 = hlfir.declare %arg0(%0) {uniq_name = "_QFEtmp"} : (!fir.ref<!fir.array<10xf32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<10xf32>>, !fir.ref<!fir.array<10xf32>>)
  %2:2 = hlfir.declare %arg1 {uniq_name = "_QFEa"} : (!fir.ref<f32>) -> (!fir.ref<f32>, !fir.ref<f32>)
  %3 = fir.shape %c10 : (index) -> !fir.shape<1>
  %4:2 = hlfir.declare %arg2(%3) {uniq_name = "_QFEx"} : (!fir.ref<!fir.array<10xf32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<10xf32>>, !fir.ref<!fir.array<10xf32>>)
  %5 = fir.shape %c10 : (index) -> !fir.shape<1>
  %6:2 = hlfir.declare %arg3(%5) {uniq_name = "_QFEy"} : (!fir.ref<!fir.array<10xf32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<10xf32>>, !fir.ref<!fir.array<10xf32>>)
  omp.teams {
    omp.workdistribute {
      %7 = fir.load %2#0 : !fir.ref<f32>
      %8 = hlfir.elemental %3 unordered : (!fir.shape<1>) -> !hlfir.expr<10xf32> {
      ^bb0(%arg4: index):
        %10 = hlfir.designate %4#0 (%arg4)  : (!fir.ref<!fir.array<10xf32>>, index) -> !fir.ref<f32>
        %11 = fir.load %10 : !fir.ref<f32>
        %12 = arith.mulf %7, %11 fastmath<contract> : f32
        hlfir.yield_element %12 : f32
      }
      hlfir.assign %8 to %1#0 : !hlfir.expr<10xf32>, !fir.ref<!fir.array<10xf32>>
      hlfir.destroy %8 : !hlfir.expr<10xf32>
      %9 = hlfir.elemental %5 unordered : (!fir.shape<1>) -> !hlfir.expr<10xf32> {
      ^bb0(%arg4: index):
        %10 = hlfir.designate %6#0 (%arg4)  : (!fir.ref<!fir.array<10xf32>>, index) -> !fir.ref<f32>
        %11 = hlfir.designate %1#0 (%arg4)  : (!fir.ref<!fir.array<10xf32>>, index) -> !fir.ref<f32>
        %12 = fir.load %10 : !fir.ref<f32>
        %13 = fir.load %11 : !fir.ref<f32>
        %14 = arith.addf %12, %13 fastmath<contract> : f32
        hlfir.yield_element %14 : f32
      }
      hlfir.assign %9 to %6#0 : !hlfir.expr<10xf32>, !fir.ref<!fir.array<10xf32>>
      hlfir.destroy %9 : !hlfir.expr<10xf32>
      omp.terminator
    }
    omp.terminator
  }
  omp.terminator
}
