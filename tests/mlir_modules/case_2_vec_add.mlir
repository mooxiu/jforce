func.func @kernel(%arg0: !fir.ref<!fir.array<10xf32>>, %arg1: !fir.ref<!fir.array<10xf32>>, %arg2: !fir.ref<!fir.array<10xf32>>) {
  %c10 = arith.constant 10 : index
  %0 = fir.shape %c10 : (index) -> !fir.shape<1>
  %1:2 = hlfir.declare %arg0(%0) {uniq_name = "_QFEout"} : (!fir.ref<!fir.array<10xf32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<10xf32>>, !fir.ref<!fir.array<10xf32>>)
  %2 = fir.shape %c10 : (index) -> !fir.shape<1>
  %3:2 = hlfir.declare %arg1(%2) {uniq_name = "_QFEaa"} : (!fir.ref<!fir.array<10xf32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<10xf32>>, !fir.ref<!fir.array<10xf32>>)
  %4 = fir.shape %c10 : (index) -> !fir.shape<1>
  %5:2 = hlfir.declare %arg2(%4) {uniq_name = "_QFEbb"} : (!fir.ref<!fir.array<10xf32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<10xf32>>, !fir.ref<!fir.array<10xf32>>)
  omp.teams {
    omp.workdistribute {
      %6 = hlfir.elemental %2 unordered : (!fir.shape<1>) -> !hlfir.expr<10xf32> {
      ^bb0(%arg3: index):
        %7 = hlfir.designate %3#0 (%arg3)  : (!fir.ref<!fir.array<10xf32>>, index) -> !fir.ref<f32>
        %8 = hlfir.designate %5#0 (%arg3)  : (!fir.ref<!fir.array<10xf32>>, index) -> !fir.ref<f32>
        %9 = fir.load %7 : !fir.ref<f32>
        %10 = fir.load %8 : !fir.ref<f32>
        %11 = arith.addf %9, %10 fastmath<contract> : f32
        hlfir.yield_element %11 : f32
      }
      hlfir.assign %6 to %1#0 : !hlfir.expr<10xf32>, !fir.ref<!fir.array<10xf32>>
      hlfir.destroy %6 : !hlfir.expr<10xf32>
      omp.terminator
    }
    omp.terminator
  }
  omp.terminator
}
