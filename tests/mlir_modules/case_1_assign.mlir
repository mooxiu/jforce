func.func @kernel(%arg0: !fir.ref<!fir.array<10xf32>>, %arg1: !fir.ref<!fir.array<10xf32>>) {
  %c10 = arith.constant 10 : index
  %0 = fir.shape %c10 : (index) -> !fir.shape<1>
  %1:2 = hlfir.declare %arg0(%0) {uniq_name = "_QFEaa"} : (!fir.ref<!fir.array<10xf32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<10xf32>>, !fir.ref<!fir.array<10xf32>>)
  %2 = fir.shape %c10 : (index) -> !fir.shape<1>
  %3:2 = hlfir.declare %arg1(%2) {uniq_name = "_QFEbb"} : (!fir.ref<!fir.array<10xf32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<10xf32>>, !fir.ref<!fir.array<10xf32>>)
  omp.teams {
    omp.workdistribute {
      hlfir.assign %3#0 to %1#0 : !fir.ref<!fir.array<10xf32>>, !fir.ref<!fir.array<10xf32>>
      omp.terminator
    }
    omp.terminator
  }
  omp.terminator
}
