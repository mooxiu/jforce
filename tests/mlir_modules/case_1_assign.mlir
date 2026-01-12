omp.target map_entries(%139 -> %arg0, %142 -> %arg1 : !fir.ref<!fir.array<10xf32>>, !fir.ref<!fir.array<10xf32>>) {
  %c10_11 = arith.constant 10 : index
  %c10_12 = arith.constant 10 : index
  %143 = fir.shape %c10_12 : (index) -> !fir.shape<1>
  %144:2 = hlfir.declare %arg0(%143) {uniq_name = "_QFEaa"} : (!fir.ref<!fir.array<10xf32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<10xf32>>, !fir.ref<!fir.array<10xf32>>)
  %145 = fir.shape %c10_11 : (index) -> !fir.shape<1>
  %146:2 = hlfir.declare %arg1(%145) {uniq_name = "_QFEbb"} : (!fir.ref<!fir.array<10xf32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<10xf32>>, !fir.ref<!fir.array<10xf32>>)
  omp.teams {
    omp.workdistribute {
      hlfir.assign %146#0 to %144#0 : !fir.ref<!fir.array<10xf32>>, !fir.ref<!fir.array<10xf32>>
      omp.terminator
    }
    omp.terminator
  }
  omp.terminator
}

