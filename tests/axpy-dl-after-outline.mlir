// ./build/src/tool/jforce-opt ./tests/axpy-dl-after-shape-infer.mlir --jforce-clean-fir-loop --canonicalize --cse --convert-hlfir-to-fir --canonicalize --fir-to-memref --fir-to-scf --enzyme-affinecfg --canonicalize --loop-invariant-code-motion --canonicalize --cse --loop-invariant-code-motion --jforce-outline-affine > ./tests/axpy-dl-after-outline.mlir

module {
  func.func @outlined_affinefor_110065004863024(%arg0: memref<i32>, %arg1: memref<i32>, %arg2: memref<f64>, %arg3: memref<4xf64>, %arg4: memref<4xf64>, %arg5: memref<4xf64>, %arg6: memref<i32>) {
    %0 = affine.load %arg0[] : memref<i32>
    %1 = arith.index_cast %0 : i32 to index
    %2 = affine.load %arg6[] : memref<i32>
    affine.for %arg7 = 0 to 4 {
      %3 = arith.addi %arg7, %1 : index
      %4 = arith.index_cast %3 : index to i32
      affine.store %4, %arg1[] : memref<i32>
      %5 = affine.load %arg2[] : memref<f64>
      %6 = affine.load %arg1[] : memref<i32>
      %7 = arith.extsi %6 : i32 to i64
      %8 = arith.index_cast %7 : i64 to index
      %9 = arith.subi %8, %1 : index
      %10 = memref.load %arg3[%9] : memref<4xf64>
      %11 = arith.mulf %5, %10 fastmath<contract> : f64
      %12 = memref.load %arg4[%9] : memref<4xf64>
      %13 = arith.addf %11, %12 fastmath<contract> : f64
      memref.store %13, %arg5[%9] : memref<4xf64>
      %14 = affine.load %arg1[] : memref<i32>
      %15 = arith.addi %14, %2 overflow<nsw> : i32
      affine.store %15, %arg1[] : memref<i32>
    }
    return
  }
  func.func @kernel(%arg0: !fir.ref<i32> {jit.arg_type = 1 : ui32, jit.literal_val = 0 : i64}, %arg1: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 4 : i64}, %arg2: !fir.ref<!fir.array<4xf64>> {jit.arg_type = 1 : ui32}, %arg3: !fir.ref<!fir.array<4xf64>> {jit.arg_type = 1 : ui32}, %arg4: !fir.ref<!fir.array<4xf64>> {jit.arg_type = 1 : ui32}, %arg5: !fir.ref<f64> {jit.arg_type = 1 : ui32, jit.literal_val = 4619567317775286272 : i64}, %arg6: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 4 : i64}, %arg7: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 4 : i64}, %arg8: !fir.ref<i32> {jit.arg_type = 0 : ui32, jit.literal_val = 4 : i64}) {
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %0 = fir.declare %arg0 {uniq_name = "_QFFrun_benchmarkEi"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %1 = fir.declare %arg1 {uniq_name = "_QFFrun_benchmarkEn"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %2 = fir.shape %c4 : (index) -> !fir.shape<1>
    %3 = fir.declare %arg2(%2) {uniq_name = "_QFFrun_benchmarkEz"} : (!fir.ref<!fir.array<4xf64>>, !fir.shape<1>) -> !fir.ref<!fir.array<4xf64>>
    %4 = fir.declare %arg3(%2) {uniq_name = "_QFFrun_benchmarkEx"} : (!fir.ref<!fir.array<4xf64>>, !fir.shape<1>) -> !fir.ref<!fir.array<4xf64>>
    %5 = fir.declare %arg4(%2) {uniq_name = "_QFFrun_benchmarkEy"} : (!fir.ref<!fir.array<4xf64>>, !fir.shape<1>) -> !fir.ref<!fir.array<4xf64>>
    %6 = fir.declare %arg5 {uniq_name = "_QFFrun_benchmarkEa"} : (!fir.ref<f64>) -> !fir.ref<f64>
    %7 = fir.convert %c1 : (index) -> i32
    %8 = fir.convert %0 : (!fir.ref<i32>) -> memref<i32>
    %9 = fir.convert %6 : (!fir.ref<f64>) -> memref<f64>
    %10 = fir.convert %4 : (!fir.ref<!fir.array<4xf64>>) -> memref<4xf64>
    %11 = fir.convert %5 : (!fir.ref<!fir.array<4xf64>>) -> memref<4xf64>
    %12 = fir.convert %3 : (!fir.ref<!fir.array<4xf64>>) -> memref<4xf64>
    %13 = arith.index_cast %c1 : index to i32
    %alloca = memref.alloca() : memref<i32>
    memref.store %13, %alloca[] : memref<i32>
    %alloca_0 = memref.alloca() : memref<i32>
    memref.store %7, %alloca_0[] : memref<i32>
    call @outlined_affinefor_110065004863024(%alloca, %8, %9, %10, %11, %12, %alloca_0) : (memref<i32>, memref<i32>, memref<f64>, memref<4xf64>, memref<4xf64>, memref<4xf64>, memref<i32>) -> ()
    omp.terminator
  }
}

