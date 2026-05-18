module {
  func.func @main(%arg0: tensor<4xf64>, %arg1: tensor<4xf64>, %arg2: tensor<4xf64>, %arg3: tensor<f64> {jit.literal_val = 7 : i64}, %arg4: tensor<i32> {jit.literal_val = 4 : i64}, %arg5: tensor<i32> {jit.literal_val = 4 : i64}, %arg6: tensor<i32> {jit.literal_val = 4 : i64}) -> (tensor<4xf64>, tensor<4xf64>, tensor<4xf64>, tensor<f64>, tensor<i32>, tensor<i32>, tensor<i32>) {
    %0 = stablehlo.broadcast_in_dim %arg3, dims = [] : (tensor<f64>) -> tensor<4xf64>
    %1 = stablehlo.multiply %arg1, %0 : tensor<4xf64>
    %2 = stablehlo.add %1, %arg2 : tensor<4xf64>
    return %2, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6 : tensor<4xf64>, tensor<4xf64>, tensor<4xf64>, tensor<f64>, tensor<i32>, tensor<i32>, tensor<i32>
  }
}

