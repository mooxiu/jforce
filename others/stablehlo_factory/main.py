import jax
from jax import export
import jax.numpy as jnp
from jax._src.interpreters import mlir as jax_mlir
from jax._src.lib.mlir import ir

@jax.jit
def addition(x, y):
    return jnp.add(x, y)

@jax.jit
def dot_product(x, y):
   return jnp.dot(x, y)

@jax.jit
def transpose(x):
   return jnp.matrix_transpose(x)

@jax.jit
def matrix_multiply(x, y):
    return jnp.matmul(x, y)


# Returns prettyprint of StableHLO module without large constants
def get_stablehlo_asm(module_str):
    with jax_mlir.make_ir_context():
        stablehlo_module = ir.Module.parse(module_str, context=jax_mlir.make_ir_context())
        return stablehlo_module.operation.get_asm(large_elements_limit=20)

def main():
  # Export the function to StableHLO
  _N: int = 409600
  x = jnp.zeros(_N, dtype=jnp.float32)
  y = jnp.zeros(_N, dtype=jnp.float32)
  
  addition_op = export.export(addition)(x, y).mlir_module()

  dot_product_op = export.export(dot_product)(x, y).mlir_module()


  print("-" * 20)
  print("Addition stableHLO: ")
  print(get_stablehlo_asm(addition_op))
  print("-" * 20)
  print("Dot product stableHLO: ")
  print(get_stablehlo_asm(dot_product_op))

  
  m1 = jnp.array([[1, 2, 3], [4, 5, 6]])
  transpose_op = export.export(transpose)(m1).mlir_module()
  print("-" * 20)
  print("Transpose stableHLO: ")
  print(get_stablehlo_asm(transpose_op))

  m2 = jnp.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
  matrix_multiply_op = export.export(matrix_multiply)(m1, m2).mlir_module()
  print("-" * 20)
  print("Matrix multiply stableHLO: ")  
  print(get_stablehlo_asm(matrix_multiply_op))

if __name__ == "__main__":
  main()