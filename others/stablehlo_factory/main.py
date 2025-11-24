import jax
from jax import export
import jax.numpy as jnp
from jax._src.interpreters import mlir as jax_mlir
from jax._src.lib.mlir import ir

@jax.jit
def addition(x, y):
    return jnp.add(x, y)

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
  stablehlo_axpy = export.export(addition)(x, y).mlir_module()
  print(get_stablehlo_asm(stablehlo_axpy))
  


if __name__ == "__main__":
  main()