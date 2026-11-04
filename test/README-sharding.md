# Sharding examples

Build and run from the `jforce` directory:

```sh
cmake --build build --target test-sharding -j2
./build/test-sharding --list-examples
./build/test-sharding --example=addition
./build/test-sharding --example=chain --prune
./build/test-sharding --example=branch --prune
./build/test-sharding --example=chain 1234
./build/test-sharding --check-tune
./build/test-sharding --check-tune --example=layers
```

`--check-tune` runs `Graph::tune` (the ILP) on every example, or only on the
one named by `--example=`. For each it prints the cost and time of the ILP, of
exhaustive search and of pruned search, then the strategy the ILP chose for
every op. It fails if the ILP cost differs from the exhaustive minimum under the
same cost model. Examples with more than 10^6 combinations skip both searches,
so their ILP result is printed but not verified; the normal runner refuses them.

Times are wall-clock in milliseconds and include generating strategies and
building the ILP, not only the HiGHS solve. Measure in a Release build
(`-DCMAKE_BUILD_TYPE=Release`); a Debug build also compiles HiGHS unoptimized.

The default example is `chain`. Without `--prune`, every operator chooses its
strategy independently. A trailing number selects one combination for inspection.
Each combination starts from a clone of the unmodified example.

For the default 2x2 mesh:

| Example | Operators | Exhaustive combinations | Pruned combinations |
| --- | --- | --- | --- |
| addition | add | 9 | 9 |
| chain | matmul, multiply, add, subtract | 6,561 | 9 |
| branch | matmul, multiply, add, subtract, add | 59,049 | 81 |
| square | matmul, matmul | 81 | 81 |
| conflict | matmul, matmul, add | 729 | 81 |
| layers | 10 matmuls, 10 add/multiply/subtract | 1.15e18 | not enumerated |

## File responsibilities

- `sharding-examples.cpp`: graph construction and the example registry.
- `sharding-test-utils.cpp`: strategy collection, SSA-based producer lookup,
  enumeration, per-use reshards, lowering, and verification.
- `sharding-test.cpp`: command-line selection and device-mesh configuration.
- `src/runtime/sharding/sharding-tuner.cpp`: production strategy generators
  and the optional pruning heuristic.

## Add an example

Write a builder in `sharding-examples.cpp` using ordinary MLIR operations.
The built-in `addition` example shows the pattern:

```cpp
OwningOpRef<ModuleOp> createAddition(MLIRContext &ctx,
                                    llvm::ArrayRef<MeshAxis> axes) {
  auto module = createModule(ctx, axes);
  OpBuilder builder(&ctx);
  auto type = RankedTensorType::get({8, 8}, builder.getF64Type());
  auto function = addFunction(builder, module.get(), {type, type}, {type});
  auto sum = stablehlo::AddOp::create(builder, builder.getUnknownLoc(),
      function.getArgument(0), function.getArgument(1));
  func::ReturnOp::create(builder, builder.getUnknownLoc(),
                         ValueRange{sum.getResult()});
  return module;
}
```

Give the new builder a unique name and add it to `getExamples()`.
No search loops or producer indices
need changing: the runner reads operations and their operand relationships from
the function's SSA graph.

Examples currently contain one single-block `@main`, static ranked-tensor
arguments, ordinary 2D matmul, and same-shaped add/multiply/subtract. Unsupported
operations report an error. Add their strategy generators before using them.

Arguments take the layout requested by their first consumer; unused arguments
are replicated. Results inherit the returned values' layouts. Explicit argument
or result sharding attributes are respected, including conversions on return
edges when required. This preserves the existing test convention; it does not
enumerate independent layouts for every function boundary.

The runner inserts reshards for individual uses, verifies selected layouts after
Shardy lowering, and removes its temporary operation-identity attributes before
printing. It prints global-shape IR; it does not execute or time device kernels.
Pruning remains a heuristic and may exclude a better plan.
