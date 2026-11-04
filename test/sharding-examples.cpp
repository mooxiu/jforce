#include "sharding-examples.h"
#include "mlir/IR/BuiltinTypes.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "stablehlo/dialect/StablehloOps.h"

using namespace mlir;

namespace sharding_test {

OwningOpRef<ModuleOp> createModule(MLIRContext &ctx,
                                   llvm::ArrayRef<MeshAxis> axes) {
  ctx.getOrLoadDialect<func::FuncDialect>();
  ctx.getOrLoadDialect<stablehlo::StablehloDialect>();
  ctx.getOrLoadDialect<sdy::SdyDialect>();
  OpBuilder builder(&ctx);
  auto module =
      OwningOpRef<ModuleOp>(ModuleOp::create(builder.getUnknownLoc()));
  builder.setInsertionPointToEnd(module->getBody());
  llvm::SmallVector<sdy::MeshAxisAttr> attrs;
  for (const auto &axis : axes)
    attrs.push_back(sdy::MeshAxisAttr::get(&ctx, axis.name, axis.size));
  sdy::MeshOp::create(builder, builder.getUnknownLoc(), meshName,
                      sdy::MeshAttr::get(&ctx, attrs));
  return module;
}

func::FuncOp addFunction(OpBuilder &builder, ModuleOp module, TypeRange inputs,
                         TypeRange results) {
  builder.setInsertionPointToEnd(module.getBody());
  auto function =
      func::FuncOp::create(builder, builder.getUnknownLoc(), "main",
                           builder.getFunctionType(inputs, results));
  builder.setInsertionPointToEnd(function.addEntryBlock());
  return function;
}

Value addMatmul(OpBuilder &builder, Value lhs, Value rhs) {
  auto a = cast<RankedTensorType>(lhs.getType());
  auto b = cast<RankedTensorType>(rhs.getType());
  auto result = RankedTensorType::get({a.getDimSize(0), b.getDimSize(1)},
                                      a.getElementType());
  auto dims = stablehlo::DotDimensionNumbersAttr::get(builder.getContext(), {},
                                                      {}, {1}, {0});
  return stablehlo::DotGeneralOp::create(builder, builder.getUnknownLoc(),
                                         result, lhs, rhs, dims, ArrayAttr{},
                                         stablehlo::DotAlgorithmAttr{})
      .getResult();
}

namespace {

OwningOpRef<ModuleOp> createAddition(MLIRContext &ctx,
                                     llvm::ArrayRef<MeshAxis> axes) {
  auto module = createModule(ctx, axes);
  OpBuilder builder(&ctx);
  auto type = RankedTensorType::get({8, 8}, builder.getF64Type());
  auto function = addFunction(builder, module.get(), {type, type}, {type});
  auto sum = stablehlo::AddOp::create(builder, builder.getUnknownLoc(),
                                      function.getArgument(0),
                                      function.getArgument(1));
  func::ReturnOp::create(builder, builder.getUnknownLoc(),
                         ValueRange{sum.getResult()});
  return module;
}

OwningOpRef<ModuleOp> createChain(MLIRContext &ctx,
                                  llvm::ArrayRef<MeshAxis> axes) {
  auto module = createModule(ctx, axes);
  OpBuilder builder(&ctx);
  auto a = RankedTensorType::get({8, 16}, builder.getF64Type());
  auto b = RankedTensorType::get({16, 8}, builder.getF64Type());
  auto c = RankedTensorType::get({8, 8}, builder.getF64Type());
  auto function = addFunction(builder, module.get(), {c, a, b}, {c, a, b});
  Value inputC = function.getArgument(0);
  auto loc = builder.getUnknownLoc();

  // P = A @ B; Q = P * C; R = Q + P; C_out = R - C.
  Value p =
      addMatmul(builder, function.getArgument(1), function.getArgument(2));
  Value q = stablehlo::MulOp::create(builder, loc, p, inputC).getResult();
  Value r = stablehlo::AddOp::create(builder, loc, q, p).getResult();
  Value output =
      stablehlo::SubtractOp::create(builder, loc, r, inputC).getResult();
  func::ReturnOp::create(
      builder, loc,
      ValueRange{output, function.getArgument(1), function.getArgument(2)});
  return module;
}

OwningOpRef<ModuleOp> createBranch(MLIRContext &ctx,
                                   llvm::ArrayRef<MeshAxis> axes) {
  auto module = createModule(ctx, axes);
  OpBuilder builder(&ctx);
  auto a = RankedTensorType::get({8, 16}, builder.getF64Type());
  auto b = RankedTensorType::get({16, 8}, builder.getF64Type());
  auto c = RankedTensorType::get({8, 8}, builder.getF64Type());
  auto function =
      addFunction(builder, module.get(), {c, a, b, c}, {c, a, b, c});
  Value inputC = function.getArgument(0), d = function.getArgument(3);
  auto loc = builder.getUnknownLoc();

  // Independent producers meet at add. A second add checks repeated op kinds.
  Value p =
      addMatmul(builder, function.getArgument(1), function.getArgument(2));
  Value q = stablehlo::MulOp::create(builder, loc, inputC, d).getResult();
  Value r = stablehlo::AddOp::create(builder, loc, p, q).getResult();
  Value s = stablehlo::SubtractOp::create(builder, loc, r, inputC).getResult();
  Value output = stablehlo::AddOp::create(builder, loc, s, d).getResult();
  func::ReturnOp::create(
      builder, loc,
      ValueRange{output, function.getArgument(1), function.getArgument(2), d});
  return module;
}

OwningOpRef<ModuleOp> createSquare(MLIRContext &ctx,
                                   llvm::ArrayRef<MeshAxis> axes) {
  auto module = createModule(ctx, axes);
  OpBuilder builder(&ctx);
  auto a = RankedTensorType::get({8, 16}, builder.getF64Type());
  auto b = RankedTensorType::get({16, 8}, builder.getF64Type());
  auto c = RankedTensorType::get({8, 8}, builder.getF64Type());
  auto function = addFunction(builder, module.get(), {a, b}, {c});

  // P = A @ B; Q = P @ P. P is both the (M, K) and the (K, N) operand of the
  // second matmul, so no plan avoids all communication.
  Value p =
      addMatmul(builder, function.getArgument(0), function.getArgument(1));
  Value q = addMatmul(builder, p, p);
  func::ReturnOp::create(builder, builder.getUnknownLoc(), ValueRange{q});
  return module;
}

OwningOpRef<ModuleOp> createConflict(MLIRContext &ctx,
                                     llvm::ArrayRef<MeshAxis> axes) {
  auto module = createModule(ctx, axes);
  OpBuilder builder(&ctx);
  auto a = RankedTensorType::get({8, 16}, builder.getF64Type());
  auto b = RankedTensorType::get({16, 8}, builder.getF64Type());
  auto c = RankedTensorType::get({8, 8}, builder.getF64Type());
  auto function = addFunction(builder, module.get(), {a, b}, {c});

  // P = A @ B; Q = P @ P; R = P + Q. The add's inputs come from two matmuls
  // that prefer different layouts, so pruning must join conflicting inputs.
  Value p =
      addMatmul(builder, function.getArgument(0), function.getArgument(1));
  Value q = addMatmul(builder, p, p);
  Value r =
      stablehlo::AddOp::create(builder, builder.getUnknownLoc(), p, q)
          .getResult();
  func::ReturnOp::create(builder, builder.getUnknownLoc(), ValueRange{r});
  return module;
}

OwningOpRef<ModuleOp> createLayers(MLIRContext &ctx,
                                   llvm::ArrayRef<MeshAxis> axes) {
  auto module = createModule(ctx, axes);
  OpBuilder builder(&ctx);
  auto loc = builder.getUnknownLoc();
  auto f64 = builder.getF64Type();
  // Batch 2: only one mesh axis can split it, so the other axis has to split
  // a feature dimension and the plans trade communication against each other.
  auto x = RankedTensorType::get({2, 32}, f64);
  auto hidden = RankedTensorType::get({2, 64}, f64);
  auto out = RankedTensorType::get({2, 16}, f64);
  auto up = RankedTensorType::get({32, 64}, f64);
  auto down = RankedTensorType::get({64, 32}, f64);
  auto square = RankedTensorType::get({32, 32}, f64);
  auto head = RankedTensorType::get({32, 16}, f64);
  auto function = addFunction(
      builder, module.get(),
      {x, up, hidden, down, square, square, square, square, square, up,
       hidden, down, head, out},
      {out});
  auto arg = [&](unsigned i) { return function.getArgument(i); };
  auto add = [&](Value a, Value b) {
    return stablehlo::AddOp::create(builder, loc, a, b).getResult();
  };
  auto mul = [&](Value a, Value b) {
    return stablehlo::MulOp::create(builder, loc, a, b).getResult();
  };
  auto sub = [&](Value a, Value b) {
    return stablehlo::SubtractOp::create(builder, loc, a, b).getResult();
  };

  // Layer 1, MLP: r1 = (((X @ W1) + B1)^2 @ W2) + X
  Value h1 = addMatmul(builder, arg(0), arg(1));
  Value h1b = add(h1, arg(2));
  Value g1 = mul(h1b, h1b);
  Value o1 = addMatmul(builder, g1, arg(3));
  Value r1 = add(o1, arg(0));

  // Layer 2, attention-like: r2 = ((((r1 Wq) * (r1 Wk)) Wa * (r1 Wv)) Wo) + r1
  Value q = addMatmul(builder, r1, arg(4));
  Value k = addMatmul(builder, r1, arg(5));
  Value v = addMatmul(builder, r1, arg(6));
  Value s = mul(q, k);
  Value a = addMatmul(builder, s, arg(7));
  Value c = mul(a, v);
  Value o2 = addMatmul(builder, c, arg(8));
  Value r2 = add(o2, r1);

  // Layer 3, MLP: r3 = (((r2 @ W3) - B3) * (r2 @ W3)) @ W4 + r2
  Value h3 = addMatmul(builder, r2, arg(9));
  Value h3b = sub(h3, arg(10));
  Value g3 = mul(h3b, h3);
  Value o3 = addMatmul(builder, g3, arg(11));
  Value r3 = add(o3, r2);

  // Output head and a loss-like difference.
  Value y = addMatmul(builder, r3, arg(12));
  Value d = sub(y, arg(13));
  func::ReturnOp::create(builder, loc, ValueRange{d});
  return module;
}

} // namespace

llvm::ArrayRef<Example> getExamples() {
  static const Example examples[] = {
      {"addition", "two inputs, one add, one result", createAddition},
      {"chain", "matmul -> multiply -> add -> subtract", createChain},
      {"branch", "independent matmul/multiply branches and two adds",
       createBranch},
      {"square", "matmul whose result feeds both operands of a second matmul",
       createSquare},
      {"conflict", "add whose inputs come from two differently sharded matmuls",
       createConflict},
      {"layers", "three residual layers, 20 ops (too large to enumerate)",
       createLayers}};
  return examples;
}

} // namespace sharding_test
