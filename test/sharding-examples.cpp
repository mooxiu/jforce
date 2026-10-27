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

} // namespace

llvm::ArrayRef<Example> getExamples() {
  static const Example examples[] = {
      {"addition", "two inputs, one add, one result", createAddition},
      {"chain", "matmul -> multiply -> add -> subtract", createChain},
      {"branch", "independent matmul/multiply branches and two adds",
       createBranch}};
  return examples;
}

} // namespace sharding_test
