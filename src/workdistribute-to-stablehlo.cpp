#include "mlir/IR/MLIRContext.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/AsmState.h"
#include <string>
#include "mlir/Parser/Parser.h"
#include "mlir/IR/BuiltinOps.h"

using namespace mlir;

std::string workdistributeToStableHLO(const std::string& rawIRStr) {
  // TODO: parse the string into moduleOp and lowering

  mlir::MLIRContext context;
  context.loadDialect<func::FuncDialect>();
  
  mlir::ParserConfig parserConfig(&context);
  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(rawIRStr, parserConfig);

  auto moduleOp = module.get();
  return rawIRStr;
}
