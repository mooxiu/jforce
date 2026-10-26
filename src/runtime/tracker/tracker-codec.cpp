#include "tracker-codec.h"

std::string serializeFunction(mlir::func::FuncOp func) {
  std::string text;
  llvm::raw_string_ostream os(text);

  mlir::OpPrintingFlags flags;
  flags.useLocalScope();
  flags.enableDebugInfo(false);
  flags.printGenericOpForm();

  func.print(os, flags);
  os.flush();
  return text;
}

