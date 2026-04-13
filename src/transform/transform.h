#include "../jit-manager.h"
#include "../profiler.h"
#include "../utilities.h"
#include "flang/Optimizer/Dialect/FIRDialect.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/HLFIR/HLFIRDialect.h"
#include "flang/Optimizer/HLFIR/HLFIROps.h"
#include "flang/Optimizer/Transforms/Passes.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <iterator>
#include <mlir/Dialect/Affine/Passes.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/LLVMIR/LLVMTypes.h>
#include <mlir/Dialect/Utils/IndexingUtils.h>
#include <mlir/IR/AsmState.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/BlockSupport.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/IR/IRMapping.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/TypeRange.h>
#include <mlir/IR/Types.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Tools/mlir-opt/MlirOptMain.h>
#include <numeric>
#include <string>
#include <unordered_map>

using namespace mlir;

void inferShape(MLIRContext *ctx, ModuleOp moduleOp, int64_t NumHostArgs,
                void **ArgBasePtrs, int64_t *ArgSizes, int64_t *ArgTypes,
                llvm::DenseMap<Value, llvm::SmallVector<int>> &sliceShiftMap);

void optimizeSignatureForXLAAliasing(MLIRContext *context,
                                     func::FuncOp &funcOp);

func::FuncOp workdistributeToStableHLO(
    MLIRContext *context, const mlir::ModuleOp &moduleOp,
    const llvm::DenseMap<Value, llvm::SmallVector<int>> &sliceShiftMap);

llvm::DenseMap<unsigned, unsigned>
trimShapeArgs(MLIRContext *context, func::FuncOp &funcOp, int64_t *ArgTypes);
