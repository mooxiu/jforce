#include "utilities.h"
#include "mlir/IR/OperationSupport.h"
#include <cstdint>

/// Copy from `offload/include/omptarget.h` as the headers are not included in this project
/// Data attributes for each data reference used in an OpenMP target region.
enum tgt_map_type {
  // No flags
  OMP_TGT_MAPTYPE_NONE = 0x000,
  // copy data from host to device
  OMP_TGT_MAPTYPE_TO = 0x001,
  // copy data from device to host
  OMP_TGT_MAPTYPE_FROM = 0x002,
  // copy regardless of the reference count
  OMP_TGT_MAPTYPE_ALWAYS = 0x004,
  // force unmapping of data
  OMP_TGT_MAPTYPE_DELETE = 0x008,
  // map the pointer as well as the pointee
  OMP_TGT_MAPTYPE_PTR_AND_OBJ = 0x010,
  // pass device base address to kernel
  OMP_TGT_MAPTYPE_TARGET_PARAM = 0x020,
  // return base device address of mapped data
  OMP_TGT_MAPTYPE_RETURN_PARAM = 0x040,
  // private variable - not mapped
  OMP_TGT_MAPTYPE_PRIVATE = 0x080,
  // copy by value - not mapped
  OMP_TGT_MAPTYPE_LITERAL = 0x100,
  // mapping is implicit
  OMP_TGT_MAPTYPE_IMPLICIT = 0x200,
  // copy data to device
  OMP_TGT_MAPTYPE_CLOSE = 0x400,
  // runtime error if not already allocated
  OMP_TGT_MAPTYPE_PRESENT = 0x1000,
  // use a separate reference counter so that the data cannot be unmapped within
  // the structured region
  // This is an OpenMP extension for the sake of OpenACC support.
  OMP_TGT_MAPTYPE_OMPX_HOLD = 0x2000,
  // Attach pointer and pointee, after processing all other maps.
  // Applicable to map-entering directives. Does not change ref-count.
  OMP_TGT_MAPTYPE_ATTACH = 0x4000,
  // descriptor for non-contiguous target-update
  OMP_TGT_MAPTYPE_NON_CONTIG = 0x100000000000,
  // member of struct, member given by [16 MSBs] - 1
  OMP_TGT_MAPTYPE_MEMBER_OF = 0xffff000000000000
};

bool isLiteralTy(int64_t argType) {
  return (bool)(argType&tgt_map_type::OMP_TGT_MAPTYPE_LITERAL);  
}

bool isMappedTy(int64_t argType) {
  return (bool)(argType&tgt_map_type::OMP_TGT_MAPTYPE_TARGET_PARAM);
}

bool isPartOfStructTy(int64_t argType) {
  return (bool)(argType&tgt_map_type::OMP_TGT_MAPTYPE_MEMBER_OF);
}

bool isPointerAndPointeeTy(int64_t argType) {
  return (bool)(argType&tgt_map_type::OMP_TGT_MAPTYPE_PTR_AND_OBJ);
} 

void logger::Log(std::string msg, logLevel level) {
  switch (level) {
  case logLevel::DEBUG:
    llvm::dbgs() << "[DEBUG] " << msg << "\n";
    break;
  case logLevel::ERROR:
    llvm::dbgs() << "[ERROR] " << msg << "\n";
    break;
  }
}

// FIXME: does not cover full situations, can be false negative.
bool isDynamicShape(mlir::Type type){
  if (auto ref = llvm::dyn_cast<fir::ReferenceType>(type)) {
    return isDynamicShape(ref.getEleTy());
  }

  if (auto box = llvm::dyn_cast<fir::BoxType>(type)) {
    return isDynamicShape(box.getEleTy());
  }

  if (auto seq = llvm::dyn_cast<fir::SequenceType>(type)){
    return seq.hasDynamicExtents();
  }

  if (auto exp = llvm::dyn_cast<hlfir::ExprType>(type)) {
    for (auto dim: exp.getShape()) {
      // TODO: is this correct usage?
      if (dim == mlir::ShapedType::kDynamic) {
        return true;
      }
    }
  }
  return false;
};

mlir::Type convertToStaticShape(mlir::Type type, llvm::ArrayRef<int64_t> shape){
  if (!isDynamicShape(type)){
    return type;
  }

  return llvm::TypeSwitch<mlir::Type, mlir::Type>(type)
    .Case<fir::ReferenceType>([&](fir::ReferenceType rType){
      return fir::ReferenceType::get(convertToStaticShape(rType.getEleTy(), shape));
    })
    .Case<fir::BoxType>([&](fir::BoxType bType){
      return fir::BoxType::get(convertToStaticShape(bType.getEleTy(), shape));
    })
    .Case<fir::SequenceType>([&](fir::SequenceType sType){
      return fir::SequenceType::get(shape, sType.getEleTy());
    })
    .Case<hlfir::ExprType>([&](hlfir::ExprType eType){
      return hlfir::ExprType::get(eType.getContext(), shape, eType.getEleTy(), eType.getPolymorphic());
    })
    .Default([&](mlir::Type t){
      return t;
    });
};

std::string getMLIROperationAsString(mlir::Operation* op) {
  std::string output;
  llvm::raw_string_ostream os(output);
  mlir::OpPrintingFlags flags;
  op->print(os, flags.useLocalScope());
  return output;
}

