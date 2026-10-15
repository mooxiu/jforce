#include "sharding-profile.h"
#include "runtime/sharding/sharding-codec.h"
#include "runtime/sharding/sharding-types.h"
#include "llvm/ADT/ArrayRef.h"
#include <nlohmann/json.hpp>

ShPGOProfile::ShPGOProfile(nlohmann::json &&data,
                           llvm::ArrayRef<MeshAxis> meshAxes) {
  if (!data.is_array()) {
    llvm::report_fatal_error("Sharding profile must be an array");
  }
  records.reserve(data.size());

  auto field = [&](const nlohmann::json &entry,
                   const char *name) -> const nlohmann::json & {
    auto it = entry.find(name);
    if (it == entry.end()) {
      llvm::errs() << "Missing profile field: " << name << "\n";
    }
    return *it;
  };

  for (const auto &entry : data) {
    ShPGOProgramRecord programRec{};
    programRec.devMeshFingerprint =
        field(entry, "devMeshFingerprint").get<std::string>();
    programRec.hwFingerprint = field(entry, "hwFingerprint").get<std::string>();
    auto kernelRecordsField = field(entry, "kernelRecords");
    std::vector<ShPGOKernelRecord> kernelRecords;
    kernelRecords.reserve(kernelRecordsField.size());
    for (const auto &kernelEntry : kernelRecordsField) {
      ShPGOKernelRecord rec{};
      rec.kernelFingerprint = static_cast<size_t>(
          field(kernelEntry, "kernelFingerprint").get<uint64_t>());
      rec.argShapes = field(kernelEntry, "argShapes")
                          .get<std::vector<std::vector<int64_t>>>();
      rec.argDTypes = field(kernelEntry, "argDTypes").get<std::vector<DType>>();
      rec.sd = toSD(field(kernelEntry, "sd"), meshAxes);
      rec.kernelExeTime =
          field(kernelEntry, "kernelExeTime").get<std::uint64_t>();
      kernelRecords.push_back(std::move(rec));
    }
    programRec.kernelRecords = std::move(kernelRecords);
    records.push_back(std::move(programRec));
  }
}
