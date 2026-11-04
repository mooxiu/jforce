#include "mlir/IR/Verifier.h"
#include "sharding-examples.h"
#include "sharding-test-utils.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace sharding_test;

namespace {

void usage(const char *program) {
  llvm::errs() << "Usage: " << program
               << " [--example=NAME] [--prune] [combination]\n"
               << "       " << program << " --list-examples\n"
               << "       " << program << " --check-tune [--example=NAME]\n";
}

} // namespace

int main(int argc, char **argv) {
  bool prune = false;
  bool checkTune = false;
  bool exampleGiven = false;
  size_t selected = 0;
  llvm::StringRef exampleName = "chain";
  for (int i = 1; i < argc; ++i) {
    llvm::StringRef argument(argv[i]);
    if (argument == "--list-examples") {
      for (const auto &example : getExamples())
        llvm::outs() << example.name << ": " << example.description << "\n";
      return 0;
    }
    if (argument == "--help") {
      usage(argv[0]);
      return 0;
    }
    if (argument == "--check-tune") {
      checkTune = true;
    } else if (argument == "--prune") {
      prune = true;
    } else if (argument.consume_front("--example=")) {
      exampleName = argument;
      exampleGiven = true;
    } else if (selected || argument.getAsInteger(10, selected) ||
               selected == 0) {
      usage(argv[0]);
      return 1;
    }
  }

  if (checkTune) {
    // Compare Graph::tune with exhaustive search on every example, or only on
    // the one named by --example=.
    MLIRContext ctx;
    llvm::SmallVector<MeshAxis> meshAxes{{"x", 0, 2}, {"y", 1, 2}};
    bool found = false;
    for (const auto &candidate : getExamples()) {
      if (exampleGiven && candidate.name != exampleName) {
        continue;
      }
      found = true;
      auto source = candidate.build(ctx, meshAxes);
      if (!source || failed(verify(source.get())) ||
          failed(checkTuner(candidate.name,
                            source->lookupSymbol<func::FuncOp>("main"),
                            meshAxes))) {
        return 1;
      }
    }
    if (!found) {
      llvm::errs() << "Unknown example: " << exampleName
                   << ". Use --list-examples to see the available examples.\n";
      return 1;
    }
    return 0;
  }

  const Example *example = nullptr;
  for (const auto &candidate : getExamples())
    if (candidate.name == exampleName)
      example = &candidate;
  if (!example) {
    llvm::errs() << "Unknown example: " << exampleName
                 << ". Use --list-examples to see the available examples.\n";
    return 1;
  }

  MLIRContext ctx;
  if (failed(checkStrategyGenerators(ctx)))
    return 1;

  // Configure the fixed device mesh here, independently of graph construction.
  llvm::SmallVector<MeshAxis> meshAxes{{"x", 0, 2}, {"y", 1, 2}};
  auto source = example->build(ctx, meshAxes);
  if (!source || failed(verify(source.get())))
    return 1;

  llvm::SmallVector<OperationChoices> choices;
  if (failed(collectChoices(source->lookupSymbol<func::FuncOp>("main"),
                            meshAxes, choices)))
    return 1;
  if (countAssignments(choices) > maxEnumeratedAssignments) {
    llvm::errs() << exampleName << " has too many combinations to enumerate; "
                 << "use --check-tune --example=" << exampleName << "\n";
    return 1;
  }
  auto assignments = enumerateAssignments(choices, prune);
  if (assignments.empty() || selected > assignments.size()) {
    llvm::errs() << "Combination must be between 1 and " << assignments.size()
                 << "\n";
    return 1;
  }

  for (size_t i = 0; i < assignments.size(); ++i) {
    if (selected && selected != i + 1)
      continue;
    auto module = applyAssignment(source.get(), choices, assignments[i]);
    if (!module ||
        failed(lowerAndCheck(module.get(), choices, assignments[i]))) {
      llvm::errs() << "Failed example " << exampleName << ", combination "
                   << i + 1 << "\n";
      if (module)
        module->print(llvm::errs());
      return 1;
    }
    printAssignment(exampleName, i, assignments.size(), prune, module.get(),
                    choices, assignments[i]);
  }
  return 0;
}
