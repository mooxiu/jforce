// RUN: %jforce-opt --help | FileCheck %s --check-prefix=HELP
// RUN: %jforce-opt %s | FileCheck %s --check-prefix=IR

// Confirm that a JForce pass is registered.
// HELP: --jforce-annotate

module {
  func.func @smoke() -> i32 {
    %zero = arith.constant 0 : i32
    return %zero : i32
  }
}

// IR-LABEL: func.func @smoke
// IR: %[[ZERO:.*]] = arith.constant 0 : i32
// IR: return %[[ZERO]] : i32
