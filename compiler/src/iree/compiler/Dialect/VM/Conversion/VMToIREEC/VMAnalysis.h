// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_VM_CONVERSION_VMTOIREEC_VMANALYSIS_H_
#define IREE_COMPILER_DIALECT_VM_CONVERSION_VMTOIREEC_VMANALYSIS_H_

#include "iree/compiler/Dialect/VM/Analysis/RegisterAllocation.h"
#include "iree/compiler/Dialect/VM/Analysis/ValueLiveness.h"
#include "iree/compiler/Dialect/VM/IR/VMTypes.h"

namespace mlir {
namespace iree_compiler {

struct VMAnalysis2 {
 public:
  VMAnalysis2(IREE::VM::FuncOp funcOp) {
    Operation *op = funcOp.getOperation();
    registerAllocation = RegisterAllocation(op);
    valueLiveness = ValueLiveness(op);
    originalFunctionType = funcOp.getFunctionType();
  }

  VMAnalysis2(VMAnalysis2 &&) = default;
  VMAnalysis2 &operator=(VMAnalysis2 &&) = default;
  VMAnalysis2(const VMAnalysis2 &) = delete;
  VMAnalysis2 &operator=(const VMAnalysis2 &) = delete;

  FunctionType getFunctionType() { return originalFunctionType; }

  int getNumRefRegisters() {
    return registerAllocation.getMaxRefRegisterOrdinal() + 1;
  }

  int getNumRefArguments() {
    assert(originalFunctionType);
    return llvm::count_if(originalFunctionType.getInputs(), [](Type inputType) {
      return inputType.isa<IREE::VM::RefType>();
    });
  }

  int getNumLocalRefs() { return getNumRefRegisters() - getNumRefArguments(); }

  uint16_t getRefRegisterOrdinal(Value ref) {
    assert(ref.getType().isa<IREE::VM::RefType>());
    return registerAllocation.mapToRegister(ref).ordinal();
  }

  bool isMove(Value ref, Operation *op) {
    assert(ref.getType().isa<IREE::VM::RefType>());
    bool lastUse = valueLiveness.isLastValueUse(ref, op);
    return lastUse && false;
  }

  void cacheLocalRef(int64_t ordinal, Value ref) {
    assert(!refs.count(ordinal));
    refs[ordinal] = ref;
  }

  Optional<Value> lookupLocalRef(int64_t ordinal) {
    if (refs.count(ordinal)) {
      return refs[ordinal];
    } else {
      return std::nullopt;
    }
  }

 private:
  RegisterAllocation registerAllocation;
  ValueLiveness valueLiveness;
  DenseMap<int64_t, Value> refs;
  FunctionType originalFunctionType;
};

using VMAnalysisCache2 = DenseMap<Operation *, VMAnalysis2>;

}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_DIALECT_VM_CONVERSION_VMTOIREEC_VMANALYSIS_H_
