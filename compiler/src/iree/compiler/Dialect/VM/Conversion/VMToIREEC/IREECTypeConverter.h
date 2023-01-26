// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_VM_CONVERSION_VMTOIREEC_IREECTYPECONVERTER_H_
#define IREE_COMPILER_DIALECT_VM_CONVERSION_VMTOIREEC_IREECTYPECONVERTER_H_

#include "iree/compiler/Dialect/IREEC/IR/IREEC.h"
#include "iree/compiler/Dialect/VM/Conversion/VMToIREEC/VMAnalysis.h"
#include "iree/compiler/Dialect/VM/IR/VMOps.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace IREEC {

class IREECTypeConverter : public mlir::TypeConverter {
 public:
  IREECTypeConverter();

  void cacheFunctionAnalysis(IREE::VM::FuncOp funcOp) {
    analysisCache.insert(
        std::make_pair(funcOp.getOperation(), VMAnalysis2(funcOp)));
  }

  LogicalResult moveFunctionAnalysis(IREE::VM::FuncOp funcOp,
                                     IREE::IREEC::FuncOp ireecFuncOp) {
    auto cachedAnalysis = lookupAnalysis(funcOp.getOperation());
    if (failed(cachedAnalysis)) {
      return failure();
    }
    analysisCache.insert(std::make_pair(
        ireecFuncOp.getOperation(), std::move(cachedAnalysis.value().get())));
    return success();
  }

 private:
  Value allocateLocalRef(IREE::IREEC::FuncOp funcOp, VMAnalysis2 &analysis,
                         int64_t ordinal);
  FailureOr<std::reference_wrapper<VMAnalysis2>> lookupAnalysis(Operation *op);
  Optional<Value> materializeRef(Value ref);

  VMAnalysisCache2 analysisCache;
};

}  // namespace IREEC
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_DIALECT_VM_CONVERSION_VMTOIREEC_IREECTYPECONVERTER_H_
