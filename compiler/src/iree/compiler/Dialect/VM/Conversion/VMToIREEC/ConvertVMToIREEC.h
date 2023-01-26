// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_VM_CONVERSION_VMTOIREEC_CONVERTVMTOIREEC_H_
#define IREE_COMPILER_DIALECT_VM_CONVERSION_VMTOIREEC_CONVERTVMTOIREEC_H_

#include "iree/compiler/Dialect/VM/Conversion/VMToIREEC/IREECTypeConverter.h"
#include "iree/compiler/Dialect/VM/IR/VMOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace iree_compiler {

void populateVMToIREECPatterns(ConversionTarget &conversionTarget,
                               IREE::IREEC::IREECTypeConverter &typeConverter,
                               RewritePatternSet &patterns);

namespace IREE {
namespace VM {

std::unique_ptr<OperationPass<IREE::VM::ModuleOp>> createConvertVMToIREECPass();

}  // namespace VM
}  // namespace IREE

}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_DIALECT_VM_CONVERSION_VMTOIREEC_CONVERTVMTOIREEC_H_
