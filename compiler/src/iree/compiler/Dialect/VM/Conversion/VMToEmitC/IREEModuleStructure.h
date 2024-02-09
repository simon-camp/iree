// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_VM_CONVERSION_VMTOEMITC_IREEMODULESTRUCTURE_H_
#define IREE_COMPILER_DIALECT_VM_CONVERSION_VMTOEMITC_IREEMODULESTRUCTURE_H_

#include "iree/compiler/Dialect/VM/Conversion/VMToEmitC/EmitCTypeConverter.h"
#include "iree/compiler/Dialect/VM/IR/VMOps.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir::iree_compiler::IREE::VM {

LogicalResult
createModuleStructure(IREE::VM::ModuleOp moduleOp,
                      IREE::VM::EmitCTypeConverter &typeConverter);

} // namespace mlir::iree_compiler::IREE::VM

#endif // IREE_COMPILER_DIALECT_VM_CONVERSION_VMTOEMITC_IREEMODULESTRUCTURE_H_
