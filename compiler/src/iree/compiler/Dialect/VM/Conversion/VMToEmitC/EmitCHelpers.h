// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_VM_CONVERSION_VMTOEMITC_EMITCHELPERS_H_
#define IREE_COMPILER_DIALECT_VM_CONVERSION_VMTOEMITC_EMITCHELPERS_H_

#include "iree/compiler/Dialect/VM/Conversion/VMToEmitC/EmitCTypeConverter.h"
#include "iree/compiler/Dialect/VM/IR/VMOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir::iree_compiler::IREE::VM {

LogicalResult clearStruct(OpBuilder builder, Value structValue);

emitc::CallOpaqueOp failContainerNull(OpBuilder &builder, Location location,
                                      Type type, StringAttr callee,
                                      ArrayAttr args, ArrayRef<Value> operands,
                                      IREE::VM::ModuleAnalysis &moduleAnalysis);

void releaseRefs(OpBuilder &builder, Location location,
                 mlir::func::FuncOp funcOp,
                 IREE::VM::ModuleAnalysis &moduleAnalysis);

emitc::CallOpaqueOp returnIfError(OpBuilder &builder, Location location,
                                  StringAttr callee, ArrayAttr args,
                                  ArrayRef<Value> operands,
                                  IREE::VM::ModuleAnalysis &moduleAnalysis);

mlir::func::CallOp returnIfError(OpBuilder &builder, Location location,
                                 mlir::func::FuncOp &callee,
                                 ArrayRef<Value> operands,
                                 IREE::VM::ModuleAnalysis &moduleAnalysis);

} // namespace mlir::iree_compiler::IREE::VM

#endif // IREE_COMPILER_DIALECT_VM_CONVERSION_VMTOEMITC_EMITCHELPERS_H_
