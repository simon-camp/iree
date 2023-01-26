// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef MLIR_DIALECT_IREEC_IR_IREEC_H
#define MLIR_DIALECT_IREEC_IR_IREEC_H
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/FunctionImplementation.h"
#include "mlir/IR/FunctionInterfaces.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

// include order matters
#include "iree/compiler/Dialect/IREEC/IR/IREECDialect.h.inc"

#define GET_ATTRDEF_CLASSES
#include "iree/compiler/Dialect/IREEC/IR/IREECAttrs.h.inc"

#define GET_TYPEDEF_CLASSES
#include "iree/compiler/Dialect/IREEC/IR/IREECTypes.h.inc"

#define GET_OP_CLASSES
#include "iree/compiler/Dialect/IREEC/IR/IREECOps.h.inc"

namespace mlir {
namespace iree_compiler {

//===----------------------------------------------------------------------===//
// custom<SymbolVisibility>($sym_visibility)
//===----------------------------------------------------------------------===//
// some.op custom<SymbolVisibility>($sym_visibility) $sym_name
// ->
// some.op @foo
// some.op private @foo

ParseResult parseSymbolVisibility(OpAsmParser &parser,
                                  StringAttr &symVisibilityAttr);
void printSymbolVisibility(OpAsmPrinter &p, Operation *op,
                           StringAttr symVisibilityAttr);
}  // namespace iree_compiler
}  // namespace mlir

#endif  // MLIR_DIALECT_IREEC_IR_IREEC_H
