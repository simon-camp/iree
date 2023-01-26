// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/IREEC/IR/IREEC.h"

#include "iree/compiler/Dialect/IREEC/IR/IREECDialect.cpp.inc"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"

//===----------------------------------------------------------------------===//
// IREECDialect
//===----------------------------------------------------------------------===//

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace IREEC {
void IREECDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "iree/compiler/Dialect/IREEC/IR/IREECOps.cpp.inc"
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "iree/compiler/Dialect/IREEC/IR/IREECTypes.cpp.inc"
      >();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "iree/compiler/Dialect/IREEC/IR/IREECAttrs.cpp.inc"
      >();
}

}  // namespace IREEC
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "iree/compiler/Dialect/IREEC/IR/IREECOps.cpp.inc"

//===----------------------------------------------------------------------===//
// IREEC Attributes
//===----------------------------------------------------------------------===//

#define GET_ATTRDEF_CLASSES
#include "iree/compiler/Dialect/IREEC/IR/IREECAttrs.cpp.inc"

//===----------------------------------------------------------------------===//
// IREEC Types
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "iree/compiler/Dialect/IREEC/IR/IREECTypes.cpp.inc"
