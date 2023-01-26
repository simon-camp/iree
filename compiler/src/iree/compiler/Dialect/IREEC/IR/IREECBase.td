// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_DIALECT_IREEC_IR_IREECBASE
#define IREE_DIALECT_IREEC_IR_IREECBASE

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// IREEC dialect definition
//===----------------------------------------------------------------------===//

def IREEC_Dialect : Dialect {
  let name = "ireec";
  let cppNamespace = "::mlir::iree_compiler::IREE::IREEC";

  let summary = "";
  let description = [{}];

  let useDefaultAttributePrinterParser = 1;
  let useDefaultTypePrinterParser = 1;
  let useFoldAPI = kEmitFoldAdaptorFolder;
}

#endif // IREE_DIALECT_IREEC_IR_IREECBASE
