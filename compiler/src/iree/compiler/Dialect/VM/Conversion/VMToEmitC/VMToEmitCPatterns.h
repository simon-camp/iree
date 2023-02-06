// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_VM_CONVERSION_VMTOEMITC_VMTOEMITCPATTERNS_H_
#define IREE_COMPILER_DIALECT_VM_CONVERSION_VMTOEMITC_VMTOEMITCPATTERNS_H_

#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace iree_compiler {

void populateVMToEmitCPatterns2(ConversionTarget &conversionTarget,
                                TypeConverter &typeConverter,
                                RewritePatternSet &patterns);
}
}  // namespace mlir
#endif  // IREE_COMPILER_DIALECT_VM_CONVERSION_VMTOEMITC_VMTOEMITCPATTERNS_H_
