// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "../PassDetail.h"
#include "iree-dialects/Dialect/IREE/IREEDialect.h"
#include "iree-dialects/Dialect/IREE/IREEOps.h"
#include "iree-dialects/Dialect/IREEPyDM/IR/Ops.h"
#include "iree-dialects/Dialect/IREEPyDM/Transforms/Passes.h"
#include "iree-dialects/Dialect/IREEPyDM/Transforms/ToIREE/Patterns.h"
#include "iree-dialects/Dialect/IREEPyDM/Transforms/ToIREE/TypeConverter.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BuiltinDialect.h"

using namespace mlir;
using namespace mlir::iree_pydm;

namespace {

struct ConvertIREEPyDMToIREEPass
    : public ConvertIREEPyDMToIREEBase<ConvertIREEPyDMToIREEPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<mlir::iree::IREEDialect, BuiltinDialect, StandardOpsDialect,
                    math::MathDialect>();
  }

  void runOnOperation() override {
    auto *context = &getContext();
    auto moduleOp = getOperation();
    LoweringTypeConverter typeConverter;
    RewritePatternSet patterns(context);
    populatePyDMToIREELoweringPatterns(context, typeConverter, patterns);

    ConversionTarget target(*context);
    target.addIllegalDialect<IREEPyDMDialect>();
    target.addLegalDialect<BuiltinDialect>();
    target.addLegalDialect<mlir::iree::IREEDialect>();
    target.addLegalDialect<mlir::math::MathDialect>();
    target.addLegalDialect<StandardOpsDialect>();
    if (failed(applyPartialConversion(moduleOp, target, std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::iree_pydm::createConvertIREEPyDMToIREEPass() {
  return std::make_unique<ConvertIREEPyDMToIREEPass>();
}
