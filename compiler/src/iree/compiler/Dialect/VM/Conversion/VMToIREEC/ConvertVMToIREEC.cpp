// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/VM/Conversion/VMToIREEC/ConvertVMToIREEC.h"

#include "iree/compiler/Dialect/IREEC/IR/IREEC.h"
#include "iree/compiler/Dialect/VM/Conversion/VMToIREEC/IREECTypeConverter.h"

namespace mlir {
namespace iree_compiler {

namespace {
class ModuleOpConversion : public OpConversionPattern<IREE::VM::ModuleOp> {
  using OpConversionPattern<IREE::VM::ModuleOp>::OpConversionPattern;
  using Adaptor = IREE::VM::ModuleOp::Adaptor;

  LogicalResult matchAndRewrite(
      IREE::VM::ModuleOp moduleOp, Adaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto loc = moduleOp.getLoc();

    auto ireec_module = rewriter.create<IREE::IREEC::ModuleOp>(
        loc, moduleOp.getSymVisibilityAttr(), moduleOp.getName());

    Region &region = ireec_module.getRegion();
    Block *block = rewriter.createBlock(&region);
    rewriter.inlineRegionBefore(moduleOp.getRegion(), block);
    rewriter.eraseOp(moduleOp);

    return success();
  }
};

class FuncOpConversion : public OpConversionPattern<IREE::VM::FuncOp> {
  using OpConversionPattern<IREE::VM::FuncOp>::OpConversionPattern;
  using Adaptor = IREE::VM::FuncOp::Adaptor;

  LogicalResult matchAndRewrite(
      IREE::VM::FuncOp funcOp, Adaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto ctx = funcOp.getContext();
    auto loc = funcOp.getLoc();

    TypeConverter::SignatureConversion signatureConverter(
        funcOp.getFunctionType().getNumInputs());
    for (const auto &arg : llvm::enumerate(funcOp.getArguments())) {
      Type convertedType =
          getTypeConverter()->convertType(arg.value().getType());
      signatureConverter.addInputs(arg.index(), convertedType);
    }

    rewriter.applySignatureConversion(&funcOp.getFunctionBody(),
                                      signatureConverter);

    SmallVector<Type> resultTypes;
    if (failed(getTypeConverter()->convertTypes(
            funcOp.getFunctionType().getResults(), resultTypes))) {
      return funcOp.emitError() << "signature conversion failed";
    }
    resultTypes.push_back(IREE::IREEC::StatusType::get(ctx));

    FunctionType functionType = rewriter.getFunctionType(
        signatureConverter.getConvertedTypes(), resultTypes);
    auto ireecFuncOp = rewriter.create<IREE::IREEC::FuncOp>(
        loc, functionType, ArrayAttr{}, ArrayAttr{});
    ireecFuncOp.setVisibility(funcOp.getVisibility());
    ireecFuncOp.setName(funcOp.getName());

    rewriter.inlineRegionBefore(funcOp.getRegion(), ireecFuncOp.getRegion(),
                                ireecFuncOp.getRegion().begin());

    IREE::IREEC::IREECTypeConverter *typeConverter =
        getTypeConverter<IREE::IREEC::IREECTypeConverter>();
    if (failed(typeConverter->moveFunctionAnalysis(funcOp, ireecFuncOp))) {
      return funcOp.emitError() << "failed to move analysis to new key";
    }

    rewriter.eraseOp(funcOp);
    return success();
  }
};

class ReturnOpConversion : public OpConversionPattern<IREE::VM::ReturnOp> {
  using OpConversionPattern<IREE::VM::ReturnOp>::OpConversionPattern;
  using Adaptor = IREE::VM::ReturnOp::Adaptor;

  LogicalResult matchAndRewrite(
      IREE::VM::ReturnOp returnOp, Adaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto ctx = returnOp.getContext();
    auto loc = returnOp.getLoc();

    Type statusType = IREE::IREEC::StatusType::get(ctx);
    Value statusOk =
        rewriter.create<IREE::IREEC::StatusOkOp>(loc, statusType).getResult();

    SmallVector<Value> operands;
    operands.append(adaptor.getOperands().begin(), adaptor.getOperands().end());
    operands.push_back(statusOk);

    rewriter.replaceOpWithNewOp<IREE::IREEC::ReturnOp>(returnOp, operands);

    return success();
  }
};

class ConstRefZeroOpConversion
    : public OpConversionPattern<IREE::VM::ConstRefZeroOp> {
  using Adaptor = IREE::VM::ConstRefZeroOp::Adaptor;
  using OpConversionPattern<IREE::VM::ConstRefZeroOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      IREE::VM::ConstRefZeroOp constRefZeroOp, Adaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto ctx = constRefZeroOp.getContext();
    auto loc = constRefZeroOp.getLoc();

    Type type = IREE::IREEC::RefType::get(ctx);
    Value ref = typeConverter->materializeTargetConversion(
        rewriter, loc, type, constRefZeroOp.getResult());

    rewriter.create<IREE::IREEC::RefReleaseOp>(loc, ref);

    rewriter.eraseOp(constRefZeroOp);
    return success();
  }
};

}  // namespace

void populateVMToIREECPatterns(ConversionTarget &conversionTarget,
                               IREE::IREEC::IREECTypeConverter &typeConverter,
                               RewritePatternSet &patterns) {
  auto context = patterns.getContext();
  patterns.add<FuncOpConversion, ModuleOpConversion, ReturnOpConversion,
               ConstRefZeroOpConversion>(typeConverter, context);
}

namespace IREE {
namespace VM {

namespace {

class ConvertVMToIREECPass
    : public PassWrapper<ConvertVMToIREECPass,
                         OperationPass<IREE::VM::ModuleOp>> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertVMToIREECPass)

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<IREE::IREEC::IREECDialect, IREE::VM::VMDialect>();
  }

  StringRef getArgument() const override { return "iree-convert-vm-to-ireec"; }

  StringRef getDescription() const override {
    return "Convert VM Ops to the IREEC dialect";
  }

  void runOnOperation() override {
    IREE::VM::ModuleOp module = getOperation();

    ConversionTarget target(getContext());
    IREEC::IREECTypeConverter typeConverter;

    RewritePatternSet patterns(&getContext());
    populateVMToIREECPatterns(target, typeConverter, patterns);

    target.addLegalDialect<IREE::IREEC::IREECDialect>();
    target.addLegalDialect<IREE::VM::VMDialect>();
    // target.addIllegalOp<IREE::VM::ModuleOp>();
    target.addIllegalOp<IREE::VM::FuncOp>();
    target.addIllegalOp<IREE::VM::ReturnOp>();
    target.addIllegalOp<IREE::VM::ConstRefZeroOp>();
    target.addIllegalOp<UnrealizedConversionCastOp>();

    for (auto funcOp : module.getOps<IREE::VM::FuncOp>()) {
      typeConverter.cacheFunctionAnalysis(funcOp);
    }

    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      return signalPassFailure();
    }

    module.walk([](UnrealizedConversionCastOp castOp) {
      if (llvm::all_of(castOp.getResults(),
                       [](Value result) { return result.use_empty(); })) {
        castOp.erase();
      }
    });
  }
};

}  // namespace

std::unique_ptr<OperationPass<IREE::VM::ModuleOp>>
createConvertVMToIREECPass() {
  return std::make_unique<ConvertVMToIREECPass>();
}

}  // namespace VM
}  // namespace IREE

static PassRegistration<IREE::VM::ConvertVMToIREECPass> pass;

}  // namespace iree_compiler
}  // namespace mlir
