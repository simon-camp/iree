// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/VM/Conversion/VMToIREEC/ConvertVMToIREEC.h"

#include "iree/compiler/Dialect/IREEC/IR/IREEC.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
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

    signatureConverter.addInputs({
        IREE::IREEC::StackType::get(ctx),
        IREE::IREEC::ModuleType::get(ctx),
        IREE::IREEC::ModuleStateType::get(ctx),
    });

    IREE::IREEC::IREECTypeConverter *typeConverter =
        getTypeConverter<IREE::IREEC::IREECTypeConverter>();

    for (const auto &arg : llvm::enumerate(funcOp.getArguments())) {
      Type convertedType = typeConverter->convertType(arg.value().getType());
      signatureConverter.addInputs(arg.index(), convertedType);
    }

    SmallVector<Type> resultTypes;
    resultTypes.push_back(IREE::IREEC::StatusType::get(ctx));
    if (failed(typeConverter->convertTypes(
            funcOp.getFunctionType().getResults(), resultTypes))) {
      return funcOp.emitError() << "signature conversion failed";
    }

    // if (!resultTypes.empty()) {
    //   signatureConverter.addInputs(resultTypes);
    // }

    FunctionType functionType = FunctionType::get(
        ctx, signatureConverter.getConvertedTypes(), resultTypes);

    auto ireecFuncOp = rewriter.replaceOpWithNewOp<IREE::IREEC::FuncOp>(
        funcOp, functionType, ArrayAttr{}, ArrayAttr{});
    ireecFuncOp.setVisibility(funcOp.getVisibility());
    ireecFuncOp.setName(funcOp.getName());

    rewriter.inlineRegionBefore(funcOp.getRegion(), ireecFuncOp.getRegion(),
                                ireecFuncOp.getRegion().begin());

    if (failed(typeConverter->moveFunctionAnalysis(funcOp, ireecFuncOp))) {
      return funcOp.emitError() << "failed to move analysis to new key";
    }

    // for (auto &block : llvm::drop_begin(ireecFuncOp.getBlocks())) {
    //   for (auto arg : block.getArguments()) {
    //     arg.dump();
    //     if (!typeConverter->isLegal(arg.getType())) {
    //       Type type = IREE::IREEC::RefType::get(ctx);
    //       Value ref = typeConverter->materializeTargetConversion(rewriter,
    //       loc,
    //                                                              type, arg);
    //       rewriter.replaceUsesOfBlockArgument(arg, ref);
    //     }
    //   }
    // }

    // TypeConverter blockArgConverter = typeConverter->blockArgConverter();
    // (void)rewriter.convertRegionTypes(&ireecFuncOp.getFunctionBody(),
    //                                   blockArgConverter,
    //                                   &signatureConverter);
    rewriter.applySignatureConversion(&ireecFuncOp.getFunctionBody(),
                                      signatureConverter, typeConverter);

    // Materialize a ref value according to the RegisterAllocation
    for (auto &op : ireecFuncOp.getOps()) {
      rewriter.setInsertionPoint(&op);
      for (auto [operandIndex, operand] : llvm::enumerate(op.getOperands())) {
        if (operand.getType().isa<IREE::VM::RefType>()) {
          Type type = IREE::IREEC::RefType::get(ctx);
          Value ireecRef = typeConverter->materializeTargetConversion(
              rewriter, loc, type, operand);
          Value vmRef = typeConverter->materializeSourceConversion(
              rewriter, loc, operand.getType(), ireecRef);
          op.setOperand(operandIndex, vmRef);
        }
      }
    }
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
    operands.push_back(statusOk);
    operands.append(adaptor.getOperands().begin(), adaptor.getOperands().end());

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
    target.addLegalOp<IREE::Util::OptimizationBarrierOp>();
    target.addLegalOp<UnrealizedConversionCastOp>();
    // target.addIllegalOp<IREE::VM::ModuleOp>();
    target.addIllegalOp<IREE::VM::FuncOp>();
    target.addIllegalOp<IREE::VM::ReturnOp>();
    // target.addIllegalOp<IREE::VM::ConstRefZeroOp>();

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
