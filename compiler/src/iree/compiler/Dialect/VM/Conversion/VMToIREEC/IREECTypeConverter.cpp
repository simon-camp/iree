// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/VM/Conversion/VMToIREEC/IREECTypeConverter.h"

#include "iree/compiler/Dialect/VM/Analysis/RegisterAllocation.h"
#include "iree/compiler/Dialect/VM/IR/VMTypes.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace IREEC {

IREECTypeConverter::IREECTypeConverter() {
  // Return the incoming type in the default case.
  addConversion([](Type type) { return type; });
  addConversion([](IntegerType type) { return type; });

  addConversion([](IREE::VM::RefType type) {
    return IREEC::RefType::get(type.getContext());
  });

  addTargetMaterialization([this](OpBuilder &builder, IREE::IREEC::RefType type,
                                  ValueRange inputs, Location loc) -> Value {
    assert(inputs.size() == 1);
    assert(inputs[0].getType().isa<IREE::VM::RefType>());
    Value ref = inputs[0];
    Optional<Value> result = materializeRef(ref);
    return result.has_value() ? result.value() : Value{};
  });

  addSourceMaterialization([](OpBuilder &builder, IREE::VM::RefType type,
                              ValueRange inputs, Location loc) -> Value {
    assert(inputs.size() == 1);
    assert(inputs[0].getType().isa<IREE::IREEC::RefType>());

    Type objectType = IREE::VM::OpaqueType::get(builder.getContext());
    Type refType = IREE::VM::RefType::get(objectType);

    auto op = builder.create<UnrealizedConversionCastOp>(loc, refType,
                                                         ValueRange{inputs[0]});

    return op.getResult(0);
  });
}

Optional<Value> IREECTypeConverter::materializeRef(Value ref) {
  assert(ref.getType().isa<IREE::VM::RefType>());

  IREE::IREEC::FuncOp funcOp;
  if (auto definingOp = ref.getDefiningOp()) {
    funcOp = definingOp->getParentOfType<IREE::IREEC::FuncOp>();
  } else {
    Operation *op = ref.cast<BlockArgument>().getOwner()->getParentOp();
    funcOp = cast<IREE::IREEC::FuncOp>(op);
  }

  auto vmAnalysis = lookupAnalysis(funcOp);
  if (failed(vmAnalysis)) {
    funcOp.emitError() << "parent func op not found in cache.";
    return std::nullopt;
  }

  int32_t ordinal = vmAnalysis.value().get().getRefRegisterOrdinal(ref);

  // Search block arguments
  int refArgCounter = 0;
  for (BlockArgument arg : funcOp.getArguments()) {
    assert(!arg.getType().isa<IREE::VM::RefType>());

    if (arg.getType().isa<IREE::IREEC::RefType>()) {
      if (ordinal == refArgCounter++) {
        return arg;
      }
    }
  }

  Optional<Value> result = vmAnalysis.value().get().lookupLocalRef(ordinal);
  if (result.has_value()) {
    return result;
  }

  return allocateLocalRef(funcOp, vmAnalysis.value().get(), ordinal);
}

Value IREECTypeConverter::allocateLocalRef(IREE::IREEC::FuncOp funcOp,
                                           VMAnalysis2 &analysis,
                                           int64_t ordinal) {
  OpBuilder builder(funcOp);
  builder.setInsertionPointToStart(funcOp.getEntryBlock());
  Value ref =
      builder
          .create<IREE::IREEC::RefNullOp>(
              funcOp.getLoc(), IREE::IREEC::RefType::get(funcOp.getContext()))
          .getResult();
  analysis.cacheLocalRef(ordinal, ref);
  return ref;
}

FailureOr<std::reference_wrapper<VMAnalysis2>>
IREECTypeConverter::lookupAnalysis(Operation *op) {
  auto ptr = analysisCache.find(op);
  if (ptr == analysisCache.end()) {
    op->emitError() << "parent func op not found in cache.";
    return failure();
  }
  return std::ref(ptr->second);
}

}  // namespace IREEC
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
