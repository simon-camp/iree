// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/VM/Conversion/VMToEmitC/EmitCHelpers.h"

#include "iree/compiler/Dialect/VM/Conversion/VMToEmitC/EmitCBuilders.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"

namespace mlir::iree_compiler::IREE::VM {

namespace {
/// Generate an emitc.call_opaque op with one result and split the current block
/// into a continuation and failure block based on the truthiness of the result
/// value, i.e. a truthy value branches to the continuation block when
/// `negateCondition` is false.
emitc::CallOpaqueOp failableCall(
    OpBuilder &builder, Location location, Type type, StringAttr callee,
    ArrayAttr args, ArrayRef<Value> operands,
    const std::function<void(emitc::CallOpaqueOp &)> &failureBlockBuilder,
    bool negateCondition = false) {
  auto callOp = builder.create<emitc::CallOpaqueOp>(
      /*location=*/location,
      /*type=*/type,
      /*callee=*/callee,
      /*args=*/args,
      /*templateArgs=*/ArrayAttr{},
      /*operands=*/operands);

  Type boolType = builder.getIntegerType(1);

  auto conditionI1 = builder.create<emitc::CastOp>(
      /*location=*/location,
      /*type=*/boolType,
      /*operand=*/callOp.getResult(0));

  // Start by splitting the block into two. The part before will contain the
  // condition, and the part after will contain the continuation point.
  Block *condBlock = builder.getInsertionBlock();
  Block::iterator opPosition = builder.getInsertionPoint();
  Block *continuationBlock = condBlock->splitBlock(opPosition);

  // Create a new block for the target of the failure.
  Block *failureBlock;
  {
    OpBuilder::InsertionGuard guard(builder);
    Region *parentRegion = condBlock->getParent();
    failureBlock = builder.createBlock(parentRegion, parentRegion->end());

    failureBlockBuilder(callOp);
  }

  builder.setInsertionPointToEnd(condBlock);
  builder.create<mlir::cf::CondBranchOp>(
      location, conditionI1.getResult(),
      negateCondition ? failureBlock : continuationBlock,
      negateCondition ? continuationBlock : failureBlock);

  builder.setInsertionPointToStart(continuationBlock);

  return callOp;
}

/// Generate a mlir.call op with one result and split the current block into a
/// continuation and failure block based on the truthiness of the result
/// value, i.e. a truthy value branches to the continuation block when
/// `negateCondition` is false.
mlir::func::CallOp failableCall(
    OpBuilder &builder, Location location, mlir::func::FuncOp &callee,
    ArrayRef<Value> operands,
    const std::function<void(mlir::func::CallOp &)> &failureBlockBuilder,
    bool negateCondition = false) {
  auto callOp = builder.create<mlir::func::CallOp>(
      /*location=*/location,
      /*callee=*/callee,
      /*operands=*/operands);

  Type boolType = builder.getIntegerType(1);

  auto conditionI1 = builder.create<emitc::CastOp>(
      /*location=*/location,
      /*type=*/boolType,
      /*operand=*/callOp.getResult(0));

  // Start by splitting the block into two. The part before will contain the
  // condition, and the part after will contain the continuation point.
  Block *condBlock = builder.getInsertionBlock();
  Block::iterator opPosition = builder.getInsertionPoint();
  Block *continuationBlock = condBlock->splitBlock(opPosition);

  // Create a new block for the target of the failure.
  Block *failureBlock;
  {
    OpBuilder::InsertionGuard guard(builder);
    Region *parentRegion = condBlock->getParent();
    failureBlock = builder.createBlock(parentRegion, parentRegion->end());

    failureBlockBuilder(callOp);
  }

  builder.setInsertionPointToEnd(condBlock);
  builder.create<mlir::cf::CondBranchOp>(
      location, conditionI1.getResult(),
      negateCondition ? failureBlock : continuationBlock,
      negateCondition ? continuationBlock : failureBlock);

  builder.setInsertionPointToStart(continuationBlock);

  return callOp;
}
} // namespace

/// Create a call to memset to clear a struct
LogicalResult clearStruct(OpBuilder builder, Value structValue) {
  auto loc = structValue.getLoc();

  if (auto ptrType =
          llvm::dyn_cast<emitc::PointerType>(structValue.getType())) {
    Value sizeValue = emitc_builders::sizeOf(
        builder, loc, TypeAttr::get(ptrType.getPointee()));
    emitc_builders::memset(builder, loc, structValue, 0, sizeValue);

    return success();
  }

  return emitError(loc, "expected pointer type");
}

emitc::CallOpaqueOp
failContainerNull(OpBuilder &builder, Location location, Type type,
                  StringAttr callee, ArrayAttr args, ArrayRef<Value> operands,
                  IREE::VM::ModuleAnalysis &moduleAnalysis) {
  auto blockBuilder = [&builder, &location,
                       &moduleAnalysis](emitc::CallOpaqueOp &callOp) {
    auto ctx = builder.getContext();

    Block *block = builder.getBlock();
    mlir::func::FuncOp funcOp = cast<mlir::func::FuncOp>(block->getParentOp());

    releaseRefs(builder, location, funcOp, moduleAnalysis);

    auto statusOp = builder.create<emitc::CallOpaqueOp>(
        /*location=*/location,
        /*type=*/emitc::OpaqueType::get(ctx, "iree_status_t"),
        /*callee=*/StringAttr::get(ctx, "iree_make_status"),
        /*args=*/
        ArrayAttr::get(
            ctx, {emitc::OpaqueAttr::get(ctx, "IREE_STATUS_INVALID_ARGUMENT")}),
        /*templateArgs=*/ArrayAttr{},
        /*operands=*/ArrayRef<Value>{});

    builder.create<mlir::func::ReturnOp>(location, statusOp.getResult(0));
  };

  return failableCall(builder, location, type, callee, args, operands,
                      blockBuilder);
}

emitc::CallOpaqueOp returnIfError(OpBuilder &builder, Location location,
                                  StringAttr callee, ArrayAttr args,
                                  ArrayRef<Value> operands,
                                  IREE::VM::ModuleAnalysis &moduleAnalysis) {
  auto blockBuilder = [&builder, &location,
                       &moduleAnalysis](emitc::CallOpaqueOp &callOp) {
    Block *block = builder.getBlock();
    mlir::func::FuncOp funcOp = cast<mlir::func::FuncOp>(block->getParentOp());

    releaseRefs(builder, location, funcOp, moduleAnalysis);

    builder.create<mlir::func::ReturnOp>(location, callOp.getResult(0));
  };

  auto ctx = builder.getContext();
  Type type = emitc::OpaqueType::get(ctx, "iree_status_t");
  return failableCall(builder, location, type, callee, args, operands,
                      blockBuilder, /*negateCondition=*/true);
}

mlir::func::CallOp returnIfError(OpBuilder &builder, Location location,
                                 mlir::func::FuncOp &callee,
                                 ArrayRef<Value> operands,
                                 IREE::VM::ModuleAnalysis &moduleAnalysis) {
  auto blockBuilder = [&builder, &location,
                       &moduleAnalysis](mlir::func::CallOp &callOp) {
    Block *block = builder.getBlock();
    mlir::func::FuncOp funcOp = cast<mlir::func::FuncOp>(block->getParentOp());

    releaseRefs(builder, location, funcOp, moduleAnalysis);

    builder.create<mlir::func::ReturnOp>(location, callOp.getResult(0));
  };

  return failableCall(builder, location, callee, operands, blockBuilder,
                      /*negateCondition=*/true);
}

/// Releases refs which are local to the function as well as ref arguments.
void releaseRefs(OpBuilder &builder, Location location,
                 mlir::func::FuncOp funcOp,
                 IREE::VM::ModuleAnalysis &moduleAnalysis) {
  auto ctx = builder.getContext();

  auto &funcAnalysis = moduleAnalysis.lookupFunction(funcOp);

  if (funcAnalysis.hasLocalRefs()) {

    for (auto pair : funcAnalysis.localRefs()) {
      Operation *op = pair.second;

      assert(isa<emitc::ApplyOp>(op));

      Value localRef = cast<emitc::ApplyOp>(op).getResult();

      emitc_builders::ireeVmRefRelease(builder, location, localRef);
    }
  }

  // We only release the original arguments not the results which were appended
  // as further operands.
  size_t refArgumentsReleased = 0;
  for (auto arg : funcOp.getArguments()) {
    if (arg.getType() ==
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, "iree_vm_ref_t"))) {
      if (funcAnalysis.getNumRefArguments() <= refArgumentsReleased++) {
        break;
      }
      emitc_builders::ireeVmRefRelease(builder, location, arg);
    }
  }
}
} // namespace mlir::iree_compiler::IREE::VM
