// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/VM/Conversion/VMToEmitC/ConvertVMToEmitC.h"

#include <optional>

#include "iree/compiler/Dialect/Util/Conversion/ConversionPatterns.h"
#include "iree/compiler/Dialect/Util/IR/UtilDialect.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "iree/compiler/Dialect/VM/Conversion/VMToEmitC/EmitCBuilders.h"
#include "iree/compiler/Dialect/VM/Conversion/VMToEmitC/VMAnalysis.h"
#include "iree/compiler/Dialect/VM/IR/VMOps.h"
#include "iree/compiler/Dialect/VM/Utils/CallingConvention.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace iree_compiler {

namespace {

enum {
  SHIM_ARGUMENT_STACK = 0,
  SHIM_ARGUMENT_FLAGS,
  SHIM_ARGUMENT_ARGS_STORAGE,
  SHIM_ARGUMENT_RETS_STORAGE,
  SHIM_ARGUMENT_MODULE,
  SHIM_ARGUMENT_MODULE_STATE,
};

/// The EmitC dialect is currently missing operations to cleanly represent some
/// constructs we need for the C target. This includes storage class specifiers
/// on functions, forward declarations of functions, globals and arrays.
/// As a workaround the conversion currently adds bits of information via
/// attributes that later get used by the CModuleTarget.
void attachAttribute(Operation *op, StringRef name, Attribute value) {
  op->setAttr(name, value);
}

/// Create a call to the sizeof operator with `value` as operand.
Value callSizeof(OpBuilder builder, Location loc, Value value) {
  auto ctx = builder.getContext();
  return builder
      .create<emitc::CallOp>(
          /*location=*/loc,
          /*type=*/emitc::OpaqueType::get(ctx, "iree_host_size_t"),
          /*callee=*/StringAttr::get(ctx, "sizeof"),
          /*args=*/ArrayAttr{},
          /*templateArgs=*/ArrayAttr{},
          /*operands=*/ArrayRef<Value>{value})
      .getResult(0);
}

/// Create a call to the sizeof operator with `attr` as operand.
Value callSizeof(OpBuilder builder, Location loc, Attribute attr) {
  auto ctx = builder.getContext();
  return builder
      .create<emitc::CallOp>(
          /*location=*/loc,
          /*type=*/emitc::OpaqueType::get(ctx, "iree_host_size_t"),
          /*callee=*/StringAttr::get(ctx, "sizeof"),
          /*args=*/ArrayAttr::get(ctx, {attr}),
          /*templateArgs=*/ArrayAttr{},
          /*operands=*/ArrayRef<Value>{})
      .getResult(0);
}

/// Create a call to memset to clear a struct
LogicalResult clearStruct(OpBuilder builder, Value structValue) {
  auto ctx = structValue.getContext();
  auto loc = structValue.getLoc();

  Value structPointerValue;
  Value sizeValue;

  if (auto ptrType = structValue.getType().dyn_cast<emitc::PointerType>()) {
    structPointerValue = structValue;
    sizeValue = callSizeof(builder, loc, TypeAttr::get(ptrType.getPointee()));
  } else {
    structPointerValue = emitc_builders::addressOf(builder, loc, structValue);
    sizeValue = callSizeof(builder, loc, structValue);
  }

  builder.create<emitc::CallOp>(
      /*location=*/loc,
      /*type=*/TypeRange{},
      /*callee=*/StringAttr::get(ctx, "memset"),
      /*args=*/
      ArrayAttr::get(ctx,
                     {builder.getIndexAttr(0), builder.getUI32IntegerAttr(0),
                      builder.getIndexAttr(1)}),
      /*templateArgs=*/ArrayAttr{},
      /*operands=*/
      ArrayRef<Value>{structPointerValue, sizeValue});

  return success();
}

LogicalResult convertFuncOp(IREE::VM::FuncOp funcOp,
                            IREE::VM::EmitCTypeConverter &typeConverter,
                            SmallVector<BlockArgument, 4> &blockArgsToRemove) {
  auto ctx = funcOp.getContext();
  auto loc = funcOp.getLoc();

  OpBuilder builder(funcOp);

  auto moduleOp = funcOp.getOperation()->getParentOfType<IREE::VM::ModuleOp>();

  FunctionType funcType = funcOp.getFunctionType();
  std::string name =
      std::string(moduleOp.getName()) + "_" + std::string(funcOp.getName());
  std::string moduleTypeName = (moduleOp.getName() + "_t").str();
  std::string moduleStateTypeName = (moduleOp.getName() + "_state_t").str();

  Type stackType =
      emitc::PointerType::get(emitc::OpaqueType::get(ctx, "iree_vm_stack_t"));
  Type moduleType =
      emitc::PointerType::get(emitc::OpaqueType::get(ctx, moduleTypeName));
  Type moduleStateType =
      emitc::PointerType::get(emitc::OpaqueType::get(ctx, moduleStateTypeName));

  SmallVector<Type, 3> inputTypes = {stackType, moduleType, moduleStateType};
  SmallVector<Type, 1> outputTypes;

  for (auto &inputType : funcType.getInputs()) {
    inputTypes.push_back(inputType);
  }

  for (auto &resultType : funcType.getResults()) {
    // We pass refs as iree_vm_ref_t* regardless of whether it is an in or out
    // parameter
    Type type = typeConverter.convertTypeAsPointer(resultType);

    inputTypes.push_back(type);
    outputTypes.push_back(type);
  }

  auto newFuncType = mlir::FunctionType::get(
      ctx, {inputTypes}, {emitc::OpaqueType::get(ctx, "iree_status_t")});

  auto newFuncOp = builder.create<mlir::func::FuncOp>(loc, name, newFuncType);

  attachAttribute(newFuncOp, "emitc.static", UnitAttr::get(ctx));

  Optional<std::string> callingConvention = makeCallingConventionString(funcOp);

  // Annotate new function with calling convention string which gets used in
  // the CModuleTarget.
  attachAttribute(newFuncOp, "vm.calling_convention",
                  StringAttr::get(ctx, callingConvention.value()));

  // This call shold be equivalent to rewriter.inlineRegionBefore()
  newFuncOp.getFunctionBody().getBlocks().splice(
      newFuncOp.end(), funcOp.getFunctionBody().getBlocks());

  Block &entryBlock = newFuncOp.getBlocks().front();

  if (!entryBlock.hasNoPredecessors()) {
    return funcOp.emitError()
           << "branches to the entry block are not supported for now.";
  }

  entryBlock.insertArgument(static_cast<unsigned>(0), stackType, loc);
  entryBlock.insertArgument(static_cast<unsigned>(1), moduleType, loc);
  entryBlock.insertArgument(static_cast<unsigned>(2), moduleStateType, loc);

  SmallVector<Location> locs(outputTypes.size(), loc);
  entryBlock.addArguments(outputTypes, locs);

  auto vmAnalysis = typeConverter.lookupAnalysis(funcOp);
  if (failed(vmAnalysis)) {
    return funcOp.emitError() << "parent func op not found in cache.";
  }

  typeConverter.analysisCache.insert(std::make_pair(
      newFuncOp.getOperation(), std::move(vmAnalysis.value().get())));

  // vmAnalysis gets invalidated, reset it
  vmAnalysis = typeConverter.lookupAnalysis(newFuncOp);
  if (failed(vmAnalysis)) {
    return funcOp.emitError()
           << "newly created mlir::func::FuncOp not found in cache.";
  }

  // Add constant ops for local refs
  const int numRefArgs = vmAnalysis.value().get().getNumRefArguments();
  const int numLocalRefs = vmAnalysis.value().get().getNumLocalRefs();

  builder.setInsertionPointToStart(&entryBlock);

  for (int i = 0; i < numLocalRefs; i++) {
    auto refOp = builder.create<emitc::VariableOp>(
        /*location=*/loc,
        /*resultType=*/emitc::OpaqueType::get(ctx, "iree_vm_ref_t"),
        /*value=*/emitc::OpaqueAttr::get(ctx, ""));

    Value refPtr = emitc_builders::addressOf(builder, loc, refOp.getResult());
    auto refPtrOp = cast<emitc::ApplyOp>(refPtr.getDefiningOp());

    // Cache local refs so that we can release them before a return operation.
    // Here we rely on the fact that the register allocation maps arguments in
    // the first slots.
    vmAnalysis.value().get().cacheLocalRef(i + numRefArgs, refPtrOp);

    if (failed(clearStruct(builder, refPtr))) {
      return failure();
    }
  }

  for (Block &block : llvm::drop_begin(newFuncOp.getBlocks(), 1)) {
    for (BlockArgument blockArg : block.getArguments()) {
      if (!blockArg.getType().isa<IREE::VM::RefType>()) {
        continue;
      }
      blockArgsToRemove.push_back(blockArg);
    }
  }

  if (failed(
          funcOp.replaceAllSymbolUses(builder.getStringAttr(name), moduleOp)))
    return funcOp.emitError() << "unable to update symbol name in module";

  return success();
}

/// Remove block arguments
LogicalResult removeBlockArguments(
    IREE::VM::ModuleOp moduleOp,
    SmallVector<BlockArgument, 4> &blockArgsToRemove) {
  for (auto &blockArg : blockArgsToRemove) {
    assert(blockArg.getType().isa<IREE::VM::RefType>());
    assert(blockArg.use_empty());
    Block *block = blockArg.getOwner();

    block->eraseArgument(blockArg.getArgNumber());
  }

  return success();
}

FailureOr<int64_t> calculateNumSpans(IREE::VM::CallVariadicOp &callOp) {
  auto isVariadic = [](APInt segmentSize) {
    return segmentSize.getSExtValue() != -1;
  };

  DenseIntElementsAttr segmentSizes = callOp.getSegmentSizes();
  size_t numSegments = segmentSizes.size();
  size_t numVariadicSegments = llvm::count_if(segmentSizes, isVariadic);

  if (numVariadicSegments != 1) {
    callOp.emitError() << "only exactly one variadic segment supported";
    return failure();
  }

  auto lastSegmentSize = *(segmentSizes.begin() + (numSegments - 1));

  if (!isVariadic(lastSegmentSize)) {
    callOp.emitError() << "expected the last segment to be variadic";
    return failure();
  }

  return lastSegmentSize.getSExtValue();
}

Optional<std::string> buildFunctionName(IREE::VM::ModuleOp &moduleOp,
                                        IREE::VM::ImportOp &importOp) {
  auto callingConvention = makeImportCallingConventionString(importOp);
  if (!callingConvention.has_value()) {
    return std::nullopt;
  }
  return moduleOp.getName().str() + "_call_" + callingConvention.value() +
         "_import_shim";
}

Optional<std::string> buildVariadicFunctionName(
    IREE::VM::ModuleOp &moduleOp, IREE::VM::ImportOp &importOp,
    DenseIntElementsAttr segmentSizes) {
  auto callingConvention = makeImportCallingConventionString(importOp);
  if (!callingConvention.has_value()) {
    return std::nullopt;
  }
  std::string result(moduleOp.getName());
  result += "_call_";
  result += callingConvention.value();
  for (int i = 0; i < importOp.getNumArguments(); i++) {
    if (importOp.isFuncArgumentVariadic(i)) {
      APInt size = *(segmentSizes.begin() + i);
      result += "_";
      result += std::to_string(size.getSExtValue());
    }
  }
  result += "_import_shim";
  return result;
}

Optional<emitc::ApplyOp> createVmTypeDefPtr(ConversionPatternRewriter &rewriter,
                                            Operation *srcOp,
                                            Type elementType) {
  auto ctx = srcOp->getContext();
  auto loc = srcOp->getLoc();

  // Map from type to enum values of type iree_vm_value_type_t and
  // iree_vm_ref_type_t
  mlir::DenseMap<Type, std::pair<std::string, std::string>> valueTypeMap = {
      {IntegerType::get(ctx, 8),
       {"IREE_VM_VALUE_TYPE_I8", "IREE_VM_REF_TYPE_NULL"}},
      {IntegerType::get(ctx, 16),
       {"IREE_VM_VALUE_TYPE_I16", "IREE_VM_REF_TYPE_NULL"}},
      {IntegerType::get(ctx, 32),
       {"IREE_VM_VALUE_TYPE_I32", "IREE_VM_REF_TYPE_NULL"}},
      {IntegerType::get(ctx, 64),
       {"IREE_VM_VALUE_TYPE_I64", "IREE_VM_REF_TYPE_NULL"}},
      {Float32Type::get(ctx),
       {"IREE_VM_VALUE_TYPE_F32", "IREE_VM_REF_TYPE_NULL"}},
      {Float64Type::get(ctx),
       {"IREE_VM_VALUE_TYPE_F64", "IREE_VM_REF_TYPE_NULL"}},
      {IREE::VM::OpaqueType::get(ctx),
       {"IREE_VM_VALUE_TYPE_NONE", "IREE_VM_REF_TYPE_NULL"}},
  };

  auto elementTypeOp = rewriter.create<emitc::VariableOp>(
      /*location=*/loc,
      /*resultType=*/emitc::OpaqueType::get(ctx, "iree_vm_type_def_t"),
      /*value=*/emitc::OpaqueAttr::get(ctx, ""));

  if (failed(clearStruct(rewriter, elementTypeOp.getResult()))) {
    return std::nullopt;
  }

  auto ptr = valueTypeMap.find((elementType));
  if (ptr != valueTypeMap.end()) {
    emitc_builders::structMemberAssign(rewriter, loc,
                                       /*memberName=*/"value_type",
                                       /*operand=*/elementTypeOp.getResult(),
                                       /*value=*/ptr->second.first);

    emitc_builders::structMemberAssign(rewriter, loc,
                                       /*memberName=*/"ref_type",
                                       /*operand=*/elementTypeOp.getResult(),
                                       /*value=*/ptr->second.second);
  } else {
    if (!elementType.isa<IREE::VM::RefType>()) {
      return std::nullopt;
    }
    Type objType = elementType.cast<IREE::VM::RefType>().getObjectType();

    std::string typeName;

    if (objType.isa<IREE::VM::ListType>()) {
      typeName = "!vm.list";
    } else {
      llvm::raw_string_ostream sstream(typeName);
      objType.print(sstream);
      sstream.flush();
    }

    // Remove leading '!' and wrap in quotes
    typeName = std::string("\"") + typeName.substr(1) + std::string("\"");

    auto typeNameCStringView = rewriter.create<emitc::CallOp>(
        /*location=*/loc,
        /*type=*/emitc::OpaqueType::get(ctx, "iree_string_view_t"),
        /*callee=*/StringAttr::get(ctx, "iree_make_cstring_view"),
        /*args=*/ArrayAttr::get(ctx, {emitc::OpaqueAttr::get(ctx, typeName)}),
        /*templateArgs=*/ArrayAttr{},
        /*operands=*/ArrayRef<Value>{});

    auto typeDescriptor = rewriter.create<emitc::CallOp>(
        /*location=*/loc,
        /*type=*/
        emitc::PointerType::get(
            emitc::OpaqueType::get(ctx, "const iree_vm_ref_type_descriptor_t")),
        /*callee=*/StringAttr::get(ctx, "iree_vm_ref_lookup_registered_type"),
        /*args=*/ArrayAttr{},
        /*templateArgs=*/ArrayAttr{},
        /*operands=*/ArrayRef<Value>{typeNameCStringView.getResult(0)});

    // TODDO(simon-camp) typeDescriptor might be NULL
    auto typeDescriptorType = emitc_builders::structPtrMember(
        rewriter, loc,
        /*type=*/emitc::OpaqueType::get(ctx, "iree_vm_ref_type_t"),
        /*memberName=*/"type",
        /*operand=*/typeDescriptor.getResult(0));

    emitc_builders::structMemberAssign(rewriter, loc,
                                       /*memberName=*/"ref_type",
                                       /*operand=*/elementTypeOp.getResult(),
                                       /*value=*/typeDescriptorType);
  }

  Value elementTypePtr =
      emitc_builders::addressOf(rewriter, loc, elementTypeOp.getResult());

  return cast<emitc::ApplyOp>(elementTypePtr.getDefiningOp());
}

/// Move multiple refs from one set of variables to another set. As these two
/// sets may alias we move the source variables into temporaries first.
/// The generated code works as follows:
/// `isMove` == true:
///    move(src_i, tmp_i); for all i
///    move(tmp_i, dest_i); for all i
/// `isMove` == false:
///    retain(src_i, tmp_i); for all i
///    assign(tmp_i, dest_i); for all i
LogicalResult retainOrMoveRefs(OpBuilder &builder, Location location,
                               BlockAndValueMapping mapping, bool isMove) {
  auto ctx = builder.getContext();

  BlockAndValueMapping tmpMapping;
  for (auto &[srcRef, destRef] : mapping.getValueMap()) {
    assert(srcRef.getType() == emitc::PointerType::get(emitc::OpaqueType::get(
                                   ctx, "iree_vm_ref_t")));

    auto tmpRef = builder.create<emitc::VariableOp>(
        /*location=*/location,
        /*resultType=*/emitc::OpaqueType::get(ctx, "iree_vm_ref_t"),
        /*value=*/emitc::OpaqueAttr::get(ctx, ""));

    Value tmpPtr =
        emitc_builders::addressOf(builder, location, tmpRef.getResult());

    if (failed(clearStruct(builder, tmpPtr))) {
      return failure();
    }

    StringRef callee = isMove ? "iree_vm_ref_move" : "iree_vm_ref_retain";
    builder.create<emitc::CallOp>(
        /*location=*/location,
        /*type=*/TypeRange{},
        /*callee=*/StringAttr::get(ctx, callee),
        /*args=*/ArrayAttr{},
        /*templateArgs=*/ArrayAttr{},
        /*operands=*/ArrayRef<Value>{srcRef, tmpPtr});

    tmpMapping.map(srcRef, tmpPtr);
  }

  for (const auto &[srcRef, destRef] : mapping.getValueMap()) {
    Value tmpRef = tmpMapping.lookup(srcRef);

    StringRef callee = isMove ? "iree_vm_ref_move" : "iree_vm_ref_assign";

    builder.create<emitc::CallOp>(
        /*location=*/location,
        /*type=*/TypeRange{},
        /*callee=*/StringAttr::get(ctx, callee),
        /*args=*/ArrayAttr{},
        /*templateArgs=*/ArrayAttr{},
        /*operands=*/ArrayRef<Value>{tmpRef, destRef});
  }

  return success();
}

/// Releases refs which are local to the function as well as ref arguments.
void releaseRefs(OpBuilder &builder, Location location,
                 mlir::func::FuncOp funcOp,
                 IREE::VM::EmitCTypeConverter &typeConverter) {
  auto ctx = builder.getContext();

  auto vmAnalysis = typeConverter.lookupAnalysis(funcOp);
  assert(succeeded(vmAnalysis));

  auto &localRefs = vmAnalysis.value().get().localRefs();
  for (auto pair : localRefs) {
    Operation *op = pair.second;

    assert(isa<emitc::ApplyOp>(op));

    Value localRef = cast<emitc::ApplyOp>(op).getResult();

    emitc_builders::ireeVmRefRelease(builder, location, localRef);
  }

  // We only release the original arguments not the results which were appended
  // as further operands.
  size_t refArgumentsReleased = 0;
  for (auto arg : funcOp.getArguments()) {
    if (arg.getType() ==
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, "iree_vm_ref_t"))) {
      if (vmAnalysis.value().get().getNumRefArguments() <=
          refArgumentsReleased++) {
        break;
      }
      emitc_builders::ireeVmRefRelease(builder, location, arg);
    }
  }
}

/// Generate an emitc.call op with one result and split the current block into a
/// continuation and failure block based on the truthiness of the result
/// value, i.e. a truthy value branches to the continuation block when
/// `negateCondition` is false.
emitc::CallOp failableCall(
    OpBuilder &builder, Location location, Type type, StringAttr callee,
    ArrayAttr args, ArrayRef<Value> operands,
    const std::function<void(emitc::CallOp &)> &failureBlockBuilder,
    bool negateCondition = false) {
  auto callOp = builder.create<emitc::CallOp>(
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
  builder.create<IREE::VM::CondBranchOp>(
      location, conditionI1.getResult(),
      negateCondition ? failureBlock : continuationBlock,
      negateCondition ? continuationBlock : failureBlock);

  builder.setInsertionPointToStart(continuationBlock);

  return callOp;
}

emitc::CallOp returnIfError(OpBuilder &builder, Location location,
                            StringAttr callee, ArrayAttr args,
                            ArrayRef<Value> operands,
                            IREE::VM::EmitCTypeConverter &typeConverter) {
  auto blockBuilder = [&builder, &location,
                       &typeConverter](emitc::CallOp &callOp) {
    Block *block = builder.getBlock();
    mlir::func::FuncOp funcOp = cast<mlir::func::FuncOp>(block->getParentOp());

    releaseRefs(builder, location, funcOp, typeConverter);

    builder.create<mlir::func::ReturnOp>(location, callOp.getResult(0));
  };

  auto ctx = builder.getContext();
  Type type = emitc::OpaqueType::get(ctx, "iree_status_t");
  return failableCall(builder, location, type, callee, args, operands,
                      blockBuilder, /*negateCondition=*/true);
}

emitc::CallOp failListNull(OpBuilder &builder, Location location, Type type,
                           StringAttr callee, ArrayAttr args,
                           ArrayRef<Value> operands,
                           IREE::VM::EmitCTypeConverter &typeConverter) {
  auto blockBuilder = [&builder, &location,
                       &typeConverter](emitc::CallOp &callOp) {
    auto ctx = builder.getContext();

    Block *block = builder.getBlock();
    mlir::func::FuncOp funcOp = cast<mlir::func::FuncOp>(block->getParentOp());

    releaseRefs(builder, location, funcOp, typeConverter);

    auto statusOp = builder.create<emitc::CallOp>(
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
  builder.create<IREE::VM::CondBranchOp>(
      location, conditionI1.getResult(),
      negateCondition ? failureBlock : continuationBlock,
      negateCondition ? continuationBlock : failureBlock);

  builder.setInsertionPointToStart(continuationBlock);

  return callOp;
}

mlir::func::CallOp returnIfError(OpBuilder &builder, Location location,
                                 mlir::func::FuncOp &callee,
                                 ArrayRef<Value> operands,
                                 IREE::VM::EmitCTypeConverter &typeConverter) {
  auto blockBuilder = [&builder, &location,
                       &typeConverter](mlir::func::CallOp &callOp) {
    Block *block = builder.getBlock();
    mlir::func::FuncOp funcOp = cast<mlir::func::FuncOp>(block->getParentOp());

    releaseRefs(builder, location, funcOp, typeConverter);

    builder.create<mlir::func::ReturnOp>(location, callOp.getResult(0));
  };

  return failableCall(builder, location, callee, operands, blockBuilder,
                      /*negateCondition=*/true);
}

LogicalResult createAPIFunctions(IREE::VM::ModuleOp moduleOp,
                                 IREE::VM::EmitCTypeConverter &typeConverter) {
  auto ctx = moduleOp.getContext();
  auto loc = moduleOp.getLoc();

  OpBuilder builder(moduleOp);
  builder.setInsertionPoint(moduleOp.getBlock().getTerminator());

  std::string moduleName{moduleOp.getName()};

  // destroy
  {
    OpBuilder::InsertionGuard guard(builder);

    auto funcType = mlir::FunctionType::get(
        ctx, {emitc::PointerType::get(emitc::OpaqueType::get(ctx, "void"))},
        {});

    auto funcOp = builder.create<mlir::func::FuncOp>(
        loc, moduleName + "_destroy", funcType);

    typeConverter.analysisCache.insert(
        std::make_pair(funcOp.getOperation(), VMAnalysis()));

    attachAttribute(funcOp, "emitc.static", UnitAttr::get(ctx));

    Block *entryBlock = funcOp.addEntryBlock();

    builder.setInsertionPointToStart(entryBlock);

    std::string moduleTypeName = moduleName + "_t";

    auto castedModuleOp = builder.create<emitc::CastOp>(
        /*location=*/loc,
        /*type=*/
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, moduleTypeName)),
        /*operand=*/funcOp.getArgument(0));

    auto allocatorOp = emitc_builders::structPtrMember(
        builder, loc,
        /*type=*/emitc::OpaqueType::get(ctx, "iree_allocator_t"),
        /*memberName=*/"allocator",
        /*operand=*/castedModuleOp.getResult());

    builder.create<emitc::CallOp>(
        /*location=*/loc,
        /*type=*/TypeRange{},
        /*callee=*/StringAttr::get(ctx, "iree_allocator_free"),
        /*args=*/ArrayAttr{},
        /*templateArgs=*/ArrayAttr{},
        /*operands=*/ArrayRef<Value>{allocatorOp, castedModuleOp.getResult()});

    builder.create<mlir::func::ReturnOp>(loc);
  }

  // alloc_state
  {
    OpBuilder::InsertionGuard guard(builder);

    auto funcType = mlir::FunctionType::get(
        ctx,
        {emitc::PointerType::get(emitc::OpaqueType::get(ctx, "void")),
         emitc::OpaqueType::get(ctx, "iree_allocator_t"),
         emitc::PointerType::get(emitc::PointerType::get(
             emitc::OpaqueType::get(ctx, "iree_vm_module_state_t")))},
        {emitc::OpaqueType::get(ctx, "iree_status_t")});

    auto funcOp = builder.create<mlir::func::FuncOp>(
        loc, moduleName + "_alloc_state", funcType);

    typeConverter.analysisCache.insert(
        std::make_pair(funcOp.getOperation(), VMAnalysis()));

    attachAttribute(funcOp, "emitc.static", UnitAttr::get(ctx));

    Block *entryBlock = funcOp.addEntryBlock();

    builder.setInsertionPointToStart(entryBlock);

    std::string moduleStateTypeName = moduleName + "_state_t";

    auto stateOp = builder.create<emitc::VariableOp>(
        /*location=*/loc,
        /*resultType=*/
        emitc::PointerType::get(
            emitc::OpaqueType::get(ctx, moduleStateTypeName)),
        /*value=*/emitc::OpaqueAttr::get(ctx, "NULL"));

    Value stateSize = callSizeof(
        builder, loc, emitc::OpaqueAttr::get(ctx, moduleStateTypeName));

    Value statePtr =
        emitc_builders::addressOf(builder, loc, stateOp.getResult());

    auto voidPtr = builder.create<emitc::CastOp>(
        /*location=*/loc,
        /*type=*/
        emitc::PointerType::get(
            emitc::PointerType::get(emitc::OpaqueType::get(ctx, "void"))),
        /*operand=*/statePtr);

    returnIfError(builder, loc, StringAttr::get(ctx, "iree_allocator_malloc"),
                  {}, {funcOp.getArgument(1), stateSize, voidPtr.getResult()},
                  /*typeConverter=*/typeConverter);

    builder.create<emitc::CallOp>(
        /*location=*/loc,
        /*type=*/TypeRange{},
        /*callee=*/StringAttr::get(ctx, "memset"),
        /*args=*/
        ArrayAttr::get(ctx,
                       {builder.getIndexAttr(0), builder.getUI32IntegerAttr(0),
                        builder.getIndexAttr(1)}),
        /*templateArgs=*/ArrayAttr{},
        /*operands=*/
        ArrayRef<Value>{stateOp.getResult(), stateSize});

    emitc_builders::structPtrMemberAssign(builder, loc,
                                          /*memberName=*/"allocator",
                                          /*operand=*/stateOp.getResult(),
                                          /*value=*/funcOp.getArgument(1));

    // Initialize buffers
    for (auto rodataOp : moduleOp.getOps<IREE::VM::RodataOp>()) {
      auto ordinal = rodataOp.getOrdinal()->getZExtValue();

      std::string bufferName = moduleName + "_" + rodataOp.getName().str();

      auto rodataPointer = builder.create<emitc::VariableOp>(
          /*location=*/loc,
          /*resultType=*/
          emitc::PointerType::get(emitc::OpaqueType::get(ctx, "const uint8_t")),
          /*value=*/emitc::OpaqueAttr::get(ctx, bufferName));

      auto bufferVoid = builder.create<emitc::CastOp>(
          /*location=*/loc,
          /*type=*/emitc::PointerType::get(emitc::OpaqueType::get(ctx, "void")),
          /*operand=*/rodataPointer.getResult());

      Value bufferSize =
          callSizeof(builder, loc, emitc::OpaqueAttr::get(ctx, bufferName));

      auto byteSpan = builder.create<emitc::CallOp>(
          /*location=*/loc,
          /*type=*/emitc::OpaqueType::get(ctx, "iree_byte_span_t"),
          /*callee=*/StringAttr::get(ctx, "iree_make_byte_span"),
          /*args=*/ArrayAttr{},
          /*templateArgs=*/ArrayAttr{},
          /*operands=*/
          ArrayRef<Value>{bufferVoid.getResult(), bufferSize});

      auto allocator = builder.create<emitc::CallOp>(
          /*location=*/loc,
          /*type=*/emitc::OpaqueType::get(ctx, "iree_allocator_t"),
          /*callee=*/StringAttr::get(ctx, "iree_allocator_null"),
          /*args=*/ArrayAttr{},
          /*templateArgs=*/ArrayAttr{},
          /*operands=*/
          ArrayRef<Value>{});

      auto buffers = emitc_builders::structPtrMember(
          builder, loc,
          /*type=*/
          emitc::PointerType::get(
              emitc::OpaqueType::get(ctx, "iree_vm_buffer_t")),
          /*memberName=*/"rodata_buffers",
          /*operand=*/stateOp.getResult());

      auto buffer = emitc_builders::arrayElementAddress(
          builder, loc,
          /*type=*/
          emitc::PointerType::get(
              emitc::OpaqueType::get(ctx, "iree_vm_buffer_t")),
          /*index=*/builder.getUI32IntegerAttr(ordinal),
          /*operand=*/buffers);

      builder.create<emitc::CallOp>(
          /*location=*/loc,
          /*type=*/TypeRange{},
          /*callee=*/StringAttr::get(ctx, "iree_vm_buffer_initialize"),
          /*args=*/
          ArrayAttr::get(ctx, {emitc::OpaqueAttr::get(
                                   ctx, "IREE_VM_BUFFER_ACCESS_ORIGIN_MODULE"),
                               builder.getIndexAttr(0), builder.getIndexAttr(1),
                               builder.getIndexAttr(2)}),
          /*templateArgs=*/ArrayAttr{},
          /*operands=*/
          ArrayRef<Value>{byteSpan.getResult(0), allocator.getResult(0),
                          buffer});
    }

    // Zero out refs from state struct.
    auto ordinalCounts = moduleOp.getOrdinalCountsAttr();
    if (!ordinalCounts) {
      return moduleOp.emitError()
             << "ordinal_counts attribute not found. The OrdinalAllocationPass "
                "must be run before.";
    }
    const int numGlobalRefs = ordinalCounts.getGlobalRefs();

    if (numGlobalRefs > 0) {
      auto refs = emitc_builders::structPtrMember(
          builder, loc,
          /*type=*/
          emitc::PointerType::get(emitc::OpaqueType::get(ctx, "iree_vm_ref_t")),
          /*memberName=*/"refs",
          /*operand=*/stateOp.getResult());

      for (int i = 0; i < numGlobalRefs; i++) {
        auto refPtrOp = emitc_builders::arrayElementAddress(
            builder, loc,
            /*type=*/
            emitc::PointerType::get(
                emitc::OpaqueType::get(ctx, "iree_vm_ref_t")),
            /*index=*/builder.getUI32IntegerAttr(i),
            /*operand=*/refs);

        if (failed(clearStruct(builder, refPtrOp))) {
          return failure();
        }
      }
    }

    auto baseStateOp = builder.create<emitc::CastOp>(
        /*location=*/loc,
        /*type=*/
        emitc::PointerType::get(
            emitc::OpaqueType::get(ctx, "iree_vm_module_state_t")),
        /*operand=*/stateOp.getResult());

    builder.create<emitc::CallOp>(
        /*location=*/loc,
        /*type=*/TypeRange{},
        /*callee=*/StringAttr::get(ctx, "EMITC_DEREF_ASSIGN_VALUE"),
        /*args=*/ArrayAttr{},
        /*templateArgs=*/ArrayAttr{},
        /*operands=*/
        ArrayRef<Value>{funcOp.getArgument(2), baseStateOp.getResult()});

    auto status = emitc_builders::ireeOkStatus(builder, loc);

    builder.create<mlir::func::ReturnOp>(loc, status);
  }

  // free_state
  {
    OpBuilder::InsertionGuard guard(builder);

    auto funcType = mlir::FunctionType::get(
        ctx,
        {emitc::PointerType::get(emitc::OpaqueType::get(ctx, "void")),
         emitc::PointerType::get(
             emitc::OpaqueType::get(ctx, "iree_vm_module_state_t"))},
        {});

    auto funcOp = builder.create<mlir::func::FuncOp>(
        loc, moduleName + "_free_state", funcType);

    typeConverter.analysisCache.insert(
        std::make_pair(funcOp.getOperation(), VMAnalysis()));

    attachAttribute(funcOp, "emitc.static", UnitAttr::get(ctx));

    Block *entryBlock = funcOp.addEntryBlock();

    builder.setInsertionPointToStart(entryBlock);

    std::string moduleStateTypeName = moduleName + "_state_t";

    auto stateOp = builder.create<emitc::CastOp>(
        /*location=*/loc,
        /*type=*/
        emitc::PointerType::get(
            emitc::OpaqueType::get(ctx, moduleStateTypeName)),
        /*operand=*/funcOp.getArgument(1));

    // Release refs from state struct.
    auto ordinalCounts = moduleOp.getOrdinalCountsAttr();
    if (!ordinalCounts) {
      return moduleOp.emitError()
             << "ordinal_counts attribute not found. The OrdinalAllocationPass "
                "must be run before.";
    }
    const int numGlobalRefs = ordinalCounts.getGlobalRefs();

    if (numGlobalRefs > 0) {
      auto refs = emitc_builders::structPtrMember(
          builder, loc,
          /*type=*/
          emitc::PointerType::get(emitc::OpaqueType::get(ctx, "iree_vm_ref_t")),
          /*memberName=*/"refs",
          /*operand=*/stateOp.getResult());

      for (int i = 0; i < numGlobalRefs; i++) {
        auto refPtrOp = emitc_builders::arrayElementAddress(
            builder, loc,
            /*type=*/
            emitc::PointerType::get(
                emitc::OpaqueType::get(ctx, "iree_vm_ref_t")),
            /*index=*/builder.getUI32IntegerAttr(i),
            /*operand=*/refs);

        emitc_builders::ireeVmRefRelease(builder, loc, refPtrOp);
      }
    }

    auto allocatorOp = emitc_builders::structPtrMember(
        builder, loc,
        /*type=*/emitc::OpaqueType::get(ctx, "iree_allocator_t"),
        /*memberName=*/"allocator",
        /*operand=*/stateOp.getResult());

    builder.create<emitc::CallOp>(
        /*location=*/loc,
        /*type=*/TypeRange{},
        /*callee=*/StringAttr::get(ctx, "iree_allocator_free"),
        /*args=*/ArrayAttr{},
        /*templateArgs=*/ArrayAttr{},
        /*operands=*/
        ArrayRef<Value>{allocatorOp, stateOp.getResult()});

    builder.create<mlir::func::ReturnOp>(loc);
  }

  // resolve_import
  {
    OpBuilder::InsertionGuard guard(builder);

    auto funcType = mlir::FunctionType::get(
        ctx,
        {
            emitc::PointerType::get(emitc::OpaqueType::get(ctx, "void")),
            emitc::PointerType::get(
                emitc::OpaqueType::get(ctx, "iree_vm_module_state_t")),
            emitc::OpaqueType::get(ctx, "iree_host_size_t"),
            emitc::PointerType::get(
                emitc::OpaqueType::get(ctx, "const iree_vm_function_t")),
            emitc::PointerType::get(emitc::OpaqueType::get(
                ctx, "const iree_vm_function_signature_t")),
        },
        {emitc::OpaqueType::get(ctx, "iree_status_t")});

    auto funcOp = builder.create<mlir::func::FuncOp>(
        loc, moduleName + "_resolve_import", funcType);

    typeConverter.analysisCache.insert(
        std::make_pair(funcOp.getOperation(), VMAnalysis()));

    attachAttribute(funcOp, "emitc.static", UnitAttr::get(ctx));

    Block *entryBlock = funcOp.addEntryBlock();

    builder.setInsertionPointToStart(entryBlock);

    std::string moduleStateTypeName = moduleName + "_state_t";

    auto stateOp = builder.create<emitc::CastOp>(
        /*location=*/loc,
        /*type=*/
        emitc::PointerType::get(
            emitc::OpaqueType::get(ctx, moduleStateTypeName)),
        /*operand=*/funcOp.getArgument(1));

    auto imports = emitc_builders::structPtrMember(
        builder, loc,
        /*type=*/
        emitc::PointerType::get(
            emitc::OpaqueType::get(ctx, "iree_vm_function_t")),
        /*memberName=*/"imports",
        /*operand=*/stateOp.getResult());

    auto import = builder.create<emitc::CallOp>(
        /*location=*/loc,
        /*type=*/
        emitc::PointerType::get(
            emitc::OpaqueType::get(ctx, "iree_vm_function_t")),
        /*callee=*/StringAttr::get(ctx, "EMITC_ARRAY_ELEMENT_ADDRESS"),
        /*args=*/ArrayAttr{},
        /*templateArgs=*/ArrayAttr{},
        /*operands=*/
        ArrayRef<Value>{imports, funcOp.getArgument(2)});

    builder.create<emitc::CallOp>(
        /*location=*/loc,
        /*type=*/TypeRange{},
        /*callee=*/StringAttr::get(ctx, "EMITC_DEREF_ASSIGN_PTR"),
        /*args=*/ArrayAttr{},
        /*templateArgs=*/ArrayAttr{},
        /*operands=*/
        ArrayRef<Value>{import.getResult(0), funcOp.getArgument(3)});

    auto status = emitc_builders::ireeOkStatus(builder, loc);

    builder.create<mlir::func::ReturnOp>(loc, status);
  }

  // create
  {
    OpBuilder::InsertionGuard guard(builder);

    auto funcType = mlir::FunctionType::get(
        ctx,
        {
            emitc::PointerType::get(
                emitc::OpaqueType::get(ctx, "iree_vm_instance_t")),
            emitc::OpaqueType::get(ctx, "iree_allocator_t"),
            emitc::PointerType::get(emitc::PointerType::get(
                emitc::OpaqueType::get(ctx, "iree_vm_module_t"))),
        },
        {
            emitc::OpaqueType::get(ctx, "iree_status_t"),
        });

    auto funcOp = builder.create<mlir::func::FuncOp>(
        loc, moduleName + "_create", funcType);

    typeConverter.analysisCache.insert(
        std::make_pair(funcOp.getOperation(), VMAnalysis()));

    // This function needs an iree_vm_native_module_descriptor_t that is emitted
    // by the CModuleTarget at the moment. So we add a marker to this function
    // and delay the printing of it.
    attachAttribute(funcOp, "vm.emit_at_end", UnitAttr::get(ctx));

    // This functions is the only one users need and it is therefore declared
    // separatly from all other functions.
    attachAttribute(funcOp, "vm.module.constructor", UnitAttr::get(ctx));

    Block *entryBlock = funcOp.addEntryBlock();

    builder.setInsertionPointToStart(entryBlock);

    std::string moduleTypeName = moduleName + "_t";

    auto module = builder.create<emitc::VariableOp>(
        /*location=*/loc,
        /*resultType=*/
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, moduleTypeName)),
        /*value=*/emitc::OpaqueAttr::get(ctx, "NULL"));

    Value moduleSize =
        callSizeof(builder, loc, emitc::OpaqueAttr::get(ctx, moduleTypeName));

    Value modulePtr =
        emitc_builders::addressOf(builder, loc, module.getResult());

    auto voidPtr = builder.create<emitc::CastOp>(
        /*location=*/loc,
        /*type=*/
        emitc::PointerType::get(
            emitc::PointerType::get(emitc::OpaqueType::get(ctx, "void"))),
        /*operand=*/modulePtr);

    returnIfError(builder, loc, StringAttr::get(ctx, "iree_allocator_malloc"),
                  {}, {funcOp.getArgument(1), moduleSize, voidPtr.getResult()},
                  /*typeConverter=*/typeConverter);

    builder.create<emitc::CallOp>(
        /*location=*/loc,
        /*type=*/TypeRange{},
        /*callee=*/StringAttr::get(ctx, "memset"),
        /*args=*/
        ArrayAttr::get(ctx,
                       {builder.getIndexAttr(0), builder.getUI32IntegerAttr(0),
                        builder.getIndexAttr(1)}),
        /*templateArgs=*/ArrayAttr{},
        /*operands=*/
        ArrayRef<Value>{module.getResult(), moduleSize});

    emitc_builders::structPtrMemberAssign(builder, loc,
                                          /*memberName=*/"allocator",
                                          /*operand=*/module.getResult(),
                                          /*value=*/funcOp.getArgument(1));

    auto vmModule = builder.create<emitc::VariableOp>(
        /*location=*/loc,
        /*resultType=*/emitc::OpaqueType::get(ctx, "iree_vm_module_t"),
        /*value=*/emitc::OpaqueAttr::get(ctx, ""));

    Value vmModulePtr =
        emitc_builders::addressOf(builder, loc, vmModule.getResult());

    auto vmInitializeStatus = builder.create<emitc::CallOp>(
        /*location=*/loc,
        /*type=*/emitc::OpaqueType::get(ctx, "iree_status_t"),
        /*callee=*/StringAttr::get(ctx, "iree_vm_module_initialize"),
        /*args=*/ArrayAttr{},
        /*templateArgs=*/ArrayAttr{},
        /*operands=*/
        ArrayRef<Value>{vmModulePtr, module.getResult()});

    Type boolType = builder.getIntegerType(1);

    auto vmInitializeIsOk = builder.create<emitc::CallOp>(
        /*location=*/loc,
        /*type=*/boolType,
        /*callee=*/StringAttr::get(ctx, "iree_status_is_ok"),
        /*args=*/ArrayAttr{},
        /*templateArgs=*/ArrayAttr{},
        /*operands=*/
        ArrayRef<Value>{vmInitializeStatus.getResult(0)});

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

      builder.create<emitc::CallOp>(
          /*location=*/loc,
          /*type=*/TypeRange{},
          /*callee=*/StringAttr::get(ctx, "iree_allocator_free"),
          /*args=*/ArrayAttr{},
          /*templateArgs=*/ArrayAttr{},
          /*operands=*/
          ArrayRef<Value>{funcOp.getArgument(1), module.getResult()});

      builder.create<mlir::func::ReturnOp>(loc,
                                           vmInitializeStatus.getResult(0));
    }

    builder.setInsertionPointToEnd(condBlock);

    builder.create<IREE::VM::CondBranchOp>(loc, vmInitializeIsOk.getResult(0),
                                           continuationBlock, failureBlock);

    builder.setInsertionPointToStart(continuationBlock);

    // Set function pointers
    for (std::string funcName :
         {"destroy", "alloc_state", "free_state", "resolve_import"}) {
      emitc_builders::structMemberAssign(builder, loc,
                                         /*memberName=*/funcName,
                                         /*operand=*/vmModule.getResult(),
                                         /*value=*/moduleName + "_" + funcName);
    }

    std::string descriptorPtr = "&" + moduleName + "_descriptor_";

    auto status = builder.create<emitc::CallOp>(
        /*location=*/loc,
        /*type=*/emitc::OpaqueType::get(ctx, "iree_status_t"),
        /*callee=*/StringAttr::get(ctx, "iree_vm_native_module_create"),
        /*args=*/
        ArrayAttr::get(ctx, {builder.getIndexAttr(0),
                             emitc::OpaqueAttr::get(ctx, descriptorPtr),
                             builder.getIndexAttr(1), builder.getIndexAttr(2),
                             builder.getIndexAttr(3)}),
        /*templateArgs=*/ArrayAttr{},
        /*operands=*/
        ArrayRef<Value>{vmModulePtr, funcOp.getArgument(0),
                        funcOp.getArgument(1), funcOp.getArgument(2)});

    builder.create<mlir::func::ReturnOp>(loc, status.getResult(0));
  }

  return success();
}

SmallVector<Attribute, 4> indexSequence(int64_t n, MLIRContext *ctx) {
  return llvm::to_vector<4>(
      llvm::map_range(llvm::seq<int64_t>(0, n), [&ctx](int64_t i) -> Attribute {
        return IntegerAttr::get(IndexType::get(ctx), i);
      }));
}

template <typename ResultOpTy>
ResultOpTy lookupSymbolRef(Operation *accessOp, StringRef attrName) {
  FlatSymbolRefAttr globalAttr =
      accessOp->getAttrOfType<FlatSymbolRefAttr>(attrName);
  ResultOpTy globalOp =
      accessOp->getParentOfType<IREE::VM::ModuleOp>().lookupSymbol<ResultOpTy>(
          globalAttr.getValue());
  return globalOp;
}

// Convert vm operations to emitc calls. The resultiong call has the ops
// operands as arguments followed by an argument for every attribute.
template <typename SrcOpTy>
class GenericOpConversion : public OpConversionPattern<SrcOpTy> {
  using Adaptor = typename SrcOpTy::Adaptor;
  using OpConversionPattern<SrcOpTy>::OpConversionPattern;

 public:
  GenericOpConversion(TypeConverter &typeConverter, MLIRContext *context,
                      StringRef funcName)
      : OpConversionPattern<SrcOpTy>(typeConverter, context),
        funcName(funcName) {}

 private:
  LogicalResult matchAndRewrite(
      SrcOpTy op, Adaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto ctx = op.getContext();

    auto type = op.getOperation()->getResultTypes();
    StringAttr callee = StringAttr::get(ctx, funcName);

    // Default to an empty args attribute, which results in the operands being
    // printed as the arguments to the function call.
    ArrayAttr args;
    ArrayAttr templateArgs;

    // If the operation has attributes, we need to explicitely build the args
    // attribute of the emitc call op. This consists of index attributes for
    // the operands, followed by the source op attributes themselves.
    if (op->getAttrs().size() > 0) {
      SmallVector<Attribute, 4> args_ =
          indexSequence(adaptor.getOperands().size(), op.getContext());

      for (NamedAttribute attr : op->getAttrs()) {
        args_.push_back(attr.getValue());
      }

      args = rewriter.getArrayAttr(args_);
    }

    rewriter.replaceOpWithNewOp<emitc::CallOp>(
        op, type, callee, args, templateArgs, adaptor.getOperands());

    return success();
  }

  StringRef funcName;
};

template <typename SrcOpTy>
class DeleteOpConversion : public OpConversionPattern<SrcOpTy> {
  using Adaptor = typename SrcOpTy::Adaptor;
  using OpConversionPattern<SrcOpTy>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      SrcOpTy op, Adaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    rewriter.eraseOp(op);
    return success();
  }
};

class FuncOpConversion : public OpConversionPattern<mlir::func::FuncOp> {
  using OpConversionPattern<mlir::func::FuncOp>::OpConversionPattern;
  using Adaptor = mlir::func::FuncOp::Adaptor;

  LogicalResult matchAndRewrite(
      mlir::func::FuncOp funcOp, Adaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    TypeConverter::SignatureConversion signatureConverter(
        funcOp.getFunctionType().getNumInputs());
    TypeConverter typeConverter;
    for (const auto &arg : llvm::enumerate(funcOp.getArguments())) {
      Type convertedType =
          getTypeConverter()->convertType(arg.value().getType());
      signatureConverter.addInputs(arg.index(), convertedType);
    }

    rewriter.applySignatureConversion(&funcOp.getFunctionBody(),
                                      signatureConverter);

    // Creates a new function with the updated signature.
    rewriter.updateRootInPlace(funcOp, [&] {
      funcOp.setType(
          rewriter.getFunctionType(signatureConverter.getConvertedTypes(),
                                   funcOp.getFunctionType().getResults()));
    });
    return success();
  }
};

class ExportOpConversion : public OpConversionPattern<IREE::VM::ExportOp> {
  using Adaptor = IREE::VM::ExportOp::Adaptor;
  using OpConversionPattern<IREE::VM::ExportOp>::OpConversionPattern;

  struct GeneratedStruct {
    Optional<Value> value = std::nullopt;
    Optional<std::string> name = std::nullopt;
    SmallVector<Value> callArguments;
  };

  LogicalResult matchAndRewrite(
      IREE::VM::ExportOp exportOp, Adaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto ctx = exportOp.getContext();
    auto loc = exportOp.getLoc();

    IREE::VM::EmitCTypeConverter *typeConverter =
        this->getTypeConverter<IREE::VM::EmitCTypeConverter>();

    mlir::func::FuncOp funcOp = lookupSymbolRef<mlir::func::FuncOp>(
        exportOp.getOperation(), "function_ref");

    auto vmAnalysis = typeConverter->lookupAnalysis(funcOp);
    if (failed(vmAnalysis)) {
      funcOp.emitError() << "func op not found in cache.";
      return failure();
    }

    std::string newFuncName = (funcOp.getName() + "_export_shim").str();

    Type stackType =
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, "iree_vm_stack_t"));
    Type flagsType = emitc::OpaqueType::get(ctx, "uint32_t");
    Type spanType = emitc::OpaqueType::get(ctx, "iree_byte_span_t");
    Type moduleType =
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, "void"));
    Type moduleStateType =
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, "void"));

    SmallVector<Type> inputTypes = {
        stackType,        // SHIM_ARGUMENT_STACK
        flagsType,        // SHIM_ARGUMENT_FLAGS
        spanType,         // SHIM_ARGUMENT_ARGS_STORAGE
        spanType,         // SHIM_ARGUMENT_RETS_STORAGE
        moduleType,       // SHIM_ARGUMENT_MODULE
        moduleStateType,  // SHIM_ARGUMENT_MODULE_STATE
    };

    auto newFuncType = mlir::FunctionType::get(
        ctx, {inputTypes}, {emitc::OpaqueType::get(ctx, "iree_status_t")});

    auto newFuncOp =
        rewriter.create<mlir::func::FuncOp>(loc, newFuncName, newFuncType);

    FunctionType functionType = vmAnalysis.value().get().getFunctionType();

    typeConverter->analysisCache.insert(
        std::make_pair(newFuncOp.getOperation(), VMAnalysis(functionType)));

    attachAttribute(newFuncOp, "emitc.static", UnitAttr::get(ctx));
    attachAttribute(newFuncOp, "vm.calling_convention",
                    funcOp.getOperation()->getAttr("vm.calling_convention"));
    attachAttribute(newFuncOp, "vm.export_name", exportOp.getExportNameAttr());

    // Populate newly generated function.
    {
      OpBuilder::InsertionGuard guard(rewriter);
      Block *block = rewriter.createBlock(&newFuncOp.getFunctionBody(),
                                          newFuncOp.getFunctionBody().end());

      // Insert arguments into block.
      block->addArgument(stackType, loc);        // SHIM_ARGUMENT_STACK
      block->addArgument(flagsType, loc);        // SHIM_ARGUMENT_FLAGS
      block->addArgument(spanType, loc);         // SHIM_ARGUMENT_ARGS_STORAGE
      block->addArgument(spanType, loc);         // SHIM_ARGUMENT_RETS_STORAGE
      block->addArgument(moduleType, loc);       // SHIM_ARGUMENT_MODULE
      block->addArgument(moduleStateType, loc);  // SHIM_ARGUMENT_MODULE_STATE

      rewriter.setInsertionPointToStart(block);

      // Create typedefs for argument and result structs.
      auto typedefs =
          typedefArgumentAndResultStructs(rewriter, exportOp, newFuncOp);

      if (failed(typedefs)) {
        return exportOp.emitError() << "struct typedef failed.";
      }

      GeneratedStruct argumentStruct;
      GeneratedStruct resultStruct;

      std::tie(argumentStruct, resultStruct) = typedefs.value();

      // Cast module and module state structs.
      auto moduleStructs =
          castModuleAndStateStructs(rewriter, exportOp, newFuncOp);

      if (failed(moduleStructs)) {
        return exportOp.emitError() << "module struct casting failed.";
      }

      Value moduleStruct;
      Value moduleStateStruct;

      std::tie(moduleStruct, moduleStateStruct) = moduleStructs.value();

      // Cast argument and result structs.
      castArgumentAndResultStructs(rewriter, exportOp, newFuncOp,
                                   argumentStruct, resultStruct);

      // Unpack arguments from struct.
      auto arguments = unpackArguments(rewriter, exportOp, argumentStruct);

      if (failed(arguments)) {
        return exportOp.emitError() << "failed to unpack arguments.";
      }

      // Unpack result pointers from struct.
      auto results = unpackResults(rewriter, exportOp, resultStruct);

      if (failed(results)) {
        return exportOp.emitError() << "failed to unpack results.";
      }

      // Call internal function and return on error.
      SmallVector<Value> operands{block->getArgument(SHIM_ARGUMENT_STACK),
                                  moduleStruct, moduleStateStruct};

      for (auto &argument : argumentStruct.callArguments) {
        operands.push_back(argument);
      }
      for (auto &result : resultStruct.callArguments) {
        operands.push_back(result);
      }

      returnIfError(rewriter, loc, funcOp, operands, *typeConverter);

      auto status = emitc_builders::ireeOkStatus(rewriter, loc);

      rewriter.create<mlir::func::ReturnOp>(loc, status);
    }

    rewriter.eraseOp(exportOp);

    return success();
  }

  FailureOr<std::pair<Value, Value>> castModuleAndStateStructs(
      ConversionPatternRewriter &rewriter, IREE::VM::ExportOp &exportOp,
      mlir::func::FuncOp &newFuncOp) const {
    auto ctx = exportOp.getContext();
    auto loc = exportOp.getLoc();

    auto module = newFuncOp.getArgument(SHIM_ARGUMENT_MODULE);
    auto moduleState = newFuncOp.getArgument(SHIM_ARGUMENT_MODULE_STATE);

    auto moduleOp =
        newFuncOp.getOperation()->getParentOfType<IREE::VM::ModuleOp>();

    std::string moduleTypeName = (moduleOp.getName() + "_t").str();
    std::string moduleStateTypeName = (moduleOp.getName() + "_state_t").str();

    auto moduleCasted = rewriter.create<emitc::CastOp>(
        /*location=*/loc,
        /*type=*/
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, moduleTypeName)),
        /*operand=*/module);

    auto moduleStateCasted = rewriter.create<emitc::CastOp>(
        /*location=*/loc,
        /*type=*/
        emitc::PointerType::get(
            emitc::OpaqueType::get(ctx, moduleStateTypeName)),
        /*operand=*/moduleState);

    return {{moduleCasted.getResult(), moduleStateCasted.getResult()}};
  }

  FailureOr<std::pair<GeneratedStruct, GeneratedStruct>>
  typedefArgumentAndResultStructs(ConversionPatternRewriter &rewriter,
                                  IREE::VM::ExportOp &exportOp,
                                  mlir::func::FuncOp &newFuncOp) const {
    auto loc = exportOp.getLoc();

    IREE::VM::EmitCTypeConverter *typeConverter =
        this->template getTypeConverter<IREE::VM::EmitCTypeConverter>();

    mlir::func::FuncOp funcOp = lookupSymbolRef<mlir::func::FuncOp>(
        exportOp.getOperation(), "function_ref");

    auto vmAnalysis = typeConverter->lookupAnalysis(funcOp);
    if (failed(vmAnalysis)) {
      funcOp.emitError() << "func op not found in cache.";
      return failure();
    }

    auto generateStructFields = [this](ArrayRef<Type> types, StringRef prefix)
        -> FailureOr<SmallVector<emitc_builders::StructField>> {
      SmallVector<emitc_builders::StructField> result;

      for (auto pair : llvm::enumerate(types)) {
        emitc::OpaqueType cType =
            getTypeConverter<IREE::VM::EmitCTypeConverter>()
                ->convertTypeAsCType(pair.value());

        if (!cType) {
          return failure();
        }

        auto fieldName = prefix.str() + std::to_string(pair.index());
        result.push_back({cType.getValue().str(), fieldName});
      }

      return result;
    };

    // TODO(simon-camp): Clean up; We generate calls to a macro that defines
    // a struct. As we declare all variables at the start of the function,
    // the macro call cannot be inlined into the function.

    // To prevent scoping issues we prefix the struct name with module and
    // function name.
    auto typedefStruct = [&rewriter, &newFuncOp, &loc](
                             std::string structName,
                             ArrayRef<emitc_builders::StructField> fields) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPoint(newFuncOp.getOperation());

      emitc_builders::structDefinition(/*builder=*/rewriter, /*location=*/loc,
                                       /*structName=*/structName,
                                       /*fields=*/fields);
    };

    FunctionType funcType = vmAnalysis.value().get().getFunctionType();

    GeneratedStruct argumentStruct;
    GeneratedStruct resultStruct;

    const bool needArgumentStruct = funcType.getNumInputs() > 0;

    if (needArgumentStruct) {
      auto structBody = generateStructFields(funcType.getInputs(), "arg");
      if (failed(structBody)) {
        return exportOp.emitError()
               << "failed to emit C type for struct definition";
      }

      std::string structName = (funcOp.getName() + "_args_t").str();
      argumentStruct.name = structName;
      typedefStruct(structName, structBody.value());
    }

    const bool needResultStruct = funcType.getNumResults() > 0;

    if (needResultStruct) {
      auto structBody = generateStructFields(funcType.getResults(), "res");

      if (failed(structBody)) {
        return failure();
      }

      std::string structName = (funcOp.getName() + "_result_t").str();
      resultStruct.name = structName;
      typedefStruct(structName, structBody.value());
    }

    return {{argumentStruct, resultStruct}};
  }

  void castArgumentAndResultStructs(ConversionPatternRewriter &rewriter,
                                    IREE::VM::ExportOp &exportOp,
                                    mlir::func::FuncOp &newFuncOp,
                                    GeneratedStruct &argumentStruct,
                                    GeneratedStruct &resultStruct) const {
    auto ctx = exportOp.getContext();
    auto loc = exportOp.getLoc();

    const bool haveArgumentStruct = argumentStruct.name.has_value();

    if (haveArgumentStruct) {
      // args_t* args = (args_t*)call->arguments.data;

      // arguments.data
      auto argumentsData = emitc_builders::structMember(
          rewriter, loc,
          /*type=*/emitc::PointerType::get(rewriter.getIntegerType(8, false)),
          /*memberName=*/"data",
          /*operand=*/newFuncOp.getArgument(SHIM_ARGUMENT_ARGS_STORAGE));

      // cast
      std::string argumentsType = argumentStruct.name.value();
      auto arguments = rewriter.create<emitc::CastOp>(
          /*location=*/loc,
          /*type=*/
          emitc::PointerType::get(emitc::OpaqueType::get(ctx, argumentsType)),
          /*operand=*/argumentsData);

      argumentStruct.value = arguments.getResult();
    }

    const bool haveResultStruct = resultStruct.name.has_value();
    if (haveResultStruct) {
      // results_t* results = (results_t*)call->results.data;

      // results.data
      auto resultsData = emitc_builders::structMember(
          rewriter, loc,
          /*type=*/emitc::PointerType::get(rewriter.getIntegerType(8, false)),
          /*memberName=*/"data",
          /*operand=*/newFuncOp.getArgument(SHIM_ARGUMENT_RETS_STORAGE));

      // cast
      std::string resultType = resultStruct.name.value();
      auto results = rewriter.create<emitc::CastOp>(
          /*location=*/loc,
          /*type=*/
          emitc::PointerType::get(emitc::OpaqueType::get(ctx, resultType)),
          /*operand=*/resultsData);

      resultStruct.value = results.getResult();
    }
  }

  LogicalResult unpackArguments(ConversionPatternRewriter &rewriter,
                                IREE::VM::ExportOp &exportOp,
                                GeneratedStruct &argumentStruct) const {
    auto ctx = exportOp.getContext();
    auto loc = exportOp.getLoc();

    // The struct is empty, nothing to do.
    if (!argumentStruct.value.has_value()) {
      return success();
    }

    IREE::VM::EmitCTypeConverter *typeConverter =
        this->template getTypeConverter<IREE::VM::EmitCTypeConverter>();

    mlir::func::FuncOp funcOp = lookupSymbolRef<mlir::func::FuncOp>(
        exportOp.getOperation(), "function_ref");

    auto vmAnalysis = typeConverter->lookupAnalysis(funcOp);
    if (failed(vmAnalysis)) {
      funcOp.emitError() << "func op not found in cache.";
      return failure();
    }

    FunctionType funcType = vmAnalysis.value().get().getFunctionType();

    for (const auto &input : llvm::enumerate(funcType.getInputs())) {
      assert(argumentStruct.value.has_value());
      auto value = argumentStruct.value.value();

      if (input.value().isa<IREE::VM::RefType>()) {
        Type ptrType = emitc::PointerType::get(
            emitc::OpaqueType::get(ctx, "iree_vm_ref_t"));
        std::string memberName = "arg" + std::to_string(input.index());
        auto memberPtr = rewriter.create<emitc::CallOp>(
            /*location=*/loc,
            /*type=*/ptrType,
            /*callee=*/StringAttr::get(ctx, "EMITC_STRUCT_PTR_MEMBER_ADDRESS"),
            /*args=*/
            ArrayAttr::get(ctx, {rewriter.getIndexAttr(0),
                                 emitc::OpaqueAttr::get(ctx, memberName)}),
            /*templateArgs=*/ArrayAttr{},
            /*operands=*/ArrayRef<Value>{value});
        argumentStruct.callArguments.push_back(memberPtr.getResult(0));
      } else {
        Type memberType = input.value();
        std::string memberName = "arg" + std::to_string(input.index());
        auto member = emitc_builders::structPtrMember(rewriter, loc,
                                                      /*type=*/memberType,
                                                      /*memberName=*/memberName,
                                                      /*operand=*/value);

        argumentStruct.callArguments.push_back(member);
      }
    }

    return success();
  }

  LogicalResult unpackResults(ConversionPatternRewriter &rewriter,
                              IREE::VM::ExportOp &exportOp,
                              GeneratedStruct &resultStruct) const {
    auto ctx = exportOp.getContext();
    auto loc = exportOp.getLoc();

    // The struct is empty, nothing to do.
    if (!resultStruct.value.has_value()) {
      return success();
    }

    IREE::VM::EmitCTypeConverter *typeConverter =
        this->template getTypeConverter<IREE::VM::EmitCTypeConverter>();

    mlir::func::FuncOp funcOp = lookupSymbolRef<mlir::func::FuncOp>(
        exportOp.getOperation(), "function_ref");

    auto vmAnalysis = typeConverter->lookupAnalysis(funcOp);
    if (failed(vmAnalysis)) {
      funcOp.emitError() << "func op not found in cache.";
      return failure();
    }

    FunctionType funcType = vmAnalysis.value().get().getFunctionType();

    for (const auto &result : llvm::enumerate(funcType.getResults())) {
      assert(resultStruct.value.has_value());
      auto value = resultStruct.value.value();

      Type ptrType = typeConverter->convertTypeAsPointer(result.value());

      std::string memberName = "res" + std::to_string(result.index());
      auto memberPtr = rewriter.create<emitc::CallOp>(
          /*location=*/loc,
          /*type=*/ptrType,
          /*callee=*/StringAttr::get(ctx, "EMITC_STRUCT_PTR_MEMBER_ADDRESS"),
          /*args=*/
          ArrayAttr::get(ctx, {rewriter.getIndexAttr(0),
                               emitc::OpaqueAttr::get(ctx, memberName)}),
          /*templateArgs=*/ArrayAttr{},
          /*operands=*/ArrayRef<Value>{value});
      resultStruct.callArguments.push_back(memberPtr.getResult(0));
    }

    return success();
  }
};

class ImportOpConverter {
 public:
  ImportOpConverter(IREE::VM::EmitCTypeConverter &typeConverter,
                    SmallVector<std::string> &importShims)
      : typeConverter(typeConverter), importShims(importShims) {}

  LogicalResult operator()(IREE::VM::ImportOp importOp) const {
    OpBuilder builder(importOp);

    auto key = makeImportCallingConventionString(importOp);
    if (!key.has_value()) {
      return importOp.emitError()
             << "Failed to build key for import shim cache.";
    }

    // The needed shim already exists.
    if (llvm::find(importShims, key) != std::end(importShims)) {
      return success();
    }

    if (importOp.isVariadic()) {
      if (failed(createVariadicImportShims(importOp, builder))) {
        return failure();
      }
    } else {
      if (failed(createImportShim(importOp, nullptr, builder))) {
        return failure();
      }
    }

    importShims.push_back(key.value());
    return success();
  }

 private:
  LogicalResult createVariadicImportShims(IREE::VM::ImportOp &importOp,
                                          OpBuilder &builder) const {
    SetVector<const void *> arities;

    for (auto caller : getCallers(importOp)) {
      DenseIntElementsAttr segmentSizes = caller.getSegmentSizes();
      const void *p = segmentSizes.getAsOpaquePointer();
      if (arities.insert(p)) {
        if (failed(createImportShim(importOp, segmentSizes, builder))) {
          return failure();
        }
      }
    }
    return success();
  }

  void failIfImportUnresolved(OpBuilder &builder, Location location,
                              Value import) const {
    auto *ctx = builder.getContext();
    Type boolType = builder.getIntegerType(1);

    // (iree_vm_function_t*)->module
    auto importModule = emitc_builders::structPtrMember(
        builder, location,
        /*type=*/
        emitc::PointerType::get(
            emitc::OpaqueType::get(ctx, "iree_vm_module_t")),
        /*memberName=*/"module",
        /*operand=*/import);

    auto conditionI1 = builder.create<emitc::CallOp>(
        /*location=*/location,
        /*type=*/boolType,
        /*callee=*/StringAttr::get(ctx, "EMITC_NOT"),
        /*args=*/
        ArrayAttr::get(ctx, {builder.getIndexAttr(0)}),
        /*templateArgs=*/ArrayAttr{},
        /*operands=*/ArrayRef<Value>{importModule});

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

      mlir::func::FuncOp funcOp =
          cast<mlir::func::FuncOp>(failureBlock->getParentOp());
      releaseRefs(builder, location, funcOp, typeConverter);

      auto statusOp = builder.create<emitc::CallOp>(
          /*location=*/location,
          /*type=*/emitc::OpaqueType::get(ctx, "iree_status_t"),
          /*callee=*/StringAttr::get(ctx, "iree_make_status"),
          /*args=*/
          ArrayAttr::get(
              ctx, {emitc::OpaqueAttr::get(ctx, "IREE_STATUS_NOT_FOUND")}),
          /*templateArgs=*/ArrayAttr{},
          /*operands=*/ArrayRef<Value>{});
      builder.create<mlir::func::ReturnOp>(location, statusOp.getResult(0));
    }

    builder.setInsertionPointToEnd(condBlock);
    builder.create<IREE::VM::CondBranchOp>(location, conditionI1.getResult(0),
                                           failureBlock, continuationBlock);

    builder.setInsertionPointToStart(continuationBlock);
  }

  LogicalResult createImportShim(IREE::VM::ImportOp &importOp,
                                 DenseIntElementsAttr segmentSizes,
                                 OpBuilder &builder) const {
    auto ctx = importOp.getContext();
    auto loc = importOp.getLoc();

    auto moduleOp =
        importOp.getOperation()->getParentOfType<IREE::VM::ModuleOp>();

    auto newFuncName =
        importOp.isVariadic()
            ? buildVariadicFunctionName(moduleOp, importOp, segmentSizes)
            : buildFunctionName(moduleOp, importOp);

    if (!newFuncName.has_value()) {
      return importOp.emitError() << "failed to build import shim name.";
    }

    auto newFuncType = buildFuncType(importOp, segmentSizes, builder, loc);

    if (failed(newFuncType)) {
      return importOp.emitError()
             << "Failed to build function type for wrapper";
    }

    auto newFuncOp = builder.create<mlir::func::FuncOp>(
        loc, newFuncName.value(), newFuncType.value());

    typeConverter.analysisCache.insert(std::make_pair(
        newFuncOp.getOperation(), VMAnalysis(importOp.getFunctionType())));

    attachAttribute(newFuncOp, "emitc.static", UnitAttr::get(ctx));

    // Populate newly generated function.
    {
      OpBuilder::InsertionGuard guard(builder);
      Block *block = builder.createBlock(&newFuncOp.getFunctionBody(),
                                         newFuncOp.getFunctionBody().end());

      for (Type type : newFuncOp.getFunctionType().getInputs()) {
        block->addArgument(type, loc);
      }

      builder.setInsertionPointToStart(block);

      auto argumentSize = buildSizeExpression(
          flattenInputTypes(importOp, segmentSizes, builder), builder, loc);
      auto resultSize =
          buildSizeExpression(importOp.getResultTypes(), builder, loc);

      if (failed(argumentSize) || failed(resultSize)) {
        return importOp.emitError()
               << "Failed to build size expressions for call struct";
      }

      auto importArg = newFuncOp.getArgument(1);
      failIfImportUnresolved(builder, loc, importArg);

      auto call = buildIreeVmFunctionCallStruct(
          importArg, argumentSize.value(), resultSize.value(), builder, loc);

      if (failed(call)) {
        return importOp.emitError() << "failed to create call struct";
      }

      if (failed(packArgumentBuffer(
              flattenInputTypes(importOp, segmentSizes, builder), newFuncOp,
              call.value(), builder, loc))) {
        return importOp.emitError() << "failed to pack argument struct";
      }

      auto stackArg = newFuncOp.getArgument(0);
      if (failed(createCall(call.value(), importArg, stackArg, builder, loc))) {
        return importOp.emitError() << "failed to create call";
      }

      if (failed(unpackResultBuffer(importOp.getResultTypes(), newFuncOp,
                                    call.value(), builder, loc))) {
        return importOp.emitError() << "failed to unpack result struct";
      }

      auto status = emitc_builders::ireeOkStatus(builder, loc);

      builder.create<mlir::func::ReturnOp>(loc, status);
    }

    return success();
  }

  FailureOr<FunctionType> buildFuncType(IREE::VM::ImportOp importOp,
                                        DenseIntElementsAttr segmentSizes,
                                        OpBuilder &builder,
                                        Location loc) const {
    auto ctx = builder.getContext();

    Type stackType =
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, "iree_vm_stack_t"));
    Type funcType = emitc::PointerType::get(
        emitc::OpaqueType::get(ctx, "iree_vm_function_t"));

    SmallVector<Type> types{stackType, funcType};

    for (Type type : flattenInputTypes(importOp, segmentSizes, builder)) {
      auto convertedType = typeConverter.convertType(type);
      types.push_back(convertedType);
    }

    for (auto &resultType : importOp.getResultTypes()) {
      Type ptrType = typeConverter.convertTypeAsPointer(resultType);

      types.push_back(ptrType);
    }

    FunctionType result = mlir::FunctionType::get(
        ctx, {types}, {emitc::OpaqueType::get(ctx, "iree_status_t")});

    return {result};
  }

  FailureOr<Value> buildSizeExpression(ArrayRef<Type> types, OpBuilder &builder,
                                       Location loc) const {
    auto ctx = builder.getContext();

    Type hostSizeType = emitc::OpaqueType::get(ctx, "iree_host_size_t");

    Value result = builder
                       .create<emitc::ConstantOp>(
                           /*location=*/loc,
                           /*resultType=*/hostSizeType,
                           /*value=*/emitc::OpaqueAttr::get(ctx, "0"))
                       .getResult();

    for (Type type : types) {
      Type valueType = typeConverter.convertTypeAsNonPointer(type);
      Value size = callSizeof(builder, loc, TypeAttr::get(valueType));

      result = builder
                   .create<emitc::CallOp>(
                       /*location=*/loc,
                       /*type=*/hostSizeType,
                       /*callee=*/StringAttr::get(ctx, "EMITC_ADD"),
                       /*args=*/ArrayAttr{},
                       /*templateArgs=*/ArrayAttr{},
                       /*operands=*/ArrayRef<Value>{result, size})
                   .getResult(0);
    }

    return {result};
  }

  FailureOr<Value> buildIreeVmFunctionCallStruct(Value import,
                                                 Value argumentSize,
                                                 Value resultSize,
                                                 OpBuilder &builder,
                                                 Location loc) const {
    auto ctx = builder.getContext();

    // iree_vm_function_call_t call;
    auto call = builder
                    .create<emitc::ConstantOp>(
                        /*location=*/loc,
                        /*resultType=*/
                        emitc::OpaqueType::get(ctx, "iree_vm_function_call_t"),
                        /*value=*/emitc::OpaqueAttr::get(ctx, ""))
                    .getResult();

    // importValue = *import;
    auto importValue = emitc_builders::contentsOf(builder, loc, import);

    // call.function = importValue;
    emitc_builders::structMemberAssign(builder, loc,
                                       /*memberName=*/"function",
                                       /*operand=*/call,
                                       /*value=*/importValue);

    allocateByteSpan(call, argumentSize, "arguments", builder, loc);
    allocateByteSpan(call, resultSize, "results", builder, loc);

    return {call};
  }

  Value allocateByteSpan(Value call, Value size, StringRef memberName,
                         OpBuilder &builder, Location loc) const {
    auto ctx = builder.getContext();

    // byteSpan = call.<memberName>;
    auto byteSpan =
        builder
            .create<emitc::CallOp>(
                /*location=*/loc,
                /*type=*/
                emitc::PointerType::get(
                    emitc::OpaqueType::get(ctx, "iree_byte_span_t")),
                /*callee=*/StringAttr::get(ctx, "EMITC_STRUCT_MEMBER_ADDRESS"),
                /*args=*/
                ArrayAttr::get(ctx, {builder.getIndexAttr(0),
                                     emitc::OpaqueAttr::get(ctx, memberName)}),
                /*templateArgs=*/ArrayAttr{},
                /*operands=*/ArrayRef<Value>{call})
            .getResult(0);

    // void *byteSpan_data_void = iree_alloca(size);
    auto byteSpanDataVoid =
        builder
            .create<emitc::CallOp>(
                /*location=*/loc,
                /*type=*/
                emitc::PointerType::get(emitc::OpaqueType::get(ctx, "void")),
                /*callee=*/StringAttr::get(ctx, "iree_alloca"),
                /*args=*/ArrayAttr{},
                /*templateArgs=*/ArrayAttr{},
                /*operands=*/ArrayRef<Value>{size})
            .getResult(0);

    // uint8_t *byteSpan_data = (uint8_t*)byteSpan_data_void;
    Type bytePtr = emitc::PointerType::get(builder.getIntegerType(8, false));
    auto byteSpanData = builder
                            .create<emitc::CastOp>(
                                /*location=*/loc,
                                /*type=*/bytePtr,
                                /*operand=*/byteSpanDataVoid)
                            .getResult();

    // byteSpan.data_length = SIZE;
    emitc_builders::structPtrMemberAssign(builder, loc,
                                          /*memberName=*/"data_length",
                                          /*operand=*/byteSpan,
                                          /*value=*/size);

    // byteSpan.data = byteSpan_data
    emitc_builders::structPtrMemberAssign(builder, loc,
                                          /*memberName=*/"data",
                                          /*operand=*/byteSpan,
                                          /*value=*/byteSpanData);

    // memset(byteSpanData, 0, SIZE);
    builder.create<emitc::CallOp>(
        /*location=*/loc,
        /*type=*/TypeRange{},
        /*callee=*/StringAttr::get(ctx, "memset"),
        /*args=*/
        ArrayAttr::get(ctx,
                       {builder.getIndexAttr(0), builder.getI32IntegerAttr(0),
                        builder.getIndexAttr(1)}),
        /*templateArgs=*/ArrayAttr{},
        /*operands=*/ArrayRef<Value>{byteSpanData, size});

    return byteSpan;
  }

  LogicalResult packArgumentBuffer(ArrayRef<Type> inputTypes,
                                   mlir::func::FuncOp &funcOp, Value call,
                                   OpBuilder &builder, Location loc) const {
    if (inputTypes.empty()) {
      return success();
    }

    auto ctx = builder.getContext();

    size_t inputOffset = 2;

    Value arguments = emitc_builders::structMember(
        builder, loc,
        /*type=*/emitc::OpaqueType::get(ctx, "iree_byte_span_t"),
        /*memberName=*/"arguments",
        /*operand=*/call);

    Type bytePtrType =
        emitc::PointerType::get(builder.getIntegerType(8, false));
    Value uint8Ptr = emitc_builders::structMember(builder, loc,
                                                  /*type=*/bytePtrType,
                                                  /*memberName=*/"data",
                                                  /*operand=*/arguments);

    for (size_t i = 0; i < inputTypes.size(); i++) {
      BlockArgument arg = funcOp.getArgument(i + inputOffset);
      Type argType = arg.getType();
      assert(!argType.isa<IREE::VM::RefType>());

      if (argType == emitc::PointerType::get(
                         emitc::OpaqueType::get(ctx, "iree_vm_ref_t"))) {
        Type refPtrType = emitc::PointerType::get(
            emitc::OpaqueType::get(ctx, "iree_vm_ref_t"));
        Value refPtr = builder
                           .create<emitc::CastOp>(
                               /*location=*/loc,
                               /*type=*/refPtrType,
                               /*operand=*/uint8Ptr)
                           .getResult();

        builder.create<emitc::CallOp>(
            /*location=*/loc,
            /*type=*/TypeRange{},
            /*callee=*/StringAttr::get(ctx, "iree_vm_ref_assign"),
            /*args=*/ArrayAttr{},
            /*templateArgs=*/ArrayAttr{},
            /*operands=*/ArrayRef<Value>{arg, refPtr});
      } else {
        assert(!argType.isa<emitc::PointerType>());
        Value size = callSizeof(builder, loc, TypeAttr::get(argType));

        // memcpy(uint8Ptr, &arg, size);
        Value argPtr = emitc_builders::addressOf(builder, loc, arg);
        builder.create<emitc::CallOp>(
            /*location=*/loc,
            /*type=*/TypeRange{},
            /*callee=*/StringAttr::get(ctx, "memcpy"),
            /*args=*/ArrayAttr{},
            /*templateArgs=*/ArrayAttr{},
            /*operands=*/ArrayRef<Value>{uint8Ptr, argPtr, size});
      }

      // Skip the addition in the last iteration.
      if (i < inputTypes.size() - 1) {
        Type valueType = typeConverter.convertTypeAsNonPointer(argType);
        Value size = callSizeof(builder, loc, TypeAttr::get(valueType));

        uint8Ptr = builder
                       .create<emitc::CallOp>(
                           /*location=*/loc,
                           /*type=*/bytePtrType,
                           /*callee=*/StringAttr::get(ctx, "EMITC_ADD"),
                           /*args=*/ArrayAttr{},
                           /*templateArgs=*/ArrayAttr{},
                           /*operands=*/ArrayRef<Value>{uint8Ptr, size})
                       .getResult(0);
      }
    }
    return success();
  }

  LogicalResult unpackResultBuffer(ArrayRef<Type> resultTypes,
                                   mlir::func::FuncOp &funcOp, Value call,
                                   OpBuilder &builder, Location loc) const {
    if (resultTypes.empty()) {
      return success();
    }

    auto ctx = builder.getContext();

    // The last N arguments are the results.
    size_t resultOffset = funcOp.getNumArguments() - resultTypes.size();

    Value results = emitc_builders::structMember(
        builder, loc,
        /*type=*/emitc::OpaqueType::get(ctx, "iree_byte_span_t"),
        /*memberName=*/"results",
        /*operand=*/call);

    Type bytePtrType =
        emitc::PointerType::get(builder.getIntegerType(8, false));
    Value uint8Ptr = emitc_builders::structMember(builder, loc,
                                                  /*type=*/bytePtrType,
                                                  /*memberName=*/"data",
                                                  /*operand=*/results);

    for (size_t i = 0; i < resultTypes.size(); i++) {
      BlockArgument arg = funcOp.getArgument(i + resultOffset);
      Type argType = arg.getType();
      assert(!argType.isa<IREE::VM::RefType>());

      if (argType == emitc::PointerType::get(
                         emitc::OpaqueType::get(ctx, "iree_vm_ref_t"))) {
        Type refPtrType = emitc::PointerType::get(
            emitc::OpaqueType::get(ctx, "iree_vm_ref_t"));
        Value refPtr = builder
                           .create<emitc::CastOp>(
                               /*location=*/loc,
                               /*type=*/refPtrType,
                               /*operand=*/uint8Ptr)
                           .getResult();

        builder.create<emitc::CallOp>(
            /*location=*/loc,
            /*type=*/TypeRange{},
            /*callee=*/StringAttr::get(ctx, "iree_vm_ref_move"),
            /*args=*/ArrayAttr{},
            /*templateArgs=*/ArrayAttr{},
            /*operands=*/ArrayRef<Value>{refPtr, arg});
      } else {
        Type valueType = argType.cast<emitc::PointerType>().getPointee();
        Value size = callSizeof(builder, loc, TypeAttr::get(valueType));

        // memcpy(arg, uint8Ptr, size);
        builder.create<emitc::CallOp>(
            /*location=*/loc,
            /*type=*/TypeRange{},
            /*callee=*/StringAttr::get(ctx, "memcpy"),
            /*args=*/ArrayAttr{},
            /*templateArgs=*/ArrayAttr{},
            /*operands=*/ArrayRef<Value>{arg, uint8Ptr, size});
      }

      // Skip the addition in the last iteration.
      if (i < resultTypes.size() - 1) {
        Type valueType = argType.cast<emitc::PointerType>().getPointee();
        Value size = callSizeof(builder, loc, TypeAttr::get(valueType));

        uint8Ptr = builder
                       .create<emitc::CallOp>(
                           /*location=*/loc,
                           /*type=*/bytePtrType,
                           /*callee=*/StringAttr::get(ctx, "EMITC_ADD"),
                           /*args=*/ArrayAttr{},
                           /*templateArgs=*/ArrayAttr{},
                           /*operands=*/ArrayRef<Value>{uint8Ptr, size})
                       .getResult(0);
      }
    }
    return success();
  }

  LogicalResult createCall(Value call, Value import, Value stack,
                           OpBuilder &builder, Location loc) const {
    auto ctx = builder.getContext();

    // RETURN_IF_ERROR(import->module->begin_call(import->module, stack, call));
    auto importModule = emitc_builders::structPtrMember(
        builder, loc,
        /*type=*/
        emitc::PointerType::get(
            emitc::OpaqueType::get(ctx, "iree_vm_module_t")),
        /*memberName=*/"module",
        /*operand=*/import);

    returnIfError(
        /*rewriter=*/builder,
        /*location=*/loc,
        /*callee=*/StringAttr::get(ctx, "EMITC_STRUCT_PTR_MEMBER_CALL"),
        /*args=*/
        ArrayAttr::get(ctx,
                       {
                           builder.getIndexAttr(0),
                           emitc::OpaqueAttr::get(ctx, "begin_call"),
                           builder.getIndexAttr(0),
                           builder.getIndexAttr(1),
                           builder.getIndexAttr(2),
                       }),
        /*operands=*/ArrayRef<Value>{importModule, stack, call},
        /*typeConverter=*/typeConverter);

    return success();
  }

  // A span count of -1 means a non variadic call
  SmallVector<Type> flattenInputTypes(IREE::VM::ImportOp importOp,
                                      DenseIntElementsAttr segmentSizes,
                                      OpBuilder &builder) const {
    assert(!segmentSizes ||
           (importOp.getNumArguments() == segmentSizes.size()));

    SmallVector<Type> result;
    auto expandType = [&result](Type type) {
      if (auto tupleType = type.dyn_cast<TupleType>()) {
        for (Type inner : tupleType) {
          result.push_back(inner);
        }
      } else {
        result.push_back(type);
      }
    };

    for (size_t i = 0; i < importOp.getNumArguments(); i++) {
      Type type = importOp.getFunctionType().getInput(i);

      if (importOp.isFuncArgumentVariadic(i)) {
        assert(segmentSizes && "segmentSizes must not be nullptr");
        APInt segmentSize = *(segmentSizes.begin() + i);
        int64_t size = segmentSize.getSExtValue();
        result.push_back(builder.getI32Type());

        assert(size >= 0);
        for (int j = 0; j < size; j++) {
          expandType(type);
        }
      } else {
        expandType(type);
      }
    }

    return result;
  }

  SmallVector<IREE::VM::CallVariadicOp> getCallers(
      IREE::VM::ImportOp &importOp) const {
    SmallVector<IREE::VM::CallVariadicOp> result;

    auto moduleOp =
        importOp.getOperation()->getParentOfType<IREE::VM::ModuleOp>();

    moduleOp.walk([&result, &importOp](Operation *op) {
      if (auto callOp = dyn_cast<IREE::VM::CallVariadicOp>(op)) {
        if (importOp == lookupSymbolRef<IREE::VM::ImportOp>(
                            callOp.getOperation(), "callee")) {
          result.push_back(callOp);
        }
      }
    });

    return result;
  }

  IREE::VM::EmitCTypeConverter &typeConverter;
  SmallVector<std::string> &importShims;
};

template <typename CallOpTy>
class CallOpConversion : public OpConversionPattern<CallOpTy> {
  using Adaptor = typename CallOpTy::Adaptor;
  using OpConversionPattern<CallOpTy>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      CallOpTy op, Adaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    mlir::func::FuncOp funcOp =
        lookupSymbolRef<mlir::func::FuncOp>(op.getOperation(), "callee");
    IREE::VM::ImportOp importOp =
        lookupSymbolRef<IREE::VM::ImportOp>(op.getOperation(), "callee");

    if (!funcOp && !importOp)
      return op.emitError() << "lookup of callee failed";

    if (funcOp && importOp)
      return op.emitError() << "lookup of callee ambiguous";

    const bool isImported = importOp != nullptr;

    return isImported ? rewriteImportedCall(op.getOperation(), adaptor,
                                            rewriter, importOp)
                      : rewriteInternalCall(op.getOperation(), adaptor,
                                            rewriter, funcOp);
  }

  LogicalResult rewriteInternalCall(Operation *op, Adaptor adaptor,
                                    ConversionPatternRewriter &rewriter,
                                    mlir::func::FuncOp funcOp) const {
    auto loc = op->getLoc();

    SmallVector<Value, 4> updatedOperands;
    SmallVector<Value, 4> resultOperands;

    auto parentFuncOp = op->getParentOfType<mlir::func::FuncOp>();

    BlockArgument stackArg = parentFuncOp.getArgument(0);
    BlockArgument moduleArg = parentFuncOp.getArgument(1);
    BlockArgument moduleStateArg = parentFuncOp.getArgument(2);

    updatedOperands = {stackArg, moduleArg, moduleStateArg};

    if (failed(updateOperands(op, nullptr, rewriter, updatedOperands,
                              resultOperands))) {
      return failure();
    };

    returnIfError(
        /*rewriter=*/rewriter, /*location=*/loc, /*callee=*/funcOp,
        /*operands=*/updatedOperands,
        /*typeConverter=*/
        *this->template getTypeConverter<IREE::VM::EmitCTypeConverter>());

    if (failed(updateResults(op, rewriter, resultOperands))) {
      return failure();
    }

    rewriter.eraseOp(op);

    return success();
  }

  LogicalResult rewriteImportedCall(Operation *op, Adaptor adaptor,
                                    ConversionPatternRewriter &rewriter,
                                    IREE::VM::ImportOp importOp) const {
    auto ctx = op->getContext();
    auto loc = op->getLoc();

    SmallVector<Value, 4> updatedOperands;
    SmallVector<Value, 4> resultOperands;

    auto moduleOp =
        importOp.getOperation()->getParentOfType<IREE::VM::ModuleOp>();

    int importOrdinal = importOp.getOrdinal()->getZExtValue();

    auto funcOp = op->getParentOfType<mlir::func::FuncOp>();
    BlockArgument stackArg = funcOp.getArgument(0);
    BlockArgument stateArg = funcOp.getArgument(2);

    auto imports = emitc_builders::structPtrMember(
        rewriter, loc,
        /*type=*/
        emitc::PointerType::get(
            emitc::OpaqueType::get(ctx, "iree_vm_function_t")),
        /*memberName=*/"imports",
        /*operand=*/stateArg);

    auto import = emitc_builders::arrayElementAddress(
        rewriter, loc,
        /*type=*/
        emitc::PointerType::get(
            emitc::OpaqueType::get(ctx, "iree_vm_function_t")),
        /*index=*/rewriter.getUI32IntegerAttr(importOrdinal),
        /*operand=*/imports);

    updatedOperands = {stackArg, import};

    Optional<std::string> funcName;
    if (auto variadicOp = dyn_cast<IREE::VM::CallVariadicOp>(op)) {
      funcName = buildVariadicFunctionName(moduleOp, importOp,
                                           variadicOp.getSegmentSizes());
    } else {
      funcName = buildFunctionName(moduleOp, importOp);
    }

    if (failed(updateOperands(op, importOp, rewriter, updatedOperands,
                              resultOperands))) {
      return failure();
    }

    if (!funcName.has_value())
      return op->emitError() << "Couldn't build name to imported function";

    auto callee = moduleOp.lookupSymbol<mlir::func::FuncOp>(funcName.value());
    if (callee == nullptr) {
      return op->emitError()
             << "Couldn't find function with name `" << funcName.value() << "`";
    }

    returnIfError(
        rewriter, loc, callee, updatedOperands,
        *this->template getTypeConverter<IREE::VM::EmitCTypeConverter>());

    if (failed(updateResults(op, rewriter, resultOperands))) {
      return failure();
    }

    rewriter.eraseOp(op);

    return success();
  }

  LogicalResult updateOperands(Operation *op, IREE::VM::ImportOp importOp,
                               ConversionPatternRewriter &rewriter,
                               SmallVector<Value, 4> &updatedOperands,
                               SmallVector<Value, 4> &resultOperands) const {
    auto ctx = op->getContext();
    auto loc = op->getLoc();

    OperandRange operands = op->getOperands();
    IREE::VM::EmitCTypeConverter *typeConverter =
        this->template getTypeConverter<IREE::VM::EmitCTypeConverter>();

    int operandIndex = 0;
    int numInputs =
        importOp ? importOp.getFunctionType().getNumInputs() : operands.size();
    for (int i = 0; i < numInputs; i++) {
      if (importOp && importOp.isFuncArgumentVariadic(i)) {
        assert(isa<IREE::VM::CallVariadicOp>(op));
        auto variadicCallOp = cast<IREE::VM::CallVariadicOp>(op);
        APInt segment = *(variadicCallOp.getSegmentSizes().begin() + i);
        int64_t size = segment.getSExtValue();

        Value segmentSize = rewriter
                                .create<emitc::ConstantOp>(
                                    /*location=*/loc,
                                    /*resultType=*/rewriter.getI32Type(),
                                    /*value=*/rewriter.getI32IntegerAttr(size))
                                .getResult();
        updatedOperands.push_back(segmentSize);

        Type type = importOp.getFunctionType().getInput(i);
        int numOps = type.isa<TupleType>() ? type.cast<TupleType>().size() : 1;
        for (int i = 0; i < size; i++) {
          for (int j = 0; j < numOps; j++) {
            FailureOr<Value> updatedOperand =
                updateOperand(operands[operandIndex], rewriter, loc);
            if (failed(updatedOperand)) {
              return failure();
            }
            updatedOperands.push_back(updatedOperand.value());
            operandIndex++;
          }
        }
      } else {
        FailureOr<Value> updatedOperand =
            updateOperand(operands[operandIndex], rewriter, loc);
        if (failed(updatedOperand)) {
          return failure();
        }
        updatedOperands.push_back(updatedOperand.value());
        operandIndex++;
      }
    }

    // Create a variable for every result and a pointer to it as output
    // parameter to the call.
    for (OpResult result : op->getResults()) {
      if (result.getType().isa<IREE::VM::RefType>()) {
        Optional<Value> ref = typeConverter->materializeRef(result);

        if (!ref.has_value()) {
          return op->emitError() << "local ref not found";
        }

        resultOperands.push_back(ref.value());
        updatedOperands.push_back(ref.value());
      } else {
        auto resultOp = rewriter.create<emitc::VariableOp>(
            /*location=*/loc,
            /*resultType=*/result.getType(),
            /*value=*/emitc::OpaqueAttr::get(ctx, ""));

        Value resultPtr =
            emitc_builders::addressOf(rewriter, loc, resultOp.getResult());

        resultOperands.push_back(resultOp.getResult());
        updatedOperands.push_back(resultPtr);
      }
    }
    return success();
  }

  FailureOr<Value> updateOperand(Value operand, OpBuilder &builder,
                                 Location loc) const {
    auto ctx = builder.getContext();

    IREE::VM::EmitCTypeConverter *typeConverter =
        this->template getTypeConverter<IREE::VM::EmitCTypeConverter>();

    assert(operand.getType() != emitc::PointerType::get(emitc::OpaqueType::get(
                                    ctx, "iree_vm_ref_t")));
    if (!operand.getType().isa<IREE::VM::RefType>()) {
      return operand;
    }

    Optional<Value> operandRef = typeConverter->materializeRef(operand);

    if (!operandRef.has_value()) {
      return emitError(loc) << "local ref not found";
    }

    auto refOp = builder.create<emitc::VariableOp>(
        /*location=*/loc,
        /*resultType=*/emitc::OpaqueType::get(ctx, "iree_vm_ref_t"),
        /*value=*/emitc::OpaqueAttr::get(ctx, ""));

    Value refPtr = emitc_builders::addressOf(builder, loc, refOp.getResult());

    if (failed(clearStruct(builder, refPtr))) {
      return failure();
    }

    builder.create<emitc::CallOp>(
        /*location=*/loc,
        /*type=*/TypeRange{},
        /*callee=*/StringAttr::get(ctx, "iree_vm_ref_assign"),
        /*args=*/ArrayAttr{},
        /*templateArgs=*/ArrayAttr{},
        /*operands=*/
        ArrayRef<Value>{operandRef.value(), refPtr});

    return refPtr;
  }

  LogicalResult updateResults(Operation *op,
                              ConversionPatternRewriter &rewriter,
                              SmallVector<Value, 4> &resultOperands) const {
    for (auto &pair : llvm::enumerate(op->getResults())) {
      size_t index = pair.index();
      OpResult result = pair.value();

      if (!result.getType().isa<IREE::VM::RefType>()) {
        result.replaceAllUsesWith(resultOperands[index]);
      }
    }
    return success();
  }
};

template <typename CmpOpTy>
class CompareRefOpConversion : public OpConversionPattern<CmpOpTy> {
  using Adaptor = typename CmpOpTy::Adaptor;
  using OpConversionPattern<CmpOpTy>::OpConversionPattern;

 public:
  CompareRefOpConversion(TypeConverter &typeConverter, MLIRContext *context,
                         StringRef funcName)
      : OpConversionPattern<CmpOpTy>(typeConverter, context),
        funcName(funcName) {}

 private:
  LogicalResult matchAndRewrite(
      CmpOpTy cmpOp, Adaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto ctx = cmpOp.getContext();
    auto loc = cmpOp.getLoc();

    auto funcOp =
        cmpOp.getOperation()->template getParentOfType<mlir::func::FuncOp>();
    IREE::VM::EmitCTypeConverter *typeConverter =
        this->template getTypeConverter<IREE::VM::EmitCTypeConverter>();

    auto vmAnalysis = typeConverter->lookupAnalysis(funcOp);
    if (failed(vmAnalysis)) {
      return cmpOp.emitError() << "parent func op not found in cache.";
    }

    bool moveLhs =
        vmAnalysis.value().get().isMove(cmpOp.getLhs(), cmpOp.getOperation());
    bool moveRhs =
        vmAnalysis.value().get().isMove(cmpOp.getRhs(), cmpOp.getOperation());

    Optional<Value> refLhs = typeConverter->materializeRef(cmpOp.getLhs());

    if (!refLhs.has_value()) {
      return cmpOp.emitError() << "local ref not found";
    }

    Optional<Value> refRhs = typeConverter->materializeRef(cmpOp.getRhs());

    if (!refRhs.has_value()) {
      return cmpOp.emitError() << "local ref not found";
    }

    rewriter.replaceOpWithNewOp<emitc::CallOp>(
        /*op=*/cmpOp,
        /*type=*/cmpOp.getType(),
        /*callee=*/StringAttr::get(ctx, funcName),
        /*args=*/ArrayAttr{},
        /*templateArgs=*/ArrayAttr{},
        /*operands=*/ArrayRef<Value>{refLhs.value(), refRhs.value()});

    if (moveLhs) {
      emitc_builders::ireeVmRefRelease(rewriter, loc, refLhs.value());
    }

    // NOTE: If lhs and rhs alias we call release twice on the same
    // argument.
    if (moveRhs) {
      emitc_builders::ireeVmRefRelease(rewriter, loc, refRhs.value());
    }

    return success();
  }

  StringRef funcName;
};

class CompareRefNotZeroOpConversion
    : public OpConversionPattern<IREE::VM::CmpNZRefOp> {
  using Adaptor = IREE::VM::CmpNZRefOp::Adaptor;
  using OpConversionPattern<IREE::VM::CmpNZRefOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      IREE::VM::CmpNZRefOp cmpOp, Adaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto ctx = cmpOp.getContext();
    auto loc = cmpOp.getLoc();

    auto funcOp = cmpOp.getOperation()->getParentOfType<mlir::func::FuncOp>();

    IREE::VM::EmitCTypeConverter *typeConverter =
        this->template getTypeConverter<IREE::VM::EmitCTypeConverter>();

    auto vmAnalysis = typeConverter->lookupAnalysis(funcOp);
    if (failed(vmAnalysis)) {
      return cmpOp.emitError() << "parent func op not found in cache.";
    }

    bool move = vmAnalysis.value().get().isMove(cmpOp.getOperand(),
                                                cmpOp.getOperation());

    Optional<Value> ref = typeConverter->materializeRef(cmpOp.getOperand());

    if (!ref.has_value()) {
      return cmpOp.emitError() << "local ref not found";
    }

    rewriter.replaceOpWithNewOp<emitc::CallOp>(
        /*op=*/cmpOp,
        /*type=*/cmpOp.getType(),
        /*callee=*/StringAttr::get(ctx, "vm_cmp_nz_ref"),
        /*args=*/ArrayAttr{},
        /*templateArgs=*/ArrayAttr{},
        /*operands=*/ArrayRef<Value>{ref.value()});

    if (move) {
      emitc_builders::ireeVmRefRelease(rewriter, loc, ref.value());
    }

    return success();
  }
};

template <typename ConstOpTy>
class ConstOpConversion : public OpConversionPattern<ConstOpTy> {
  using Adaptor = typename ConstOpTy::Adaptor;
  using OpConversionPattern<ConstOpTy>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      ConstOpTy constOp, Adaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<emitc::ConstantOp>(constOp, constOp.getType(),
                                                   constOp.getValue());
    return success();
  }
};

template <typename ConstZeroOpTy>
class ConstZeroOpConversion : public OpConversionPattern<ConstZeroOpTy> {
  using Adaptor = typename ConstZeroOpTy::Adaptor;
  using OpConversionPattern<ConstZeroOpTy>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      ConstZeroOpTy constZeroOp, Adaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto type = constZeroOp.getType();

    Attribute value = rewriter.getZeroAttr(type);

    rewriter.replaceOpWithNewOp<emitc::ConstantOp>(constZeroOp, type, value);
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
    auto loc = constRefZeroOp.getLoc();

    IREE::VM::EmitCTypeConverter *typeConverter =
        getTypeConverter<IREE::VM::EmitCTypeConverter>();

    Optional<Value> ref =
        typeConverter->materializeRef(constRefZeroOp.getResult());

    if (!ref.has_value()) {
      return constRefZeroOp.emitError() << "local ref not found";
    }

    emitc_builders::ireeVmRefRelease(rewriter, loc, ref.value());

    rewriter.replaceOp(constRefZeroOp, ref.value());

    return success();
  }
};

class ConstRefRodataOpConversion
    : public OpConversionPattern<IREE::VM::ConstRefRodataOp> {
  using Adaptor = IREE::VM::ConstRefRodataOp::Adaptor;
  using OpConversionPattern<IREE::VM::ConstRefRodataOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      IREE::VM::ConstRefRodataOp constRefRodataOp, Adaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto ctx = constRefRodataOp.getContext();
    auto loc = constRefRodataOp.getLoc();

    auto rodataOp = lookupSymbolRef<IREE::VM::RodataOp>(
        constRefRodataOp.getOperation(), "rodata");
    if (!rodataOp) {
      return constRefRodataOp.emitError() << "Unable to find RodataOp";
    }

    auto funcOp =
        constRefRodataOp.getOperation()->getParentOfType<mlir::func::FuncOp>();

    BlockArgument stateArg = funcOp.getArgument(2);
    auto rodataBuffersPtr = emitc_builders::structPtrMember(
        rewriter, loc,
        /*type=*/
        emitc::PointerType::get(
            emitc::OpaqueType::get(ctx, "iree_vm_buffer_t")),
        /*memberName=*/"rodata_buffers",
        /*operand=*/stateArg);

    auto byteBufferPtrOp = emitc_builders::arrayElementAddress(
        rewriter, loc,
        /*type=*/
        emitc::PointerType::get(
            emitc::OpaqueType::get(ctx, "iree_vm_buffer_t")),
        /*index=*/
        rewriter.getUI32IntegerAttr(
            static_cast<uint32_t>(rodataOp.getOrdinal()->getZExtValue())),
        /*operand=*/rodataBuffersPtr);

    auto typeIdOp = rewriter.create<emitc::CallOp>(
        /*location=*/loc,
        /*type=*/emitc::OpaqueType::get(ctx, "iree_vm_ref_type_t"),
        /*callee=*/StringAttr::get(ctx, "iree_vm_buffer_type_id"),
        /*args=*/ArrayAttr{},
        /*templateArgs=*/ArrayAttr{},
        /*operands=*/ArrayRef<Value>{});

    IREE::VM::EmitCTypeConverter *typeConverter =
        this->template getTypeConverter<IREE::VM::EmitCTypeConverter>();

    Optional<Value> ref =
        typeConverter->materializeRef(constRefRodataOp.getResult());

    if (!ref.has_value()) {
      return constRefRodataOp.emitError() << "local ref not found";
    }

    returnIfError(
        /*rewriter=*/rewriter,
        /*location=*/loc,
        /*callee=*/StringAttr::get(ctx, "iree_vm_ref_wrap_retain"),
        /*args=*/ArrayAttr{},
        /*operands=*/
        ArrayRef<Value>{byteBufferPtrOp, typeIdOp.getResult(0), ref.value()},
        /*typeConverter=*/*typeConverter);

    rewriter.replaceOp(constRefRodataOp, ref.value());

    return success();
  }
};

class BranchOpConversion : public OpConversionPattern<IREE::VM::BranchOp> {
  using Adaptor = IREE::VM::BranchOp::Adaptor;
  using OpConversionPattern<IREE::VM::BranchOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      IREE::VM::BranchOp op, Adaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    assert(op.getOperands().size() == adaptor.getOperands().size());

    auto isNotRefOperand = [](Value operand) {
      return !operand.getType().isa<IREE::VM::RefType>();
    };

    SmallVector<Value> nonRefOperands;
    for (Value operand : op.getOperands()) {
      if (isNotRefOperand(operand)) {
        nonRefOperands.push_back(operand);
      }
    }

    Block *dest = op.getDest();

    // If we don't have ref block arguments, we can convert the operation
    // directly.
    if (adaptor.getOperands().size() == nonRefOperands.size()) {
      rewriter.replaceOpWithNewOp<mlir::cf::BranchOp>(op, op.getDest(),
                                                      op.getOperands());
      return success();
    }

    auto funcOp = op.getOperation()->getParentOfType<mlir::func::FuncOp>();

    IREE::VM::EmitCTypeConverter *typeConverter =
        this->template getTypeConverter<IREE::VM::EmitCTypeConverter>();

    auto vmAnalysis = typeConverter->lookupAnalysis(funcOp);
    if (failed(vmAnalysis)) {
      return op->emitError() << "parent func op not found in cache.";
    }

    Block *destDispatch;
    {
      OpBuilder::InsertionGuard guard(rewriter);
      destDispatch = rewriter.createBlock(dest);

      BlockAndValueMapping refMapping;
      for (auto pair :
           llvm::zip_equal(op.getOperands(), dest->getArguments())) {
        Value operand = std::get<0>(pair);
        BlockArgument blockArg = std::get<1>(pair);

        if (isNotRefOperand(operand)) {
          continue;
        }

        assert(operand.getType().isa<IREE::VM::RefType>());
        assert(blockArg.getType().isa<IREE::VM::RefType>());

        Optional<Value> operandRef = typeConverter->materializeRef(operand);
        Optional<Value> blockArgRef = typeConverter->materializeRef(blockArg);

        if (!operandRef.has_value()) {
          return op.emitError() << "local ref not found";
        }
        if (!blockArgRef.has_value()) {
          return op.emitError() << "local ref not found";
        }

        refMapping.map(operandRef.value(), blockArgRef.value());
      }
      if (failed(retainOrMoveRefs(rewriter, loc, refMapping,
                                  /*isMove=*/false))) {
        return op.emitError() << "moving of multiple refs failed";
      }
      rewriter.create<mlir::cf::BranchOp>(loc, op.getDest(), nonRefOperands);
    }

    rewriter.replaceOpWithNewOp<mlir::cf::BranchOp>(op, destDispatch);

    return success();
  }
};

// Basic block arguments are emitted as variable assignments in EmitC. Because
// of that we need to treat ref operands separately here. We remove ref
// arguments from the basic blocks and use the ref C API to set the ref
// variables. The generated IR looks roughly as follows:

// clang-format off
// vm.cond_br %cond, ^bb1(%ref : !vm.ref<?>, %int : i32), ^bb2(%ref : !vm.ref<?>, %int : i32)
// ^bb1(%ref_arg_1 : !vm.ref<?>, %int_arg : i32):
//   ...
// ^bb2(%ref_arg_2 : !vm.ref<?>, %int_arg : i32):
//   ...
// =>
// cond_br %cond, ^bb1_dispatch, ^bb2_dispatch
// ^bb1_dispatch:
//   // populate the variable corresponding to ordinal(%ref_arg_1)
//   br ^bb1(%int : i32)
// ^bb2_dispatch:
//   // populate the variable corresponding to ordinal(%ref_arg_2)
//   br ^bb2(%int : i32)
// ^bb1(%int_arg : i32):
//   ...
// ^bb2(%int_arg : i32):
//   ...
// clang-format on
class CondBranchOpConversion
    : public OpConversionPattern<IREE::VM::CondBranchOp> {
  using Adaptor = IREE::VM::CondBranchOp::Adaptor;
  using OpConversionPattern<IREE::VM::CondBranchOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      IREE::VM::CondBranchOp op, Adaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    assert(op.getOperands().size() == adaptor.getOperands().size());

    auto isNotRefOperand = [](Value operand) {
      return !operand.getType().isa<IREE::VM::RefType>();
    };

    SmallVector<Value> nonRefOperands;
    for (Value operand : op.getOperands()) {
      if (isNotRefOperand(operand)) {
        nonRefOperands.push_back(operand);
      }
    }

    Block *trueDest = op.getTrueDest();
    Block *falseDest = op.getFalseDest();

    Type boolType = rewriter.getI1Type();

    auto condition = rewriter.create<IREE::VM::CmpNZI32Op>(
        loc, rewriter.getI32Type(), op.getCondition());
    auto conditionI1 = rewriter.create<emitc::CastOp>(
        /*location=*/loc,
        /*type=*/boolType,
        /*operand=*/condition.getResult());

    // If we don't have ref block arguments, we can convert the operation
    // directly.
    if (adaptor.getOperands().size() == nonRefOperands.size()) {
      rewriter.replaceOpWithNewOp<mlir::cf::CondBranchOp>(
          op, conditionI1.getResult(), op.getTrueDest(), op.getTrueOperands(),
          op.getFalseDest(), op.getFalseOperands());
      return success();
    }

    auto funcOp = op.getOperation()->getParentOfType<mlir::func::FuncOp>();
    IREE::VM::EmitCTypeConverter *typeConverter =
        getTypeConverter<IREE::VM::EmitCTypeConverter>();

    auto vmAnalysis = typeConverter->lookupAnalysis(funcOp);
    if (failed(vmAnalysis)) {
      return op->emitError() << "parent func op not found in cache.";
    }

    Block *trueDestDispatch;
    {
      OpBuilder::InsertionGuard guard(rewriter);
      trueDestDispatch = rewriter.createBlock(trueDest);

      // Let the BranchOpConversion handle ref block arguments.
      rewriter.create<IREE::VM::BranchOp>(loc, op.getTrueDest(),
                                          op.getTrueOperands());
    }

    Block *falseDestDispatch;
    {
      OpBuilder::InsertionGuard guard(rewriter);
      falseDestDispatch = rewriter.createBlock(falseDest);

      // Let the BranchOpConversion handle ref block arguments.
      rewriter.create<IREE::VM::BranchOp>(loc, op.getFalseDest(),
                                          op.getFalseOperands());
    }

    rewriter.replaceOpWithNewOp<mlir::cf::CondBranchOp>(
        op, conditionI1.getResult(), trueDestDispatch, falseDestDispatch);

    return success();
  }
};

class ReturnOpConversion : public OpConversionPattern<IREE::VM::ReturnOp> {
  using Adaptor = IREE::VM::ReturnOp::Adaptor;
  using OpConversionPattern<IREE::VM::ReturnOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      IREE::VM::ReturnOp op, Adaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto ctx = op.getContext();
    auto loc = op.getLoc();

    auto funcOp = op.getOperation()->getParentOfType<mlir::func::FuncOp>();
    IREE::VM::EmitCTypeConverter *typeConverter =
        getTypeConverter<IREE::VM::EmitCTypeConverter>();

    // The result variables are the last N arguments of the function.
    unsigned int firstOutputArgumentIndex =
        funcOp.getNumArguments() - op.getOperands().size();

    BlockAndValueMapping refMapping;
    for (auto &pair : llvm::enumerate(op.getOperands())) {
      Value operand = pair.value();
      size_t index = pair.index();

      unsigned int argumentIndex = firstOutputArgumentIndex + index;
      BlockArgument resultArgument = funcOp.getArgument(argumentIndex);

      if (operand.getType().isa<IREE::VM::RefType>()) {
        assert(operand.getType() !=
               emitc::PointerType::get(
                   emitc::OpaqueType::get(ctx, "iree_vm_ref_t")));

        Optional<Value> operandRef = typeConverter->materializeRef(operand);

        if (!operandRef.has_value()) {
          return op->emitError() << "local ref not found";
        }
        refMapping.map(operandRef.value(), resultArgument);
      } else {
        rewriter.create<emitc::CallOp>(
            /*location=*/loc,
            /*type=*/TypeRange{},
            /*callee=*/StringAttr::get(ctx, "EMITC_DEREF_ASSIGN_VALUE"),
            /*args=*/ArrayAttr{},
            /*templateArgs=*/ArrayAttr{},
            /*operands=*/ArrayRef<Value>{resultArgument, operand});
      }
    }

    if (failed(retainOrMoveRefs(rewriter, loc, refMapping, /*isMove=*/true))) {
      return op.emitError() << "moving of multiple refs failed";
    }
    releaseRefs(rewriter, loc, funcOp, *typeConverter);

    auto status = emitc_builders::ireeOkStatus(rewriter, loc);

    rewriter.replaceOpWithNewOp<mlir::func::ReturnOp>(op, status);

    return success();
  }
};

class ImportResolvedOpConversion
    : public OpConversionPattern<IREE::VM::ImportResolvedOp> {
  using Adaptor = IREE::VM::ImportResolvedOp::Adaptor;
  using OpConversionPattern<IREE::VM::ImportResolvedOp>::OpConversionPattern;

 public:
  ImportResolvedOpConversion(TypeConverter &typeConverter, MLIRContext *context)
      : OpConversionPattern(typeConverter, context) {}

 private:
  LogicalResult matchAndRewrite(
      IREE::VM::ImportResolvedOp op, Adaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto ctx = op.getContext();
    auto loc = op.getLoc();

    IREE::VM::ImportOp importOp =
        lookupSymbolRef<IREE::VM::ImportOp>(op.getOperation(), "import");
    int importOrdinal = importOp.getOrdinal()->getZExtValue();

    auto funcOp = op->getParentOfType<mlir::func::FuncOp>();
    BlockArgument stateArg = funcOp.getArgument(2);

    auto imports = emitc_builders::structPtrMember(
        rewriter, loc,
        /*type=*/
        emitc::PointerType::get(
            emitc::OpaqueType::get(ctx, "iree_vm_function_t")),
        /*memberName=*/"imports",
        /*operand=*/stateArg);

    auto import = emitc_builders::arrayElementAddress(
        rewriter, loc,
        /*type=*/
        emitc::PointerType::get(
            emitc::OpaqueType::get(ctx, "iree_vm_function_t")),
        /*index=*/rewriter.getUI32IntegerAttr(importOrdinal),
        /*operand=*/imports);

    // (iree_vm_function_t*)->module
    auto importModule = emitc_builders::structPtrMember(
        rewriter, loc,
        /*type=*/
        emitc::PointerType::get(
            emitc::OpaqueType::get(ctx, "iree_vm_module_t")),
        /*memberName=*/"module",
        /*operand=*/import);

    Type boolType = rewriter.getIntegerType(1);
    auto conditionI1 = rewriter
                           .create<emitc::CallOp>(
                               /*location=*/loc,
                               /*type=*/boolType,
                               /*callee=*/StringAttr::get(ctx, "EMITC_NOT"),
                               /*args=*/
                               ArrayAttr::get(ctx, {rewriter.getIndexAttr(0)}),
                               /*templateArgs=*/ArrayAttr{},
                               /*operands=*/ArrayRef<Value>{importModule})
                           .getResult(0);
    auto invConditionI1 =
        rewriter
            .create<emitc::CallOp>(
                /*location=*/loc,
                /*type=*/boolType,
                /*callee=*/StringAttr::get(ctx, "EMITC_NOT"),
                /*args=*/
                ArrayAttr::get(ctx, {rewriter.getIndexAttr(0)}),
                /*templateArgs=*/ArrayAttr{},
                /*operands=*/ArrayRef<Value>{conditionI1})
            .getResult(0);

    auto i32Type = rewriter.getIntegerType(32);
    auto conditionI32 = rewriter.create<emitc::CastOp>(
        /*location=*/loc,
        /*type=*/i32Type,
        /*operand=*/invConditionI1);

    rewriter.replaceOp(op, {conditionI32.getResult()});

    return success();
  }
};

class FailOpConversion : public OpConversionPattern<IREE::VM::FailOp> {
  using Adaptor = IREE::VM::FailOp::Adaptor;
  using OpConversionPattern<IREE::VM::FailOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      IREE::VM::FailOp op, Adaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto ctx = op.getContext();
    auto loc = op.getLoc();

    Block *block = rewriter.getInsertionBlock();
    Region *parentRegion = block->getParent();
    Block *passthroughBlock;
    {
      OpBuilder::InsertionGuard guard(rewriter);
      passthroughBlock =
          rewriter.createBlock(parentRegion, parentRegion->end());

      auto funcOp = op.getOperation()->getParentOfType<mlir::func::FuncOp>();
      IREE::VM::EmitCTypeConverter *typeConverter =
          getTypeConverter<IREE::VM::EmitCTypeConverter>();

      releaseRefs(rewriter, loc, funcOp, *typeConverter);

      auto status = emitc_builders::ireeOkStatus(rewriter, loc);

      rewriter.create<mlir::func::ReturnOp>(loc, status);
    }
    Block *failureBlock;
    {
      OpBuilder::InsertionGuard guard(rewriter);
      failureBlock = rewriter.createBlock(parentRegion, parentRegion->end());

      auto funcOp = op.getOperation()->getParentOfType<mlir::func::FuncOp>();
      IREE::VM::EmitCTypeConverter *typeConverter =
          getTypeConverter<IREE::VM::EmitCTypeConverter>();

      releaseRefs(rewriter, loc, funcOp, *typeConverter);

      std::string message = std::string("\"") +
                            op.getMessage().value_or("").str() +
                            std::string("\"");

      auto messageOp = rewriter.create<emitc::CallOp>(
          /*location=*/loc,
          /*type=*/emitc::OpaqueType::get(ctx, "iree_string_view_t"),
          /*callee=*/StringAttr::get(ctx, "iree_make_cstring_view"),
          /*args=*/
          ArrayAttr::get(ctx, {emitc::OpaqueAttr::get(ctx, message)}),
          /*templateArgs=*/ArrayAttr{},
          /*operands=*/ArrayRef<Value>{});

      auto messageSizeOp = emitc_builders::structMember(
          rewriter, loc,
          /*type=*/emitc::OpaqueType::get(ctx, "iree_host_size_t"),
          /*memberName=*/"size",
          /*operand=*/messageOp.getResult(0));

      auto messageSizeIntOp = rewriter.create<emitc::CastOp>(
          /*location=*/loc,
          /*type=*/rewriter.getIntegerType(32),
          /*operand=*/messageSizeOp);

      auto messageDataOp = emitc_builders::structMember(
          rewriter, loc,
          /*type=*/
          emitc::PointerType::get(emitc::OpaqueType::get(ctx, "const char")),
          /*memberName=*/"data",
          /*operand=*/messageOp.getResult(0));

      auto status = rewriter.create<emitc::CallOp>(
          /*location=*/loc,
          /*type=*/emitc::OpaqueType::get(ctx, "iree_status_t"),
          /*callee=*/StringAttr::get(ctx, "iree_status_allocate_f"),
          /*args=*/
          ArrayAttr::get(
              ctx,
              {emitc::OpaqueAttr::get(ctx, "IREE_STATUS_FAILED_PRECONDITION"),
               emitc::OpaqueAttr::get(ctx, "\"<vm>\""),
               rewriter.getI32IntegerAttr(0),
               emitc::OpaqueAttr::get(ctx, "\"%.*s\""),
               rewriter.getIndexAttr(0), rewriter.getIndexAttr(1)}),
          /*templateArgs=*/ArrayAttr{},
          /*operands=*/
          ArrayRef<Value>{messageSizeIntOp.getResult(), messageDataOp});

      rewriter.create<mlir::func::ReturnOp>(loc, status.getResult(0));
    }

    Type boolType = rewriter.getIntegerType(1);
    auto condition = rewriter.create<emitc::CastOp>(
        /*location=*/loc,
        /*type=*/boolType,
        /*operand=*/op.getStatus());

    rewriter.replaceOpWithNewOp<mlir::cf::CondBranchOp>(
        op, condition.getResult(), failureBlock, passthroughBlock);

    return success();
  }
};

template <typename LoadOpTy, typename GlobalOpTy>
class GlobalLoadOpConversion : public OpConversionPattern<LoadOpTy> {
  using Adaptor = typename LoadOpTy::Adaptor;
  using OpConversionPattern<LoadOpTy>::OpConversionPattern;

 public:
  GlobalLoadOpConversion(TypeConverter &typeConverter, MLIRContext *context,
                         StringRef funcName)
      : OpConversionPattern<LoadOpTy>(typeConverter, context),
        funcName(funcName) {}

 private:
  LogicalResult matchAndRewrite(
      LoadOpTy loadOp, Adaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto ctx = loadOp.getContext();
    auto loc = loadOp.getLoc();

    GlobalOpTy globalOp =
        lookupSymbolRef<GlobalOpTy>(loadOp.getOperation(), "global");
    if (!globalOp) {
      return loadOp.emitError() << "Unable to find GlobalOp";
    }

    auto funcOp =
        loadOp.getOperation()->template getParentOfType<mlir::func::FuncOp>();

    BlockArgument stateArg = funcOp.getArgument(2);
    auto rwDataPtr = emitc_builders::structPtrMember(
        rewriter, loc,
        /*type=*/emitc::PointerType::get(rewriter.getIntegerType(8, false)),
        /*memberName=*/"rwdata",
        /*operand=*/stateArg);

    rewriter.replaceOpWithNewOp<emitc::CallOp>(
        /*op=*/loadOp,
        /*type=*/loadOp.getOperation()->getResultTypes(),
        /*callee=*/StringAttr::get(ctx, funcName),
        /*args=*/
        rewriter.getArrayAttr(
            {rewriter.getIndexAttr(0),
             rewriter.getUI32IntegerAttr(static_cast<uint32_t>(
                 globalOp.getOrdinal()->getZExtValue()))}),
        /*templateArgs=*/ArrayAttr{},
        /*operands=*/ArrayRef<Value>{rwDataPtr});

    return success();
  }

  StringRef funcName;
};

template <typename LoadStoreOpTy>
class GlobalLoadStoreRefOpConversion
    : public OpConversionPattern<LoadStoreOpTy> {
  using Adaptor = typename LoadStoreOpTy::Adaptor;
  using OpConversionPattern<LoadStoreOpTy>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      LoadStoreOpTy op, Adaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    if (isa<IREE::VM::GlobalLoadRefOp>(op)) {
      return rewriteOp(op.getOperation(), adaptor, rewriter, true);
    } else if (isa<IREE::VM::GlobalStoreRefOp>(op)) {
      return rewriteOp(op.getOperation(), adaptor, rewriter, false);
    }

    return op.emitError() << "op must be one of `vm.global.load.ref` or "
                             "`vm.global.store.ref`";
  }

  LogicalResult rewriteOp(Operation *op, Adaptor adaptor,
                          ConversionPatternRewriter &rewriter,
                          bool isLoad) const {
    auto ctx = op->getContext();
    auto loc = op->getLoc();

    IREE::VM::GlobalRefOp globalOp =
        lookupSymbolRef<IREE::VM::GlobalRefOp>(op, "global");
    if (!globalOp) {
      return op->emitError() << "Unable to find GlobalOp";
    }

    auto globalOrdinal = globalOp.getOrdinal()->getZExtValue();

    auto funcOp = op->getParentOfType<mlir::func::FuncOp>();
    IREE::VM::EmitCTypeConverter *typeConverter =
        this->template getTypeConverter<IREE::VM::EmitCTypeConverter>();

    auto vmAnalysis = typeConverter->lookupAnalysis(funcOp);
    if (failed(vmAnalysis)) {
      return op->emitError() << "parent func op not found in cache.";
    }

    Value localValue = isLoad ? op->getResult(0) : op->getOperand(0);

    Optional<Value> localRef = typeConverter->materializeRef(localValue);

    if (!localRef.has_value()) {
      return op->emitError() << "local ref not found";
    }

    BlockArgument stateArg = funcOp.getArgument(2);
    auto refs = emitc_builders::structPtrMember(
        rewriter, loc,
        /*type=*/
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, "iree_vm_ref_t")),
        /*memberName=*/"refs",
        /*operand=*/stateArg);

    auto stateRef = emitc_builders::arrayElementAddress(
        rewriter, loc,
        /*type=*/
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, "iree_vm_ref_t")),
        /*index=*/rewriter.getUI32IntegerAttr(globalOrdinal),
        /*operand=*/refs);

    Type elementType = localValue.getType();

    auto elementTypePtrOp = createVmTypeDefPtr(rewriter, op, elementType);

    if (!elementTypePtrOp.has_value()) {
      return op->emitError() << "generating iree_vm_type_def_t* failed";
    }

    auto typedefRefType = emitc_builders::structPtrMember(
        rewriter, loc,
        /*type=*/emitc::OpaqueType::get(ctx, "iree_vm_ref_type_t"),
        /*memberName=*/"ref_type",
        /*operand=*/elementTypePtrOp.value().getResult());

    Value srcRef = isLoad ? stateRef : localRef.value();
    Value destRef = isLoad ? localRef.value() : stateRef;

    bool move = vmAnalysis.value().get().isMove(localValue, op);

    returnIfError(
        /*rewriter=*/rewriter,
        /*location=*/loc,
        /*callee=*/StringAttr::get(ctx, "iree_vm_ref_retain_or_move_checked"),
        /*args=*/
        ArrayAttr::get(ctx,
                       {rewriter.getBoolAttr(move), rewriter.getIndexAttr(0),
                        rewriter.getIndexAttr(1), rewriter.getIndexAttr(2)}),
        /*operands=*/ArrayRef<Value>{srcRef, typedefRefType, destRef},
        /*typeConverter=*/*typeConverter);

    if (isLoad) {
      rewriter.replaceOp(op, localRef.value());
    } else {
      rewriter.eraseOp(op);
    }

    return success();
  }
};

template <typename StoreOpTy, typename GlobalOpTy>
class GlobalStoreOpConversion : public OpConversionPattern<StoreOpTy> {
  using Adaptor = typename StoreOpTy::Adaptor;
  using OpConversionPattern<StoreOpTy>::OpConversionPattern;

 public:
  GlobalStoreOpConversion(TypeConverter &typeConverter, MLIRContext *context,
                          StringRef funcName)
      : OpConversionPattern<StoreOpTy>(typeConverter, context),
        funcName(funcName) {}

 private:
  LogicalResult matchAndRewrite(
      StoreOpTy storeOp, Adaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto ctx = storeOp.getContext();
    auto loc = storeOp.getLoc();

    GlobalOpTy globalOp =
        lookupSymbolRef<GlobalOpTy>(storeOp.getOperation(), "global");
    if (!globalOp) {
      return storeOp.emitError() << "Unable to find GlobalOp";
    }

    auto funcOp =
        storeOp.getOperation()->template getParentOfType<mlir::func::FuncOp>();

    BlockArgument stateArg = funcOp.getArgument(2);
    auto rwDataPtr = emitc_builders::structPtrMember(
        rewriter, loc,
        /*type=*/emitc::PointerType::get(rewriter.getIntegerType(8, false)),
        /*memberName=*/"rwdata",
        /*operand=*/stateArg);

    rewriter.replaceOpWithNewOp<emitc::CallOp>(
        /*op=*/storeOp,
        /*type=*/storeOp.getOperation()->getResultTypes(),
        /*callee=*/StringAttr::get(ctx, funcName),
        /*args=*/
        rewriter.getArrayAttr(
            {rewriter.getIndexAttr(0),
             rewriter.getUI32IntegerAttr(
                 static_cast<uint32_t>(globalOp.getOrdinal()->getZExtValue())),
             rewriter.getIndexAttr(1)}),
        /*templateArgs=*/ArrayAttr{},
        /*operands=*/
        ArrayRef<Value>{rwDataPtr, storeOp.getValue()});

    return success();
  }

  StringRef funcName;
};

// Convert vm list operations to two emitc calls. The wrapping ref pointer
// is first dereferenced and the result is used as the argument of the
// specified function name.
template <typename SrcOpTy>
class ListOpConversion : public OpConversionPattern<SrcOpTy> {
  using Adaptor = typename SrcOpTy::Adaptor;
  using OpConversionPattern<SrcOpTy>::OpConversionPattern;

 public:
  ListOpConversion(TypeConverter &typeConverter, MLIRContext *context,
                   StringRef funcName, size_t listArgumentIndex, bool failable)
      : OpConversionPattern<SrcOpTy>(typeConverter, context),
        funcName(funcName),
        listArgumentIndex(listArgumentIndex),
        failable(failable) {}

 private:
  LogicalResult matchAndRewrite(
      SrcOpTy op, Adaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto ctx = op.getContext();
    auto loc = op.getLoc();

    IREE::VM::EmitCTypeConverter *typeConverter =
        this->template getTypeConverter<IREE::VM::EmitCTypeConverter>();

    if (listArgumentIndex >= adaptor.getOperands().size()) {
      return op.emitError() << " index for list argument out of range";
    }

    Value listOperand = adaptor.getOperands()[listArgumentIndex];

    Value refValue = emitc_builders::contentsOf(rewriter, loc, listOperand);

    auto listDerefOp = failListNull(
        /*rewriter=*/rewriter,
        /*location=*/loc,
        /*type=*/
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, "iree_vm_list_t")),
        /*callee=*/StringAttr::get(ctx, "iree_vm_list_deref"),
        /*args=*/ArrayAttr{},
        /*operands=*/ArrayRef<Value>{refValue},
        /*typeConverter=*/*typeConverter);

    // Replace the one list argument (which is wrapped in a ref) with the
    // unwrapped list.
    SmallVector<Value, 4> updatedOperands;
    for (auto &operand : llvm::enumerate(adaptor.getOperands())) {
      if (operand.index() == listArgumentIndex) {
        updatedOperands.push_back(listDerefOp.getResult(0));
      } else {
        updatedOperands.push_back(operand.value());
      }
    }

    if (failable) {
      returnIfError(
          /*rewriter=*/rewriter,
          /*location=*/loc,
          /*callee=*/StringAttr::get(ctx, funcName),
          /*args=*/ArrayAttr{},
          /*operands=*/ArrayRef<Value>(updatedOperands),
          /*typeConverter=*/*typeConverter);

      rewriter.replaceOp(op, ArrayRef<Value>{});
    } else {
      rewriter.replaceOpWithNewOp<emitc::CallOp>(
          /*op=*/op,
          /*type=*/op.getOperation()->getResultTypes(),
          /*callee=*/StringAttr::get(ctx, funcName),
          /*args=*/ArrayAttr{},
          /*templateArgs=*/ArrayAttr{},
          /*operands=*/ArrayRef<Value>(updatedOperands));
    }

    return success();
  }

  StringRef funcName;

  // The index of the list argument. This gets replaced in the conversion.
  size_t listArgumentIndex;

  // Whether the function call can fail, i.e. it returns an iree_status_t.
  bool failable;
};

class ListAllocOpConversion
    : public OpConversionPattern<IREE::VM::ListAllocOp> {
  using Adaptor = IREE::VM::ListAllocOp::Adaptor;
  using OpConversionPattern<IREE::VM::ListAllocOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      IREE::VM::ListAllocOp allocOp, Adaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto ctx = allocOp.getContext();
    auto loc = allocOp.getLoc();

    Type convertedType = typeConverter->convertType(allocOp.getType());

    if (!convertedType) {
      return allocOp.emitOpError() << "type conversion failed";
    }

    auto elementType = allocOp.getType()
                           .cast<IREE::VM::RefType>()
                           .getObjectType()
                           .cast<IREE::VM::ListType>()
                           .getElementType();

    Optional<emitc::ApplyOp> elementTypePtrOp =
        createVmTypeDefPtr(rewriter, allocOp.getOperation(), elementType);

    if (!elementTypePtrOp.has_value()) {
      return allocOp.emitError() << "generating iree_vm_type_def_t* failed";
    }

    auto listOp = rewriter.create<emitc::VariableOp>(
        /*location=*/loc,
        /*resultType=*/
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, "iree_vm_list_t")),
        /*value=*/emitc::OpaqueAttr::get(ctx, "NULL"));

    Value listPtr =
        emitc_builders::addressOf(rewriter, loc, listOp.getResult());

    auto funcOp = allocOp.getOperation()->getParentOfType<mlir::func::FuncOp>();
    IREE::VM::EmitCTypeConverter *typeConverter =
        getTypeConverter<IREE::VM::EmitCTypeConverter>();

    BlockArgument stateArg = funcOp.getArgument(2);
    auto allocatorOp = emitc_builders::structPtrMember(
        rewriter, loc,
        /*type=*/emitc::OpaqueType::get(ctx, "iree_allocator_t"),
        /*memberName=*/"allocator",
        /*operand=*/stateArg);

    returnIfError(
        /*rewriter=*/rewriter,
        /*location=*/loc,
        /*callee=*/StringAttr::get(ctx, "iree_vm_list_create"),
        /*args=*/ArrayAttr{},
        /*operands=*/
        ArrayRef<Value>{elementTypePtrOp.value().getResult(),
                        adaptor.getOperands()[0], allocatorOp, listPtr},
        /*typeConverter=*/*typeConverter);

    auto ref = typeConverter->materializeRef(allocOp.getResult());

    if (!ref.has_value()) {
      return allocOp.emitError() << "local ref not found";
    }

    auto refTypeOp = rewriter.create<emitc::CallOp>(
        /*location=*/loc,
        /*type=*/emitc::OpaqueType::get(ctx, "iree_vm_ref_type_t"),
        /*callee=*/StringAttr::get(ctx, "iree_vm_list_type_id"),
        /*args=*/ArrayAttr{},
        /*templateArgs=*/ArrayAttr{},
        /*operands=*/ArrayRef<Value>{});

    returnIfError(
        /*rewriter=*/rewriter,
        /*location=*/loc,
        /*callee=*/StringAttr::get(ctx, "iree_vm_ref_wrap_assign"),
        /*args=*/ArrayAttr{},
        /*operands=*/
        ArrayRef<Value>{listOp.getResult(), refTypeOp.getResult(0),
                        ref.value()},
        /*typeConverter=*/*typeConverter);

    rewriter.replaceOp(allocOp, ref.value());

    return success();
  }
};

template <typename GetOpTy>
class ListGetOpConversion : public OpConversionPattern<GetOpTy> {
  using Adaptor = typename GetOpTy::Adaptor;
  using OpConversionPattern<GetOpTy>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      GetOpTy getOp, Adaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto ctx = getOp.getContext();
    auto loc = getOp.getLoc();

    Optional<StringRef> valueTypeEnum;
    Optional<StringRef> valueExtractor;

    std::tie(valueTypeEnum, valueExtractor) =
        TypeSwitch<Operation *,
                   std::pair<Optional<StringRef>, Optional<StringRef>>>(
            getOp.getOperation())
            .Case<IREE::VM::ListGetI32Op>([&](auto op) {
              return std::make_pair(StringRef("IREE_VM_VALUE_TYPE_I32"),
                                    StringRef("iree_vm_value_get_i32"));
            })
            .template Case<IREE::VM::ListGetI64Op>([&](auto op) {
              return std::make_pair(StringRef("IREE_VM_VALUE_TYPE_I64"),
                                    StringRef("iree_vm_value_get_i64"));
            })
            .Default([](Operation *) {
              return std::make_pair(std::nullopt, std::nullopt);
            });

    if (!valueTypeEnum.has_value() || !valueExtractor.has_value()) {
      return getOp.emitOpError() << "element type not handled";
    }

    auto valueOp = rewriter.create<emitc::VariableOp>(
        /*location=*/loc,
        /*resultType=*/emitc::OpaqueType::get(ctx, "iree_vm_value_t"),
        /*value=*/emitc::OpaqueAttr::get(ctx, ""));

    Value valuePtr =
        emitc_builders::addressOf(rewriter, loc, valueOp.getResult());

    Value refValue =
        emitc_builders::contentsOf(rewriter, loc, adaptor.getOperands()[0]);

    IREE::VM::EmitCTypeConverter *typeConverter =
        this->template getTypeConverter<IREE::VM::EmitCTypeConverter>();

    auto listDerefOp = failListNull(
        /*rewriter=*/rewriter,
        /*location=*/loc,
        /*type=*/
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, "iree_vm_list_t")),
        /*callee=*/StringAttr::get(ctx, "iree_vm_list_deref"),
        /*args=*/ArrayAttr{},
        /*operands=*/ArrayRef<Value>{refValue},
        /*typeConverter=*/*typeConverter);

    returnIfError(
        /*rewriter=*/rewriter,
        /*location=*/loc,
        /*callee=*/StringAttr::get(ctx, "iree_vm_list_get_value_as"),
        /*args=*/
        ArrayAttr::get(ctx, {rewriter.getIndexAttr(0), rewriter.getIndexAttr(1),
                             emitc::OpaqueAttr::get(ctx, valueTypeEnum.value()),
                             rewriter.getIndexAttr(2)}),
        /*operands=*/
        ArrayRef<Value>{listDerefOp.getResult(0), getOp.getIndex(), valuePtr},
        /*typeConverter=*/*typeConverter);

    rewriter.replaceOpWithNewOp<emitc::CallOp>(
        /*op=*/getOp,
        /*type=*/getOp.getType(),
        /*callee=*/StringAttr::get(ctx, valueExtractor.value()),
        /*args=*/ArrayAttr{},
        /*templateArgs=*/ArrayAttr{},
        /*operands=*/ArrayRef<Value>{valuePtr});

    return success();
  }
};

class ListGetRefOpConversion
    : public OpConversionPattern<IREE::VM::ListGetRefOp> {
  using Adaptor = IREE::VM::ListGetRefOp::Adaptor;
  using OpConversionPattern<IREE::VM::ListGetRefOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      IREE::VM::ListGetRefOp getOp, Adaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto ctx = getOp.getContext();
    auto loc = getOp.getLoc();

    Value listRefValue =
        emitc_builders::contentsOf(rewriter, loc, adaptor.getOperands()[0]);

    IREE::VM::EmitCTypeConverter *typeConverter =
        getTypeConverter<IREE::VM::EmitCTypeConverter>();

    auto listDerefOp = failListNull(
        /*rewriter=*/rewriter,
        /*location=*/loc,
        /*type=*/
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, "iree_vm_list_t")),
        /*callee=*/StringAttr::get(ctx, "iree_vm_list_deref"),
        /*args=*/ArrayAttr{},
        /*operands=*/ArrayRef<Value>{listRefValue},
        /*typeConverter=*/*typeConverter);

    auto ref = typeConverter->materializeRef(getOp.getResult());

    if (!ref.has_value()) {
      return getOp.emitError() << "local ref not found";
    }

    returnIfError(
        /*rewriter=*/rewriter,
        /*location=*/loc,
        /*callee=*/StringAttr::get(ctx, "iree_vm_list_get_ref_retain"),
        /*args=*/ArrayAttr{},
        /*operands=*/
        ArrayRef<Value>{listDerefOp.getResult(0), getOp.getIndex(),
                        ref.value()},
        /*typeConverter=*/*typeConverter);

    Type elementType = getOp.getResult().getType();

    auto elementTypePtrOp =
        createVmTypeDefPtr(rewriter, getOp.getOperation(), elementType);

    if (!elementTypePtrOp.has_value()) {
      return getOp.emitError() << "generating iree_vm_type_def_t* failed";
    }

    // Build the following expression:
    // (ref->type != IREE_VM_REF_TYPE_NULL &&
    // (iree_vm_type_def_is_value(type_def) || ref->type !=
    // type_def->ref_type))
    emitc::CallOp invalidType;
    {
      auto refType = emitc_builders::structPtrMember(
          rewriter, loc,
          /*type=*/emitc::OpaqueType::get(ctx, "iree_vm_ref_type_t"),
          /*memberName=*/"type",
          /*operand=*/ref.value());

      auto refTypeNull = rewriter.create<emitc::ConstantOp>(
          /*location=*/loc,
          /*resultType=*/emitc::OpaqueType::get(ctx, "iree_vm_ref_type_t"),
          /*value=*/emitc::OpaqueAttr::get(ctx, "IREE_VM_REF_TYPE_NULL"));

      auto typedefIsValue = rewriter.create<emitc::CallOp>(
          /*location=*/loc,
          /*type=*/rewriter.getI1Type(),
          /*callee=*/StringAttr::get(ctx, "iree_vm_type_def_is_value"),
          /*args=*/ArrayAttr{},
          /*templateArgs=*/ArrayAttr{},
          /*operands=*/
          ArrayRef<Value>{elementTypePtrOp.value().getResult()});

      auto typedefRefType = emitc_builders::structPtrMember(
          rewriter, loc,
          /*type=*/emitc::OpaqueType::get(ctx, "iree_vm_ref_type_t"),
          /*memberName=*/"ref_type",
          /*operand=*/elementTypePtrOp.value().getResult());

      auto refTypeIsNotNull = rewriter.create<emitc::CallOp>(
          /*location=*/loc,
          /*type=*/rewriter.getI1Type(),
          /*callee=*/StringAttr::get(ctx, "EMITC_NE"),
          /*args=*/ArrayAttr{},
          /*templateArgs=*/ArrayAttr{},
          /*operands=*/
          ArrayRef<Value>{refType, refTypeNull.getResult()});

      auto refTypesDontMatch = rewriter.create<emitc::CallOp>(
          /*location=*/loc,
          /*type=*/rewriter.getI1Type(),
          /*callee=*/StringAttr::get(ctx, "EMITC_NE"),
          /*args=*/ArrayAttr{},
          /*templateArgs=*/ArrayAttr{},
          /*operands=*/
          ArrayRef<Value>{refType, typedefRefType});

      auto invalidRefType = rewriter.create<emitc::CallOp>(
          /*location=*/loc,
          /*type=*/rewriter.getI1Type(),
          /*callee=*/StringAttr::get(ctx, "EMITC_OR"),
          /*args=*/ArrayAttr{},
          /*templateArgs=*/ArrayAttr{},
          /*operands=*/
          ArrayRef<Value>{typedefIsValue.getResult(0),
                          refTypesDontMatch.getResult(0)});

      invalidType = rewriter.create<emitc::CallOp>(
          /*location=*/loc,
          /*type=*/rewriter.getI1Type(),
          /*callee=*/StringAttr::get(ctx, "EMITC_AND"),
          /*args=*/ArrayAttr{},
          /*templateArgs=*/ArrayAttr{},
          /*operands=*/
          ArrayRef<Value>{refTypeIsNotNull.getResult(0),
                          invalidRefType.getResult(0)});
    }

    // Start by splitting the block into two. The part before will contain
    // the condition, and the part after will contain the continuation
    // point.
    Block *condBlock = rewriter.getInsertionBlock();
    Block::iterator opPosition = rewriter.getInsertionPoint();
    Block *continuationBlock = condBlock->splitBlock(opPosition);

    // Create a new block for the target of the failure.
    Block *failureBlock;
    {
      OpBuilder::InsertionGuard guard(rewriter);
      Region *parentRegion = condBlock->getParent();
      failureBlock = rewriter.createBlock(parentRegion, parentRegion->end());

      emitc_builders::ireeVmRefRelease(rewriter, loc, ref.value());

      rewriter.create<mlir::cf::BranchOp>(loc, continuationBlock);
    }

    rewriter.setInsertionPointToEnd(condBlock);
    rewriter.create<IREE::VM::CondBranchOp>(loc, invalidType.getResult(0),
                                            failureBlock, continuationBlock);

    rewriter.replaceOp(getOp, ref.value());

    return success();
  }
};

template <typename SetOpTy>
class ListSetOpConversion : public OpConversionPattern<SetOpTy> {
  using Adaptor = typename SetOpTy::Adaptor;
  using OpConversionPattern<SetOpTy>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      SetOpTy setOp, Adaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto ctx = setOp.getContext();
    auto loc = setOp.getLoc();

    Optional<StringRef> valueConstructor =
        TypeSwitch<Operation *, Optional<StringRef>>(setOp.getOperation())
            .Case<IREE::VM::ListSetI32Op>(
                [&](auto op) { return StringRef("iree_vm_value_make_i32"); })
            .template Case<IREE::VM::ListSetI64Op>(
                [&](auto op) { return StringRef("iree_vm_value_make_i64"); })
            .Default([](Operation *) { return std::nullopt; });

    if (!valueConstructor.has_value()) {
      return setOp.emitOpError() << " not handled";
    }

    auto valueOp = rewriter.create<emitc::CallOp>(
        /*location=*/loc,
        /*type=*/emitc::OpaqueType::get(ctx, "iree_vm_value_t"),
        /*callee=*/StringAttr::get(ctx, valueConstructor.value()),
        /*args=*/ArrayAttr{},
        /*templateArgs=*/ArrayAttr{},
        /*operands=*/ArrayRef<Value>{setOp.getValue()});

    Value valuePtr =
        emitc_builders::addressOf(rewriter, loc, valueOp.getResult(0));

    Value refValue =
        emitc_builders::contentsOf(rewriter, loc, adaptor.getOperands()[0]);

    IREE::VM::EmitCTypeConverter *typeConverter =
        this->template getTypeConverter<IREE::VM::EmitCTypeConverter>();

    auto listDerefOp = failListNull(
        /*rewriter=*/rewriter,
        /*location=*/loc,
        /*type=*/
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, "iree_vm_list_t")),
        /*callee=*/StringAttr::get(ctx, "iree_vm_list_deref"),
        /*args=*/ArrayAttr{},
        /*operands=*/ArrayRef<Value>{refValue},
        /*typeConverter=*/*typeConverter);

    returnIfError(
        /*rewriter=*/rewriter,
        /*location=*/loc,
        /*callee=*/StringAttr::get(ctx, "iree_vm_list_set_value"),
        /*args=*/ArrayAttr{},
        /*operands=*/
        ArrayRef<Value>{listDerefOp.getResult(0), setOp.getIndex(), valuePtr},
        /*typeConverter=*/*typeConverter);

    rewriter.eraseOp(setOp);

    return success();
  }
};

class ListSetRefOpConversion
    : public OpConversionPattern<IREE::VM::ListSetRefOp> {
  using Adaptor = IREE::VM::ListSetRefOp::Adaptor;
  using OpConversionPattern<IREE::VM::ListSetRefOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      IREE::VM::ListSetRefOp setOp, Adaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto ctx = setOp.getContext();
    auto loc = setOp.getLoc();

    Value refValue =
        emitc_builders::contentsOf(rewriter, loc, adaptor.getOperands()[0]);

    IREE::VM::EmitCTypeConverter *typeConverter =
        this->template getTypeConverter<IREE::VM::EmitCTypeConverter>();

    auto listDerefOp = failListNull(
        /*rewriter=*/rewriter,
        /*location=*/loc,
        /*type=*/
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, "iree_vm_list_t")),
        /*callee=*/StringAttr::get(ctx, "iree_vm_list_deref"),
        /*args=*/ArrayAttr{},
        /*operands=*/ArrayRef<Value>{refValue},
        /*typeConverter=*/*typeConverter);

    auto funcOp = setOp.getOperation()->getParentOfType<mlir::func::FuncOp>();

    auto vmAnalysis = typeConverter->lookupAnalysis(funcOp);
    if (failed(vmAnalysis)) {
      return setOp.emitError() << "parent func op not found in cache.";
    }
    bool move =
        vmAnalysis.value().get().isMove(setOp.getValue(), setOp.getOperation());

    StringRef callee =
        move ? "iree_vm_list_set_ref_move" : "iree_vm_list_set_ref_retain";

    returnIfError(
        /*rewriter=*/rewriter,
        /*location=*/loc,
        /*callee=*/StringAttr::get(ctx, callee),
        /*args=*/ArrayAttr{},
        /*operands=*/
        ArrayRef<Value>{listDerefOp.getResult(0), setOp.getIndex(),
                        adaptor.getValue()},
        /*typeConverter=*/*typeConverter);

    rewriter.eraseOp(setOp);

    return success();
  }
};
}  // namespace

void populateVMToEmitCPatterns(ConversionTarget &conversionTarget,
                               IREE::VM::EmitCTypeConverter &typeConverter,
                               RewritePatternSet &patterns) {
  auto context = patterns.getContext();
  populateUtilConversionPatterns(context, conversionTarget, typeConverter,
                                 patterns);

  // CFG
  patterns.add<BranchOpConversion>(typeConverter, context);
  patterns.add<CallOpConversion<IREE::VM::CallOp>>(typeConverter, context);
  patterns.add<CallOpConversion<IREE::VM::CallVariadicOp>>(typeConverter,
                                                           context);
  patterns.add<CondBranchOpConversion>(typeConverter, context);
  patterns.add<FailOpConversion>(typeConverter, context);
  patterns.add<FuncOpConversion>(typeConverter, context);
  patterns.add<ExportOpConversion>(typeConverter, context);
  patterns.add<ReturnOpConversion>(typeConverter, context);
  patterns.add<ImportResolvedOpConversion>(typeConverter, context);

  // Globals
  // Globals get packed into the state struct and are referenced by their
  // ordinal only after the conversion.
  patterns.add<DeleteOpConversion<IREE::VM::GlobalI32Op>>(typeConverter,
                                                          context);
  patterns.add<DeleteOpConversion<IREE::VM::GlobalI64Op>>(typeConverter,
                                                          context);
  patterns.add<DeleteOpConversion<IREE::VM::GlobalF32Op>>(typeConverter,
                                                          context);
  patterns.add<DeleteOpConversion<IREE::VM::GlobalRefOp>>(typeConverter,
                                                          context);

  patterns.add<
      GlobalLoadOpConversion<IREE::VM::GlobalLoadI32Op, IREE::VM::GlobalI32Op>>(
      typeConverter, context, "vm_global_load_i32");
  patterns.add<GlobalStoreOpConversion<IREE::VM::GlobalStoreI32Op,
                                       IREE::VM::GlobalI32Op>>(
      typeConverter, context, "vm_global_store_i32");

  patterns.add<GlobalLoadStoreRefOpConversion<IREE::VM::GlobalLoadRefOp>>(
      typeConverter, context);
  patterns.add<GlobalLoadStoreRefOpConversion<IREE::VM::GlobalStoreRefOp>>(
      typeConverter, context);

  // Constants
  patterns.add<ConstOpConversion<IREE::VM::ConstI32Op>>(typeConverter, context);
  patterns.add<ConstZeroOpConversion<IREE::VM::ConstI32ZeroOp>>(typeConverter,
                                                                context);
  patterns.add<ConstRefZeroOpConversion>(typeConverter, context);
  patterns.add<ConstRefRodataOpConversion>(typeConverter, context);

  // List ops
  patterns.add<ListAllocOpConversion>(typeConverter, context);
  patterns.add<ListOpConversion<IREE::VM::ListReserveOp>>(
      typeConverter, context, "iree_vm_list_reserve", 0, true);
  patterns.add<ListOpConversion<IREE::VM::ListResizeOp>>(
      typeConverter, context, "iree_vm_list_resize", 0, true);
  patterns.add<ListOpConversion<IREE::VM::ListSizeOp>>(
      typeConverter, context, "iree_vm_list_size", 0, false);
  patterns.add<ListGetOpConversion<IREE::VM::ListGetI32Op>>(typeConverter,
                                                            context);
  patterns.add<ListGetRefOpConversion>(typeConverter, context);
  patterns.add<ListSetOpConversion<IREE::VM::ListSetI32Op>>(typeConverter,
                                                            context);
  patterns.add<ListSetRefOpConversion>(typeConverter, context);

  // Conditional assignment ops
  patterns.add<GenericOpConversion<IREE::VM::SelectI32Op>>(
      typeConverter, context, "vm_select_i32");

  // Native integer arithmetic ops
  patterns.add<GenericOpConversion<IREE::VM::AddI32Op>>(typeConverter, context,
                                                        "vm_add_i32");
  patterns.add<GenericOpConversion<IREE::VM::SubI32Op>>(typeConverter, context,
                                                        "vm_sub_i32");
  patterns.add<GenericOpConversion<IREE::VM::MulI32Op>>(typeConverter, context,
                                                        "vm_mul_i32");
  patterns.add<GenericOpConversion<IREE::VM::DivI32SOp>>(typeConverter, context,
                                                         "vm_div_i32s");
  patterns.add<GenericOpConversion<IREE::VM::DivI32UOp>>(typeConverter, context,
                                                         "vm_div_i32u");
  patterns.add<GenericOpConversion<IREE::VM::RemI32SOp>>(typeConverter, context,
                                                         "vm_rem_i32s");
  patterns.add<GenericOpConversion<IREE::VM::RemI32UOp>>(typeConverter, context,
                                                         "vm_rem_i32u");
  patterns.add<GenericOpConversion<IREE::VM::FMAI32Op>>(typeConverter, context,
                                                        "vm_fma_i32");
  patterns.add<GenericOpConversion<IREE::VM::AbsI32Op>>(typeConverter, context,
                                                        "vm_abs_i32");
  patterns.add<GenericOpConversion<IREE::VM::NotI32Op>>(typeConverter, context,
                                                        "vm_not_i32");
  patterns.add<GenericOpConversion<IREE::VM::AndI32Op>>(typeConverter, context,
                                                        "vm_and_i32");
  patterns.add<GenericOpConversion<IREE::VM::OrI32Op>>(typeConverter, context,
                                                       "vm_or_i32");
  patterns.add<GenericOpConversion<IREE::VM::XorI32Op>>(typeConverter, context,
                                                        "vm_xor_i32");
  patterns.add<GenericOpConversion<IREE::VM::CtlzI32Op>>(typeConverter, context,
                                                         "vm_ctlz_i32");

  // Casting and type conversion/emulation ops
  patterns.add<GenericOpConversion<IREE::VM::TruncI32I8Op>>(
      typeConverter, context, "vm_trunc_i32i8");
  patterns.add<GenericOpConversion<IREE::VM::TruncI32I16Op>>(
      typeConverter, context, "vm_trunc_i32i16");
  patterns.add<GenericOpConversion<IREE::VM::ExtI8I32SOp>>(
      typeConverter, context, "vm_ext_i8i32s");
  patterns.add<GenericOpConversion<IREE::VM::ExtI8I32UOp>>(
      typeConverter, context, "vm_ext_i8i32u");
  patterns.add<GenericOpConversion<IREE::VM::ExtI16I32SOp>>(
      typeConverter, context, "vm_ext_i16i32s");
  patterns.add<GenericOpConversion<IREE::VM::ExtI16I32UOp>>(
      typeConverter, context, "vm_ext_i16i32u");

  // Native bitwise shift and rotate ops
  patterns.add<GenericOpConversion<IREE::VM::ShlI32Op>>(typeConverter, context,
                                                        "vm_shl_i32");
  patterns.add<GenericOpConversion<IREE::VM::ShrI32SOp>>(typeConverter, context,
                                                         "vm_shr_i32s");
  patterns.add<GenericOpConversion<IREE::VM::ShrI32UOp>>(typeConverter, context,
                                                         "vm_shr_i32u");

  // Comparison ops
  patterns.add<GenericOpConversion<IREE::VM::CmpEQI32Op>>(
      typeConverter, context, "vm_cmp_eq_i32");
  patterns.add<GenericOpConversion<IREE::VM::CmpNEI32Op>>(
      typeConverter, context, "vm_cmp_ne_i32");
  patterns.add<GenericOpConversion<IREE::VM::CmpLTI32SOp>>(
      typeConverter, context, "vm_cmp_lt_i32s");
  patterns.add<GenericOpConversion<IREE::VM::CmpLTI32UOp>>(
      typeConverter, context, "vm_cmp_lt_i32u");
  patterns.add<GenericOpConversion<IREE::VM::CmpNZI32Op>>(
      typeConverter, context, "vm_cmp_nz_i32");
  patterns.add<CompareRefOpConversion<IREE::VM::CmpEQRefOp>>(
      typeConverter, context, "vm_cmp_eq_ref");
  patterns.add<CompareRefOpConversion<IREE::VM::CmpNERefOp>>(
      typeConverter, context, "vm_cmp_ne_ref");
  patterns.add<CompareRefNotZeroOpConversion>(typeConverter, context);

  // ExtF32: Globals
  patterns.add<
      GlobalLoadOpConversion<IREE::VM::GlobalLoadF32Op, IREE::VM::GlobalF32Op>>(
      typeConverter, context, "vm_global_load_f32");
  patterns.add<GlobalStoreOpConversion<IREE::VM::GlobalStoreF32Op,
                                       IREE::VM::GlobalF32Op>>(
      typeConverter, context, "vm_global_store_f32");

  // ExtF32: Native floating-point constants
  patterns.add<ConstOpConversion<IREE::VM::ConstF32Op>>(typeConverter, context);
  patterns.add<ConstZeroOpConversion<IREE::VM::ConstF32ZeroOp>>(typeConverter,
                                                                context);

  // ExtF32: Conditional assignment
  patterns.add<GenericOpConversion<IREE::VM::SelectF32Op>>(
      typeConverter, context, "vm_select_f32");

  // ExtF32: Native floating-point arithmetic
  patterns.add<GenericOpConversion<IREE::VM::AddF32Op>>(typeConverter, context,
                                                        "vm_add_f32");
  patterns.add<GenericOpConversion<IREE::VM::SubF32Op>>(typeConverter, context,
                                                        "vm_sub_f32");
  patterns.add<GenericOpConversion<IREE::VM::MulF32Op>>(typeConverter, context,
                                                        "vm_mul_f32");
  patterns.add<GenericOpConversion<IREE::VM::DivF32Op>>(typeConverter, context,
                                                        "vm_div_f32");
  patterns.add<GenericOpConversion<IREE::VM::RemF32Op>>(typeConverter, context,
                                                        "vm_rem_f32");
  patterns.add<GenericOpConversion<IREE::VM::FMAF32Op>>(typeConverter, context,
                                                        "vm_fma_f32");
  patterns.add<GenericOpConversion<IREE::VM::AbsF32Op>>(typeConverter, context,
                                                        "vm_abs_f32");
  patterns.add<GenericOpConversion<IREE::VM::NegF32Op>>(typeConverter, context,
                                                        "vm_neg_f32");
  patterns.add<GenericOpConversion<IREE::VM::CeilF32Op>>(typeConverter, context,
                                                         "vm_ceil_f32");
  patterns.add<GenericOpConversion<IREE::VM::FloorF32Op>>(
      typeConverter, context, "vm_floor_f32");
  patterns.add<GenericOpConversion<IREE::VM::RoundF32Op>>(
      typeConverter, context, "vm_round_f32");

  patterns.add<GenericOpConversion<IREE::VM::AtanF32Op>>(typeConverter, context,
                                                         "vm_atan_f32");
  patterns.add<GenericOpConversion<IREE::VM::Atan2F32Op>>(
      typeConverter, context, "vm_atan2_f32");
  patterns.add<GenericOpConversion<IREE::VM::CosF32Op>>(typeConverter, context,
                                                        "vm_cos_f32");
  patterns.add<GenericOpConversion<IREE::VM::SinF32Op>>(typeConverter, context,
                                                        "vm_sin_f32");
  patterns.add<GenericOpConversion<IREE::VM::ExpF32Op>>(typeConverter, context,
                                                        "vm_exp_f32");
  patterns.add<GenericOpConversion<IREE::VM::Exp2F32Op>>(typeConverter, context,
                                                         "vm_exp2_f32");
  patterns.add<GenericOpConversion<IREE::VM::ExpM1F32Op>>(
      typeConverter, context, "vm_expm1_f32");
  patterns.add<GenericOpConversion<IREE::VM::LogF32Op>>(typeConverter, context,
                                                        "vm_log_f32");
  patterns.add<GenericOpConversion<IREE::VM::Log10F32Op>>(
      typeConverter, context, "vm_log10_f32");
  patterns.add<GenericOpConversion<IREE::VM::Log1pF32Op>>(
      typeConverter, context, "vm_log1p_f32");
  patterns.add<GenericOpConversion<IREE::VM::Log2F32Op>>(typeConverter, context,
                                                         "vm_log2_f32");
  patterns.add<GenericOpConversion<IREE::VM::PowF32Op>>(typeConverter, context,
                                                        "vm_pow_f32");
  patterns.add<GenericOpConversion<IREE::VM::RsqrtF32Op>>(
      typeConverter, context, "vm_rsqrt_f32");
  patterns.add<GenericOpConversion<IREE::VM::SqrtF32Op>>(typeConverter, context,
                                                         "vm_sqrt_f32");
  patterns.add<GenericOpConversion<IREE::VM::TanhF32Op>>(typeConverter, context,
                                                         "vm_tanh_f32");
  patterns.add<GenericOpConversion<IREE::VM::ErfF32Op>>(typeConverter, context,
                                                        "vm_erf_f32");

  // ExtF32: Casting and type conversion/emulation
  patterns.add<GenericOpConversion<IREE::VM::CastSI32F32Op>>(
      typeConverter, context, "vm_cast_si32f32");
  patterns.add<GenericOpConversion<IREE::VM::CastUI32F32Op>>(
      typeConverter, context, "vm_cast_ui32f32");
  patterns.add<GenericOpConversion<IREE::VM::CastF32SI32Op>>(
      typeConverter, context, "vm_cast_f32si32");
  patterns.add<GenericOpConversion<IREE::VM::CastF32UI32Op>>(
      typeConverter, context, "vm_cast_f32ui32");
  patterns.add<GenericOpConversion<IREE::VM::BitcastI32F32Op>>(
      typeConverter, context, "vm_bitcast_i32f32");
  patterns.add<GenericOpConversion<IREE::VM::BitcastF32I32Op>>(
      typeConverter, context, "vm_bitcast_f32i32");

  // ExtF32: Comparison ops
  patterns.add<GenericOpConversion<IREE::VM::CmpEQF32OOp>>(
      typeConverter, context, "vm_cmp_eq_f32o");
  patterns.add<GenericOpConversion<IREE::VM::CmpEQF32UOp>>(
      typeConverter, context, "vm_cmp_eq_f32u");
  patterns.add<GenericOpConversion<IREE::VM::CmpNEF32OOp>>(
      typeConverter, context, "vm_cmp_ne_f32o");
  patterns.add<GenericOpConversion<IREE::VM::CmpNEF32UOp>>(
      typeConverter, context, "vm_cmp_ne_f32u");
  patterns.add<GenericOpConversion<IREE::VM::CmpLTF32OOp>>(
      typeConverter, context, "vm_cmp_lt_f32o");
  patterns.add<GenericOpConversion<IREE::VM::CmpLTF32UOp>>(
      typeConverter, context, "vm_cmp_lt_f32u");
  patterns.add<GenericOpConversion<IREE::VM::CmpLTEF32OOp>>(
      typeConverter, context, "vm_cmp_lte_f32o");
  patterns.add<GenericOpConversion<IREE::VM::CmpLTEF32UOp>>(
      typeConverter, context, "vm_cmp_lte_f32u");
  patterns.add<GenericOpConversion<IREE::VM::CmpNaNF32Op>>(
      typeConverter, context, "vm_cmp_nan_f32");

  // ExtI64: Globals
  patterns.add<
      GlobalLoadOpConversion<IREE::VM::GlobalLoadI64Op, IREE::VM::GlobalI64Op>>(
      typeConverter, context, "vm_global_load_i64");
  patterns.add<GlobalStoreOpConversion<IREE::VM::GlobalStoreI64Op,
                                       IREE::VM::GlobalI64Op>>(
      typeConverter, context, "vm_global_store_i64");

  // ExtI64: Constants
  patterns.add<ConstOpConversion<IREE::VM::ConstI64Op>>(typeConverter, context);
  patterns.add<ConstZeroOpConversion<IREE::VM::ConstI64ZeroOp>>(typeConverter,
                                                                context);

  // ExtI64: List ops
  patterns.add<ListGetOpConversion<IREE::VM::ListGetI64Op>>(typeConverter,
                                                            context);
  patterns.add<ListSetOpConversion<IREE::VM::ListSetI64Op>>(typeConverter,
                                                            context);

  // ExtI64: Conditional assignment ops
  patterns.add<GenericOpConversion<IREE::VM::SelectI64Op>>(
      typeConverter, context, "vm_select_i64");
  // ExtI64: Native integer arithmetic ops
  patterns.add<GenericOpConversion<IREE::VM::AddI64Op>>(typeConverter, context,
                                                        "vm_add_i64");
  patterns.add<GenericOpConversion<IREE::VM::SubI64Op>>(typeConverter, context,
                                                        "vm_sub_i64");
  patterns.add<GenericOpConversion<IREE::VM::MulI64Op>>(typeConverter, context,
                                                        "vm_mul_i64");
  patterns.add<GenericOpConversion<IREE::VM::DivI64SOp>>(typeConverter, context,
                                                         "vm_div_i64s");
  patterns.add<GenericOpConversion<IREE::VM::DivI64UOp>>(typeConverter, context,
                                                         "vm_div_i64u");
  patterns.add<GenericOpConversion<IREE::VM::RemI64SOp>>(typeConverter, context,
                                                         "vm_rem_i64s");
  patterns.add<GenericOpConversion<IREE::VM::RemI64UOp>>(typeConverter, context,
                                                         "vm_rem_i64u");
  patterns.add<GenericOpConversion<IREE::VM::FMAI64Op>>(typeConverter, context,
                                                        "vm_fma_i64");
  patterns.add<GenericOpConversion<IREE::VM::AbsI64Op>>(typeConverter, context,
                                                        "vm_abs_i64");
  patterns.add<GenericOpConversion<IREE::VM::NotI64Op>>(typeConverter, context,
                                                        "vm_not_i64");
  patterns.add<GenericOpConversion<IREE::VM::AndI64Op>>(typeConverter, context,
                                                        "vm_and_i64");
  patterns.add<GenericOpConversion<IREE::VM::OrI64Op>>(typeConverter, context,
                                                       "vm_or_i64");
  patterns.add<GenericOpConversion<IREE::VM::XorI64Op>>(typeConverter, context,
                                                        "vm_xor_i64");
  patterns.add<GenericOpConversion<IREE::VM::CtlzI64Op>>(typeConverter, context,
                                                         "vm_ctlz_i64");

  // ExtI64: Casting and type conversion/emulation ops
  patterns.add<GenericOpConversion<IREE::VM::TruncI64I32Op>>(
      typeConverter, context, "vm_trunc_i64i32");
  patterns.add<GenericOpConversion<IREE::VM::ExtI32I64SOp>>(
      typeConverter, context, "vm_ext_i32i64s");
  patterns.add<GenericOpConversion<IREE::VM::ExtI32I64UOp>>(
      typeConverter, context, "vm_ext_i32i64u");

  // ExtI64: Native bitwise shift and rotate ops
  patterns.add<GenericOpConversion<IREE::VM::ShlI64Op>>(typeConverter, context,
                                                        "vm_shl_i64");
  patterns.add<GenericOpConversion<IREE::VM::ShrI64SOp>>(typeConverter, context,
                                                         "vm_shr_i64s");
  patterns.add<GenericOpConversion<IREE::VM::ShrI64UOp>>(typeConverter, context,
                                                         "vm_shr_i64u");

  // ExtI64: Comparison ops
  patterns.add<GenericOpConversion<IREE::VM::CmpEQI64Op>>(
      typeConverter, context, "vm_cmp_eq_i64");
  patterns.add<GenericOpConversion<IREE::VM::CmpNEI64Op>>(
      typeConverter, context, "vm_cmp_ne_i64");
  patterns.add<GenericOpConversion<IREE::VM::CmpLTI64SOp>>(
      typeConverter, context, "vm_cmp_lt_i64s");
  patterns.add<GenericOpConversion<IREE::VM::CmpLTI64UOp>>(
      typeConverter, context, "vm_cmp_lt_i64u");
  patterns.add<GenericOpConversion<IREE::VM::CmpNZI64Op>>(
      typeConverter, context, "vm_cmp_nz_i64");
}

namespace IREE {
namespace VM {

namespace {

// A pass converting IREE VM operations into the EmitC dialect.
// vm.func ops get converted to std.func with the calling convention used by
// EmitC. Each function gets three additional arguments a `iree_vm_stack_t*` as
// well as two module specific struct pointers (`{module_name}_t*` and
// `{module_name}_state_t`). These are followed by the original function
// arguments and out arguments for the vm.func results. The result type of the
// function is `iree_status_t`. Ref types are always passed as pointers.
//
// Examples:
//   () -> () => (iree_vm_stack_t*, module_t*, module_state_t*) -> iree_status_t
//
//   (i) -> () => (iree_vm_stack_t*, module_t*, module_state_t*, int32_t) ->
//                  iree_status_t
//
//   (r) -> () => (iree_vm_stack_t*, module_t*, module_state_t*, iree_vm_ref_t*)
//                  -> iree_status_t
//
//   () -> (r) => (iree_vm_stack_t*, module_t*, module_state_t*, iree_vm_ref_t*)
//                  -> iree_status_t
//
//   (iir) -> (ri) => (iree_vm_stack_t*, module_t*, module_state_t*, int32_t,
//                      int32_t, iree_vm_ref_t*, iree_vm_ref_t*, int32_t*) ->
//                      iree_status_t
class ConvertVMToEmitCPass
    : public PassWrapper<ConvertVMToEmitCPass,
                         OperationPass<IREE::VM::ModuleOp>> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertVMToEmitCPass)

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<mlir::emitc::EmitCDialect, mlir::BuiltinDialect,
                    mlir::func::FuncDialect, IREE::Util::UtilDialect>();
  }

  StringRef getArgument() const override { return "iree-convert-vm-to-emitc"; }

  StringRef getDescription() const override {
    return "Convert VM Ops to the EmitC dialect";
  }

  void runOnOperation() override {
    IREE::VM::ModuleOp module = getOperation();

    ConversionTarget target(getContext());
    EmitCTypeConverter typeConverter;

    // Convert vm.func ops to std.func with the calling convention used by
    // EmitC. We convert these upfront to make sure vm.call ops always
    // reference std.func ops with the correct calling convention during the
    // conversion.
    SmallVector<IREE::VM::FuncOp, 4> funcsToRemove;
    SmallVector<BlockArgument, 4> blockArgsToRemove;
    for (auto funcOp : module.getOps<IREE::VM::FuncOp>()) {
      Operation *op = funcOp.getOperation();
      typeConverter.analysisCache.insert(
          std::make_pair(op, VMAnalysis(funcOp)));

      if (failed(convertFuncOp(funcOp, typeConverter, blockArgsToRemove))) {
        return signalPassFailure();
      }
      funcsToRemove.push_back(funcOp);
    }

    for (auto &funcOp : funcsToRemove) {
      funcOp.erase();
    }

    // Generate func ops that implement the C API.
    if (failed(createAPIFunctions(module, typeConverter))) {
      return signalPassFailure();
    }

    SmallVector<std::string> importShims;

    // The conversion of `call/call.variadic` ops on imported functions expects
    // import ops to be rewritten to compiler generated shim functions. To
    // ensure this we only rewrite `import` ops first.
    ImportOpConverter importOpConverter(typeConverter, importShims);
    for (auto importOp : module.getOps<IREE::VM::ImportOp>()) {
      if (failed(importOpConverter(importOp))) {
        return signalPassFailure();
      }
    }

    RewritePatternSet patterns(&getContext());
    populateVMToEmitCPatterns(target, typeConverter, patterns);

    target.addLegalDialect<emitc::EmitCDialect, mlir::BuiltinDialect,
                           mlir::cf::ControlFlowDialect,
                           mlir::func::FuncDialect>();

    target.addDynamicallyLegalOp<mlir::func::FuncOp>(
        [&](mlir::func::FuncOp op) {
          return typeConverter.isSignatureLegal(op.getFunctionType());
        });

    // Structural ops
    target.addLegalOp<IREE::VM::ModuleOp>();
    target.addLegalOp<IREE::VM::ModuleTerminatorOp>();
    target.addLegalOp<IREE::VM::ImportOp>();

    // This op is needed in the printer to emit an array holding the data.
    target.addLegalOp<IREE::VM::RodataOp>();

    if (failed(applyFullConversion(module, target, std::move(patterns)))) {
      return signalPassFailure();
    }

    // Remove unused block arguments from refs
    if (failed(removeBlockArguments(module, blockArgsToRemove))) {
      return signalPassFailure();
    }

    SetVector<Operation *> &materializations =
        typeConverter.sourceMaterializations;

    module.walk([&materializations](Operation *op) {
      // Remove dead basic block arguments
      if (materializations.contains(op)) {
        assert(isa<emitc::VariableOp>(op));
        assert(op->use_empty());

        materializations.remove(op);
        op->erase();
        return;
      }
    });
  }
};

}  // namespace

std::unique_ptr<OperationPass<IREE::VM::ModuleOp>>
createConvertVMToEmitCPass() {
  return std::make_unique<ConvertVMToEmitCPass>();
}

}  // namespace VM
}  // namespace IREE

static PassRegistration<IREE::VM::ConvertVMToEmitCPass> pass;

}  // namespace iree_compiler
}  // namespace mlir
