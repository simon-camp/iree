// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/VM/Conversion/VMToEmitC/IREEModuleStructure.h"

#include <optional>
#include "iree/compiler/Dialect/VM/Conversion/VMToEmitC/EmitCBuilders.h"
#include "iree/compiler/Dialect/VM/Conversion/VMToEmitC/EmitCHelpers.h"

#include <optional>
#include "iree/compiler/Dialect/Util/Conversion/ConversionPatterns.h"
#include "iree/compiler/Dialect/Util/IR/UtilDialect.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "iree/compiler/Dialect/VM/Conversion/VMToEmitC/EmitCBuilders.h"
#include "iree/compiler/Dialect/VM/Conversion/VMToEmitC/IREEModuleStructure.h"
#include "iree/compiler/Dialect/VM/Conversion/VMToEmitC/VMAnalysis.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::iree_compiler::IREE::VM {

namespace {
static LogicalResult
createAPIFunctions(IREE::VM::ModuleOp moduleOp,
                   IREE::VM::ModuleAnalysis &moduleAnalysis) {
  auto ctx = moduleOp.getContext();
  auto loc = moduleOp.getLoc();

  OpBuilder builder(moduleOp);
  builder.setInsertionPoint(moduleOp.getBlock().getTerminator());

  std::string moduleName{moduleOp.getName()};

  // void destroy(void*)
  {
    OpBuilder::InsertionGuard guard(builder);

    const int moduleArgIndex = 0;

    auto funcType = mlir::FunctionType::get(
        ctx, {emitc::PointerType::get(emitc::OpaqueType::get(ctx, "void"))},
        {});

    auto funcOp = builder.create<mlir::func::FuncOp>(
        loc, moduleName + "_destroy", funcType);
    funcOp.setPrivate();

    moduleAnalysis.addDummy(funcOp, /*emitAtEnd=*/false);

    Block *entryBlock = funcOp.addEntryBlock();
    const BlockArgument moduleArg = funcOp.getArgument(moduleArgIndex);

    builder.setInsertionPointToStart(entryBlock);

    std::string moduleTypeName = std::string("struct ") + moduleName + "_t";

    auto castedModuleOp = builder.create<emitc::CastOp>(
        /*location=*/loc,
        /*type=*/
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, moduleTypeName)),
        /*operand=*/moduleArg);

    auto allocatorOp = emitc_builders::structPtrMember(
        builder, loc,
        /*type=*/emitc::OpaqueType::get(ctx, "iree_allocator_t"),
        /*memberName=*/"allocator",
        /*operand=*/castedModuleOp.getResult());

    builder.create<emitc::CallOpaqueOp>(
        /*location=*/loc,
        /*type=*/TypeRange{},
        /*callee=*/StringAttr::get(ctx, "iree_allocator_free"),
        /*args=*/ArrayAttr{},
        /*templateArgs=*/ArrayAttr{},
        /*operands=*/ArrayRef<Value>{allocatorOp, castedModuleOp.getResult()});

    builder.create<mlir::func::ReturnOp>(loc);
  }

  // iree_status_t alloc_state(void*, iree_allocator_t,
  // iree_vm_module_state_t**)
  {
    OpBuilder::InsertionGuard guard(builder);

    const int allocatorArgIndex = 1;
    const int moduleStateArgIndex = 2;

    auto funcType = mlir::FunctionType::get(
        ctx,
        {emitc::PointerType::get(emitc::OpaqueType::get(ctx, "void")),
         emitc::OpaqueType::get(ctx, "iree_allocator_t"),
         emitc::PointerType::get(emitc::PointerType::get(
             emitc::OpaqueType::get(ctx, "iree_vm_module_state_t")))},
        {emitc::OpaqueType::get(ctx, "iree_status_t")});

    auto funcOp = builder.create<mlir::func::FuncOp>(
        loc, moduleName + "_alloc_state", funcType);
    funcOp.setPrivate();

    moduleAnalysis.addDummy(funcOp, /*emitAtEnd=*/false);

    Block *entryBlock = funcOp.addEntryBlock();

    const BlockArgument allocatorArg = funcOp.getArgument(allocatorArgIndex);
    const BlockArgument moduleStateArg =
        funcOp.getArgument(moduleStateArgIndex);

    builder.setInsertionPointToStart(entryBlock);

    std::string moduleStateTypeName =
        std::string("struct ") + moduleName + "_state_t";

    Value state = emitc_builders::allocateVariable(
        builder, loc,
        emitc::PointerType::get(
            emitc::OpaqueType::get(ctx, moduleStateTypeName)),
        {"NULL"});

    Value stateSize = emitc_builders::sizeOf(
        builder, loc, emitc::OpaqueAttr::get(ctx, moduleStateTypeName));

    Value statePtr = emitc_builders::addressOf(builder, loc, state);

    auto voidPtr = builder.create<emitc::CastOp>(
        /*location=*/loc,
        /*type=*/
        emitc::PointerType::get(
            emitc::PointerType::get(emitc::OpaqueType::get(ctx, "void"))),
        /*operand=*/statePtr);

    returnIfError(builder, loc, StringAttr::get(ctx, "iree_allocator_malloc"),
                  {}, {allocatorArg, stateSize, voidPtr.getResult()},
                  moduleAnalysis);

    emitc_builders::memset(builder, loc, state, 0, stateSize);

    emitc_builders::structPtrMemberAssign(builder, loc,
                                          /*memberName=*/"allocator",
                                          /*operand=*/state,
                                          /*value=*/allocatorArg);

    // Initialize buffers
    for (auto rodataOp : moduleOp.getOps<IREE::VM::RodataOp>()) {
      auto ordinal = rodataOp.getOrdinal()->getZExtValue();

      std::string bufferName = moduleName + "_" + rodataOp.getName().str();

      Value rodataPointer = emitc_builders::allocateVariable(
          builder, loc,
          emitc::PointerType::get(emitc::OpaqueType::get(ctx, "const uint8_t")),
          {bufferName});

      auto bufferVoid = builder.create<emitc::CastOp>(
          /*location=*/loc,
          /*type=*/emitc::PointerType::get(emitc::OpaqueType::get(ctx, "void")),
          /*operand=*/rodataPointer);

      Value bufferSize = emitc_builders::sizeOf(
          builder, loc, emitc::OpaqueAttr::get(ctx, bufferName));

      auto byteSpan = builder.create<emitc::CallOpaqueOp>(
          /*location=*/loc,
          /*type=*/emitc::OpaqueType::get(ctx, "iree_byte_span_t"),
          /*callee=*/StringAttr::get(ctx, "iree_make_byte_span"),
          /*args=*/ArrayAttr{},
          /*templateArgs=*/ArrayAttr{},
          /*operands=*/
          ArrayRef<Value>{bufferVoid.getResult(), bufferSize});

      auto allocator = builder.create<emitc::CallOpaqueOp>(
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
          /*operand=*/state);

      auto buffer = emitc_builders::arrayElementAddress(
          builder, loc,
          /*type=*/
          emitc::PointerType::get(
              emitc::OpaqueType::get(ctx, "iree_vm_buffer_t")),
          /*index=*/builder.getUI32IntegerAttr(ordinal),
          /*operand=*/buffers);

      builder.create<emitc::CallOpaqueOp>(
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
          /*operand=*/state);

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
        /*operand=*/state);

    builder.create<emitc::CallOpaqueOp>(
        /*location=*/loc,
        /*type=*/TypeRange{},
        /*callee=*/StringAttr::get(ctx, "EMITC_DEREF_ASSIGN_VALUE"),
        /*args=*/ArrayAttr{},
        /*templateArgs=*/ArrayAttr{},
        /*operands=*/
        ArrayRef<Value>{moduleStateArg, baseStateOp.getResult()});

    auto status = emitc_builders::ireeOkStatus(builder, loc);

    builder.create<mlir::func::ReturnOp>(loc, status);
  }

  // void free_state(void*, iree_vm_module_state_t*)
  {
    OpBuilder::InsertionGuard guard(builder);

    const int moduleStateArgIndex = 1;

    auto funcType = mlir::FunctionType::get(
        ctx,
        {emitc::PointerType::get(emitc::OpaqueType::get(ctx, "void")),
         emitc::PointerType::get(
             emitc::OpaqueType::get(ctx, "iree_vm_module_state_t"))},
        {});

    auto funcOp = builder.create<mlir::func::FuncOp>(
        loc, moduleName + "_free_state", funcType);
    funcOp.setPrivate();

    moduleAnalysis.addDummy(funcOp, /*emitAtEnd=*/false);

    Block *entryBlock = funcOp.addEntryBlock();

    const BlockArgument moduleStateArg =
        funcOp.getArgument(moduleStateArgIndex);

    builder.setInsertionPointToStart(entryBlock);

    std::string moduleStateTypeName =
        std::string("struct ") + moduleName + "_state_t";

    auto stateOp = builder.create<emitc::CastOp>(
        /*location=*/loc,
        /*type=*/
        emitc::PointerType::get(
            emitc::OpaqueType::get(ctx, moduleStateTypeName)),
        /*operand=*/moduleStateArg);

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

    builder.create<emitc::CallOpaqueOp>(
        /*location=*/loc,
        /*type=*/TypeRange{},
        /*callee=*/StringAttr::get(ctx, "iree_allocator_free"),
        /*args=*/ArrayAttr{},
        /*templateArgs=*/ArrayAttr{},
        /*operands=*/
        ArrayRef<Value>{allocatorOp, stateOp.getResult()});

    builder.create<mlir::func::ReturnOp>(loc);
  }

  // iree_status_t resolve_import(
  //   void*,
  //   iree_vm_module_state_t*,
  //   iree_host_size_t,
  //   const iree_vm_function_t*,
  //   const iree_vm_function_signature_t*
  // )
  {
    OpBuilder::InsertionGuard guard(builder);

    const int moduleStateArgIndex = 1;
    const int ordinalArgIndex = 2;
    const int functionArgIndex = 3;

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
    funcOp.setPrivate();

    moduleAnalysis.addDummy(funcOp, /*emitAtEnd=*/false);

    Block *entryBlock = funcOp.addEntryBlock();

    const BlockArgument moduleStateArg =
        funcOp.getArgument(moduleStateArgIndex);
    const BlockArgument ordinalArg = funcOp.getArgument(ordinalArgIndex);
    const BlockArgument functionArg = funcOp.getArgument(functionArgIndex);

    builder.setInsertionPointToStart(entryBlock);

    std::string moduleStateTypeName =
        std::string("struct ") + moduleName + "_state_t";

    auto stateOp = builder.create<emitc::CastOp>(
        /*location=*/loc,
        /*type=*/
        emitc::PointerType::get(
            emitc::OpaqueType::get(ctx, moduleStateTypeName)),
        /*operand=*/moduleStateArg);

    auto imports = emitc_builders::structPtrMember(
        builder, loc,
        /*type=*/
        emitc::PointerType::get(
            emitc::OpaqueType::get(ctx, "iree_vm_function_t")),
        /*memberName=*/"imports",
        /*operand=*/stateOp.getResult());

    auto import = emitc_builders::arrayElementAddress(
        builder, loc, /*type=*/
        emitc::PointerType::get(
            emitc::OpaqueType::get(ctx, "iree_vm_function_t")),
        /*index=*/ordinalArg, /*operand=*/imports);

    builder.create<emitc::CallOpaqueOp>(
        /*location=*/loc,
        /*type=*/TypeRange{},
        /*callee=*/StringAttr::get(ctx, "EMITC_DEREF_ASSIGN_PTR"),
        /*args=*/ArrayAttr{},
        /*templateArgs=*/ArrayAttr{},
        /*operands=*/
        ArrayRef<Value>{import, functionArg});

    auto status = emitc_builders::ireeOkStatus(builder, loc);

    builder.create<mlir::func::ReturnOp>(loc, status);
  }

  // iree_status_t create(
  //   iree_vm_instance_t*,
  //   iree_allocator_t,
  //   iree_vm_module_t**
  // );
  {
    OpBuilder::InsertionGuard guard(builder);

    const int instanceArgIndex = 0;
    const int allocatorArgIndex = 1;
    const int moduleArgIndex = 2;

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
    funcOp.setPublic();

    // This function needs an iree_vm_native_module_descriptor_t. THe IR for it
    // is generated after the dialect conversion. So we mark this function to
    // move it to the correct spot later.
    moduleAnalysis.addDummy(funcOp, /*emitAtEnd=*/true);

    Block *entryBlock = funcOp.addEntryBlock();

    const BlockArgument instanceArg = funcOp.getArgument(instanceArgIndex);
    const BlockArgument allocatorArg = funcOp.getArgument(allocatorArgIndex);
    const BlockArgument moduleArg = funcOp.getArgument(moduleArgIndex);

    builder.setInsertionPointToStart(entryBlock);

    std::string moduleTypeName = std::string("struct ") + moduleName + "_t";

    Value module = emitc_builders::allocateVariable(
        builder, loc,
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, moduleTypeName)),
        {"NULL"});

    Value moduleSize = emitc_builders::sizeOf(
        builder, loc, emitc::OpaqueAttr::get(ctx, moduleTypeName));

    Value modulePtr = emitc_builders::addressOf(builder, loc, module);

    auto voidPtr = builder.create<emitc::CastOp>(
        /*location=*/loc,
        /*type=*/
        emitc::PointerType::get(
            emitc::PointerType::get(emitc::OpaqueType::get(ctx, "void"))),
        /*operand=*/modulePtr);

    returnIfError(builder, loc, StringAttr::get(ctx, "iree_allocator_malloc"),
                  {}, {allocatorArg, moduleSize, voidPtr.getResult()},
                  moduleAnalysis);

    emitc_builders::memset(builder, loc, module, 0, moduleSize);

    emitc_builders::structPtrMemberAssign(builder, loc,
                                          /*memberName=*/"allocator",
                                          /*operand=*/module,
                                          /*value=*/allocatorArg);

    auto &typeTable = moduleAnalysis.typeTable;
    if (!typeTable.empty()) {
      Type typeRefType = emitc::OpaqueType::get(ctx, "iree_vm_ref_type_t");
      Type typeRefArrayType = emitc::PointerType::get(typeRefType);
      Value moduleTypes = emitc_builders::structPtrMember(
          builder, loc, typeRefArrayType, "types", module);

      std::string listType = "!vm.list";
      for (auto [index, typeDef] : llvm::enumerate(typeTable)) {
        std::string typeName = typeDef.full_name;
        std::string listPrefix = typeName.substr(0, listType.size());
        if (listType == listPrefix) {
          typeName = listPrefix;
        }

        // Remove leading '!' and wrap in quotes
        if (typeName[0] == '!') {
          typeName = typeName.substr(1);
        }
        typeName = std::string("\"") + typeName + std::string("\"");

        Value stringView =
            emitc_builders::ireeMakeCstringView(builder, loc, typeName);
        Value refType = emitc_builders::ireeVmInstanceLookupType(
            builder, loc, instanceArg, stringView);
        emitc_builders::arrayElementAssign(builder, loc, moduleTypes, index,
                                           refType);
      }
    }

    Value vmModule = emitc_builders::allocateVariable(
        builder, loc, emitc::OpaqueType::get(ctx, "iree_vm_module_t"));

    Value vmModulePtr = emitc_builders::addressOf(builder, loc, vmModule);

    auto vmInitializeStatus = builder.create<emitc::CallOpaqueOp>(
        /*location=*/loc,
        /*type=*/emitc::OpaqueType::get(ctx, "iree_status_t"),
        /*callee=*/StringAttr::get(ctx, "iree_vm_module_initialize"),
        /*args=*/ArrayAttr{},
        /*templateArgs=*/ArrayAttr{},
        /*operands=*/
        ArrayRef<Value>{vmModulePtr, module});

    Type boolType = builder.getIntegerType(1);

    auto vmInitializeIsOk = builder.create<emitc::CallOpaqueOp>(
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

      builder.create<emitc::CallOpaqueOp>(
          /*location=*/loc,
          /*type=*/TypeRange{},
          /*callee=*/StringAttr::get(ctx, "iree_allocator_free"),
          /*args=*/ArrayAttr{},
          /*templateArgs=*/ArrayAttr{},
          /*operands=*/
          ArrayRef<Value>{allocatorArg, module});

      builder.create<mlir::func::ReturnOp>(loc,
                                           vmInitializeStatus.getResult(0));
    }

    builder.setInsertionPointToEnd(condBlock);

    builder.create<mlir::cf::CondBranchOp>(loc, vmInitializeIsOk.getResult(0),
                                           continuationBlock, failureBlock);

    builder.setInsertionPointToStart(continuationBlock);

    // Set function pointers
    for (std::string funcName :
         {"destroy", "alloc_state", "free_state", "resolve_import"}) {
      emitc_builders::structMemberAssign(builder, loc,
                                         /*memberName=*/funcName,
                                         /*operand=*/vmModule,
                                         /*value=*/moduleName + "_" + funcName);
    }

    std::string descriptorPtr = "&" + moduleName + "_descriptor_";

    auto status = builder.create<emitc::CallOpaqueOp>(
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
        ArrayRef<Value>{vmModulePtr, instanceArg, allocatorArg, moduleArg});

    builder.create<mlir::func::ReturnOp>(loc, status.getResult(0));
  }

  return success();
}
} // namespace

/// TODO:
///   - replace string concatenation with Twines
///   - be more thoughtful of the location to use
///   - move things to conversion patterns if possible
///   - create helper functions in emitc_builders.escape_hatch
///     - function declaration
///     - global
///     - structured preprocessor directive
///     - struct definition
LogicalResult
createModuleStructure(IREE::VM::ModuleOp moduleOp,
                      IREE::VM::EmitCTypeConverter &typeConverter) {
  if (failed(createAPIFunctions(moduleOp, typeConverter.analysis))) {
    return failure();
  }

  auto loc = moduleOp.getLoc();

  OpBuilder builder(moduleOp);

  SmallVector<Operation *> opsToRemove;
  {
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(&moduleOp.getBlock());

    std::string includeGuard = moduleOp.getName().upper() + "_H_";

    emitc_builders::preprocessorDirective(builder, loc, emitc_builders::IFNDEF,
                                          includeGuard);

    emitc_builders::preprocessorDirective(builder, loc, emitc_builders::DEFINE,
                                          includeGuard);

    emitc_builders::preprocessorDirective(builder, loc, emitc_builders::INCLUDE,
                                          "\"iree/vm/api.h\"");

    emitc_builders::preprocessorDirective(builder, loc, emitc_builders::IFDEF,
                                          "__cplusplus");
    builder.create<emitc::VerbatimOp>(loc, "extern \"C\" {");
    emitc_builders::preprocessorDirective(builder, loc, emitc_builders::ENDIF,
                                          "//  __cplusplus");

    // Emit declarations for public functions.
    for (auto funcOp : moduleOp.getOps<mlir::func::FuncOp>()) {
      if (funcOp.isPublic()) {
        auto declOp =
            emitc_builders::func_decl(builder, loc, funcOp, typeConverter);
        if (failed(declOp))
          return failure();
      }
    }

    emitc_builders::preprocessorDirective(builder, loc, emitc_builders::IFDEF,
                                          "__cplusplus");
    builder.create<emitc::VerbatimOp>(loc, "}  // extern \"C\" {");
    emitc_builders::preprocessorDirective(builder, loc, emitc_builders::ENDIF,
                                          "//  __cplusplus");
    emitc_builders::preprocessorDirective(builder, loc, emitc_builders::ENDIF,
                                          std::string("//  ") + includeGuard);
    emitc_builders::preprocessorDirective(builder, loc, emitc_builders::IF,
                                          "defined(EMITC_IMPLEMENTATION)");
    emitc_builders::preprocessorDirective(builder, loc, emitc_builders::INCLUDE,
                                          "\"iree/vm/ops.h\"");
    emitc_builders::preprocessorDirective(builder, loc, emitc_builders::INCLUDE,
                                          "\"iree/vm/ops_emitc.h\"");
    emitc_builders::preprocessorDirective(builder, loc, emitc_builders::INCLUDE,
                                          "\"iree/vm/shims_emitc.h\"");

    // rodata ops
    // TODO: Make this a conversion
    for (auto rodataOp : moduleOp.getOps<IREE::VM::RodataOp>()) {
      auto value = llvm::dyn_cast<IREE::Util::SerializableAttrInterface>(
          rodataOp.getValue());
      assert(value && "expected a serializable rodata value");
      SmallVector<char> byteBuffer;
      if (failed(value.serializeToVector(
              rodataOp.getLoc(), llvm::endianness::little, byteBuffer))) {
        return rodataOp.emitError() << "error during serialization";
      }

      constexpr size_t kDefaultRodataAlignment = 16;
      size_t alignment =
          rodataOp.getAlignment()
              ? static_cast<size_t>(rodataOp.getAlignment().value())
              : 0;
      if (alignment == 0)
        alignment = kDefaultRodataAlignment;

      std::string bufferName =
          moduleOp.getName().str() + "_" + rodataOp.getName().str();

      std::string stmt = "iree_alignas(" + std::to_string(alignment) +
                         ") static const uint8_t " + bufferName + "[] = {";
      size_t index = 0;
      for (char value : byteBuffer) {
        if (index++ > 0)
          stmt += ", ";
        stmt += std::to_string(
            static_cast<unsigned int>(static_cast<unsigned char>(value)));
      }
      stmt += "};";
      builder.create<emitc::VerbatimOp>(loc, stmt);
      opsToRemove.push_back(rodataOp.getOperation());
    }

    // structs
    // Returns |count| or 1 if |count| == 0.
    // Some compilers (MSVC) don't support zero-length struct fields on the
    // interior of structs (just VLA at the tail).
    auto countOrEmpty = [](uint32_t count) { return count ? count : 1; };

    const int64_t numTypes = typeConverter.analysis.typeTable.size();

    std::string moduleStructName = moduleOp.getName().str() + "_t";
    SmallVector<emitc_builders::StructField> moduleStructFields{
        {"iree_allocator_t", "allocator"},
        {"iree_vm_ref_type_t", "types", countOrEmpty(numTypes)}};

    emitc_builders::structDefinition(builder, loc, moduleStructName,
                                     moduleStructFields);

    auto ordinalCounts = moduleOp.getOrdinalCountsAttr();

    std::string moduleStructStateName = moduleOp.getName().str() + "_state_t";
    SmallVector<emitc_builders::StructField> moduleStructStateFields{
        {"iree_allocator_t", "allocator"},
        {"uint8_t", "rwdata", countOrEmpty(ordinalCounts.getGlobalBytes())},
        {"iree_vm_ref_t", "refs", countOrEmpty(ordinalCounts.getGlobalRefs())},
        {"iree_vm_buffer_t", "rodata_buffers",
         countOrEmpty(ordinalCounts.getRodatas())},
        {"iree_vm_function_t", "imports",
         countOrEmpty(ordinalCounts.getImportFuncs())},
    };

    emitc_builders::structDefinition(builder, loc, moduleStructStateName,
                                     moduleStructStateFields);

    // Emit declarations for private functions.
    for (auto funcOp : moduleOp.getOps<mlir::func::FuncOp>()) {
      if (funcOp.isPrivate()) {
        auto declOp =
            emitc_builders::func_decl(builder, loc, funcOp, typeConverter);
        if (failed(declOp))
          return failure();
      }
    }

    // global descriptors
    // TODO: Move this to a structured helper
    //   - define structs for each entity etc.
    auto printStringView = [](StringRef s) -> std::string {
      // We can't use iree_make_string_view because function calls are not
      // allowed for constant expressions in C.
      // TODO(#7605): Switch to IREE_SVL. We can't use IREE_SVL today because it
      // uses designated initializers, which cause issues when compiled as C++.
      return ("{\"" + s + "\", " + std::to_string(s.size()) + "}").str();
    };

    // dependencies
    std::string dependenciesName = moduleOp.getName().str() + "_dependencies_";
    std::string deps;
    deps += "static const iree_vm_module_dependency_t " + dependenciesName +
            "[] = {";
    auto dependencies = moduleOp.getDependencies();
    if (dependencies.empty()) {
      // Empty list placeholder.
      deps += "{{0}},";
    } else {
      for (auto &dependency : dependencies) {
        deps += "{" + printStringView(dependency.name) + ", " +
                std::to_string(dependency.minimumVersion) + ", " +
                (dependency.isOptional
                     ? "IREE_VM_MODULE_DEPENDENCY_FLAG_OPTIONAL"
                     : "IREE_VM_MODULE_DEPENDENCY_FLAG_REQUIRED") +
                "},";
      }
    }
    deps += "};";
    builder.create<emitc::VerbatimOp>(loc, deps);

    // imports
    SmallVector<IREE::VM::ImportOp> importOps(
        moduleOp.getOps<IREE::VM::ImportOp>());
    std::string importName = moduleOp.getName().str() + "_imports_";
    std::string imports;
    imports += "static const iree_vm_native_import_descriptor_t " + importName +
               "[] = {";
    if (importOps.empty()) {
      // Empty list placeholder.
      imports += "{0},";
    } else {
      // sort import ops by ordinal
      llvm::sort(importOps, [](auto &lhs, auto &rhs) {
        return lhs.getOrdinal()->getZExtValue() <
               rhs.getOrdinal()->getZExtValue();
      });
      for (auto importOp : importOps) {
        imports +=
            std::string("{") +
            (importOp.getIsOptional() ? "IREE_VM_NATIVE_IMPORT_OPTIONAL"
                                      : "IREE_VM_NATIVE_IMPORT_REQUIRED") +
            ", " + printStringView(importOp.getName()) + "},";
      }
    }
    imports += "};";
    builder.create<emitc::VerbatimOp>(loc, imports);

    for (auto op : moduleOp.getOps<IREE::VM::ImportOp>()) {
      opsToRemove.push_back(op);
    }

    // exports
    SmallVector<func::FuncOp> exportedFunctions;
    for (auto func : moduleOp.getOps<func::FuncOp>()) {
      if (typeConverter.analysis.lookupFunction(func).isExported()) {
        exportedFunctions.push_back(func);
      }
    }
    auto extractExportName = [&typeConverter](func::FuncOp funcOp) {
      return typeConverter.analysis.lookupFunction(funcOp).getExportName();
    };
    std::string exportName = moduleOp.getName().str() + "_exports_";
    std::string exports;
    exports += "static const iree_vm_native_export_descriptor_t " + exportName +
               "[] = {";
    if (exportedFunctions.empty()) {
      // Empty list placeholder.
      exports += "{{0}},";
    } else {
      // sort export ops
      llvm::sort(exportedFunctions, [&extractExportName](auto &lhs, auto &rhs) {
        return extractExportName(lhs).compare(extractExportName(rhs)) < 0;
      });
      for (auto funcOp : exportedFunctions) {
        StringRef exportName = extractExportName(funcOp);
        StringRef callingConvention =
            typeConverter.analysis.lookupFunction(funcOp)
                .getCallingConvention();

        // TODO(simon-camp): support function-level reflection attributes
        exports += "{" + printStringView(exportName) + ", " +
                   printStringView(callingConvention) + ", 0, NULL},";
      }
    }
    exports += "};";
    builder.create<emitc::VerbatimOp>(loc, exports);

    // functions
    std::string functionName = moduleOp.getName().str() + "_funcs_";
    std::string functions;
    functions +=
        "static const iree_vm_native_function_ptr_t " + functionName + "[] = {";
    if (exportedFunctions.empty()) {
      // Empty list placeholder.
      functions += "{0},";
    } else {
      // We only add exported functions to the table, as calls to internal
      // functions are directly mapped to C function calls of the generated
      // implementation.
      for (auto funcOp : exportedFunctions) {
        auto funcName = funcOp.getName();
        functions += std::string("{") +
                     "(iree_vm_native_function_shim_t)iree_emitc_shim, " +
                     "(iree_vm_native_function_target_t)" + funcName.str() +
                     "},";
      }
    }
    functions += "};";
    builder.create<emitc::VerbatimOp>(loc, functions);

    // module descriptor
    // TODO(simon-camp): support module-level reflection attributes
    std::string descriptorName = moduleOp.getName().str() + "_descriptor_";
    std::string descriptor;
    descriptor +=
        "static const iree_vm_native_module_descriptor_t " + descriptorName +
        " = {"
        // name:
        + printStringView(moduleOp.getName()) +
        ","
        // version:
        + std::to_string(moduleOp.getVersion().value_or(0u)) +
        ","
        // attrs:
        + "0," +
        "NULL,"
        // dependencies:
        + std::to_string(dependencies.size()) + "," + dependenciesName +
        ","
        // imports:
        + std::to_string(importOps.size()) + "," + importName +
        ","
        // exports:
        + std::to_string(exportedFunctions.size()) + "," + exportName +
        ","
        // functions:
        + std::to_string(exportedFunctions.size()) + "," + functionName + "," +
        "};";

    builder.create<emitc::VerbatimOp>(loc, descriptor);

    // move marked functions to the end of the module
    auto funcs =
        SmallVector<mlir::func::FuncOp>(moduleOp.getOps<mlir::func::FuncOp>());
    for (auto func : funcs) {
      if (typeConverter.analysis.lookupFunction(func).shouldEmitAtEnd()) {
        func->moveBefore(moduleOp.getBlock().getTerminator());
      }
    }

    builder.setInsertionPoint(moduleOp.getBlock().getTerminator());
    emitc_builders::preprocessorDirective(builder, loc, emitc_builders::ENDIF,
                                          "  // EMITC_IMPLEMENTATION");

    // insert a verbatim op with the value `static` before private functions.
    // This can be removed when we switch to emitc.func ops.
    for (auto func : moduleOp.getOps<mlir::func::FuncOp>()) {
      emitc_builders::makeFuncStatic(builder, loc, func);
    }
  }

  for (auto op : opsToRemove) {
    op->erase();
  }

  return success();
}

} // namespace mlir::iree_compiler::IREE::VM
