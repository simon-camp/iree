// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_DIALECT_IREEC_IR_IREEC
#define IREE_DIALECT_IREEC_IR_IREEC

include "iree/compiler/Dialect/IREEC/IR/IREECBase.td"
include "iree/compiler/Dialect/IREEC/IR/IREECAttributes.td"
include "iree/compiler/Dialect/IREEC/IR/IREECTypes.td"

include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/FunctionInterfaces.td"
include "mlir/IR/RegionKindInterface.td"
include "mlir/IR/SymbolInterfaces.td"

//===----------------------------------------------------------------------===//
// IREEC op definitions
//===----------------------------------------------------------------------===//

// Base class for IREEC dialect ops.
class IREEC_Op<string mnemonic, list<Trait> traits = []>
    : Op<IREEC_Dialect, mnemonic, traits>;


def IREEC_ModuleOp : IREEC_Op<"module", [
    IsolatedFromAbove,
    NoRegionArguments,
    SymbolTable,
    Symbol,
  ] # GraphRegionNoTerminator.traits> {

  let arguments = (ins
    OptionalAttr<StrAttr>:$sym_visibility,
    SymbolNameAttr:$sym_name
  );

  let regions = (region SizedRegion<1>:$body);

  let assemblyFormat = [{
    custom<SymbolVisibility>($sym_visibility)
    $sym_name
    attr-dict-with-keyword
    regions
  }];
}

def IREEC_FuncOp : IREEC_Op<"func", [
    IsolatedFromAbove,
    // HasParent<"IREE::IREEC::ModuleOp">,
    FunctionOpInterface,
    CallableOpInterface,
    Symbol,
]> {
  let arguments = (ins
    TypeAttrOf<FunctionType>:$function_type,
    OptionalAttr<DictArrayAttr>:$arg_attrs,
    OptionalAttr<DictArrayAttr>:$res_attrs    
  );

  let regions = (region AnyRegion:$body);

  let extraClassDeclaration = [{
    /// Returns the argument types of this function.
    ArrayRef<Type> getArgumentTypes() { return getFunctionType().getInputs(); }

    /// Returns the result types of this function.
    ArrayRef<Type> getResultTypes() { return getFunctionType().getResults(); }

    Region *getCallableRegion() { return &getBody(); }
    ArrayRef<Type> getCallableResults() {
      assert(!isExternal() && "invalid callable");
      return getFunctionType().getResults();
    }

    /// Add an entry block to an empty function and set up the block arguments
    /// to match the signature of the function.
    Block *addEntryBlock() {
      assert(empty() && "function already has an entry block");
      auto *entry = new Block();
      push_back(entry);
      SmallVector<Location> locs(getFunctionType().getNumInputs(), getLoc());
      entry->addArguments(getFunctionType().getInputs(), locs);
      return entry;
    }

    Block *getEntryBlock() {
      assert(!empty() && "function must have an entry block");
      return &getRegion().getBlocks().front();
    }

    static ParseResult parse(OpAsmParser &parser, OperationState &result) {
      auto buildFuncType =
          [](Builder &builder, ArrayRef<Type> argTypes, ArrayRef<Type> results,
            function_interface_impl::VariadicFlag,
            std::string &) { return builder.getFunctionType(argTypes, results); };

      return function_interface_impl::parseFunctionOp(
          parser, result, /*allowVariadic=*/false,
          getFunctionTypeAttrName(result.name), buildFuncType,
          getArgAttrsAttrName(result.name), getResAttrsAttrName(result.name));
    }

    void print(OpAsmPrinter &p) {
      function_interface_impl::printFunctionOp(
          p, *this, /*isVariadic=*/false, getFunctionTypeAttrName(),
          getArgAttrsAttrName(), getResAttrsAttrName());
    }
  }];
}

def IREEC_ReturnOp : IREEC_Op<"return", [
  Pure,
  HasParent<"IREE::IREEC::FuncOp">,
  ReturnLike,
  Terminator
]> {
  let arguments = (ins Variadic<AnyType>:$operands);

  let assemblyFormat = "attr-dict ($operands^ `:` type($operands))?";
  
  let extraClassDeclaration = [{
    LogicalResult verify() {
      auto function = cast<FuncOp>((*this)->getParentOp());

      // The operand number and types must match the function signature.
      const auto &results = function.getFunctionType().getResults();
      if (getNumOperands() != results.size())
        return emitOpError("has ")
              << getNumOperands() << " operands, but enclosing function (@"
              << function.getName() << ") returns " << results.size();

      for (unsigned i = 0, e = results.size(); i != e; ++i)
        if (getOperand(i).getType() != results[i])
          return emitError() << "type of return operand " << i << " ("
                            << getOperand(i).getType()
                            << ") doesn't match function result type ("
                            << results[i] << ")"
                            << " in function @" << function.getName();

      return success();
    }
  }];
}

//===----------------------------------------------------------------------===//
// Ops related to Pointer type
//===----------------------------------------------------------------------===//

def IREEC_PointerWriteOp : IREEC_Op<"ptr.write",
     [TypesMatchWith<"type of 'value' matches pointee type of 'ptr'",
                     "ptr", "value",
                     "$_self.cast<PointerType>().getPointee()">]> {

  let arguments = (ins AnyType:$value,
                       Arg<IREEC_PointerType, "the reference to store to",
                           [MemWrite]>:$ptr);

  let assemblyFormat = [{
    operands attr-dict `:` qualified(type($ptr))
  }];
}

//===----------------------------------------------------------------------===//
// Ops related to global variables
//===----------------------------------------------------------------------===//

def IREEC_GlobalBufferOp : IREEC_Op<"global.byte_buffer", [Symbol]> {
  let arguments = (ins SymbolNameAttr:$sym_name,
                       DenseI8ArrayAttr:$value,
                       I64Attr:$alignment);

  let assemblyFormat = "`alignas` $alignment $sym_name attr-dict `=` $value";
}

def IREEC_GlobalStructOp : IREEC_Op<"global.struct", [Symbol]> {
  let arguments = (ins SymbolNameAttr:$sym_name,
                       IREEC_StructFieldArrayAttr:$members);

  let assemblyFormat = "$sym_name attr-dict `=` $members";
}

//===----------------------------------------------------------------------===//
// `iree_allocator_t` related Ops
//===----------------------------------------------------------------------===//

def IREEC_AllocatorNullOp : IREEC_Op<"allocator.null", []> {
  let results = (outs IREEC_AllocatorType:$result);
  let assemblyFormat = "attr-dict `:` type($result)";
}

//===----------------------------------------------------------------------===//
// `iree_buffer_t` related Ops
//===----------------------------------------------------------------------===//

def IREEC_BufferInitOp : IREEC_Op<"buffer.init", []> {
  let arguments = (ins
    IREEC_BufferType:$buffer,
    IREEC_AllocatorType:$allocator,
    IREEC_ByteSpanType:$byte_span
  );
  let assemblyFormat = "operands attr-dict";
}

//===----------------------------------------------------------------------===//
// `iree_byte_span_t` related Ops
//===----------------------------------------------------------------------===//

def IREEC_ByteSpanFromBufferOp : IREEC_Op<"byte_span.from_buffer", [
  DeclareOpInterfaceMethods<SymbolUserOpInterface>
]> {
  let arguments = (ins FlatSymbolRefAttr:$bufferRef);
  let results = (outs IREEC_ByteSpanType:$result);
  let assemblyFormat = "$bufferRef attr-dict `:` type($result)";

  let extraClassDefinition = [{
    LogicalResult ByteSpanFromBufferOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
      Operation *op = getOperation();
      if (!symbolTable.lookupNearestSymbolFrom<IREE::IREEC::GlobalBufferOp>(op, getBufferRefAttr())) {
        return op->emitError() << "ireec.global.byte_buffer op named '" << getBufferRef() << "' not found";
      }
      return success();
    }
  }];
}

//===----------------------------------------------------------------------===//
// `iree_vm_module_t` related Ops
//===----------------------------------------------------------------------===//

def IREEC_ModuleCastOp : IREEC_Op<"module.cast", []> {
  let arguments = (ins Arg<IREEC_VoidPointerType, "", [MemRead]>:$ptr);
  let results = (outs Res<IREEC_ModuleType, "", [MemWrite]>:$result);
  let assemblyFormat = "operands attr-dict `:` type($result)";
}

def IREEC_ModuleFreeOp : IREEC_Op<"module.free", []> {
  let arguments = (ins Arg<IREEC_ModuleType, "", [MemWrite]>:$module);
  let assemblyFormat = "operands attr-dict";
}

//===----------------------------------------------------------------------===//
// `iree_vm_module_state_t` related Ops
//===----------------------------------------------------------------------===//

def IREEC_ModuleStateDerivedAllocOp : IREEC_Op<"module_state.derived.alloc", []> {
  let arguments = (ins IREEC_AllocatorType:$allocator);
  let results = (outs Res<IREEC_ModuleStateDerivedType, "", [MemWrite]>:$result);
  let assemblyFormat = "operands attr-dict `:` qualified(type($result))";
}

def IREEC_ModuleStateDerivedBufferGetOp : IREEC_Op<"module_state.derived.buffer.get", []> {
  let arguments = (
    ins IREEC_ModuleStateDerivedType:$module_state,
    I32Attr:$index
  );
  let results = (outs IREEC_BufferType:$result);
  let assemblyFormat = "$module_state `[` $index `]` attr-dict `:` qualified(type($module_state)) `->` qualified(type($result))";
}

def IREEC_ModuleStateDerivedCastToBaseOp : IREEC_Op<"module_state.derived.cast.base", []> {
  let arguments = (ins IREEC_ModuleStateDerivedType:$module_state);
  let results = (outs IREEC_ModuleStateType:$result);
  let assemblyFormat = "operands attr-dict `:` qualified(type($module_state)) `->` qualified(type($result))";
}

def IREEC_ModuleStateDerivedRefGetOp : IREEC_Op<"module_state.derived.ref.get", []> {
  let arguments = (
    ins IREEC_ModuleStateDerivedType:$module_state,
    I32Attr:$index
  );
  let results = (outs IREEC_RefType:$result);
  let assemblyFormat = "$module_state `[` $index `]` attr-dict `:` qualified(type($module_state)) `->` qualified(type($result))";
}

//===----------------------------------------------------------------------===//
// `iree_vm_ref_t` related Ops
//===----------------------------------------------------------------------===//

def IREEC_RefClearOp : IREEC_Op<"ref.clear", []> {
  let arguments = (ins Arg<IREEC_RefType, "", [MemWrite]>:$ref);
  let assemblyFormat = "operands attr-dict";
}

def IREEC_RefNullOp : IREEC_Op<"ref.null", []> {
  let results = (outs Arg<IREEC_RefType, "", [MemAlloc, MemWrite]>:$result);
  let assemblyFormat = "attr-dict `:` type($result)";
}

def IREEC_RefReleaseOp : IREEC_Op<"ref.release", []> {
  let arguments = (ins Arg<IREEC_RefType, "", [MemWrite]>:$ref);
  let assemblyFormat = "operands attr-dict";
}

def IREEC_RefMoveOp : IREEC_Op<"ref.move", []> {
  let arguments = (ins
    Arg<IREEC_RefType, "", [MemRead, MemWrite]>:$ref,
    Arg<IREEC_RefType, "", [MemRead, MemWrite]>:$out_ref
  );
  let assemblyFormat = "$ref `->` $out_ref attr-dict";
}

def IREEC_RefMoveParallelOp : IREEC_Op<"ref.move.parallel", [SameVariadicOperandSize]> {
  let arguments = (ins
    Variadic<IREEC_RefType>:$refs,
    Variadic<IREEC_RefType>:$out_refs
  );
  let assemblyFormat = "`(` $refs `)` `->` `(` $out_refs `)` attr-dict";
}

def IREEC_RefRetainOp : IREEC_Op<"ref.retain", []> {
  let arguments = (ins
    Arg<IREEC_RefType, "", [MemRead, MemWrite]>:$ref,
    Arg<IREEC_RefType, "", [MemRead, MemWrite]>:$out_ref
  );
  let assemblyFormat = "$ref `->` $out_ref attr-dict";
}

def IREEC_RefRetainParallelOp : IREEC_Op<"ref.retain.parallel", [SameVariadicOperandSize]> {
  let arguments = (ins
    Variadic<IREEC_RefType>:$refs,
    Variadic<IREEC_RefType>:$out_refs
  );
  let assemblyFormat = "`(` $refs `)` `->` `(` $out_refs `)` attr-dict";
}

def IREEC_RefAssignOp : IREEC_Op<"ref.assign", []> {
  let arguments = (ins
    Arg<IREEC_RefType, "", [MemRead]>:$ref,
    Arg<IREEC_RefType, "", [MemRead, MemWrite]>:$out_ref
  );
  let assemblyFormat = "$ref `->` $out_ref attr-dict";
}

def IREEC_RefAssignParallelOp : IREEC_Op<"ref.assign.parallel", [SameVariadicOperandSize]> {
  let arguments = (ins
    Variadic<IREEC_RefType>:$refs,
    Variadic<IREEC_RefType>:$out_refs
  );
  let assemblyFormat = "`(` $refs `)` `->` `(` $out_refs `)` attr-dict";
}

//===----------------------------------------------------------------------===//
// `iree_status_t` related Ops
//===----------------------------------------------------------------------===//

def IREEC_StatusOkOp : IREEC_Op<"status.ok", []> {
  let results = (outs IREEC_StatusType:$result);
  let assemblyFormat = "attr-dict `:` type($result)";
}

def IREEC_StatusIsOkOp : IREEC_Op<"status.is_ok", []> {
  let arguments = (ins IREEC_StatusType:$status);
  let results = (outs I1:$result);
  let assemblyFormat = "operands attr-dict `:` type($result)";
}

def IREEC_StatusReturnIfErrorOp : IREEC_Op<"status.return_if_error", []> {
  let arguments = (ins IREEC_StatusType:$status);
  let assemblyFormat = "operands attr-dict";
  let extraClassDeclaration = [{
    LogicalResult verify() {
      auto function = cast<FunctionOpInterface>((*this)->getParentOp());
      const auto &result_types = function.getResultTypes();
      if (result_types.size() != 1) {
        return emitOpError("expected enclosing function (@")
          << function.getName() << ") to return one result";
      }
      auto resultType = result_types[0];
      if (!resultType.isa<StatusType>()) {
        return emitOpError("expected enclosing function (@")
          << function.getName() << ") to return '!ireec.status', got " 
          << resultType;
      }
      return success();
    }
  }];
}

#endif // IREE_DIALECT_IREEC_IR_IREEC
