// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


#ifndef IREE_DIALECT_IREEC_IR_IREECTYPES
#define IREE_DIALECT_IREEC_IR_IREECTYPES

include "mlir/IR/AttrTypeBase.td"
include "iree/compiler/Dialect/IREEC/IR/IREECBase.td"

//===----------------------------------------------------------------------===//
// IREEC type definitions
//===----------------------------------------------------------------------===//

class IREEC_Type<string name, string typeMnemonic>
    : TypeDef<IREEC_Dialect, name> {
  let mnemonic = typeMnemonic;
}

def IREEC_PointerType : IREEC_Type<"Pointer", "ptr"> {
  let parameters = (ins "Type":$pointee);
  let builders = [
    TypeBuilderWithInferredContext<(ins "Type":$pointee), [{
      return $_get(pointee.getContext(), pointee);
    }]>
  ];
  let assemblyFormat = "`<` qualified($pointee) `>`";
}

def IREEC_VoidPointerType : IREEC_Type<"VoidPointer", "voidptr"> {
}

def IREEC_StructRefType : IREEC_Type<"StructRef", "struct_ref"> {
  let parameters = (ins "FlatSymbolRefAttr":$struct_ref);
  let builders = [
    TypeBuilderWithInferredContext<(ins "FlatSymbolRefAttr":$struct_ref), [{
      return $_get(struct_ref.getContext(), struct_ref);
    }]>
  ];
  let assemblyFormat = "`<` $struct_ref `>`";
}

def IREEC_AllocatorType : IREEC_Type<"Allocator", "allocator"> {
}

def IREEC_BufferType : IREEC_Type<"Buffer", "buffer"> {
}

def IREEC_ModuleType : IREEC_Type<"Module", "module"> {
}

def IREEC_ModuleStateType : IREEC_Type<"ModuleState", "module_state"> {
}

def IREEC_ModuleStateDerivedType : IREEC_Type<"ModuleStateDerived", "module_state.derived"> {
  let parameters = (ins StringRefParameter<"the name of the generated struct type">:$name);
  let assemblyFormat = "`<``` `type_name``` `=``` $name `>`";
}

def IREEC_ByteSpanType : IREEC_Type<"ByteSpan", "byte_span"> {
}

def IREEC_RefType : IREEC_Type<"Ref", "ref"> {
}

def IREEC_StackType : IREEC_Type<"Stack", "stack"> {
}

def IREEC_StatusType : IREEC_Type<"Status", "status"> {
}

#endif // IREE_DIALECT_IREEC_IR_IREECTYPES
