// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_DIALECT_IREEC_IR_IREECATTRIBUTES
#define IREE_DIALECT_IREEC_IR_IREECATTRIBUTES

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"
include "iree/compiler/Dialect/IREEC/IR/IREECBase.td"

//===----------------------------------------------------------------------===//
// IREEC attribute definitions
//===----------------------------------------------------------------------===//

class IREEC_Attr<string name, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<IREEC_Dialect, name, traits> {
  let mnemonic = attrMnemonic;
}

def IREEC_StructFieldAttr : IREEC_Attr<"StructField", "struct_field"> {
  let parameters = (ins 
      "Type":$member_type,
      StringRefParameter<"the opaque value">:$member_name
  );
  let assemblyFormat = "`<` $member_name `:` $member_type `>`";
}

def IREEC_StructFieldArrayAttr :
    TypedArrayAttrBase<IREEC_StructFieldAttr,
                       "Array of struct fields">;

#endif // IREE_DIALECT_IREEC_IR_IREECATTRIBUTES
