// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_VM_CONVERSION_VMTOEMITC_EMITCBUILDERS_H_
#define IREE_COMPILER_DIALECT_VM_CONVERSION_VMTOEMITC_EMITCBUILDERS_H_

#include <optional>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"

#include "iree/compiler/Dialect/VM/Conversion/VMToEmitC/EmitCTypeConverter.h"

namespace mlir::iree_compiler::emitc_builders {

struct StructField {
  std::string type;
  std::string name;
  std::optional<size_t> arraySize = std::nullopt;

  bool isArray() const { return arraySize.has_value(); }
};

enum PreprocessorDirective {
  DEFINE = 0,
  UNDEF,
  IFDEF,
  IFNDEF,
  IF,
  ENDIF,
  ELSE,
  ELIF,
  LINE,
  ERROR,
  INCLUDE,
  PRAGMA
};

enum UnaryOperator {
  // arithmetic
  PLUS = 0,
  MINUS,
  BITWISE_NOT,
  // logical
  LOGICAL_NOT,
};

enum BinaryOperator {
  // arithmetic
  ADDITION = 0,
  SUBTRACTION,
  PRODUCT,
  DIVISION,
  REMAINDER,
  BITWISE_AND,
  BITWISE_OR,
  BITWISE_XOR,
  BITWISE_LEFT_SHIFT,
  BITWISE_RIGHT_SHIFT,
  // logical
  LOGICAL_AND,
  LOGICAL_OR,
  // comparison
  EQUAL_TO,
  NOT_EQUAL_TO,
  LESS_THAN,
  GREATER_THAN,
  LESS_THAN_OR_EQUAL,
  GREATER_THAN_OR_EQUAL,
};

Value unaryOperator(OpBuilder builder, Location location, UnaryOperator op,
                    Value operand, Type resultType);

TypedValue<emitc::LValueType> allocateVariable(OpBuilder builder,
                                               Location location, Type type,
                                               Attribute initializer);

TypedValue<emitc::LValueType>
allocateVariable(OpBuilder builder, Location location, Type type,
                 std::optional<StringRef> initializer = std::nullopt);

TypedValue<emitc::PointerType> addressOf(OpBuilder builder, Location location,
                                         TypedValue<emitc::LValueType> operand);

/// Materialize a value as an emitc LValue by assigning to a local variable. As
/// this generates a variable declaration followed by an assignment padding
/// bytes of the result have an indetermined value.
TypedValue<emitc::LValueType> asLValue(OpBuilder builder, Location location,
                                       Value value);

Value contentsOf(OpBuilder builder, Location location,
                 TypedValue<emitc::PointerType> operand);

Value sizeOf(OpBuilder builder, Location location, Attribute attr);

Value sizeOf(OpBuilder builder, Location location, Value value);

void memcpy(OpBuilder builder, Location location,
            TypedValue<emitc::PointerType> dest,
            TypedValue<emitc::PointerType> src, Value count);

void memset(OpBuilder builder, Location location,
            TypedValue<emitc::PointerType> dest, int ch, Value count);

Value arrayElement(OpBuilder builder, Location location, Type type,
                   size_t index, Value operand);

Value arrayElementAddress(OpBuilder builder, Location location, Type type,
                          IntegerAttr index, Value operand);

Value arrayElementAddress(OpBuilder builder, Location location, Type type,
                          Value index, Value operand);

void arrayElementAssign(OpBuilder builder, Location location, Value array,
                        size_t index, Value value);

Value loadLValue(OpBuilder builder, Location location,
                 TypedValue<emitc::LValueType> operand);

void structDefinition(OpBuilder builder, Location location,
                      StringRef structName, ArrayRef<StructField> fields);

Value structMember(OpBuilder builder, Location location, Type type,
                   StringRef memberName, Value operand);

TypedValue<emitc::PointerType>
structMemberAddress(OpBuilder builder, Location location, Type type,
                    StringRef memberName,
                    TypedValue<emitc::LValueType> operand);

void structMemberAssign(OpBuilder builder, Location location,
                        StringRef memberName,
                        TypedValue<emitc::LValueType> operand, Value data);

void structMemberAssign(OpBuilder builder, Location location,
                        StringRef memberName,
                        TypedValue<emitc::LValueType> operand, StringRef data);

Value structPtrMember(OpBuilder builder, Location location, Type type,
                      StringRef memberName, Value operand);

void structPtrMemberAssign(OpBuilder builder, Location location,
                           StringRef memberName,
                           TypedValue<emitc::LValueType> operand, Value data);

Value ireeMakeCstringView(OpBuilder builder, Location location,
                          std::string str);

Value ireeOkStatus(OpBuilder builder, Location location);

Value ireeVmInstanceLookupType(OpBuilder builder, Location location,
                               Value instance, Value stringView);

void ireeVmRefRelease(OpBuilder builder, Location location, Value operand);

emitc::VerbatimOp preprocessorDirective(OpBuilder builder, Location location,
                                        PreprocessorDirective directive,
                                        StringRef value);

} // namespace mlir::iree_compiler::emitc_builders

#endif // IREE_COMPILER_DIALECT_VM_CONVERSION_VMTOEMITC_EMITCBUILDERS_H_
