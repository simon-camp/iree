// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/VM/Conversion/VMToEmitC/EmitCBuilders.h"

#include "llvm/ADT/ArrayRef.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"

namespace mlir {
namespace iree_compiler {
namespace emitc_builders {

namespace {
std::string mapUnaryOperator(UnaryOperator op) {
  switch (op) {
    case UnaryOperator::PLUS:
      return "+";
    case UnaryOperator::MINUS:
      return "-";
    case UnaryOperator::BITWISE_NOT:
      return "~";
    case UnaryOperator::LOGICAL_NOT:
      return "!";
  }
}

std::string mapBinaryOperator(BinaryOperator op) {
  switch (op) {
    case BinaryOperator::ADDITION:
      return "+";
    case BinaryOperator::SUBTRACTION:
      return "-";
    case BinaryOperator::PRODUCT:
      return "*";
    case BinaryOperator::DIVISION:
      return "/";
    case BinaryOperator::REMAINDER:
      return "%";
    case BinaryOperator::BITWISE_AND:
      return "&";
    case BinaryOperator::BITWISE_OR:
      return "|";
    case BinaryOperator::BITWISE_XOR:
      return "^";
    case BinaryOperator::BITWISE_LEFT_SHIFT:
      return "<<";
    case BinaryOperator::BITWISE_RIGHT_SHIFT:
      return ">>";
    case BinaryOperator::LOGICAL_AND:
      return "&&";
    case BinaryOperator::LOGICAL_OR:
      return "||";
    case BinaryOperator::EQUAL_TO:
      return "==";
    case BinaryOperator::NOT_EQUAL_TO:
      return "!=";
    case BinaryOperator::LESS_THAN:
      return "<";
    case BinaryOperator::GREATER_THAN:
      return ">";
    case BinaryOperator::LESS_THAN_OR_EQUAL:
      return "<=";
    case BinaryOperator::GREATER_THAN_OR_EQUAL:
      return ">=";
  }
}
}  // namespace

Value unaryOperator(OpBuilder builder, Location location, UnaryOperator op,
                    Value operand, Type resultType) {
  auto ctx = builder.getContext();

  return builder
      .create<emitc::CallOp>(
          /*location=*/location,
          /*type=*/resultType,
          /*callee=*/StringAttr::get(ctx, "EMITC_UNARY"),
          /*args=*/
          ArrayAttr::get(ctx,
                         {emitc::OpaqueAttr::get(ctx, mapUnaryOperator(op)),
                          builder.getIndexAttr(0)}),
          /*templateArgs=*/ArrayAttr{},
          /*operands=*/ArrayRef<Value>{operand})
      .getResult(0);
}

Value binaryOperator(OpBuilder builder, Location location, BinaryOperator op,
                     Value lhs, Value rhs, Type resultType) {
  auto ctx = builder.getContext();

  return builder
      .create<emitc::CallOp>(
          /*location=*/location,
          /*type=*/resultType,
          /*callee=*/StringAttr::get(ctx, "EMITC_BINARY"),
          /*args=*/
          ArrayAttr::get(ctx,
                         {emitc::OpaqueAttr::get(ctx, mapBinaryOperator(op)),
                          builder.getIndexAttr(0), builder.getIndexAttr(1)}),
          /*templateArgs=*/ArrayAttr{},
          /*operands=*/ArrayRef<Value>{lhs, rhs})
      .getResult(0);
}

Value allocateVariable(OpBuilder builder, Location location, Type type,
                       Optional<StringRef> initializer) {
  auto ctx = builder.getContext();
  return builder
      .create<emitc::VariableOp>(
          /*location=*/location,
          /*resultType=*/type,
          /*value=*/emitc::OpaqueAttr::get(ctx, initializer.value_or("")))
      .getResult();
}

Value addressOf(OpBuilder builder, Location location, Value operand) {
  auto ctx = builder.getContext();

  return builder
      .create<emitc::ApplyOp>(
          /*location=*/location,
          /*result=*/emitc::PointerType::get(operand.getType()),
          /*applicableOperator=*/StringAttr::get(ctx, "&"),
          /*operand=*/operand)
      .getResult();
}

Value contentsOf(OpBuilder builder, Location location, Value operand) {
  auto ctx = builder.getContext();

  Type type = operand.getType();
  assert(type.isa<emitc::PointerType>());

  return builder
      .create<emitc::ApplyOp>(
          /*location=*/location,
          /*result=*/type.cast<emitc::PointerType>().getPointee(),
          /*applicableOperator=*/StringAttr::get(ctx, "*"),
          /*operand=*/operand)
      .getResult();
}

Value sizeOf(OpBuilder builder, Location loc, Value value) {
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

Value sizeOf(OpBuilder builder, Location loc, Attribute attr) {
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

void memcpy(OpBuilder builder, Location location, Value dest, Value src,
            Value count) {
  auto ctx = builder.getContext();
  builder.create<emitc::CallOp>(
      /*location=*/location,
      /*type=*/TypeRange{},
      /*callee=*/StringAttr::get(ctx, "memcpy"),
      /*args=*/ArrayAttr{},
      /*templateArgs=*/ArrayAttr{},
      /*operands=*/ArrayRef<Value>{dest, src, count});
}

void memset(OpBuilder builder, Location location, Value dest, int ch,
            Value count) {
  auto ctx = builder.getContext();
  builder.create<emitc::CallOp>(
      /*location=*/location,
      /*type=*/TypeRange{},
      /*callee=*/StringAttr::get(ctx, "memset"),
      /*args=*/
      ArrayAttr::get(ctx,
                     {builder.getIndexAttr(0), builder.getUI32IntegerAttr(ch),
                      builder.getIndexAttr(1)}),
      /*templateArgs=*/ArrayAttr{},
      /*operands=*/
      ArrayRef<Value>{dest, count});
}

Value arrayElementAddress(OpBuilder builder, Location location, Type type,
                          IntegerAttr index, Value operand) {
  auto ctx = builder.getContext();
  return builder
      .create<emitc::CallOp>(
          /*location=*/location,
          /*type=*/type,
          /*callee=*/StringAttr::get(ctx, "EMITC_ARRAY_ELEMENT_ADDRESS"),
          /*args=*/
          ArrayAttr::get(ctx, {builder.getIndexAttr(0), index}),
          /*templateArgs=*/ArrayAttr{},
          /*operands=*/ArrayRef<Value>{operand})
      .getResult(0);
}

Value arrayElementAddress(OpBuilder builder, Location location, Type type,
                          Value index, Value operand) {
  auto ctx = builder.getContext();
  return builder
      .create<emitc::CallOp>(
          /*location=*/location,
          /*type=*/type,
          /*callee=*/StringAttr::get(ctx, "EMITC_ARRAY_ELEMENT_ADDRESS"),
          /*args=*/
          ArrayAttr::get(ctx,
                         {builder.getIndexAttr(0), builder.getIndexAttr(1)}),
          /*templateArgs=*/ArrayAttr{},
          /*operands=*/ArrayRef<Value>{operand, index})
      .getResult(0);
}

void structDefinition(OpBuilder builder, Location location,
                      StringRef structName, ArrayRef<StructField> fields) {
  std::string structBody;

  for (auto &field : fields) {
    structBody += field.type + " " + field.name + ";";
  }

  auto ctx = builder.getContext();

  builder.create<emitc::CallOp>(
      /*location=*/location, /*type=*/TypeRange{},
      /*callee=*/StringAttr::get(ctx, "EMITC_TYPEDEF_STRUCT"), /*args=*/
      ArrayAttr::get(ctx, {emitc::OpaqueAttr::get(ctx, structName),
                           emitc::OpaqueAttr::get(ctx, structBody)}),
      /*templateArgs=*/ArrayAttr{}, /*operands=*/ArrayRef<Value>{});
}

Value structMember(OpBuilder builder, Location location, Type type,
                   StringRef memberName, Value operand) {
  auto ctx = builder.getContext();
  return builder
      .create<emitc::CallOp>(
          /*location=*/location,
          /*type=*/type,
          /*callee=*/StringAttr::get(ctx, "EMITC_STRUCT_MEMBER"),
          /*args=*/
          ArrayAttr::get(ctx, {builder.getIndexAttr(0),
                               emitc::OpaqueAttr::get(ctx, memberName)}),
          /*templateArgs=*/ArrayAttr{},
          /*operands=*/ArrayRef<Value>{operand})
      .getResult(0);
}

void structMemberAssign(OpBuilder builder, Location location,
                        StringRef memberName, Value operand, Value data) {
  auto ctx = builder.getContext();
  builder.create<emitc::CallOp>(
      /*location=*/location,
      /*type=*/TypeRange{},
      /*callee=*/StringAttr::get(ctx, "EMITC_STRUCT_MEMBER_ASSIGN"),
      /*args=*/
      ArrayAttr::get(ctx, {builder.getIndexAttr(0),
                           emitc::OpaqueAttr::get(ctx, memberName),
                           builder.getIndexAttr(1)}),
      /*templateArgs=*/ArrayAttr{},
      /*operands=*/ArrayRef<Value>{operand, data});
}

void structMemberAssign(OpBuilder builder, Location location,
                        StringRef memberName, Value operand, StringRef data) {
  auto ctx = builder.getContext();
  builder.create<emitc::CallOp>(
      /*location=*/location,
      /*type=*/TypeRange{},
      /*callee=*/StringAttr::get(ctx, "EMITC_STRUCT_MEMBER_ASSIGN"),
      /*args=*/
      ArrayAttr::get(ctx, {builder.getIndexAttr(0),
                           emitc::OpaqueAttr::get(ctx, memberName),
                           emitc::OpaqueAttr::get(ctx, data)}),
      /*templateArgs=*/ArrayAttr{},
      /*operands=*/ArrayRef<Value>{operand});
}

Value structPtrMember(OpBuilder builder, Location location, Type type,
                      StringRef memberName, Value operand) {
  auto ctx = builder.getContext();
  return builder
      .create<emitc::CallOp>(
          /*location=*/location,
          /*type=*/type,
          /*callee=*/StringAttr::get(ctx, "EMITC_STRUCT_PTR_MEMBER"),
          /*args=*/
          ArrayAttr::get(ctx, {builder.getIndexAttr(0),
                               emitc::OpaqueAttr::get(ctx, memberName)}),
          /*templateArgs=*/ArrayAttr{},
          /*operands=*/ArrayRef<Value>{operand})
      .getResult(0);
}

void structPtrMemberAssign(OpBuilder builder, Location location,
                           StringRef memberName, Value operand, Value data) {
  auto ctx = builder.getContext();
  builder.create<emitc::CallOp>(
      /*location=*/location,
      /*type=*/TypeRange{},
      /*callee=*/StringAttr::get(ctx, "EMITC_STRUCT_PTR_MEMBER_ASSIGN"),
      /*args=*/
      ArrayAttr::get(ctx, {builder.getIndexAttr(0),
                           emitc::OpaqueAttr::get(ctx, memberName),
                           builder.getIndexAttr(1)}),
      /*templateArgs=*/ArrayAttr{},
      /*operands=*/ArrayRef<Value>{operand, data});
}

Value ireeOkStatus(OpBuilder builder, Location location) {
  auto ctx = builder.getContext();
  return builder
      .create<emitc::CallOp>(
          /*location=*/location,
          /*type=*/emitc::OpaqueType::get(ctx, "iree_status_t"),
          /*callee=*/StringAttr::get(ctx, "iree_ok_status"),
          /*args=*/ArrayAttr{},
          /*templateArgs=*/ArrayAttr{},
          /*operands=*/ArrayRef<Value>{})
      .getResult(0);
}

void ireeVmRefRelease(OpBuilder builder, Location location, Value operand) {
  auto ctx = builder.getContext();
  builder.create<emitc::CallOp>(
      /*location=*/location,
      /*type=*/TypeRange{},
      /*callee=*/StringAttr::get(ctx, "iree_vm_ref_release"),
      /*args=*/ArrayAttr{},
      /*templateArgs=*/ArrayAttr{},
      /*operands=*/ArrayRef<Value>{operand});
}

void ireeVmRefMove(OpBuilder builder, Location location, Value operand,
                   Value out_operand) {
  auto ctx = builder.getContext();
  builder.create<emitc::CallOp>(
      /*location=*/location,
      /*type=*/TypeRange{},
      /*callee=*/StringAttr::get(ctx, "iree_vm_ref_move"),
      /*args=*/ArrayAttr{},
      /*templateArgs=*/ArrayAttr{},
      /*operands=*/ArrayRef<Value>{operand, out_operand});
}

void ireeVmRefRetain(OpBuilder builder, Location location, Value operand,
                     Value out_operand) {
  auto ctx = builder.getContext();
  builder.create<emitc::CallOp>(
      /*location=*/location,
      /*type=*/TypeRange{},
      /*callee=*/StringAttr::get(ctx, "iree_vm_ref_retain"),
      /*args=*/ArrayAttr{},
      /*templateArgs=*/ArrayAttr{},
      /*operands=*/ArrayRef<Value>{operand, out_operand});
}

void ireeVmRefAssign(OpBuilder builder, Location location, Value operand,
                     Value out_operand) {
  auto ctx = builder.getContext();
  builder.create<emitc::CallOp>(
      /*location=*/location,
      /*type=*/TypeRange{},
      /*callee=*/StringAttr::get(ctx, "iree_vm_ref_assign"),
      /*args=*/ArrayAttr{},
      /*templateArgs=*/ArrayAttr{},
      /*operands=*/ArrayRef<Value>{operand, out_operand});
}

}  // namespace emitc_builders
}  // namespace iree_compiler
}  // namespace mlir
