// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_VM_SHIMS_EMITC_H_
#define IREE_VM_SHIMS_EMITC_H_

#include "iree/vm/module.h"
#include "iree/vm/shims.h"
#include "iree/vm/stack.h"
#include "string.h"

// see Calling convetion in module.h

#define CONCAT_(a, b) a##b
#define CONCAT(a, b) CONCAT_(a, b)

#define COPY_VALUE_AND_ADVANCE(dest, src, type) \
  memcpy(dest, src, sizeof(type));              \
  dest += sizeof(type);
#define COPY_REF_AND_ADVANCE(dest, src)            \
  iree_vm_ref_assign(src, (iree_vm_ref_t*)(dest)); \
  dest += sizeof(iree_vm_ref_t);

#define COPY_VARARG_VALUE_AND_ADVANCE(varargs, dest, type) \
  COPY_VARARG_VALUE_AND_ADVANCE_(varargs, dest, type,      \
                                 CONCAT(_temp_, __COUNTER__))
#define COPY_VARARG_VALUE_AND_ADVANCE_(varargs, dest, type, temp) \
  type temp = va_arg(varargs, type);                              \
  memcpy(dest, &temp, sizeof(type));                              \
  dest += sizeof(type);

#define COPY_VARARG_REF_AND_ADVANCE(varargs, dest) \
  COPY_VARARG_REF_AND_ADVANCE_(varargs, dest, CONCAT(_temp_, __COUNTER__))
#define COPY_VARARG_REF_AND_ADVANCE_(varargs, dest, temp) \
  iree_vm_ref_t* temp = va_arg(varargs, iree_vm_ref_t*);  \
  iree_vm_ref_assign(temp, (iree_vm_ref_t*)(dest));       \
  dest += sizeof(iree_vm_ref_t);

// 0v_v
typedef iree_status_t (*call_0v_v_t)(iree_vm_stack_t* stack, void* module_ptr,
                                     void* module_state);

static iree_status_t call_0v_v_shim(iree_vm_stack_t* stack,
                                    const iree_vm_function_call_t* call,
                                    call_0v_v_t target_fn, void* module,
                                    void* module_state,
                                    iree_vm_execution_result_t* out_result) {
  return target_fn(stack, module, module_state);
}

// 0v_i
typedef iree_status_t (*call_0v_i_t)(iree_vm_stack_t* stack, void* module_ptr,
                                     void* module_state, int32_t* res0);

static iree_status_t call_0v_i_shim(iree_vm_stack_t* stack,
                                    const iree_vm_function_call_t* call,
                                    call_0v_i_t target_fn, void* module,
                                    void* module_state,
                                    iree_vm_execution_result_t* out_result) {
  iree_vm_abi_i_t* results = (iree_vm_abi_i_t*)call->results.data;

  return target_fn(stack, module, module_state, &results->i0);
}

// 0v_r
typedef iree_status_t (*call_0v_r_t)(iree_vm_stack_t* stack, void* module_ptr,
                                     void* module_state, iree_vm_ref_t* res0);

static iree_status_t call_0v_r_shim(iree_vm_stack_t* stack,
                                    const iree_vm_function_call_t* call,
                                    call_0v_r_t target_fn, void* module,
                                    void* module_state,
                                    iree_vm_execution_result_t* out_result) {
  iree_vm_abi_r_t* results = (iree_vm_abi_r_t*)call->results.data;

  return target_fn(stack, module, module_state, &results->r0);
}

// 0i_i
typedef iree_status_t (*call_0i_i_t)(iree_vm_stack_t* stack, void* module_ptr,
                                     void* module_state, int32_t arg0,
                                     int32_t* res0);

static iree_status_t call_0i_i_shim(iree_vm_stack_t* stack,
                                    const iree_vm_function_call_t* call,
                                    call_0i_i_t target_fn, void* module,
                                    void* module_state,
                                    iree_vm_execution_result_t* out_result) {
  iree_vm_abi_i_t* args = (iree_vm_abi_i_t*)call->arguments.data;
  iree_vm_abi_i_t* results = (iree_vm_abi_i_t*)call->results.data;

  return target_fn(stack, module, module_state, args->i0, &results->i0);
}

// 0ii_i
typedef iree_status_t (*call_0ii_i_t)(iree_vm_stack_t* stack, void* module_ptr,
                                      void* module_state, int32_t arg0,
                                      int32_t arg1, int32_t* res0);

static iree_status_t call_0ii_i_shim(iree_vm_stack_t* stack,
                                     const iree_vm_function_call_t* call,
                                     call_0ii_i_t target_fn, void* module,
                                     void* module_state,
                                     iree_vm_execution_result_t* out_result) {
  iree_vm_abi_ii_t* args = (iree_vm_abi_ii_t*)call->arguments.data;
  iree_vm_abi_i_t* results = (iree_vm_abi_i_t*)call->results.data;

  return target_fn(stack, module, module_state, args->i0, args->i1,
                   &results->i0);
}

// 0rr_r
typedef iree_status_t (*call_0rr_r_t)(iree_vm_stack_t* stack, void* module_ptr,
                                      void* module_state, iree_vm_ref_t* arg0,
                                      iree_vm_ref_t* arg1, iree_vm_ref_t* res0);

static iree_status_t call_0rr_r_shim(iree_vm_stack_t* stack,
                                     const iree_vm_function_call_t* call,
                                     call_0rr_r_t target_fn, void* module,
                                     void* module_state,
                                     iree_vm_execution_result_t* out_result) {
  iree_vm_abi_rr_t* args = (iree_vm_abi_rr_t*)call->arguments.data;
  iree_vm_abi_r_t* results = (iree_vm_abi_r_t*)call->results.data;

  return target_fn(stack, module, module_state, &args->r0, &args->r1,
                   &results->r0);
}

// fixed imports

// 0i_i
static iree_status_t call_0i_i_import(iree_vm_stack_t* stack,
                                      const iree_vm_function_t* import,
                                      int32_t arg0, int32_t* res0) {
  iree_vm_abi_i_t arguments;
  arguments.i0 = arg0;

  iree_vm_abi_i_t results;
  memset(&results, 0, sizeof(results));

  iree_vm_function_call_t call;
  call.function = *import;
  call.arguments = iree_make_byte_span(&arguments, sizeof(arguments));
  call.results = iree_make_byte_span(&results, sizeof(results));

  iree_vm_execution_result_t result;
  memset(&result, 0, sizeof(result));

  iree_status_t status =
      import->module->begin_call(import->module, stack, &call, &result);
  *res0 = results.i0;
  return status;
}

// 0r_r
static iree_status_t call_0r_r_import(iree_vm_stack_t* stack,
                                      const iree_vm_function_t* import,
                                      iree_vm_ref_t* arg0,
                                      iree_vm_ref_t* res0) {
  iree_vm_abi_r_t arguments;
  iree_vm_ref_assign(arg0, &arguments.r0);

  iree_vm_abi_r_t results;
  memset(&results, 0, sizeof(results));

  iree_vm_function_call_t call;
  call.function = *import;
  call.arguments = iree_make_byte_span(&arguments, sizeof(arguments));
  call.results = iree_make_byte_span(&results, sizeof(results));

  iree_vm_execution_result_t result;
  memset(&result, 0, sizeof(result));

  iree_status_t status =
      import->module->begin_call(import->module, stack, &call, &result);
  iree_vm_ref_move(&results.r0, res0);
  return status;
}

// 0r_v
static iree_status_t call_0r_v_import(iree_vm_stack_t* stack,
                                      const iree_vm_function_t* import,
                                      iree_vm_ref_t* arg0) {
  iree_vm_abi_r_t arguments;
  iree_vm_ref_assign(arg0, &arguments.r0);

  iree_vm_abi_v_t results;
  memset(&results, 0, sizeof(results));

  iree_vm_function_call_t call;
  call.function = *import;
  call.arguments = iree_make_byte_span(&arguments, sizeof(arguments));
  call.results = iree_make_byte_span(&results, sizeof(results));

  iree_vm_execution_result_t result;
  memset(&result, 0, sizeof(result));

  iree_status_t status =
      import->module->begin_call(import->module, stack, &call, &result);

  return status;
}

// 0rii_r
static iree_status_t call_0rii_r_import(iree_vm_stack_t* stack,
                                        const iree_vm_function_t* import,
                                        iree_vm_ref_t* arg0, int32_t arg1,
                                        int32_t arg2, iree_vm_ref_t* res0) {
  iree_vm_abi_rii_t arguments;
  iree_vm_ref_assign(arg0, &arguments.r0);
  arguments.i1 = arg1;
  arguments.i2 = arg2;

  iree_vm_abi_r_t results;
  memset(&results, 0, sizeof(results));

  iree_vm_function_call_t call;
  call.function = *import;
  call.arguments = iree_make_byte_span(&arguments, sizeof(arguments));
  call.results = iree_make_byte_span(&results, sizeof(results));

  iree_vm_execution_result_t result;
  memset(&result, 0, sizeof(result));

  iree_status_t status =
      import->module->begin_call(import->module, stack, &call, &result);
  iree_vm_ref_move(&results.r0, res0);
  return status;
}

// 0riii_r
static iree_status_t call_0riii_r_import(iree_vm_stack_t* stack,
                                         const iree_vm_function_t* import,
                                         iree_vm_ref_t* arg0, int32_t arg1,
                                         int32_t arg2, int32_t arg3,
                                         iree_vm_ref_t* res0) {
  iree_vm_abi_riii_t arguments;
  iree_vm_ref_assign(arg0, &arguments.r0);
  arguments.i1 = arg1;
  arguments.i2 = arg2;
  arguments.i3 = arg3;

  iree_vm_abi_r_t results;
  memset(&results, 0, sizeof(results));

  iree_vm_function_call_t call;
  call.function = *import;
  call.arguments = iree_make_byte_span(&arguments, sizeof(arguments));
  call.results = iree_make_byte_span(&results, sizeof(results));

  iree_vm_execution_result_t result;
  memset(&result, 0, sizeof(result));

  iree_status_t status =
      import->module->begin_call(import->module, stack, &call, &result);
  iree_vm_ref_move(&results.r0, res0);
  return status;
}

// 0riii_v
static iree_status_t call_0riii_v_import(iree_vm_stack_t* stack,
                                         const iree_vm_function_t* import,
                                         iree_vm_ref_t* arg0, int32_t arg1,
                                         int32_t arg2, int32_t arg3) {
  iree_vm_abi_riii_t arguments;
  iree_vm_ref_assign(arg0, &arguments.r0);
  arguments.i1 = arg1;
  arguments.i2 = arg2;
  arguments.i3 = arg3;

  iree_vm_abi_v_t results;
  memset(&results, 0, sizeof(results));

  iree_vm_function_call_t call;
  call.function = *import;
  call.arguments = iree_make_byte_span(&arguments, sizeof(arguments));
  call.results = iree_make_byte_span(&results, sizeof(results));

  iree_vm_execution_result_t result;
  memset(&result, 0, sizeof(result));

  iree_status_t status =
      import->module->begin_call(import->module, stack, &call, &result);
  return status;
}

// 0rr_v
static iree_status_t call_0rr_v_import(iree_vm_stack_t* stack,
                                       const iree_vm_function_t* import,
                                       iree_vm_ref_t* arg0,
                                       iree_vm_ref_t* arg1) {
  iree_vm_abi_rr_t arguments;
  iree_vm_ref_assign(arg0, &arguments.r0);
  iree_vm_ref_assign(arg1, &arguments.r1);

  iree_vm_abi_v_t results;
  memset(&results, 0, sizeof(results));

  iree_vm_function_call_t call;
  call.function = *import;
  call.arguments = iree_make_byte_span(&arguments, sizeof(arguments));
  call.results = iree_make_byte_span(&results, sizeof(results));

  iree_vm_execution_result_t result;
  memset(&result, 0, sizeof(result));

  iree_status_t status =
      import->module->begin_call(import->module, stack, &call, &result);
  return status;
}

// 0rriiii_v
static iree_status_t call_0rriiii_v_import(iree_vm_stack_t* stack,
                                           const iree_vm_function_t* import,
                                           iree_vm_ref_t* arg0,
                                           iree_vm_ref_t* arg1, int32_t arg2,
                                           int32_t arg3, int32_t arg4,
                                           int32_t arg5) {
  iree_vm_abi_rriiii_t arguments;
  iree_vm_ref_assign(arg0, &arguments.r0);
  iree_vm_ref_assign(arg1, &arguments.r1);
  arguments.i2 = arg2;
  arguments.i3 = arg3;
  arguments.i4 = arg4;
  arguments.i5 = arg5;

  iree_vm_abi_v_t results;
  memset(&results, 0, sizeof(results));

  iree_vm_function_call_t call;
  call.function = *import;
  call.arguments = iree_make_byte_span(&arguments, sizeof(arguments));
  call.results = iree_make_byte_span(&results, sizeof(results));

  iree_vm_execution_result_t result;
  memset(&result, 0, sizeof(result));

  iree_status_t status =
      import->module->begin_call(import->module, stack, &call, &result);
  return status;
}

// 0rrr_ii
static iree_status_t call_0rrr_ii_import(iree_vm_stack_t* stack,
                                         const iree_vm_function_t* import,
                                         iree_vm_ref_t* arg0,
                                         iree_vm_ref_t* arg1,
                                         iree_vm_ref_t* arg2, int32_t* res0,
                                         int32_t* res1) {
  iree_vm_abi_rrr_t arguments;
  iree_vm_ref_assign(arg0, &arguments.r0);
  iree_vm_ref_assign(arg1, &arguments.r1);
  iree_vm_ref_assign(arg2, &arguments.r2);

  iree_vm_abi_ii_t results;
  memset(&results, 0, sizeof(results));

  iree_vm_function_call_t call;
  call.function = *import;
  call.arguments = iree_make_byte_span(&arguments, sizeof(arguments));
  call.results = iree_make_byte_span(&results, sizeof(results));

  iree_vm_execution_result_t result;
  memset(&result, 0, sizeof(result));

  iree_status_t status =
      import->module->begin_call(import->module, stack, &call, &result);
  *res0 = results.i0;
  *res1 = results.i1;
  return status;
}

// 0v_r
static iree_status_t call_0v_r_import(iree_vm_stack_t* stack,
                                      const iree_vm_function_t* import,
                                      iree_vm_ref_t* res0) {
  iree_vm_abi_v_t arguments;
  iree_vm_abi_r_t results;
  memset(&results, 0, sizeof(results));
  iree_vm_function_call_t call;

  call.function = *import;
  call.arguments = iree_make_byte_span(&arguments, sizeof(arguments));
  call.results = iree_make_byte_span(&results, sizeof(results));

  iree_vm_execution_result_t result;
  memset(&result, 0, sizeof(result));
  iree_status_t status =
      import->module->begin_call(import->module, stack, &call, &result);

  iree_vm_ref_move(&results.r0, res0);
  return status;
}

// 0v_v
static iree_status_t call_0v_v_import(iree_vm_stack_t* stack,
                                      const iree_vm_function_t* import) {
  iree_vm_abi_v_t arguments;
  iree_vm_abi_v_t results;
  memset(&results, 0, sizeof(results));
  iree_vm_function_call_t call;

  call.function = *import;
  call.arguments = iree_make_byte_span(&arguments, sizeof(arguments));
  call.results = iree_make_byte_span(&results, sizeof(results));

  iree_vm_execution_result_t result;
  memset(&result, 0, sizeof(result));
  iree_status_t status =
      import->module->begin_call(import->module, stack, &call, &result);

  return status;
}

// variadic imports

// 0riCiiiD_r
static iree_status_t call_0riCiiiD_r_import(iree_vm_stack_t* stack,
                                            const iree_vm_function_t* import,
                                            int32_t spanCount,
                                            iree_vm_ref_t* arg0, int32_t arg1,
                                            ...) {
  int32_t spanSize = sizeof(int32_t) + sizeof(int32_t) + sizeof(int32_t);
  int32_t dataLength = sizeof(iree_vm_ref_t) + sizeof(int32_t) +
                       sizeof(int32_t) + spanCount * spanSize;

  iree_vm_abi_r_t results;
  memset(&results, 0, sizeof(results));
  iree_vm_function_call_t call;

  call.function = *import;
  call.arguments.data_length = dataLength;
  call.arguments.data = (uint8_t*)iree_alloca(call.arguments.data_length);
  memset(call.arguments.data, 0, call.arguments.data_length);

  uint8_t* ptr = call.arguments.data;
  COPY_REF_AND_ADVANCE(ptr, arg0);
  COPY_VALUE_AND_ADVANCE(ptr, &arg1, int32_t);
  COPY_VALUE_AND_ADVANCE(ptr, &spanCount, int32_t);

  va_list varargs;
  va_start(varargs, arg1);
  for (int i = 0; i < spanCount; i++) {
    COPY_VARARG_VALUE_AND_ADVANCE(varargs, ptr, int32_t);
    COPY_VARARG_VALUE_AND_ADVANCE(varargs, ptr, int32_t);
    COPY_VARARG_VALUE_AND_ADVANCE(varargs, ptr, int32_t);
  }
  iree_vm_ref_t* res0 = va_arg(varargs, iree_vm_ref_t*);
  va_end(varargs);

  call.results = iree_make_byte_span(&results, sizeof(results));

  iree_vm_execution_result_t result;
  memset(&result, 0, sizeof(result));
  iree_status_t status =
      import->module->begin_call(import->module, stack, &call, &result);

  iree_vm_ref_move(&results.r0, res0);
  return status;
}

// 0riCrD_r
static iree_status_t call_0riCrD_r_import(iree_vm_stack_t* stack,
                                          const iree_vm_function_t* import,
                                          int32_t spanCount,
                                          iree_vm_ref_t* arg0, int32_t arg1,
                                          ...) {
  int32_t spanSize = sizeof(iree_vm_ref_t);
  int32_t dataLength = sizeof(iree_vm_ref_t) + sizeof(int32_t) +
                       sizeof(int32_t) + spanCount * spanSize;

  iree_vm_abi_r_t results;
  memset(&results, 0, sizeof(results));
  iree_vm_function_call_t call;

  call.function = *import;
  call.arguments.data_length = dataLength;
  call.arguments.data = (uint8_t*)iree_alloca(call.arguments.data_length);
  memset(call.arguments.data, 0, call.arguments.data_length);

  uint8_t* ptr = call.arguments.data;
  COPY_REF_AND_ADVANCE(ptr, arg0);
  COPY_VALUE_AND_ADVANCE(ptr, &arg1, int32_t);
  COPY_VALUE_AND_ADVANCE(ptr, &spanCount, int32_t);

  va_list varargs;
  va_start(varargs, arg1);
  for (int i = 0; i < spanCount; i++) {
    COPY_VARARG_REF_AND_ADVANCE(varargs, ptr);
  }
  iree_vm_ref_t* res0 = va_arg(varargs, iree_vm_ref_t*);
  va_end(varargs);

  call.results = iree_make_byte_span(&results, sizeof(results));

  iree_vm_execution_result_t result;
  memset(&result, 0, sizeof(result));
  iree_status_t status =
      import->module->begin_call(import->module, stack, &call, &result);

  iree_vm_ref_move(&results.r0, res0);
  return status;
}

// 0rrrCrD_r
static iree_status_t call_0rrrCrD_r_import(
    iree_vm_stack_t* stack, const iree_vm_function_t* import, int32_t spanCount,
    iree_vm_ref_t* arg0, iree_vm_ref_t* arg1, iree_vm_ref_t* arg2, ...) {
  int32_t spanSize = sizeof(iree_vm_ref_t);
  int32_t dataLength = sizeof(iree_vm_ref_t) + sizeof(iree_vm_ref_t) +
                       sizeof(iree_vm_ref_t) + sizeof(int32_t) +
                       spanCount * spanSize;

  iree_vm_abi_r_t results;
  memset(&results, 0, sizeof(results));
  iree_vm_function_call_t call;

  call.function = *import;
  call.arguments.data_length = dataLength;
  call.arguments.data = (uint8_t*)iree_alloca(call.arguments.data_length);
  memset(call.arguments.data, 0, call.arguments.data_length);

  uint8_t* ptr = call.arguments.data;
  COPY_REF_AND_ADVANCE(ptr, arg0);
  COPY_REF_AND_ADVANCE(ptr, arg1);
  COPY_REF_AND_ADVANCE(ptr, arg2);
  COPY_VALUE_AND_ADVANCE(ptr, &spanCount, int32_t);

  va_list varargs;
  va_start(varargs, arg2);
  for (int i = 0; i < spanCount; i++) {
    COPY_VARARG_REF_AND_ADVANCE(varargs, ptr);
  }
  iree_vm_ref_t* res0 = va_arg(varargs, iree_vm_ref_t*);
  va_end(varargs);

  call.results = iree_make_byte_span(&results, sizeof(results));

  iree_vm_execution_result_t result;
  memset(&result, 0, sizeof(result));
  iree_status_t status =
      import->module->begin_call(import->module, stack, &call, &result);

  iree_vm_ref_move(&results.r0, res0);
  return status;
}

// 0rriCiriiD_v
static iree_status_t call_0rriCiriiD_v_import(
    iree_vm_stack_t* stack, const iree_vm_function_t* import, int32_t spanCount,
    iree_vm_ref_t* arg0, iree_vm_ref_t* arg1, int32_t arg2, ...) {
  int32_t spanSize = sizeof(int32_t) + sizeof(iree_vm_ref_t) + sizeof(int32_t) +
                     sizeof(int32_t);
  int32_t dataLength = sizeof(iree_vm_ref_t) + sizeof(iree_vm_ref_t) +
                       sizeof(int32_t) + sizeof(int32_t) + spanCount * spanSize;

  iree_vm_abi_v_t results;
  memset(&results, 0, sizeof(results));
  iree_vm_function_call_t call;

  call.function = *import;
  call.arguments.data_length = dataLength;
  call.arguments.data = (uint8_t*)iree_alloca(call.arguments.data_length);
  memset(call.arguments.data, 0, call.arguments.data_length);

  uint8_t* ptr = call.arguments.data;
  COPY_REF_AND_ADVANCE(ptr, arg0);
  COPY_REF_AND_ADVANCE(ptr, arg1);
  COPY_VALUE_AND_ADVANCE(ptr, &arg2, int32_t);
  COPY_VALUE_AND_ADVANCE(ptr, &spanCount, int32_t);

  va_list varargs;
  va_start(varargs, arg2);
  for (int i = 0; i < spanCount; i++) {
    COPY_VARARG_VALUE_AND_ADVANCE(varargs, ptr, int32_t);
    COPY_VARARG_REF_AND_ADVANCE(varargs, ptr);
    COPY_VARARG_VALUE_AND_ADVANCE(varargs, ptr, int32_t);
    COPY_VARARG_VALUE_AND_ADVANCE(varargs, ptr, int32_t);
  }
  va_end(varargs);

  call.results = iree_make_byte_span(&results, sizeof(results));

  iree_vm_execution_result_t result;
  memset(&result, 0, sizeof(result));
  iree_status_t status =
      import->module->begin_call(import->module, stack, &call, &result);

  return status;
}

// 0riiCiD_r
static iree_status_t call_0riiCiD_r_import(iree_vm_stack_t* stack,
                                           const iree_vm_function_t* import,
                                           int32_t spanCount,
                                           iree_vm_ref_t* arg0, int32_t arg1,
                                           int32_t arg2, ...) {
  int32_t spanSize = sizeof(int32_t);
  int32_t dataLength = sizeof(iree_vm_ref_t) + sizeof(int32_t) +
                       sizeof(int32_t) + sizeof(int32_t) + spanCount * spanSize;

  iree_vm_abi_r_t results;
  memset(&results, 0, sizeof(results));
  iree_vm_function_call_t call;

  call.function = *import;
  call.arguments.data_length = dataLength;
  call.arguments.data = (uint8_t*)iree_alloca(call.arguments.data_length);
  memset(call.arguments.data, 0, call.arguments.data_length);

  uint8_t* ptr = call.arguments.data;
  COPY_REF_AND_ADVANCE(ptr, arg0);
  COPY_VALUE_AND_ADVANCE(ptr, &arg1, int32_t);
  COPY_VALUE_AND_ADVANCE(ptr, &arg2, int32_t);
  COPY_VALUE_AND_ADVANCE(ptr, &spanCount, int32_t);

  va_list varargs;
  va_start(varargs, arg1);
  for (int i = 0; i < spanCount; i++) {
    COPY_VARARG_VALUE_AND_ADVANCE(varargs, ptr, int32_t);
  }
  iree_vm_ref_t* res0 = va_arg(varargs, iree_vm_ref_t*);
  va_end(varargs);

  call.results = iree_make_byte_span(&results, sizeof(results));

  iree_vm_execution_result_t result;
  memset(&result, 0, sizeof(result));
  iree_status_t status =
      import->module->begin_call(import->module, stack, &call, &result);

  iree_vm_ref_move(&results.r0, res0);
  return status;
}

#undef CONCAT_
#undef CONCAT

#undef COPY_VALUE_AND_ADVANCE
#undef COPY_VARARG_VALUE_AND_ADVANCE
#undef COPY_VARARG_VALUE_AND_ADVANCE_
#undef COPY_REF_AND_ADVANCE
#undef COPY_VARARG_REF_AND_ADVANCE
#undef COPY_VARARG_REF_AND_ADVANCE_

#endif  // IREE_VM_SHIMS_EMITC_H_#endif  // IREE_VM_SHIMS_EMITC_H_
