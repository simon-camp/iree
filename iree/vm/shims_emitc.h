// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_VM_SHIMS_EMITC_H_
#define IREE_VM_SHIMS_EMITC_H_

#include "iree/vm/module.h"
#include "iree/vm/stack.h"

// see Calling convetion in module.h
// Variadic arguments are not supported

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
  typedef struct {
    int32_t ret0;
  } results_t;

  results_t* results = (results_t*)call->results.data;

  return target_fn(stack, module, module_state, &results->ret0);
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
  typedef struct {
    int32_t arg0;
  } args_t;
  typedef struct {
    int32_t ret0;
  } results_t;

  const args_t* args = (const args_t*)call->arguments.data;
  results_t* results = (results_t*)call->results.data;

  return target_fn(stack, module, module_state, args->arg0, &results->ret0);
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
  typedef struct {
    int32_t arg0;
    int32_t arg1;
  } args_t;
  typedef struct {
    int32_t ret0;
  } results_t;

  const args_t* args = (const args_t*)call->arguments.data;
  results_t* results = (results_t*)call->results.data;
  return target_fn(stack, module, module_state, args->arg0, args->arg1,
                   &results->ret0);
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
  typedef struct {
    iree_vm_ref_t arg0;
    iree_vm_ref_t arg1;
  } args_t;
  typedef struct {
    iree_vm_ref_t ret0;
  } results_t;

  const args_t* args = (const args_t*)call->arguments.data;
  results_t* results = (results_t*)call->results.data;

  return target_fn(stack, module, module_state, &args->arg0, &args->arg1,
                   &results->ret0);
}

// fixed imports

// 0i_i
static iree_status_t call_0i_i_import(iree_vm_stack_t* stack,
                                      const iree_vm_function_t* import,
                                      int32_t arg0, int32_t* res0) {
  iree_vm_abi_i_t arguments;
  arguments.i0 = arg0;

  iree_vm_abi_i_t results;

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
  return iree_ok_status();
}

// 0r_v
static iree_status_t call_0r_v_import(iree_vm_stack_t* stack,
                                      const iree_vm_function_t* import,
                                      iree_vm_ref_t* arg0) {
  return iree_ok_status();
}

// 0rii_r
static iree_status_t call_0rii_r_import(iree_vm_stack_t* stack,
                                        const iree_vm_function_t* import,
                                        iree_vm_ref_t* arg0, int32_t arg1,
                                        int32_t arg2, iree_vm_ref_t* res0) {
  return iree_ok_status();
}

// 0riii_r
static iree_status_t call_0riii_r_import(iree_vm_stack_t* stack,
                                         const iree_vm_function_t* import,
                                         iree_vm_ref_t* arg0, int32_t arg2,
                                         int32_t arg3, int32_t arg4,
                                         iree_vm_ref_t* res0) {
  return iree_ok_status();
}

// 0riii_v
static iree_status_t call_0riii_v_import(iree_vm_stack_t* stack,
                                         const iree_vm_function_t* import,
                                         iree_vm_ref_t* arg0, int32_t arg2,
                                         int32_t arg3, int32_t arg4) {
  return iree_ok_status();
}

// 0rr_v
static iree_status_t call_0rr_v_import(iree_vm_stack_t* stack,
                                       const iree_vm_function_t* import,
                                       iree_vm_ref_t* arg0,
                                       iree_vm_ref_t* arg1) {
  return iree_ok_status();
}

// 0rriiii_v
static iree_status_t call_0rriiii_v_import(iree_vm_stack_t* stack,
                                           const iree_vm_function_t* import,
                                           iree_vm_ref_t* arg0,
                                           iree_vm_ref_t* arg1, int32_t arg2,
                                           int32_t arg3, int32_t arg4,
                                           int32_t arg5) {
  return iree_ok_status();
}

// 0rrr_ii
static iree_status_t call_0rrr_ii_import(iree_vm_stack_t* stack,
                                         const iree_vm_function_t* import,
                                         iree_vm_ref_t* arg0,
                                         iree_vm_ref_t* arg1,
                                         iree_vm_ref_t* arg2, int32_t* res0,
                                         int32_t* res1) {
  return iree_ok_status();
}

// 0v_r
static iree_status_t call_0v_r_import(iree_vm_stack_t* stack,
                                      const iree_vm_function_t* import,
                                      iree_vm_ref_t* res0) {
  iree_vm_abi_v_t arguments;
  iree_vm_abi_r_t results;
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

// variadic imports

// 0riCiiiD_r
static iree_status_t call_0riCiiiD_r_import(iree_vm_stack_t* stack,
                                            const iree_vm_function_t* import,
                                            int32_t varArgsCount,
                                            iree_vm_ref_t* arg0, int32_t arg1,
                                            ...) {
  return iree_ok_status();
}

// 0riCrD_r
static iree_status_t call_0riCrD_r_import(iree_vm_stack_t* stack,
                                          const iree_vm_function_t* import,
                                          int32_t varArgsCount,
                                          iree_vm_ref_t* arg0, int32_t arg1,
                                          ...) {
  return iree_ok_status();
}

// 0rrrCrD_r
static iree_status_t call_0rrrCrD_r_import(iree_vm_stack_t* stack,
                                           const iree_vm_function_t* import,
                                           int32_t varArgsCount,
                                           iree_vm_ref_t* arg0,
                                           iree_vm_ref_t* arg1,
                                           iree_vm_ref_t* arg2, ...) {
  return iree_ok_status();
}

// 0rriCiriiD_v
static iree_status_t call_0rriCiriiD_v_import(iree_vm_stack_t* stack,
                                              const iree_vm_function_t* import,
                                              int32_t varArgsCount,
                                              iree_vm_ref_t* arg0,
                                              iree_vm_ref_t* arg1, int32_t arg2,
                                              ...) {
  return iree_ok_status();
}

// 0riiCiD_r
static iree_status_t call_0riiCiD_r_import(iree_vm_stack_t* stack,
                                           const iree_vm_function_t* import,
                                           int32_t varArgsCount,
                                           iree_vm_ref_t* arg0, int32_t arg1,
                                           int32_t arg2, ...) {
  return iree_ok_status();
}

#endif  // IREE_VM_SHIMS_EMITC_H_#endif  // IREE_VM_SHIMS_EMITC_H_
