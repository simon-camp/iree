// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "iree/base/api.h"
#include "iree/base/internal/cpu.h"
#include "iree/base/internal/flags.h"
#include "iree/base/internal/math.h"
#include "iree/base/internal/path.h"
#include "iree/base/target_platform.h"
#include "iree/hal/api.h"
#include "iree/modules/hal/module.h"
#include "iree/tooling/device_util.h"
#include "iree/tooling/trace_replay.h"
#include "iree/tooling/yaml_util.h"
#include "iree/vm/api.h"

IREE_FLAG(bool, trace_execution, false, "Traces VM execution to stderr.");

static const char* emoji(bool good) { return good ? "🦄" : "🐞"; }

// Defines the type of a primitive value.
typedef enum iree_e2e_test_value_type_e {
  // Not a value type.
  IREE_E2E_TEST_VALUE_TYPE_NONE = 0,
  // int8_t.
  IREE_E2E_TEST_VALUE_TYPE_I8 = 1,
  // int16_t.
  IREE_E2E_TEST_VALUE_TYPE_I16 = 2,
  // int32_t.
  IREE_E2E_TEST_VALUE_TYPE_I32 = 3,
  // int64_t.
  IREE_E2E_TEST_VALUE_TYPE_I64 = 4,
  // halft_t.
  IREE_E2E_TEST_VALUE_TYPE_F16 = 5,
  // float.
  IREE_E2E_TEST_VALUE_TYPE_F32 = 6,
  // double.
  IREE_E2E_TEST_VALUE_TYPE_F64 = 7,
} iree_e2e_test_value_type_t;

// Maximum size, in bytes, of any value type we can represent.
#define IREE_E2E_TEST_VALUE_STORAGE_SIZE 8

// A variant value type.
typedef struct iree_e2e_test_value_t {
  iree_e2e_test_value_type_t type;
  union {
    int8_t i8;
    int16_t i16;
    int32_t i32;
    int64_t i64;
    float f32;
    uint16_t f16_u16;
    double f64;
    uint8_t value_storage[IREE_E2E_TEST_VALUE_STORAGE_SIZE];  // max size of all
                                                              // value types
  };
} iree_e2e_test_value_t;

static inline iree_e2e_test_value_t iree_e2e_test_value_make_none() {
  iree_e2e_test_value_t result;
  result.type = IREE_E2E_TEST_VALUE_TYPE_NONE;
  return result;
}

static inline iree_e2e_test_value_t iree_e2e_test_value_make_i8(int8_t value) {
  iree_e2e_test_value_t result;
  result.type = IREE_E2E_TEST_VALUE_TYPE_I8;
  result.i8 = value;
  return result;
}

static inline iree_e2e_test_value_t iree_e2e_test_value_make_i16(
    int16_t value) {
  iree_e2e_test_value_t result;
  result.type = IREE_E2E_TEST_VALUE_TYPE_I16;
  result.i16 = value;
  return result;
}

static inline iree_e2e_test_value_t iree_e2e_test_value_make_i32(
    int32_t value) {
  iree_e2e_test_value_t result;
  result.type = IREE_E2E_TEST_VALUE_TYPE_I32;
  result.i32 = value;
  return result;
}

// TODO(#5542): check the value type before accessing the union.
static inline int32_t iree_e2e_test_value_get_i32(
    iree_e2e_test_value_t* value) {
  return value->i32;
}

static inline iree_e2e_test_value_t iree_e2e_test_value_make_i64(
    int64_t value) {
  iree_e2e_test_value_t result;
  result.type = IREE_E2E_TEST_VALUE_TYPE_I64;
  result.i64 = value;
  return result;
}

// TODO(#5542): check the value type before accessing the union.
static inline int64_t iree_e2e_test_value_get_i64(
    iree_e2e_test_value_t* value) {
  return value->i64;
}

static inline iree_e2e_test_value_t iree_e2e_test_value_make_f16(
    uint16_t value) {
  iree_e2e_test_value_t result;
  result.type = IREE_E2E_TEST_VALUE_TYPE_F16;
  result.f16_u16 = value;
  return result;
}

static inline iree_e2e_test_value_t iree_e2e_test_value_make_f32(float value) {
  iree_e2e_test_value_t result;
  result.type = IREE_E2E_TEST_VALUE_TYPE_F32;
  result.f32 = value;
  return result;
}

// TODO(#5542): check the value type before accessing the union.
static inline float iree_e2e_test_value_get_f32(iree_e2e_test_value_t* value) {
  return value->f32;
}

// TODO(#5542): check the value type before accessing the union.
static inline uint16_t iree_e2e_test_value_get_f16(
    iree_e2e_test_value_t* value) {
  return value->f16_u16;
}

static inline iree_e2e_test_value_t iree_e2e_test_value_make_f64(double value) {
  iree_e2e_test_value_t result;
  result.type = IREE_E2E_TEST_VALUE_TYPE_F64;
  result.f64 = value;
  return result;
}

// TODO(#5542): check the value type before accessing the union.
static inline double iree_e2e_test_value_get_f64(iree_e2e_test_value_t* value) {
  return value->f64;
}

/*****************************************************************************
 *
 * Part 1:
 *
 * Generic helper functions to deal with buffer_view's.
 *
 *****************************************************************************/

// Get list[i] as a buffer_view.
static iree_status_t get_item_as_buffer_view(
    iree_vm_list_t* list, iree_host_size_t i,
    iree_hal_buffer_view_t** out_value) {
  iree_vm_variant_t variant = iree_vm_variant_empty();
  IREE_RETURN_IF_ERROR(iree_vm_list_get_variant_assign(list, i, &variant));
  if (!iree_vm_variant_is_ref(variant)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "expected list item %zu to be a ref", i);
  }
  return iree_hal_buffer_view_check_deref(variant.ref, out_value);
}

// Validates that |buffer_view|'s memory type satisfies |expected|.
static iree_status_t validate_memory_type(iree_hal_buffer_view_t* buffer_view,
                                          iree_hal_memory_type_t expected) {
  return iree_hal_buffer_validate_memory_type(
      iree_hal_buffer_memory_type(iree_hal_buffer_view_buffer(buffer_view)),
      expected);
}

// Map dense row-major data in a host-local buffer_view.
static iree_status_t map_host_local_row_major_data(
    iree_hal_buffer_view_t* buffer_view,
    enum iree_hal_memory_access_bits_t access,
    iree_hal_buffer_mapping_t* mapping) {
  // Really validate host-local, not just host-visible: callers may rely on
  // host-coherency.
  IREE_RETURN_IF_ERROR(
      validate_memory_type(buffer_view, IREE_HAL_MEMORY_TYPE_HOST_LOCAL));
  if (iree_hal_buffer_view_encoding_type(buffer_view) !=
      IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "buffer_view is not dense row major");
  }
  IREE_RETURN_IF_ERROR(iree_hal_buffer_map_range(
      iree_hal_buffer_view_buffer(buffer_view),
      IREE_HAL_MAPPING_MODE_PERSISTENT, access, 0, IREE_WHOLE_BUFFER, mapping));
  return iree_ok_status();
}

// Allocates host-local |dst| to have the same shape as |src|.
// Implicitly zero-filled.
static iree_status_t allocate_host_buffer_view_like(
    iree_hal_allocator_t* hal_allocator, iree_hal_buffer_view_t* src,
    iree_hal_buffer_view_t** dst) {
  return iree_hal_buffer_view_allocate_buffer(
      hal_allocator, iree_hal_buffer_view_shape_rank(src),
      iree_hal_buffer_view_shape_dims(src),
      iree_hal_buffer_view_element_type(src),
      iree_hal_buffer_view_encoding_type(src),
      (iree_hal_buffer_params_t){
          .type = IREE_HAL_MEMORY_TYPE_HOST_LOCAL,
          .usage =
              IREE_HAL_BUFFER_USAGE_TRANSFER | IREE_HAL_BUFFER_USAGE_MAPPING,
      },
      iree_const_byte_span_empty(), dst);
}

// Allocates device-local |dst| to have the same shape as |src|.
// Implicitly zero-filled.
static iree_status_t allocate_device_buffer_view_like(
    iree_hal_allocator_t* hal_allocator, iree_hal_buffer_view_t* src,
    iree_const_byte_span_t initial_data, iree_hal_buffer_view_t** dst) {
  return iree_hal_buffer_view_allocate_buffer(
      hal_allocator, iree_hal_buffer_view_shape_rank(src),
      iree_hal_buffer_view_shape_dims(src),
      iree_hal_buffer_view_element_type(src),
      iree_hal_buffer_view_encoding_type(src),
      (iree_hal_buffer_params_t){
          .type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
          .usage = IREE_HAL_BUFFER_USAGE_DEFAULT,
      },
      initial_data, dst);
}

// Performs a deep copy of device-local |src| into host-local |dst|.
// Allocates |dst|.
static iree_status_t copy_device_buffer_view_to_host(
    iree_hal_device_t* device, iree_hal_allocator_t* hal_allocator,
    iree_hal_buffer_view_t* src, iree_hal_buffer_view_t** dst) {
  IREE_RETURN_IF_ERROR(
      validate_memory_type(src, IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL));
  IREE_RETURN_IF_ERROR(allocate_host_buffer_view_like(hal_allocator, src, dst));
  iree_hal_buffer_mapping_t dst_mapping;
  iree_status_t status = map_host_local_row_major_data(
      *dst, IREE_HAL_MEMORY_ACCESS_WRITE, &dst_mapping);
  if (iree_status_is_ok(status)) {
    status = iree_hal_device_transfer_d2h(
        device, iree_hal_buffer_view_buffer(src), 0, dst_mapping.contents.data,
        iree_hal_buffer_view_byte_length(src),
        IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT, iree_infinite_timeout());
  }
  IREE_RETURN_IF_ERROR(iree_hal_buffer_unmap_range(&dst_mapping));
  return status;
}

// Performs a deep copy of host-local |src| into a device-local |dst|.
// Allocates |dst|.
static iree_status_t copy_host_buffer_view_to_device(
    iree_hal_device_t* device, iree_hal_allocator_t* hal_allocator,
    iree_hal_buffer_view_t* src, iree_hal_buffer_view_t** dst) {
  iree_hal_buffer_mapping_t src_mapping;
  IREE_RETURN_IF_ERROR(map_host_local_row_major_data(
      src, IREE_HAL_MEMORY_ACCESS_READ, &src_mapping));
  iree_const_byte_span_t const_src_bytes = iree_make_const_byte_span(
      src_mapping.contents.data, src_mapping.contents.data_length);
  IREE_RETURN_IF_ERROR(allocate_device_buffer_view_like(hal_allocator, src,
                                                        const_src_bytes, dst));
  IREE_RETURN_IF_ERROR(iree_hal_buffer_unmap_range(&src_mapping));
  return iree_ok_status();
}

// Performs a deep copy of device-local |src| into a device-local |dst|.
// Allocates |dst|.
static iree_status_t copy_device_buffer_view_to_device(
    iree_hal_device_t* device, iree_hal_allocator_t* hal_allocator,
    iree_hal_buffer_view_t* src, iree_hal_buffer_view_t** dst) {
  IREE_RETURN_IF_ERROR(
      validate_memory_type(src, IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL));
  IREE_RETURN_IF_ERROR(allocate_device_buffer_view_like(
      hal_allocator, src, iree_const_byte_span_empty(), dst));
  iree_status_t status = iree_hal_device_transfer_d2d(
      device, iree_hal_buffer_view_buffer(src), 0,
      iree_hal_buffer_view_buffer(*dst), 0,
      iree_hal_buffer_view_byte_length(src),
      IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT, iree_infinite_timeout());
  if (!iree_status_is_ok(status)) {
    iree_hal_buffer_view_release(*dst);
  }
  return status;
}

// Deep-copy the list of device-local buffer-views |src| into |dst|.
// Allocates |dst|.
static iree_status_t copy_device_buffer_views_to_host(
    iree_hal_device_t* device, iree_hal_allocator_t* hal_allocator,
    iree_vm_list_t* src, iree_vm_list_t** dst) {
  iree_vm_type_def_t elem_type = iree_vm_list_element_type(src);
  iree_host_size_t size = iree_vm_list_size(src);
  iree_allocator_t allocator = iree_hal_allocator_host_allocator(hal_allocator);
  IREE_RETURN_IF_ERROR(iree_vm_list_create(elem_type, size, allocator, dst));
  IREE_RETURN_IF_ERROR(iree_vm_list_resize(*dst, size));
  for (iree_host_size_t i = 0; i < size; ++i) {
    iree_hal_buffer_view_t* src_elem = NULL;
    IREE_RETURN_IF_ERROR(get_item_as_buffer_view(src, i, &src_elem));
    iree_hal_buffer_view_t* dst_elem = NULL;
    IREE_RETURN_IF_ERROR(copy_device_buffer_view_to_host(device, hal_allocator,
                                                         src_elem, &dst_elem));
    iree_vm_ref_t dst_elem_ref = {0};
    IREE_RETURN_IF_ERROR(iree_vm_ref_wrap_assign(
        dst_elem, iree_hal_buffer_view_type(), &dst_elem_ref));
    IREE_RETURN_IF_ERROR(iree_vm_list_set_ref_move(*dst, i, &dst_elem_ref));
  }
  return iree_ok_status();
}

// Helper to write an int value to a single buffer element.
static void write_int_element(iree_hal_element_type_t element_type, int value,
                              void* dst) {
#define WRITE_INT_ELEMENT_CASE(ETYPE, CTYPE) \
  case IREE_HAL_ELEMENT_TYPE_##ETYPE:        \
    *(CTYPE*)dst = (CTYPE)value;             \
    break;

  switch (element_type) {
    WRITE_INT_ELEMENT_CASE(INT_8, int8_t)
    WRITE_INT_ELEMENT_CASE(INT_32, int32_t)
    case IREE_HAL_ELEMENT_TYPE_FLOAT_16:
      *(uint16_t*)dst = iree_math_f32_to_f16((float)value);
      break;
      WRITE_INT_ELEMENT_CASE(FLOAT_32, float)
    default:
      IREE_ASSERT(false, "unhandled element type");
      break;
  }

#undef WRITE_INT_ELEMENT_CASE
}

/*****************************************************************************
 *
 * Part 2:
 *
 * Helper functions to deal with matrices and matrix multiplications.
 *
 * Much of this is the |reference_matmul| function, a reference implementation
 * of matrix multiplication on host-mapped buffers, and helpers for it.
 *
 * Still generic in the sense that none of the high-level logic of this
 * particular test program is entrenched here.
 *
 *****************************************************************************/
// Write an int32_t element to a mapped row-major matrix buffer.
static void write_int_to_matrix_element(int32_t value, iree_hal_dim_t m_size,
                                        iree_hal_dim_t n_size,
                                        iree_hal_element_type_t result_type,
                                        void* data, iree_hal_dim_t m,
                                        iree_hal_dim_t n) {
  iree_host_size_t index = n + m * n_size;
  (void)m_size;
  if (iree_hal_element_type_is_integer(result_type, 32)) {
    ((int32_t*)data)[index] = value;
    return;
  } else if (result_type == IREE_HAL_ELEMENT_TYPE_FLOAT_16) {
    ((uint16_t*)data)[index] = iree_math_f32_to_f16((float)value);
    return;
  } else if (result_type == IREE_HAL_ELEMENT_TYPE_FLOAT_32) {
    ((float*)data)[index] = value;
    return;
  }
  IREE_CHECK_OK(iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                 "unhandled matmul result type"));
}

// Reads an element from a mapped row-major matrix buffer.
static iree_e2e_test_value_t read_matrix_element(
    iree_hal_dim_t m_size, iree_hal_dim_t n_size,
    iree_hal_element_type_t result_type, void* data, iree_hal_dim_t m,
    iree_hal_dim_t n) {
  iree_host_size_t index = n + m * n_size;
  (void)m_size;
  if (iree_hal_element_type_is_integer(result_type, 8)) {
    return iree_e2e_test_value_make_i8(((int8_t*)data)[index]);
  } else if (iree_hal_element_type_is_integer(result_type, 16)) {
    return iree_e2e_test_value_make_i16(((int16_t*)data)[index]);
  } else if (iree_hal_element_type_is_integer(result_type, 32)) {
    return iree_e2e_test_value_make_i32(((int32_t*)data)[index]);
  } else if (result_type == IREE_HAL_ELEMENT_TYPE_FLOAT_16) {
    return iree_e2e_test_value_make_f16(((uint16_t*)data)[index]);
  } else if (result_type == IREE_HAL_ELEMENT_TYPE_FLOAT_32) {
    return iree_e2e_test_value_make_f32(((float*)data)[index]);
  }
  iree_status_abort(iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                     "unhandled matmul result type"));
  return iree_e2e_test_value_make_none();
}

// Get the shape of a buffer_view that is a matrix, i.e. 2D shape.
static iree_status_t get_matrix_shape(iree_hal_buffer_view_t* buffer_view,
                                      iree_hal_dim_t* dims) {
  iree_host_size_t shape_rank = iree_hal_buffer_view_shape_rank(buffer_view);
  if (shape_rank != 2) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "expected a matrix (2D tensor) shape, got a %zu-dimensional shape",
        shape_rank);
  }
  dims[0] = iree_hal_buffer_view_shape_dim(buffer_view, 0);
  dims[1] = iree_hal_buffer_view_shape_dim(buffer_view, 1);
  if (!(dims[0] > 0 && dims[1] > 0)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "expected matrix dims to be positive, got %" PRIdim
                            "x%" PRIdim,
                            dims[0], dims[1]);
  }
  return iree_ok_status();
}

// Get the {m,k,n}_size values of the shape of a matmul
static iree_status_t get_matmul_sizes(
    iree_hal_buffer_view_t* lhs, iree_hal_buffer_view_t* rhs,
    iree_hal_buffer_view_t* acc, iree_hal_buffer_view_t* result,
    iree_hal_dim_t* m_size, iree_hal_dim_t* k_size, iree_hal_dim_t* n_size) {
  iree_hal_dim_t lhs_dims[2] = {0};
  iree_hal_dim_t rhs_dims[2] = {0};
  iree_hal_dim_t acc_dims[2] = {0};
  iree_hal_dim_t result_dims[2] = {0};
  IREE_RETURN_IF_ERROR(get_matrix_shape(lhs, lhs_dims));
  IREE_RETURN_IF_ERROR(get_matrix_shape(rhs, rhs_dims));
  IREE_RETURN_IF_ERROR(get_matrix_shape(acc, acc_dims));
  IREE_RETURN_IF_ERROR(get_matrix_shape(result, result_dims));
  *m_size = lhs_dims[0];
  *k_size = lhs_dims[1];
  *n_size = rhs_dims[1];
  if (!(lhs_dims[0] == *m_size && lhs_dims[1] == *k_size &&
        rhs_dims[0] == *k_size && rhs_dims[1] == *n_size &&
        acc_dims[0] == *m_size && acc_dims[1] == *n_size &&
        result_dims[0] == *m_size && result_dims[1] == *n_size)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "mismatched matrix shapes in matmul: %" PRIdim "x%" PRIdim " * %" PRIdim
        "x%" PRIdim " + %" PRIdim "x%" PRIdim " -> %" PRIdim "x%" PRIdim,
        lhs_dims[0], lhs_dims[1], rhs_dims[0], rhs_dims[1], acc_dims[0],
        acc_dims[1], result_dims[0], result_dims[1]);
  }
  return iree_ok_status();
}

#define IREE_TRACE_REPLAY_REFERENCE_MATMUL(LHSTYPE, RHSTYPE, RESTYPE, ACCTYPE) \
  static void reference_matmul_##LHSTYPE##_##RHSTYPE##_##RESTYPE##_##ACCTYPE(  \
      iree_hal_dim_t m_size, iree_hal_dim_t k_size, iree_hal_dim_t n_size,     \
      iree_hal_element_type_t lhs_type, iree_hal_element_type_t rhs_type,      \
      iree_hal_element_type_t acc_type, LHSTYPE* lhs_data, RHSTYPE* rhs_data,  \
      ACCTYPE* acc_data, RESTYPE* result_data, iree_hal_dim_t m,               \
      iree_hal_dim_t n) {                                                      \
    ACCTYPE acc = acc_data[n + m * n_size];                                    \
    for (iree_hal_dim_t k = 0; k < k_size; ++k) {                              \
      LHSTYPE lhs_value = lhs_data[k + m * k_size];                            \
      RHSTYPE rhs_value = rhs_data[n + k * n_size];                            \
      acc += (ACCTYPE)lhs_value * (ACCTYPE)rhs_value;                          \
    }                                                                          \
    result_data[n + m * n_size] = acc;                                         \
  }

// Reference mamtul instantiations from macro IREE_TRACE_REPLAY_REFERENCE_MATMUL
// for the f32 input, f32 accumlation, and f32 result.
// [float <= float * float + float]
IREE_TRACE_REPLAY_REFERENCE_MATMUL(float, float, float, float)

// Reference mamtul instantiations from macro IREE_TRACE_REPLAY_REFERENCE_MATMUL
// for the int8_t input, int32_t accumlation, and int32_t result.
// [i32 <= i8 * i8 + i32]
IREE_TRACE_REPLAY_REFERENCE_MATMUL(int8_t, int8_t, int32_t, int32_t)

// Reference mamtul for the half_t input, half_t accumlation, and half_t result.
// [f16 <= f16 * f16 + f16]
static void reference_matmul_f16_f16_f16_f16(
    iree_hal_dim_t m_size, iree_hal_dim_t k_size, iree_hal_dim_t n_size,
    iree_hal_element_type_t lhs_type, iree_hal_element_type_t rhs_type,
    iree_hal_element_type_t acc_type, uint16_t* lhs_data, uint16_t* rhs_data,
    uint16_t* acc_data, uint16_t* result_data, iree_hal_dim_t m,
    iree_hal_dim_t n) {
  float acc = iree_math_f16_to_f32(acc_data[n + m * n_size]);
  for (iree_hal_dim_t k = 0; k < k_size; ++k) {
    acc = iree_math_round_to_nearest_f16(
        iree_math_round_to_nearest_f16(
            (iree_math_f16_to_f32(lhs_data[k + m * k_size]) *
             iree_math_f16_to_f32(rhs_data[n + k * n_size]))) +
        acc);
  }
  result_data[n + m * n_size] = iree_math_f32_to_f16(acc);
}

// Helper for reference_matmul.
// Computes one element in the result matrix.
static void reference_matmul_element(
    iree_hal_dim_t m_size, iree_hal_dim_t k_size, iree_hal_dim_t n_size,
    iree_hal_element_type_t lhs_type, iree_hal_element_type_t rhs_type,
    iree_hal_element_type_t acc_type, void* lhs_data, void* rhs_data,
    void* acc_data, void* result_data, iree_hal_dim_t m, iree_hal_dim_t n) {
  if (lhs_type == IREE_HAL_ELEMENT_TYPE_FLOAT_32 &&
      rhs_type == IREE_HAL_ELEMENT_TYPE_FLOAT_32 &&
      acc_type == IREE_HAL_ELEMENT_TYPE_FLOAT_32) {
    reference_matmul_float_float_float_float(
        m_size, k_size, n_size, lhs_type, rhs_type, acc_type, (float*)lhs_data,
        (float*)rhs_data, (float*)acc_data, (float*)result_data, m, n);
  } else if (iree_hal_element_type_is_integer(lhs_type, 8) &&
             iree_hal_element_type_is_integer(rhs_type, 8) &&
             iree_hal_element_type_is_integer(acc_type, 32)) {
    reference_matmul_int8_t_int8_t_int32_t_int32_t(
        m_size, k_size, n_size, lhs_type, rhs_type, acc_type, (int8_t*)lhs_data,
        (int8_t*)rhs_data, (int32_t*)acc_data, (int32_t*)result_data, m, n);
  } else if (lhs_type == IREE_HAL_ELEMENT_TYPE_FLOAT_16 &&
             rhs_type == IREE_HAL_ELEMENT_TYPE_FLOAT_16 &&
             acc_type == IREE_HAL_ELEMENT_TYPE_FLOAT_16) {
    reference_matmul_f16_f16_f16_f16(m_size, k_size, n_size, lhs_type, rhs_type,
                                     acc_type, (uint16_t*)lhs_data,
                                     (uint16_t*)rhs_data, (uint16_t*)acc_data,
                                     (uint16_t*)result_data, m, n);
  } else {
    iree_status_abort(
        iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                         "unhandled combination of element types in matmul"));
  }
}

// Reference matmul implementation, used to compare matmul results against.
static iree_status_t reference_matmul(iree_vm_list_t* input_list,
                                      iree_hal_buffer_view_t* result) {
  iree_hal_buffer_view_t* lhs = NULL;
  iree_hal_buffer_view_t* rhs = NULL;
  iree_hal_buffer_view_t* acc = NULL;
  IREE_RETURN_IF_ERROR(get_item_as_buffer_view(input_list, 0, &lhs));
  IREE_RETURN_IF_ERROR(get_item_as_buffer_view(input_list, 1, &rhs));
  IREE_RETURN_IF_ERROR(get_item_as_buffer_view(input_list, 2, &acc));

  iree_hal_dim_t m_size, k_size, n_size;
  IREE_RETURN_IF_ERROR(
      get_matmul_sizes(lhs, rhs, acc, result, &m_size, &k_size, &n_size));
  iree_hal_buffer_mapping_t lhs_mapping;
  iree_hal_buffer_mapping_t rhs_mapping;
  iree_hal_buffer_mapping_t acc_mapping;
  iree_hal_buffer_mapping_t result_mapping;
  IREE_RETURN_IF_ERROR(map_host_local_row_major_data(
      lhs, IREE_HAL_MEMORY_ACCESS_READ, &lhs_mapping));
  IREE_RETURN_IF_ERROR(map_host_local_row_major_data(
      rhs, IREE_HAL_MEMORY_ACCESS_READ, &rhs_mapping));
  IREE_RETURN_IF_ERROR(map_host_local_row_major_data(
      acc, IREE_HAL_MEMORY_ACCESS_READ, &acc_mapping));
  IREE_RETURN_IF_ERROR(map_host_local_row_major_data(
      result, IREE_HAL_MEMORY_ACCESS_WRITE, &result_mapping));
  iree_hal_element_type_t lhs_type = iree_hal_buffer_view_element_type(lhs);
  iree_hal_element_type_t rhs_type = iree_hal_buffer_view_element_type(rhs);
  iree_hal_element_type_t acc_type = iree_hal_buffer_view_element_type(acc);
  for (iree_hal_dim_t m = 0; m < m_size; ++m) {
    for (iree_hal_dim_t n = 0; n < n_size; ++n) {
      reference_matmul_element(
          m_size, k_size, n_size, lhs_type, rhs_type, acc_type,
          lhs_mapping.contents.data, rhs_mapping.contents.data,
          acc_mapping.contents.data, result_mapping.contents.data, m, n);
    }
  }
  IREE_RETURN_IF_ERROR(iree_hal_buffer_unmap_range(&lhs_mapping));
  IREE_RETURN_IF_ERROR(iree_hal_buffer_unmap_range(&rhs_mapping));
  IREE_RETURN_IF_ERROR(iree_hal_buffer_unmap_range(&acc_mapping));
  IREE_RETURN_IF_ERROR(iree_hal_buffer_unmap_range(&result_mapping));
  return iree_ok_status();
}

/*****************************************************************************
 *
 * Part 3:
 *
 * Helper functions to validate matmul test results and pretty-print matrices.
 *
 * The only entry point in to this part is |check_matmul_results|, the other
 * functions are only helpers for it.
 *
 * Still generic in the sense that none of the high-level logic of this
 * particular test program is entrenched here.
 *
 *****************************************************************************/

// Enum controlling how many decimals to print floats with.
typedef enum precision_e {
  PRECISION_LOW,
  PRECISION_HIGH,
} precision_t;

// Prints a iree_e2e_test_value_t to a string buffer. Returns the number of
// characters written. Like snprintf.
static int snprintf_value(char* buf, size_t bufsize,
                          iree_e2e_test_value_t value, precision_t precision) {
  switch (value.type) {
    case IREE_E2E_TEST_VALUE_TYPE_I8:
      return snprintf(buf, bufsize, "%" PRIi8, value.i8);
    case IREE_E2E_TEST_VALUE_TYPE_I16:
      return snprintf(buf, bufsize, "%" PRIi16, value.i16);
    case IREE_E2E_TEST_VALUE_TYPE_I32:
      return snprintf(buf, bufsize, "%" PRIi32, value.i32);
    case IREE_E2E_TEST_VALUE_TYPE_I64:
      return snprintf(buf, bufsize, "%" PRIi64, value.i64);
    case IREE_E2E_TEST_VALUE_TYPE_F16:
      return snprintf(buf, bufsize,
                      precision == PRECISION_HIGH ? "%.5g" : "%.4g",
                      iree_math_f16_to_f32(value.f16_u16));
    case IREE_E2E_TEST_VALUE_TYPE_F32:
      return snprintf(buf, bufsize,
                      precision == PRECISION_HIGH ? "%.8g" : "%.4g", value.f32);
    case IREE_E2E_TEST_VALUE_TYPE_F64:
      return snprintf(buf, bufsize,
                      precision == PRECISION_HIGH ? "%.16g" : "%.4g",
                      value.f64);
    default:
      iree_status_abort(iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                         "unhandled value type"));
      return 0;
  }
}

// Returns true if |expected| and |actual| agree to tolerable accuracy.
static bool matmul_result_elements_agree(iree_e2e_test_value_t expected,
                                         iree_e2e_test_value_t actual) {
  if (expected.type != actual.type) {
    iree_status_abort(
        iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "mismatched types"));
    return false;
  }
  switch (expected.type) {
    case IREE_E2E_TEST_VALUE_TYPE_I32:
      return actual.i32 == expected.i32;
    // Since we fill buffers with small integers for floating point GEMMs
    // functional testing, we test for bit-exactness on the actual and
    // expected values.
    case IREE_E2E_TEST_VALUE_TYPE_F16:
      return actual.f16_u16 == expected.f16_u16;
    case IREE_E2E_TEST_VALUE_TYPE_F32:
      return actual.f32 == expected.f32;
    default:
      iree_status_abort(iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                         "unhandled value type"));
      return false;
  }
}

// Returns the largest number of characters to print any matrix element.
static int get_max_elem_width(precision_t precision, int row_start, int row_end,
                              int col_start, int col_end,
                              iree_hal_buffer_view_t* matrix) {
  iree_hal_dim_t dims[2] = {0};
  get_matrix_shape(matrix, dims);
  int rows = dims[0];
  int cols = dims[1];
  iree_hal_element_type_t elem_type = iree_hal_buffer_view_element_type(matrix);
  iree_hal_buffer_mapping_t mapping;
  IREE_CHECK_OK(map_host_local_row_major_data(
      matrix, IREE_HAL_MEMORY_ACCESS_READ, &mapping));
  int max_elem_width = 0;
  for (int row = row_start; row < row_end; row++) {
    for (int col = col_start; col < col_end; col++) {
      iree_e2e_test_value_t elem = read_matrix_element(
          rows, cols, elem_type, mapping.contents.data, row, col);
      char buf[64];
      int this_elem_width = snprintf_value(buf, sizeof buf, elem, precision);
      // iree_max is a macro, may evaluate its args twice. Give it plain ints.
      max_elem_width = iree_max(max_elem_width, this_elem_width);
    }
  }
  IREE_CHECK_OK(iree_hal_buffer_unmap_range(&mapping));
  return max_elem_width;
}

// Prints |matrix| to |file|, with |label| as caption.
// |precision| controls how many decimals are printed for float values.
//
// If |other_matrix| is not NULL, then any matrix entries that disagree
// between |matrix| and |other_matrix| (according to
// matmul_result_elements_agree) are highlighted.
//
// |highlight| is either NULL or is a UTF-8 string that will be printed next to
// any entry of |matrix| that disagrees with the corresponding entry of
// |other_matrix|.
//
// |highlight| should be NULL if and only if |other_matrix| is NULL.
//
// In order for matrix columns to be properly laid out, the rendering of
// |highlight| in a fixed-width font should have the width of two regular Latin
// characters. According to
// https://www.unicode.org/reports/tr11/#Recommendations, a single emoji
// character should meet that requirement.
static void print_matrix(FILE* file, const char* label, precision_t precision,
                         int row_start, int row_end, int col_start, int col_end,
                         iree_hal_buffer_view_t* matrix,
                         iree_hal_buffer_view_t* other_matrix,
                         const char* highlight) {
  assert((other_matrix == NULL) == (highlight == NULL));
  iree_hal_dim_t dims[2] = {0};
  get_matrix_shape(matrix, dims);
  int rows = dims[0];
  int cols = dims[1];
  iree_hal_element_type_t elem_type = iree_hal_buffer_view_element_type(matrix);
  iree_hal_buffer_mapping_t mapping;
  IREE_CHECK_OK(map_host_local_row_major_data(
      matrix, IREE_HAL_MEMORY_ACCESS_READ, &mapping));
  int max_elem_width = get_max_elem_width(precision, row_start, row_end,
                                          col_start, col_end, matrix);
  iree_hal_buffer_mapping_t other_mapping;
  if (other_matrix) {
    IREE_CHECK_OK(map_host_local_row_major_data(
        other_matrix, IREE_HAL_MEMORY_ACCESS_READ, &other_mapping));
    int other_matrix_max_elem_width = get_max_elem_width(
        precision, row_start, row_end, col_start, col_end, other_matrix);
    // iree_max is a macro, may evaluate its args twice. Give it plain ints.
    max_elem_width = iree_max(max_elem_width, other_matrix_max_elem_width);
  }

  fprintf(file,
          "%s (rows %d..%d out of %d..%d, columns %d..%d out of %d..%d)\n",
          label, row_start, row_end - 1, 0, rows - 1, col_start, col_end - 1, 0,
          cols - 1);
  for (int row = row_start; row < row_end; row++) {
    for (int col = col_start; col < col_end; col++) {
      iree_e2e_test_value_t elem = read_matrix_element(
          rows, cols, elem_type, mapping.contents.data, row, col);
      bool disagree = false;
      if (other_matrix) {
        iree_e2e_test_value_t other_elem = read_matrix_element(
            rows, cols, elem_type, other_mapping.contents.data, row, col);
        disagree = !matmul_result_elements_agree(elem, other_elem);
      }
      char buf[64];
      snprintf_value(buf, sizeof buf, elem, precision);
      fprintf(file, "%*s", max_elem_width, buf);
      // See comment on |highlight| function parameter for why 2 spaces.
      // A 3rd space is added unconditionally to make it clear that a highlight
      // concerns the matrix entry to its left.
      fprintf(file, "%s ", disagree ? highlight : "  ");
    }
    fprintf(file, "\n");
  }

  IREE_CHECK_OK(iree_hal_buffer_unmap_range(&mapping));
  if (other_matrix) {
    IREE_CHECK_OK(iree_hal_buffer_unmap_range(&mapping));
  }
}

// Helper for check_matmul_results: handler for the failure case.
// If |file| is not NULL, detailed logging is written to it.
static iree_status_t check_matmul_failure(
    FILE* file, iree_e2e_test_value_t actual_value,
    iree_e2e_test_value_t expected_value, iree_hal_dim_t row,
    iree_hal_dim_t col, iree_hal_buffer_view_t* lhs,
    iree_hal_buffer_view_t* rhs, iree_hal_buffer_view_t* acc,
    iree_hal_buffer_view_t* actual_result,
    iree_hal_buffer_view_t* expected_result) {
  if (!file) {
    return iree_make_status(IREE_STATUS_ABORTED);
  }
  fprintf(file,
          "\n\nerror: the actual and expected result matrices disagree "
          "at row %" PRIdim ", column %" PRIdim ".\n\n",
          row, col);
  char actual_value_buf[32];
  char expected_value_buf[32];
  snprintf_value(actual_value_buf, sizeof actual_value_buf, actual_value,
                 PRECISION_HIGH);
  snprintf_value(expected_value_buf, sizeof expected_value_buf, expected_value,
                 PRECISION_HIGH);
  fprintf(file, "actual value: %s\n", actual_value_buf);
  fprintf(file, "expected value: %s\n", expected_value_buf);

  iree_hal_dim_t m_size, k_size, n_size;
  IREE_RETURN_IF_ERROR(get_matmul_sizes(lhs, rhs, acc, actual_result, &m_size,
                                        &k_size, &n_size));
  iree_hal_dim_t context = 8;
  const char* context_env = getenv("IREE_MATMUL_TEST_SHOW_CONTEXT");
  if (context_env) {
    if (1 != sscanf(context_env, "%" PRIdim, &context)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "Failed to parse IREE_MATMUL_TEST_SHOW_CONTEXT "
                              "as \"%%" PRIdim "\". Got \"%s\"",
                              context_env);
    }
  }
  int m_start = iree_max(0, (int)row - (int)context);
  int m_end = iree_min(m_size, row + context);
  int n_start = iree_max(0, (int)col - (int)context);
  int n_end = iree_min(n_size, col + context);
  // We have a lot more freedom to pick k_start, k_end, since these parameters
  // only affect which regions of the input lhs and rhs matrices are printed.
  // If we were only testing random lhs and rhs, we would just pick
  // k_start = 0 and any reasonable k_end value. Since we are often using
  // identity matrices for lhs and rhs, and we expect the majority of
  // test failures to occur with such identity matrices, we try to pick
  // k_start and k_end so that nontrivial regions of identity matrices will be
  // printed. That means that we try to have [k_start, k_end) intervals
  // overlap [m_start, m_end) and [n_start, n_end).
  int k_start = iree_max(0, iree_min(m_start, n_start));
  int k_end = iree_min(k_size, iree_max(m_end, n_end));
  // [k_start, k_end) could be arbitrarily long at this point. Constrain it a
  // bit to avoid huge output.
  k_end = iree_min(k_end, k_start + 4 * context);

  fprintf(file, "\n");
  print_matrix(file, "left-hand side", PRECISION_LOW, m_start, m_end, k_start,
               k_end, lhs, NULL, NULL);
  fprintf(file, "\n");
  print_matrix(file, "right-hand side", PRECISION_LOW, k_start, k_end, n_start,
               n_end, rhs, NULL, NULL);
  fprintf(file, "\n");
  print_matrix(file, "input accumulator", PRECISION_LOW, m_start, m_end,
               n_start, n_end, acc, NULL, NULL);
  fprintf(file, "\n");
  print_matrix(file, "expected result", PRECISION_LOW, m_start, m_end, n_start,
               n_end, expected_result, actual_result, emoji(true));
  fprintf(file, "\n");
  print_matrix(file, "actual result", PRECISION_LOW, m_start, m_end, n_start,
               n_end, actual_result, expected_result, emoji(false));
  fprintf(file, "\n");
  return iree_make_status(IREE_STATUS_ABORTED);
}

// Helper for check_matmul_results: the actual interesting part once we've
// obtained and validated the {m,k,n}_size values. On error, detailed logging is
// written to |file| if it is not NULL.
static iree_status_t check_matmul_results_impl(
    FILE* file, iree_hal_dim_t m_size, iree_hal_dim_t k_size,
    iree_hal_dim_t n_size, iree_hal_buffer_view_t* lhs,
    iree_hal_buffer_view_t* rhs, iree_hal_buffer_view_t* acc,
    iree_hal_buffer_view_t* actual_result,
    iree_hal_buffer_view_t* expected_result) {
  iree_hal_buffer_mapping_t actual_result_mapping;
  iree_hal_buffer_mapping_t expected_result_mapping;
  IREE_RETURN_IF_ERROR(map_host_local_row_major_data(
      actual_result, IREE_HAL_MEMORY_ACCESS_READ, &actual_result_mapping));
  IREE_RETURN_IF_ERROR(map_host_local_row_major_data(
      expected_result, IREE_HAL_MEMORY_ACCESS_READ, &expected_result_mapping));
  iree_hal_element_type_t result_type =
      iree_hal_buffer_view_element_type(actual_result);
  for (iree_hal_dim_t m = 0; m < m_size; ++m) {
    for (iree_hal_dim_t n = 0; n < n_size; ++n) {
      iree_e2e_test_value_t actual_value =
          read_matrix_element(m_size, n_size, result_type,
                              actual_result_mapping.contents.data, m, n);
      iree_e2e_test_value_t expected_value =
          read_matrix_element(m_size, n_size, result_type,
                              expected_result_mapping.contents.data, m, n);
      if (!matmul_result_elements_agree(actual_value, expected_value)) {
        return check_matmul_failure(file, actual_value, expected_value, m, n,
                                    lhs, rhs, acc, actual_result,
                                    expected_result);
      }
    }
  }
  IREE_RETURN_IF_ERROR(iree_hal_buffer_unmap_range(&actual_result_mapping));
  IREE_RETURN_IF_ERROR(iree_hal_buffer_unmap_range(&expected_result_mapping));
  return iree_ok_status();
}

// Given an actual matmul's inputs and output (all host-local), uses a reference
// matmul implementation on the same inputs to check if the output is correct.
// On error, detailed logging is written to |file| if it is not NULL.
static iree_status_t check_matmul_results(
    FILE* file, iree_vm_list_t* input_list,
    iree_hal_buffer_view_t* actual_result,
    iree_hal_buffer_view_t* expected_result) {
  iree_hal_buffer_view_t* lhs = NULL;
  iree_hal_buffer_view_t* rhs = NULL;
  iree_hal_buffer_view_t* acc = NULL;
  IREE_RETURN_IF_ERROR(get_item_as_buffer_view(input_list, 0, &lhs));
  IREE_RETURN_IF_ERROR(get_item_as_buffer_view(input_list, 1, &rhs));
  IREE_RETURN_IF_ERROR(get_item_as_buffer_view(input_list, 2, &acc));

  iree_hal_dim_t m_size, k_size, n_size;
  IREE_RETURN_IF_ERROR(get_matmul_sizes(lhs, rhs, acc, actual_result, &m_size,
                                        &k_size, &n_size));

  return check_matmul_results_impl(file, m_size, k_size, n_size, lhs, rhs, acc,
                                   actual_result, expected_result);
}

/*****************************************************************************
 *
 * Part 4:
 *
 * Core matmul test logic.
 *
 * The entry point into this part is |replay_event_call_matmul|, which is the
 * handler for each matmul testcase in the trace. Other functions are only
 * helpers for it.
 *
 * |replay_event_call_matmul| calls |do_matmul_and_check_results| to actually
 * perform a matmul. In normal cases, each |replay_event_call_matmul| performs
 * one call to |do_matmul_and_check_results|, but when that generates an error,
 * it will make additional calls to |do_matmul_and_check_results| to evaluate
 * variants of the failed testcase to generate a more helpful log.
 *
 * The |matrix_mask_t| stuff is only used to generate these variants of failed
 * testcases.
 *
 *****************************************************************************/

// Enumerates ways that we may mask matrices in list of matrix inputs to matmul
// testcases.
typedef enum {
  MATRIX_MASK_NONE,      // no-op: leave the existing matrix unchanged.
  MATRIX_MASK_ZERO,      // overwrite the matrix with zeros.
  MATRIX_MASK_IDENTITY,  // overwrite with (general rectangular) identity matrix
} matrix_mask_t;

static iree_status_t make_identity_matrix_callback(
    iree_hal_buffer_mapping_t* mapping, void* user_data) {
  iree_hal_buffer_view_t* src = (iree_hal_buffer_view_t*)user_data;
  iree_hal_element_type_t elem_type = iree_hal_buffer_view_element_type(src);
  iree_host_size_t elem_byte_count =
      iree_hal_element_dense_byte_count(elem_type);
  iree_hal_dim_t dims[2] = {0};
  IREE_RETURN_IF_ERROR(get_matrix_shape(src, dims));
  int rows = dims[0];
  int cols = dims[1];
  // Write 1 to matrix elements on the main diagonal.
  int diagonal_size = iree_min(rows, cols);
  memset(mapping->contents.data, 0, mapping->contents.data_length);
  intptr_t diagonal_elem_addr = (intptr_t)mapping->contents.data;
  for (int i = 0; i < diagonal_size; ++i) {
    write_int_element(elem_type, 1, (void*)diagonal_elem_addr);
    // Due to the row-major storage, the diagonal entries are every
    // (cols + 1)-th buffer elements.
    diagonal_elem_addr += elem_byte_count * (cols + 1);
  }
  return iree_ok_status();
}

// Allocates device-local |dst| and initializes it as an identity-matrix shaped
// like |src|.
static iree_status_t make_device_identity_matrix_like(
    iree_hal_device_t* device, iree_hal_allocator_t* hal_allocator,
    iree_hal_buffer_view_t* src, iree_hal_buffer_view_t** dst) {
  return iree_hal_buffer_view_generate_buffer(
      hal_allocator, iree_hal_buffer_view_shape_rank(src),
      iree_hal_buffer_view_shape_dims(src),
      iree_hal_buffer_view_element_type(src),
      iree_hal_buffer_view_encoding_type(src),
      (iree_hal_buffer_params_t){
          .type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
          .usage = IREE_HAL_BUFFER_USAGE_DEFAULT,
      },
      make_identity_matrix_callback, src, dst);
}

// Allocates device-local |dst| shaped like |src|, and:
// - If |mask| is MATRIX_MASK_NONE, copies device-local |src| into |dst|.
// - If |mask| is MATRIX_MASK_ZERO, leaves |dst| zero-filled.
// - If |mask| is MATRIX_MASK_IDENTITY, makes |dst| an identity-matrix
static iree_status_t mask_and_copy_device_buffer_view_to_device(
    iree_hal_device_t* device, iree_hal_allocator_t* hal_allocator,
    iree_hal_buffer_view_t* src, matrix_mask_t mask,
    iree_hal_buffer_view_t** dst) {
  if (mask == MATRIX_MASK_NONE) {
    IREE_RETURN_IF_ERROR(
        copy_device_buffer_view_to_device(device, hal_allocator, src, dst));
  } else if (mask == MATRIX_MASK_ZERO) {
    IREE_RETURN_IF_ERROR(allocate_device_buffer_view_like(
        hal_allocator, src, iree_const_byte_span_empty(), dst));
  } else if (mask == MATRIX_MASK_IDENTITY) {
    IREE_RETURN_IF_ERROR(
        make_device_identity_matrix_like(device, hal_allocator, src, dst));
  } else {
    iree_status_abort(iree_make_status(IREE_STATUS_INTERNAL, "bad mask enum"));
  }
  return iree_ok_status();
}

// Deep-copies device-local list of buffer_views |src| into |dst|, applying
// mask[i] to the i-th list element as in
// |mask_and_copy_device_buffer_view_to_device|.
// Requirement: |mask| must point to an array of the same length as |src|.
static iree_status_t mask_and_copy_device_buffer_views_to_device(
    iree_hal_device_t* device, iree_hal_allocator_t* hal_allocator,
    iree_vm_list_t* src_list, matrix_mask_t* mask, iree_vm_list_t** dst_list) {
  iree_vm_type_def_t elem_type = iree_vm_list_element_type(src_list);
  iree_host_size_t size = iree_vm_list_size(src_list);
  iree_allocator_t allocator = iree_hal_allocator_host_allocator(hal_allocator);
  IREE_RETURN_IF_ERROR(
      iree_vm_list_create(elem_type, size, allocator, dst_list));
  IREE_RETURN_IF_ERROR(iree_vm_list_resize(*dst_list, size));
  for (iree_host_size_t i = 0; i < size; ++i) {
    iree_hal_buffer_view_t* src = NULL;
    IREE_RETURN_IF_ERROR(get_item_as_buffer_view(src_list, i, &src));
    iree_hal_buffer_view_t* dst = NULL;
    IREE_RETURN_IF_ERROR(mask_and_copy_device_buffer_view_to_device(
        device, hal_allocator, src, mask[i], &dst));
    iree_vm_ref_t dst_ref = {0};
    IREE_RETURN_IF_ERROR(
        iree_vm_ref_wrap_assign(dst, iree_hal_buffer_view_type(), &dst_ref));
    IREE_RETURN_IF_ERROR(iree_vm_list_set_ref_move(*dst_list, i, &dst_ref));
  }
  return iree_ok_status();
}

// Performs one matmul test, on the device-local input matrices given in
// |original_device_inputs|, applying the masks given in |mask| as in
// |mask_and_copy_device_buffer_view_to_device|.
// Both |input_list| and |mask| should have length 3. The 3 input matrices are
// LHS, RHS, Accumulator, in that order.
//
// The contents of |original_device_inputs| are preserved, even if the
// |function| would overwrite input-output arguments (e.g. the accumulator).
static iree_status_t do_matmul_and_check_results(
    FILE* file, iree_trace_replay_t* replay, iree_vm_function_t function,
    matrix_mask_t* mask, iree_vm_list_t* original_device_inputs) {
  iree_hal_allocator_t* device_allocator =
      iree_hal_device_allocator(replay->device);

  // Perform a deep copy of the inputs to pass to the test function.
  // Needed as the test function may mutate some of the input list elements,
  // e.g. input-output parameters. For instance, the accumulator input of a
  // linalg.matmul. We need to preserve the original test inputs to perform
  // reruns on variants in the failure case (see |replay_event_call_matmul|).
  iree_vm_list_t* device_inputs = NULL;
  iree_status_t status = mask_and_copy_device_buffer_views_to_device(
      replay->device, device_allocator, original_device_inputs, mask,
      &device_inputs);

  // Perform a deep copy of the device-local inputs into host-local buffers.
  // Needed to pass to the reference matmul implementation and to logging
  // in the failure case.
  iree_vm_list_t* host_inputs = NULL;
  if (iree_status_is_ok(status)) {
    status = copy_device_buffer_views_to_host(replay->device, device_allocator,
                                              device_inputs, &host_inputs);
  }

  // Invoke the function to produce the actual result.
  iree_vm_list_t* device_outputs = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_vm_list_create(iree_vm_make_undefined_type_def(),
                                 /*initial_capacity=*/8, replay->host_allocator,
                                 &device_outputs);
  }

  if (iree_status_is_ok(status)) {
    status = iree_vm_invoke(
        replay->context, function, IREE_VM_INVOCATION_FLAG_NONE,
        /*policy=*/NULL, device_inputs, device_outputs, replay->host_allocator);
  }

  iree_vm_list_release(device_inputs);

  // Get the device_actual_result from the device_outputs.
  iree_hal_buffer_view_t* device_actual_result;
  if (iree_status_is_ok(status)) {
    status = get_item_as_buffer_view(device_outputs, 0, &device_actual_result);
  }

  // Copy the results to a host local buffer to be able to map it.
  iree_hal_buffer_view_t* host_actual_result = NULL;
  if (iree_status_is_ok(status)) {
    status = copy_device_buffer_view_to_host(replay->device, device_allocator,
                                             device_actual_result,
                                             &host_actual_result);
  }

  // Allocate host_expected_result with same shape as host_actual_result.
  iree_hal_buffer_view_t* host_expected_result = NULL;
  if (iree_status_is_ok(status)) {
    status = allocate_host_buffer_view_like(
        device_allocator, host_actual_result, &host_expected_result);
  }

  // Use the reference matmul implementation to fill host_expected_result
  if (iree_status_is_ok(status)) {
    status = reference_matmul(host_inputs, host_expected_result);
  }

  // Check that host_actual_result and host_expected_result agree.
  if (iree_status_is_ok(status)) {
    status = check_matmul_results(file, host_inputs, host_actual_result,
                                  host_expected_result);
  }

  iree_vm_list_release(device_outputs);  // releases device_actual_result
  iree_vm_list_release(host_inputs);
  iree_hal_buffer_view_release(host_actual_result);
  iree_hal_buffer_view_release(host_expected_result);
  return status;
}

const char* matrix_form(matrix_mask_t mask) {
  switch (mask) {
    case MATRIX_MASK_NONE:
      return "GENERAL";
    case MATRIX_MASK_ZERO:
      return "ZERO";
    case MATRIX_MASK_IDENTITY:
      return "IDENTITY";
  }
  assert(false);
  return NULL;
}

// Prints to |file| a message about the matmul shape. Useful as testcases
// otherwise only print the function name, and in the dynamic-shapes cases, that
// doesn't tell the actual shape.
static iree_status_t print_matmul_shape(FILE* file,
                                        iree_vm_list_t* input_list) {
  iree_hal_buffer_view_t* lhs = NULL;
  iree_hal_buffer_view_t* rhs = NULL;
  IREE_RETURN_IF_ERROR(get_item_as_buffer_view(input_list, 0, &lhs));
  IREE_RETURN_IF_ERROR(get_item_as_buffer_view(input_list, 1, &rhs));
  iree_hal_dim_t lhs_dims[2] = {0};
  iree_hal_dim_t rhs_dims[2] = {0};
  IREE_RETURN_IF_ERROR(get_matrix_shape(lhs, lhs_dims));
  IREE_RETURN_IF_ERROR(get_matrix_shape(rhs, rhs_dims));
  fprintf(file, "Matmul shape (MxKxN): %" PRIdim "x%" PRIdim "x%" PRIdim "\n",
          lhs_dims[0], lhs_dims[1], rhs_dims[1]);
  return iree_ok_status();
}

// Special handler for function calls in a e2e matmul test trace.
// Assumes that all calls are to functions that take 3 inputs (lhs, rhs, acc)
// and return the result of a matmul (lhs*rhs+acc).
static iree_status_t replay_event_call_matmul(iree_trace_replay_t* replay,
                                              yaml_document_t* document,
                                              yaml_node_t* event_node) {
  yaml_node_t* function_node = NULL;
  IREE_RETURN_IF_ERROR(iree_yaml_mapping_find(
      document, event_node, iree_make_cstring_view("function"),
      &function_node));
  iree_string_view_t function_name = iree_yaml_node_as_string(function_node);
  fprintf(stderr, "--- CALL[%.*s] ---\n", (int)function_name.size,
          function_name.data);

  iree_vm_function_t function;
  iree_vm_list_t* device_inputs = NULL;
  IREE_RETURN_IF_ERROR(iree_trace_replay_event_call_prepare(
      replay, document, event_node, &function, &device_inputs));

  IREE_CHECK_OK(print_matmul_shape(stderr, device_inputs));

  // Perform the matmul test. So far we are using pseudorandom matrices (as
  // specified in the YAML trace and interpreted above in
  // |iree_trace_replay_event_call_prepare|). So this is a test on general
  // random matrices: great for test coverage (if this succeeds, any variant on
  // more special matrices would also succeed) but bad for debugging (if this
  // fails, having to debug that would involve staring at arrays of random
  // numbers). So for now we pass NULL as the |file| param, keeping errors
  // silent for now.
  matrix_mask_t none_masks[3] = {MATRIX_MASK_NONE, MATRIX_MASK_NONE,
                                 MATRIX_MASK_NONE};
  iree_status_t status = do_matmul_and_check_results(NULL, replay, function,
                                                     none_masks, device_inputs);
  if (!iree_status_is_ok(status)) {
    // The matmul test failed. So whatever we do now is only for the sake of
    // generating the most undertandable possible error log. We are going to
    // retry the matmul but on more special, easy-to-understand matrices,
    // gradually increasing generality, and we will abort and log details on
    // the first error that we encounter.
    iree_string_builder_t sb;
    iree_string_builder_initialize(replay->host_allocator, &sb);
    matrix_mask_t all_debug_masks[6][3] = {
        // Try Zero * Zero + Zero. Expected result: Zero.
        {MATRIX_MASK_ZERO, MATRIX_MASK_ZERO, MATRIX_MASK_ZERO},
        // Try Identity * Identity + Zero. Expected result: Identity.
        {MATRIX_MASK_IDENTITY, MATRIX_MASK_IDENTITY, MATRIX_MASK_ZERO},
        // Try RandomLHS * Identity + Zero. Expected result: RandomLHS.
        {MATRIX_MASK_NONE, MATRIX_MASK_IDENTITY, MATRIX_MASK_ZERO},
        // Try Identity * RandomRHS + Zero. Expected result: RandomRHS.
        {MATRIX_MASK_IDENTITY, MATRIX_MASK_NONE, MATRIX_MASK_ZERO},
        // Try Identity * Identity + RandomAccum.
        // Expected result: Identity + RandomAccum.
        {MATRIX_MASK_IDENTITY, MATRIX_MASK_IDENTITY, MATRIX_MASK_NONE},
        // Finally run the general case again. If none of the above special
        // cases
        // failed, then that at least must fail, since we already ran that and
        // it
        // had failed.
        {MATRIX_MASK_NONE, MATRIX_MASK_NONE, MATRIX_MASK_NONE}};
    bool reproduced_failure = false;
    for (int i = 0; i < IREE_ARRAYSIZE(all_debug_masks); ++i) {
      matrix_mask_t* masks = all_debug_masks[i];
      iree_status_code_t rerun_status =
          iree_status_consume_code(do_matmul_and_check_results(
              stderr, replay, function, masks, device_inputs));
      bool good = iree_status_is_ok(rerun_status);
      reproduced_failure |= !good;
      iree_string_builder_append_format(
          &sb, "%s LHS:%-10s * RHS:%-10s + ACCUMULATOR:%-10s\n", emoji(good),
          matrix_form(masks[0]), matrix_form(masks[1]), matrix_form(masks[2]));
      if (!good) break;
    }
    if (!reproduced_failure) {
      iree_status_abort(iree_make_status(
          IREE_STATUS_INTERNAL,
          "Internal error: a matmul test failed, but subsequent reruns for "
          "logging purposes were not able to reproduce the failure."));
    }
    fprintf(stderr,
            "Summary of reruns, pinpointing how general matrices need to be to "
            "reproduce this failure:\n%s\n",
            iree_string_builder_buffer(&sb));
    iree_string_builder_deinitialize(&sb);
  }

  // Clean up.
  iree_vm_list_release(device_inputs);

  return status;
}

/*****************************************************************************
 *
 * Part 5:
 *
 * main function and high-level logic before one enters matmul test details.
 *
 *****************************************************************************/

// Helper for |replay_event_requirements|.
static iree_status_t iree_cpu_has_required_target_features(
    yaml_document_t* document, yaml_node_t* target_features_node) {
  for (yaml_node_item_t* item = target_features_node->data.sequence.items.start;
       item != target_features_node->data.sequence.items.top; ++item) {
    yaml_node_t* item_node = yaml_document_get_node(document, *item);
    iree_string_view_t required_feature = iree_yaml_node_as_string(item_node);
    if (iree_string_view_is_empty(required_feature)) continue;
    int64_t feature_is_supported = 0;
    IREE_RETURN_IF_ERROR(
        iree_cpu_lookup_data_by_key(required_feature, &feature_is_supported));
    if (!feature_is_supported) {
      return iree_make_status(
          // The error status matters. We distinguish "feature not supported",
          // which is a normal thing to happen, from actual errors.
          IREE_STATUS_UNAVAILABLE,
          "The target device does not have the required feature '%.*s'.\n",
          (int)required_feature.size, required_feature.data);
    }
  }
  return iree_ok_status();
}

// returns UNAVAILABLE if the required CPU feature is not supported by the CPU.
static iree_status_t replay_event_requirements(iree_trace_replay_t* replay,
                                               yaml_document_t* document,
                                               yaml_node_t* event_node) {
  yaml_node_t* target_features_node = NULL;
  IREE_RETURN_IF_ERROR(iree_yaml_mapping_find(
      document, event_node, iree_make_cstring_view("target_features"),
      &target_features_node));
  return iree_cpu_has_required_target_features(document, target_features_node);
}

static iree_status_t iree_e2e_matmul_test_trace_replay_event(
    iree_trace_replay_t* replay, yaml_document_t* document,
    yaml_node_t* event_node) {
  if (event_node->type != YAML_MAPPING_NODE) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "(%zu): expected mapping node",
                            event_node->start_mark.line);
  }
  yaml_node_t* type_node = NULL;
  IREE_RETURN_IF_ERROR(iree_yaml_mapping_find(
      document, event_node, iree_make_cstring_view("type"), &type_node));
  if (iree_yaml_string_equal(type_node, iree_make_cstring_view("call"))) {
    return replay_event_call_matmul(replay, document, event_node);
  } else if (iree_yaml_string_equal(type_node,
                                    iree_make_cstring_view("requirements"))) {
    return replay_event_requirements(replay, document, event_node);
  } else {
    return iree_trace_replay_event(replay, document, event_node);
  }
}

// Runs the trace in |file| using |root_path| as the base for any path lookups
// required for external files referenced in |file|.
static iree_status_t run_trace_file(iree_string_view_t root_path, FILE* file,
                                    iree_vm_instance_t* instance) {
  iree_trace_replay_t replay;
  IREE_RETURN_IF_ERROR(iree_trace_replay_initialize(
      root_path, instance, IREE_TRACE_REPLAY_FLAG_NONE,
      FLAG_trace_execution ? IREE_VM_CONTEXT_FLAG_TRACE_EXECUTION
                           : IREE_VM_CONTEXT_FLAG_NONE,
      iree_hal_available_driver_registry(), iree_allocator_system(), &replay));

  // Query device overrides, if any. When omitted the devices from the trace
  // file will be used.
  // TODO(#5724): remove this and instead provide a device set on initialize.
  iree_host_size_t device_uri_count = 0;
  const iree_string_view_t* device_uris = NULL;
  iree_hal_get_devices_flag_list(&device_uri_count, &device_uris);
  iree_trace_replay_set_hal_devices_override(&replay, device_uri_count,
                                             device_uris);

  yaml_parser_t parser;
  if (!yaml_parser_initialize(&parser)) {
    iree_trace_replay_deinitialize(&replay);
    return iree_make_status(IREE_STATUS_INTERNAL,
                            "yaml_parser_initialize failed");
  }
  yaml_parser_set_input_file(&parser, file);

  iree_status_t status = iree_ok_status();
  for (bool document_eof = false; !document_eof;) {
    yaml_document_t document;
    if (!yaml_parser_load(&parser, &document)) {
      status = iree_status_from_yaml_parser_error(&parser);
      break;
    }
    yaml_node_t* event_node = yaml_document_get_root_node(&document);
    if (event_node) {
      status = iree_e2e_matmul_test_trace_replay_event(&replay, &document,
                                                       event_node);
    } else {
      document_eof = true;
    }
    yaml_document_delete(&document);
    if (!iree_status_is_ok(status)) break;
  }

  yaml_parser_delete(&parser);
  iree_trace_replay_deinitialize(&replay);
  return status;
}

// Runs each of the given traces files sequentially in isolated contexts.
static iree_status_t run_trace_files(int file_count, char** file_paths,
                                     iree_vm_instance_t* instance) {
  for (int i = 0; i < file_count; ++i) {
    iree_string_view_t file_path = iree_make_cstring_view(file_paths[i]);
    iree_string_view_t root_path = iree_file_path_dirname(file_path);
    FILE* file = fopen(file_paths[i], "rb");
    if (!file) {
      return iree_make_status(iree_status_code_from_errno(errno),
                              "failed to open trace file '%.*s'",
                              (int)file_path.size, file_path.data);
    }
    iree_status_t status = run_trace_file(root_path, file, instance);
    fclose(file);
    IREE_RETURN_IF_ERROR(status, "replaying trace file '%.*s'",
                         (int)file_path.size, file_path.data);
  }
  return iree_ok_status();
}

int main(int argc, char** argv) {
  iree_flags_parse_checked(IREE_FLAGS_PARSE_MODE_DEFAULT, &argc, &argv);
  if (argc <= 1) {
    fprintf(stderr,
            "no trace files provided; pass one or more yaml file paths\n");
    return 1;
  }

  iree_vm_instance_t* instance = NULL;
  iree_status_t status = iree_vm_instance_create(
      IREE_VM_TYPE_CAPACITY_DEFAULT, iree_allocator_system(), &instance);
  if (iree_status_is_ok(status)) {
    status = run_trace_files(argc - 1, argv + 1, instance);
  }
  iree_vm_instance_release(instance);
  if (!iree_status_is_ok(status)) {
    iree_status_fprint(stderr, status);
    bool is_unavailable = iree_status_is_unavailable(status);
    iree_status_free(status);
    return is_unavailable ? EXIT_SUCCESS : EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
