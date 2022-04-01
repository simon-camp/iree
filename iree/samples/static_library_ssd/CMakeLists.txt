# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

if(NOT ${IREE_TARGET_BACKEND_DYLIB_LLVM_AOT}
  OR NOT ${IREE_HAL_DRIVER_DYLIB}
  OR NOT ${IREE_BUILD_COMPILER})
  return()
endif()

# Set iree-translate binary.
set(_TRANSLATE_TOOL_EXECUTABLE $<TARGET_FILE:iree_tools_iree-translate>)

## Example with VM bytecode module.
# Setup args for iree-translate.
set(_TRANSLATE_ARGS)
list(APPEND _TRANSLATE_ARGS "-iree-input-type=tosa")
list(APPEND _TRANSLATE_ARGS "-iree-mlir-to-vm-bytecode-module")
list(APPEND _TRANSLATE_ARGS "-iree-hal-target-backends=dylib-llvm-aot")
list(APPEND _TRANSLATE_ARGS "-iree-llvm-link-embedded=false")
list(APPEND _TRANSLATE_ARGS "-iree-llvm-link-static")
list(APPEND _TRANSLATE_ARGS "-iree-llvm-static-library-output-path=mobilenet_ssd.o")
list(APPEND _TRANSLATE_ARGS "${CMAKE_CURRENT_SOURCE_DIR}/mobilenet_ssd_quant.mlir")
list(APPEND _TRANSLATE_ARGS "-o")
list(APPEND _TRANSLATE_ARGS "mobilenet_ssd.vmfb")

# Custom command for iree-translate to generate static library and bytecode.
add_custom_command(
  OUTPUT
    ${CMAKE_CURRENT_BINARY_DIR}/mobilenet_ssd.h
    ${CMAKE_CURRENT_BINARY_DIR}/mobilenet_ssd.o
    ${CMAKE_CURRENT_BINARY_DIR}/mobilenet_ssd.vmfb
  COMMAND ${_TRANSLATE_TOOL_EXECUTABLE} ${_TRANSLATE_ARGS}
  DEPENDS ${_TRANSLATE_TOOL_EXECUTABLE} "mobilenet_ssd_quant.mlir"
)

# Tell cmake about the mobilenet_ssd library so it will link it.
add_library(mobilenet_ssd
  STATIC
  ${CMAKE_CURRENT_BINARY_DIR}/mobilenet_ssd.o)

SET_TARGET_PROPERTIES(
  mobilenet_ssd
  PROPERTIES
  LINKER_LANGUAGE C)

# Note: If you're cross compiling the mobilenet_ssd for a different backend, you'll
# need to run iree-translate manually to produce the desired '.vmfb' and '.h/.o'
# files. Substitute the 'mobilenet_ssd' dependency in iree_cc_binary() below with
# your own static library and the `mobilenet_ssd.vmfb` in the iree_c_embed_data()
# rule. You can use paths to files, i.e. 'path/to/generated/output.vmfb' in
# place of CMake targets.

# Generate the embed data with the bytecode module
iree_c_embed_data(
  NAME
    mobilenet_ssd_c
  IDENTIFIER
    iree_samples_static_library_mobilenet_ssd
  GENERATED_SRCS
    mobilenet_ssd.vmfb
  C_FILE_OUTPUT
    mobilenet_ssd_c.c
  H_FILE_OUTPUT
    mobilenet_ssd_c.h
  FLATTEN
  PUBLIC
)

iree_cc_binary(
NAME
  static_library_demo
SRCS
  "create_bytecode_module.c"
  "static_library_demo.c"
DEPS
  ::mobilenet_ssd_c
  iree::runtime
  iree::hal::local::loaders::static_library_loader
  iree::hal::local::sync_driver
  mobilenet_ssd
)

iree_lit_test(
  NAME
    static_library_demo_test
  TEST_FILE
    "static_library_demo_test.txt"
  TOOLS
    ::static_library_demo
    FileCheck
  LABELS
    "hostonly"
)


if(NOT (${IREE_ENABLE_EMITC} OR DEFINED IREE_HOST_BINARY_ROOT))
  return()
endif()

## Example with VM C module.
# Setup args for iree-translate.
set(_TRANSLATE_ARGS)
list(APPEND _TRANSLATE_ARGS "-iree-input-type=tosa")
list(APPEND _TRANSLATE_ARGS "-iree-mlir-to-vm-c-module")
list(APPEND _TRANSLATE_ARGS "-iree-hal-target-backends=dylib-llvm-aot")
list(APPEND _TRANSLATE_ARGS "-iree-llvm-link-embedded=false")
list(APPEND _TRANSLATE_ARGS "-iree-llvm-link-static")
list(APPEND _TRANSLATE_ARGS "-iree-llvm-static-library-output-path=mobilenet_ssd_c_module.o")
list(APPEND _TRANSLATE_ARGS "${CMAKE_CURRENT_SOURCE_DIR}/mobilenet_ssd_quant.mlir")
list(APPEND _TRANSLATE_ARGS "-o")
list(APPEND _TRANSLATE_ARGS "mobilenet_ssd_emitc.h")

# Custom command for iree-translate to generate static library and C module.
add_custom_command(
  OUTPUT
    ${CMAKE_CURRENT_BINARY_DIR}/mobilenet_ssd_c_module.h
    ${CMAKE_CURRENT_BINARY_DIR}/mobilenet_ssd_c_module.o
    ${CMAKE_CURRENT_BINARY_DIR}/mobilenet_ssd_emitc.h
  COMMAND ${_TRANSLATE_TOOL_EXECUTABLE} ${_TRANSLATE_ARGS}
  DEPENDS ${_TRANSLATE_TOOL_EXECUTABLE} "mobilenet_ssd_quant.mlir"
)

# TODO(marbre): Cleanup custom targets and libraries.
add_custom_target(
  mobilenet_ssd_gen
  DEPENDS
    "mobilenet_ssd_emitc.h"
)

add_library(mobilenet_ssd_c_module
 STATIC
 ${CMAKE_CURRENT_BINARY_DIR}/mobilenet_ssd_c_module.o
)
add_dependencies(mobilenet_ssd_c_module mobilenet_ssd_gen)

SET_TARGET_PROPERTIES(
  mobilenet_ssd_c_module
  PROPERTIES
  LINKER_LANGUAGE C
)

# TODO(marbre): Cleanup SRCS and DEPS.
iree_cc_library(
  NAME
    mobilenet_ssd_emitc
  HDRS
    "mobilenet_ssd_emitc.h"
  DEFINES
    "EMITC_IMPLEMENTATION"
)

iree_cc_binary(
  NAME
    static_library_demo_c
  SRCS
    "create_c_module.c"
    "static_library_demo.c"
  DEPS
    ::mobilenet_ssd_emitc
    iree::runtime
    iree::hal::local::loaders::static_library_loader
    iree::hal::local::sync_driver
    iree::vm::shims_emitc
    mobilenet_ssd_c_module
)

iree_lit_test(
  NAME
    static_library_demo_c_test
  TEST_FILE
    "static_library_demo_c_test.txt"
  TOOLS
    ::static_library_demo_c
    FileCheck
  LABELS
    "hostonly"
)
