# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

load("@llvm-project//mlir:tblgen.bzl", "td_library")
load("//build_tools/bazel:build_defs.oss.bzl", "iree_compiler_cc_library")
load("//build_tools/bazel:enforce_glob.bzl", "enforce_glob")
load("//build_tools/bazel:iree_tablegen.bzl", "iree_gentbl_cc_library", "iree_tablegen_doc")

package(
    default_visibility = ["//visibility:public"],
    features = ["layering_check"],
    licenses = ["notice"],  # Apache 2.0
)

td_library(
    name = "td_files",
    srcs = enforce_glob(
        [
            "IREEC.td",
            "IREECAttributes.td",
            "IREECBase.td",
            "IREECTypes.td",
        ],
        include = ["*.td"],
    ),
    deps = [
        "//compiler/src/iree/compiler/Dialect/Util/IR:td_files",
        "@llvm-project//mlir:ControlFlowInterfacesTdFiles",
        "@llvm-project//mlir:FuncTdFiles",
        "@llvm-project//mlir:InferTypeOpInterfaceTdFiles",
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:SideEffectInterfacesTdFiles",
        "@llvm-project//mlir:ViewLikeInterfaceTdFiles",
    ],
)

iree_compiler_cc_library(
    name = "IR",
    srcs = [
        "IREEC.cpp",
        "IREECAttrs.cpp.inc",
        "IREECDialect.cpp.inc",
        "IREECOps.cpp.inc",
        "IREECTypes.cpp.inc",
        # "IREECDialect.cpp",
        # "IREECEnums.cpp.inc",
        # "IREECOpFolders.cpp",
        # "IREECOpInterfaces.cpp.inc",
        # "IREECOps.cpp",
        # "IREECTypeInterfaces.cpp.inc",
        # "IREECTypes.cpp",
    ],
    hdrs = [
        "IREEC.h",
        "IREECAttrs.h.inc",
        "IREECDialect.h.inc",
        "IREECOps.h.inc",
        "IREECTypes.h.inc",
        # "IREECDialect.h",
        # "IREECEnums.h.inc",
        # "IREECOpInterfaces.h.inc",
        # "IREECOps.h",
        # "IREECTypeInterfaces.h.inc",
        # "IREECTypes.h",
    ],
    deps = [
        # ":IREECEnumsGen",
        # ":IREECInterfacesGen",
        ":IREECAttrsGen",
        ":IREECDialectGen",
        ":IREECOpsGen",
        ":IREECTypesGen",
        "//compiler/src/iree/compiler/Dialect/Util/IR",
        "//compiler/src/iree/compiler/Utils",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:AffineDialect",
        "@llvm-project//mlir:AffineUtils",
        "@llvm-project//mlir:Analysis",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:ArithUtils",
        "@llvm-project//mlir:ControlFlowInterfaces",
        "@llvm-project//mlir:DialectUtils",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:InferTypeOpInterface",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:Parser",
        "@llvm-project//mlir:SideEffectInterfaces",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:TransformUtils",
        "@llvm-project//mlir:ViewLikeInterface",
    ],
)

# iree_gentbl_cc_library(
#     name = "IREECEnumsGen",
#     tbl_outs = [
#         (
#             ["--gen-enum-decls"],
#             "IREECEnums.h.inc",
#         ),
#         (
#             ["--gen-enum-defs"],
#             "IREECEnums.cpp.inc",
#         ),
#     ],
#     tblgen = "@llvm-project//mlir:mlir-tblgen",
#     td_file = "IREECBase.td",
#     deps = [":td_files"],
# )

# iree_gentbl_cc_library(
#     name = "IREECInterfacesGen",
#     tbl_outs = [
#         (
#             ["--gen-op-interface-decls"],
#             "IREECOpInterfaces.h.inc",
#         ),
#         (
#             ["--gen-op-interface-defs"],
#             "IREECOpInterfaces.cpp.inc",
#         ),
#         (
#             ["--gen-type-interface-decls"],
#             "IREECTypeInterfaces.h.inc",
#         ),
#         (
#             ["--gen-type-interface-defs"],
#             "IREECTypeInterfaces.cpp.inc",
#         ),
#     ],
#     tblgen = "@llvm-project//mlir:mlir-tblgen",
#     td_file = "IREECInterfaces.td",
#     deps = [":td_files"],
# )

iree_gentbl_cc_library(
    name = "IREECDialectGen",
    tbl_outs = [
        (
            ["--gen-dialect-decls"],
            "IREECDialect.h.inc",
        ),
        (
            ["--gen-dialect-defs"],
            "IREECDialect.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "IREEC.td",
    deps = [":td_files"],
)

iree_gentbl_cc_library(
    name = "IREECOpsGen",
    tbl_outs = [
        (
            ["--gen-op-decls"],
            "IREECOps.h.inc",
        ),
        (
            ["--gen-op-defs"],
            "IREECOps.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "IREEC.td",
    deps = [":td_files"],
)

iree_gentbl_cc_library(
    name = "IREECTypesGen",
    tbl_outs = [
        (
            [
                "--gen-typedef-decls",
                "--typedefs-dialect=ireec",
            ],
            "IREECTypes.h.inc",
        ),
        (
            [
                "--gen-typedef-defs",
                "--typedefs-dialect=ireec",
            ],
            "IREECTypes.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "IREECTypes.td",
    deps = [":td_files"],
)

iree_gentbl_cc_library(
    name = "IREECAttrsGen",
    tbl_outs = [
        (
            [
                "--gen-attrdef-decls",
                "--attrdefs-dialect=ireec",
            ],
            "IREECAttrs.h.inc",
        ),
        (
            [
                "--gen-attrdef-defs",
                "--attrdefs-dialect=ireec",
            ],
            "IREECAttrs.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "IREECAttributes.td",
    deps = [":td_files"],
)

iree_tablegen_doc(
    name = "IREECDialectDocGen",
    tbl_outs = [
        (
            [
                "--gen-dialect-doc",
                "-dialect=ireec",
            ],
            "IREECDialect.md",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "IREECOps.td",
    deps = [":td_files"],
)
