// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(vm.module(iree-vm-ordinal-allocation),vm.module(iree-convert-vm-to-emitc))" %s | FileCheck %s

vm.module @my_module {
  vm.global.i32 private @c42 = 42 : i32

  // CHECK-LABEL: @my_module_global_load_i32
  vm.func @global_load_i32() -> i32 {
    // CHECK-NEXT: %0 = emitc.call "EMITC_STRUCT_PTR_MEMBER"(%arg2) {args = [0 : index, #emitc.opaque<"rwdata">]} : (!emitc.ptr<!emitc.opaque<"my_module_state_t">>) -> !emitc.ptr<ui8>
    // CHECK-NEXT: %1 = emitc.call "vm_global_load_i32"(%0) {args = [0 : index, 0 : ui32]} : (!emitc.ptr<ui8>) -> i32
    %0 = vm.global.load.i32 @c42 : i32
    vm.return %0 : i32
  }
}

// -----

vm.module @my_module {
  vm.global.i32 private mutable @c107_mut = 107 : i32

  // CHECK-LABEL: @my_module_global_store_i32
  vm.func @global_store_i32(%arg0 : i32) {
    // CHECK-NEXT: %0 = emitc.call "EMITC_STRUCT_PTR_MEMBER"(%arg2) {args = [0 : index, #emitc.opaque<"rwdata">]} : (!emitc.ptr<!emitc.opaque<"my_module_state_t">>) -> !emitc.ptr<ui8>
    // CHECK-NEXT: emitc.call "vm_global_store_i32"(%0, %arg3) {args = [0 : index, 0 : ui32, 1 : index]} : (!emitc.ptr<ui8>, i32) -> ()
    vm.global.store.i32 %arg0, @c107_mut : i32
    vm.return
  }
}

// -----

vm.module @my_module {
  vm.global.ref private @g0 : !vm.buffer

  // CHECK-LABEL: @my_module_global_load_ref
  vm.func @global_load_ref() -> !vm.buffer {
    // CHECK: %[[A:.+]] = emitc.call "EMITC_STRUCT_PTR_MEMBER"(%arg2) {args = [0 : index, #emitc.opaque<"refs">]} : (!emitc.ptr<!emitc.opaque<"my_module_state_t">>) -> !emitc.ptr<!emitc.opaque<"iree_vm_ref_t">>
    // CHECK: %[[B:.+]] = emitc.call "EMITC_ARRAY_ELEMENT_ADDRESS"(%[[A]]) {args = [0 : index, 0 : ui32]} : (!emitc.ptr<!emitc.opaque<"iree_vm_ref_t">>) -> !emitc.ptr<!emitc.opaque<"iree_vm_ref_t">>
    // CHECK: %[[C:.+]] = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.opaque<"iree_vm_type_def_t">
    // CHECK: %[[D:.+]] = emitc.apply "&"(%[[C]]) : (!emitc.opaque<"iree_vm_type_def_t">) -> !emitc.ptr<!emitc.opaque<"iree_vm_type_def_t">>
    // CHECK: emitc.call "EMITC_STRUCT_MEMBER_ASSIGN"{{.*}}
    // CHECK: %[[E:.+]] = emitc.call "EMITC_STRUCT_PTR_MEMBER"(%[[D]]) {args = [0 : index, #emitc.opaque<"ref_type">]} : (!emitc.ptr<!emitc.opaque<"iree_vm_type_def_t">>) -> !emitc.opaque<"iree_vm_ref_type_t">
    // CHECK: %{{.+}} = emitc.call "iree_vm_ref_retain_or_move_checked"(%[[B]], %[[E]], %arg3) {args = [false, 0 : index, 1 : index, 2 : index]} : (!emitc.ptr<!emitc.opaque<"iree_vm_ref_t">>, !emitc.opaque<"iree_vm_ref_type_t">, !emitc.ptr<!emitc.opaque<"iree_vm_ref_t">>) -> !emitc.opaque<"iree_status_t">
    %0 = vm.global.load.ref @g0 : !vm.buffer
    vm.return %0 : !vm.buffer
  }
}

// -----

vm.module @my_module {
  vm.global.ref private mutable @g0_mut : !vm.buffer

  // CHECK-LABEL: @my_module_global_store_ref
  vm.func @global_store_ref(%arg0 : !vm.buffer) {
    // CHECK: %[[A:.+]] = emitc.call "EMITC_STRUCT_PTR_MEMBER"(%arg2) {args = [0 : index, #emitc.opaque<"refs">]} : (!emitc.ptr<!emitc.opaque<"my_module_state_t">>) -> !emitc.ptr<!emitc.opaque<"iree_vm_ref_t">>
    // CHECK: %[[B:.+]] = emitc.call "EMITC_ARRAY_ELEMENT_ADDRESS"(%[[A]]) {args = [0 : index, 0 : ui32]} : (!emitc.ptr<!emitc.opaque<"iree_vm_ref_t">>) -> !emitc.ptr<!emitc.opaque<"iree_vm_ref_t">>
    // CHECK: %[[C:.+]] = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.opaque<"iree_vm_type_def_t">
    // CHECK: %[[D:.+]] = emitc.apply "&"(%[[C]]) : (!emitc.opaque<"iree_vm_type_def_t">) -> !emitc.ptr<!emitc.opaque<"iree_vm_type_def_t">>
    // CHECK: emitc.call "EMITC_STRUCT_MEMBER_ASSIGN"{{.*}}
    // CHECK: %[[E:.+]] = emitc.call "EMITC_STRUCT_PTR_MEMBER"(%[[D]]) {args = [0 : index, #emitc.opaque<"ref_type">]} : (!emitc.ptr<!emitc.opaque<"iree_vm_type_def_t">>) -> !emitc.opaque<"iree_vm_ref_type_t">
    // CHECK: %{{.+}} = emitc.call "iree_vm_ref_retain_or_move_checked"(%arg3, %[[E]], %[[B]]) {args = [false, 0 : index, 1 : index, 2 : index]} : (!emitc.ptr<!emitc.opaque<"iree_vm_ref_t">>, !emitc.opaque<"iree_vm_ref_type_t">, !emitc.ptr<!emitc.opaque<"iree_vm_ref_t">>) -> !emitc.opaque<"iree_status_t">
    vm.global.store.ref %arg0, @g0_mut : !vm.buffer
    vm.return
  }
}
