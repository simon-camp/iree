// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(vm.module(iree-vm-ordinal-allocation),vm.module(iree-convert-vm-to-emitc))" %s | FileCheck %s

vm.module @my_module {
  // CHECK-LABEL: emitc.func private @my_module_list_get_f64
  vm.func @list_get_f64(%arg0: !vm.list<f64>, %arg1: i32) -> f64 {
    // CHECK-NEXT: %0 = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.lvalue<!emitc.opaque<"iree_vm_value_t">>
    // CHECK-NEXT: %1 = emitc.apply "&"(%0) : (!emitc.lvalue<!emitc.opaque<"iree_vm_value_t">>) -> !emitc.ptr<!emitc.opaque<"iree_vm_value_t">>
    // CHECK-NEXT: %2 = emitc.apply "*"(%arg3) : (!emitc.ptr<!emitc.opaque<"iree_vm_ref_t">>) -> !emitc.opaque<"iree_vm_ref_t">
    // CHECK-NEXT: %3 = emitc.call_opaque "iree_vm_list_deref"(%2) : (!emitc.opaque<"iree_vm_ref_t">) -> !emitc.ptr<!emitc.opaque<"iree_vm_list_t">>
    // CHECK: %{{.+}} = emitc.call_opaque "iree_vm_list_get_value_as"(%3, %arg4, %1) {args = [0 : index, 1 : index, #emitc.opaque<"IREE_VM_VALUE_TYPE_F64">, 2 : index]} : (!emitc.ptr<!emitc.opaque<"iree_vm_list_t">>, i32, !emitc.ptr<!emitc.opaque<"iree_vm_value_t">>) -> !emitc.opaque<"iree_status_t">
    %0 = vm.list.get.f64 %arg0, %arg1 : (!vm.list<f64>, i32) -> f64
    vm.return %0 : f64
  }
}

// -----

vm.module @my_module {
  // CHECK-LABEL: emitc.func private @my_module_list_set_f64
  vm.func @list_set_f64(%arg0: !vm.list<f64>, %arg1: i32, %arg2: f64) {
    // CHECK-NEXT: %0 = emitc.call_opaque "iree_vm_value_make_f64"(%arg5) : (f64) -> !emitc.opaque<"iree_vm_value_t">
    // CHECK-NEXT: %1 = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.lvalue<!emitc.opaque<"iree_vm_value_t">>
    // CHECK-NEXT: emitc.assign %0 : !emitc.opaque<"iree_vm_value_t"> to %1 : <!emitc.opaque<"iree_vm_value_t">>
    // CHECK-NEXT: %2 = emitc.apply "&"(%1) : (!emitc.lvalue<!emitc.opaque<"iree_vm_value_t">>) -> !emitc.ptr<!emitc.opaque<"iree_vm_value_t">>
    // CHECK-NEXT: %3 = emitc.apply "*"(%arg3) : (!emitc.ptr<!emitc.opaque<"iree_vm_ref_t">>) -> !emitc.opaque<"iree_vm_ref_t">
    // CHECK-NEXT: %4 = emitc.call_opaque "iree_vm_list_deref"(%3) : (!emitc.opaque<"iree_vm_ref_t">) -> !emitc.ptr<!emitc.opaque<"iree_vm_list_t">>
    // CHECK: %{{.+}} = emitc.call_opaque "iree_vm_list_set_value"(%4, %arg4, %2) : (!emitc.ptr<!emitc.opaque<"iree_vm_list_t">>, i32, !emitc.ptr<!emitc.opaque<"iree_vm_value_t">>) -> !emitc.opaque<"iree_status_t">
    vm.list.set.f64 %arg0, %arg1, %arg2 : (!vm.list<f64>, i32, f64)
    vm.return
  }
}
