// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(vm.module(iree-vm-ordinal-allocation),vm.module(iree-convert-vm-to-emitc))" %s | FileCheck %s

vm.module @my_module {
  // CHECK-LABEL: @my_module_list_alloc
  vm.func @list_alloc(%arg0: i32) -> !vm.list<i32> {
    // CHECK: %[[A:.+]] = emitc.apply "&"(%{{.+}}) : (!emitc.opaque<"iree_vm_type_def_t">) -> !emitc.ptr<!emitc.opaque<"iree_vm_type_def_t">>
    // CHECK: emitc.call "EMITC_STRUCT_MEMBER_ASSIGN"
    // CHECK: %[[B:.+]] = "emitc.variable"() {value = #emitc.opaque<"NULL">} : () -> !emitc.ptr<!emitc.opaque<"iree_vm_list_t">>
    // CHECK: %[[C:.+]] = emitc.apply "&"(%[[B]]) : (!emitc.ptr<!emitc.opaque<"iree_vm_list_t">>) -> !emitc.ptr<!emitc.ptr<!emitc.opaque<"iree_vm_list_t">>>
    // CHECK: %[[D:.+]] = emitc.call "EMITC_STRUCT_PTR_MEMBER"(%arg2) {args = [0 : index, #emitc.opaque<"allocator">]} : (!emitc.ptr<!emitc.opaque<"my_module_state_t">>) -> !emitc.opaque<"iree_allocator_t">
    // CHECK: %{{.+}} = emitc.call "iree_vm_list_create"(%[[A]], %arg3, %[[D]], %[[C]]) : (!emitc.ptr<!emitc.opaque<"iree_vm_type_def_t">>, i32, !emitc.opaque<"iree_allocator_t">, !emitc.ptr<!emitc.ptr<!emitc.opaque<"iree_vm_list_t">>>) -> !emitc.opaque<"iree_status_t">
    // CHECK: %[[E:.+]] = emitc.call "iree_vm_list_type_id"() : () -> !emitc.opaque<"iree_vm_ref_type_t">
    // CHECK: %{{.+}} = emitc.call "iree_vm_ref_wrap_assign"(%[[B]], %[[E]], %arg4) : (!emitc.ptr<!emitc.opaque<"iree_vm_list_t">>, !emitc.opaque<"iree_vm_ref_type_t">, !emitc.ptr<!emitc.opaque<"iree_vm_ref_t">>) -> !emitc.opaque<"iree_status_t">
    %0 = vm.list.alloc %arg0 : (i32) -> !vm.list<i32>
    vm.return %0 : !vm.list<i32>
  }
}

// -----

vm.module @my_module {
  // CHECK-LABEL: @my_module_list_reserve
  vm.func @list_reserve(%arg0: !vm.list<i32>, %arg1: i32) {
    // CHECK-NEXT: %0 = emitc.apply "*"(%arg3) : (!emitc.ptr<!emitc.opaque<"iree_vm_ref_t">>) -> !emitc.opaque<"iree_vm_ref_t">
    // CHECK-NEXT: %1 = emitc.call "iree_vm_list_deref"(%0) : (!emitc.opaque<"iree_vm_ref_t">) -> !emitc.ptr<!emitc.opaque<"iree_vm_list_t">>
    // CHECK: %{{.+}} = emitc.call "iree_vm_list_reserve"(%1, %arg4) : (!emitc.ptr<!emitc.opaque<"iree_vm_list_t">>, i32) -> !emitc.opaque<"iree_status_t">
    vm.list.reserve %arg0, %arg1 : (!vm.list<i32>, i32)
    vm.return
  }
}

// -----

vm.module @my_module {
  // CHECK-LABEL: @my_module_list_resize
  vm.func @list_resize(%arg0: !vm.list<i32>, %arg1: i32) {
    // CHECK-NEXT: %0 = emitc.apply "*"(%arg3) : (!emitc.ptr<!emitc.opaque<"iree_vm_ref_t">>) -> !emitc.opaque<"iree_vm_ref_t">
    // CHECK-NEXT: %1 = emitc.call "iree_vm_list_deref"(%0) : (!emitc.opaque<"iree_vm_ref_t">) -> !emitc.ptr<!emitc.opaque<"iree_vm_list_t">>
    // CHECK: %{{.+}} = emitc.call "iree_vm_list_resize"(%1, %arg4) : (!emitc.ptr<!emitc.opaque<"iree_vm_list_t">>, i32) -> !emitc.opaque<"iree_status_t">
    vm.list.resize %arg0, %arg1 : (!vm.list<i32>, i32)
    vm.return
  }
}

// -----

vm.module @my_module {
  // CHECK-LABEL: @my_module_list_size
  vm.func @list_size(%arg0: !vm.list<i32>) -> i32 {
    // CHECK-NEXT: %0 = emitc.apply "*"(%arg3) : (!emitc.ptr<!emitc.opaque<"iree_vm_ref_t">>) -> !emitc.opaque<"iree_vm_ref_t">
    // CHECK-NEXT: %1 = emitc.call "iree_vm_list_deref"(%0) : (!emitc.opaque<"iree_vm_ref_t">) -> !emitc.ptr<!emitc.opaque<"iree_vm_list_t">>
    // CHECK: %{{.+}} = emitc.call "iree_vm_list_size"(%1) : (!emitc.ptr<!emitc.opaque<"iree_vm_list_t">>) -> i32
    %0 = vm.list.size %arg0 : (!vm.list<i32>) -> i32
    vm.return %0 : i32
  }
}

// -----

vm.module @my_module {
  // CHECK-LABEL: @my_module_list_get_i32
  vm.func @list_get_i32(%arg0: !vm.list<i32>, %arg1: i32) -> i32 {
    // CHECK-NEXT: %0 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.opaque<"iree_vm_value_t">
    // CHECK-NEXT: %1 = emitc.apply "&"(%0) : (!emitc.opaque<"iree_vm_value_t">) -> !emitc.ptr<!emitc.opaque<"iree_vm_value_t">>
    // CHECK-NEXT: %2 = emitc.apply "*"(%arg3) : (!emitc.ptr<!emitc.opaque<"iree_vm_ref_t">>) -> !emitc.opaque<"iree_vm_ref_t">
    // CHECK-NEXT: %3 = emitc.call "iree_vm_list_deref"(%2) : (!emitc.opaque<"iree_vm_ref_t">) -> !emitc.ptr<!emitc.opaque<"iree_vm_list_t">>
    // CHECK: %{{.+}} = emitc.call "iree_vm_list_get_value_as"(%3, %arg4, %1) {args = [0 : index, 1 : index, #emitc.opaque<"IREE_VM_VALUE_TYPE_I32">, 2 : index]} : (!emitc.ptr<!emitc.opaque<"iree_vm_list_t">>, i32, !emitc.ptr<!emitc.opaque<"iree_vm_value_t">>) -> !emitc.opaque<"iree_status_t">
    %0 = vm.list.get.i32 %arg0, %arg1 : (!vm.list<i32>, i32) -> i32
    vm.return %0 : i32
  }
}

// -----

vm.module @my_module {
  // CHECK-LABEL: @my_module_list_get_ref
  vm.func @list_get_ref(%arg0: !vm.list<!vm.ref<?>>, %arg1: i32) -> !vm.buffer {
    // CHECK-NEXT: %0 = emitc.apply "*"(%arg3) : (!emitc.ptr<!emitc.opaque<"iree_vm_ref_t">>) -> !emitc.opaque<"iree_vm_ref_t">
    // CHECK-NEXT: %1 = emitc.call "iree_vm_list_deref"(%0) : (!emitc.opaque<"iree_vm_ref_t">) -> !emitc.ptr<!emitc.opaque<"iree_vm_list_t">>
    // CHECK: %{{.+}} = emitc.call "iree_vm_list_get_ref_retain"(%1, %arg4, %arg3) : (!emitc.ptr<!emitc.opaque<"iree_vm_list_t">>, i32, !emitc.ptr<!emitc.opaque<"iree_vm_ref_t">>) -> !emitc.opaque<"iree_status_t">
    // CHECK: %[[A:.+]] = emitc.call "EMITC_STRUCT_PTR_MEMBER"(%arg3) {args = [0 : index, #emitc.opaque<"type">]} : (!emitc.ptr<!emitc.opaque<"iree_vm_ref_t">>) -> !emitc.opaque<"iree_vm_ref_type_t">
    // CHECK: %[[B:.+]] = "emitc.constant"() {value = #emitc.opaque<"IREE_VM_REF_TYPE_NULL">} : () -> !emitc.opaque<"iree_vm_ref_type_t">
    // CHECK: %[[C:.+]] = emitc.call "iree_vm_type_def_is_value"(%{{.+}}) : (!emitc.ptr<!emitc.opaque<"iree_vm_type_def_t">>) -> i1
    // CHECK: %[[D:.+]] = emitc.call "EMITC_STRUCT_PTR_MEMBER"(%{{.+}}) {args = [0 : index, #emitc.opaque<"ref_type">]} : (!emitc.ptr<!emitc.opaque<"iree_vm_type_def_t">>) -> !emitc.opaque<"iree_vm_ref_type_t">
    // CHECK: %[[E:.+]] = emitc.call "EMITC_BINARY"(%[[A]], %[[B]]) {args = [#emitc.opaque<"!=">, 0 : index, 1 : index]} : (!emitc.opaque<"iree_vm_ref_type_t">, !emitc.opaque<"iree_vm_ref_type_t">) -> i1
    // CHECK: %[[F:.+]] = emitc.call "EMITC_BINARY"(%[[A]], %[[D]]) {args = [#emitc.opaque<"!=">, 0 : index, 1 : index]} : (!emitc.opaque<"iree_vm_ref_type_t">, !emitc.opaque<"iree_vm_ref_type_t">) -> i1
    // CHECK: %[[G:.+]] = emitc.call "EMITC_BINARY"(%[[C]], %[[F]]) {args = [#emitc.opaque<"||">, 0 : index, 1 : index]} : (i1, i1) -> i1
    // CHECK: %{{.+}} = emitc.call "EMITC_BINARY"(%[[E]], %[[G]]) {args = [#emitc.opaque<"&&">, 0 : index, 1 : index]} : (i1, i1) -> i1
    // CHECK: cf.cond_br %{{.+}}, ^[[FAIL:.+]], ^[[CONTINUE:.+]]
    // CHECK: ^[[FAIL]]:
    // CHECK-NEXT: emitc.call "iree_vm_ref_release"(%arg3) : (!emitc.ptr<!emitc.opaque<"iree_vm_ref_t">>) -> ()
    // CHECK-NEXT: cf.br ^[[CONTINUE]]
    %0 = vm.list.get.ref %arg0, %arg1 : (!vm.list<!vm.ref<?>>, i32) -> !vm.buffer
    vm.return %0 : !vm.buffer
  }
}

// -----

vm.module @my_module {
  // CHECK-LABEL: @my_module_list_set_i32
  vm.func @list_set_i32(%arg0: !vm.list<i32>, %arg1: i32, %arg2: i32) {
    // CHECK-NEXT: %0 = emitc.call "iree_vm_value_make_i32"(%arg5) : (i32) -> !emitc.opaque<"iree_vm_value_t">
    // CHECK-NEXT: %1 = emitc.apply "&"(%0) : (!emitc.opaque<"iree_vm_value_t">) -> !emitc.ptr<!emitc.opaque<"iree_vm_value_t">>
    // CHECK-NEXT: %2 = emitc.apply "*"(%arg3) : (!emitc.ptr<!emitc.opaque<"iree_vm_ref_t">>) -> !emitc.opaque<"iree_vm_ref_t">
    // CHECK-NEXT: %3 = emitc.call "iree_vm_list_deref"(%2) : (!emitc.opaque<"iree_vm_ref_t">) -> !emitc.ptr<!emitc.opaque<"iree_vm_list_t">>
    // CHECK: %{{.+}} = emitc.call "iree_vm_list_set_value"(%3, %arg4, %1) : (!emitc.ptr<!emitc.opaque<"iree_vm_list_t">>, i32, !emitc.ptr<!emitc.opaque<"iree_vm_value_t">>) -> !emitc.opaque<"iree_status_t">
    vm.list.set.i32 %arg0, %arg1, %arg2 : (!vm.list<i32>, i32, i32)
    vm.return
  }
}

// -----

vm.module @my_module {
  // CHECK-LABEL: @my_module_list_set_ref
  vm.func @list_set_ref(%arg0: !vm.list<!vm.ref<?>>, %arg1: i32, %arg2: !vm.buffer) {
    // CHECK-NEXT: %0 = emitc.apply "*"(%arg3) : (!emitc.ptr<!emitc.opaque<"iree_vm_ref_t">>) -> !emitc.opaque<"iree_vm_ref_t">
    // CHECK-NEXT: %1 = emitc.call "iree_vm_list_deref"(%0) : (!emitc.opaque<"iree_vm_ref_t">) -> !emitc.ptr<!emitc.opaque<"iree_vm_list_t">>
    // CHECK: %{{.+}} = emitc.call "iree_vm_list_set_ref_retain"(%1, %arg4, %arg5) : (!emitc.ptr<!emitc.opaque<"iree_vm_list_t">>, i32, !emitc.ptr<!emitc.opaque<"iree_vm_ref_t">>) -> !emitc.opaque<"iree_status_t">
    vm.list.set.ref %arg0, %arg1, %arg2 : (!vm.list<!vm.ref<?>>, i32, !vm.buffer)
    vm.return
  }
}
