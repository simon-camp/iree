// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(vm.module(iree-vm-ordinal-allocation),vm.module(iree-convert-vm-to-ireec))" %s | FileCheck %s

vm.module @empty_module {
}

// -----

vm.module @func_empty {
  vm.func @func() {
    vm.return
  }
}

// -----

vm.module @func_arg_i32 {
  vm.func @func(%arg0 : i32) {
    vm.return
  }
}

// -----

vm.module @func_arg_ref {
  vm.func @func(%arg0 : !vm.ref<?>) {
    vm.return
  }
}

// -----

vm.module @func_res_i32 {
  vm.func @func() -> i32 {
    %0 = vm.const.i32 0
    vm.return %0 : i32
  }
}

// -----

vm.module @func_res_ref {
  vm.func @func() -> !vm.ref<?> {
    %0 = vm.const.ref.zero : !vm.ref<?>
    vm.return %0 : !vm.ref<?>
  }
}
