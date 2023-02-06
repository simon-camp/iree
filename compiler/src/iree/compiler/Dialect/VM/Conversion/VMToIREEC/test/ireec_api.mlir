// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(vm.module(iree-vm-ordinal-allocation),vm.module(iree-convert-vm-to-ireec))" %s | FileCheck %s

// // CHECK: vm.module public @empty_module
// vm.module @empty_module {
// }

// // -----

// CHECK: vm.module public @func_empty
vm.module @func_empty {
  // CHECK ireec.func
  // vm.func @func() {
  //   vm.return
  // }

  // vm.func @func_arg_i32(%arg0 : i32) {
  //   vm.return
  // }

  // vm.func @func_arg_ref(%arg0 : !vm.ref<?>) {
  //   vm.return
  // }

  // vm.func @func_res_i32() -> i32 {
  //   %0 = vm.const.i32 0
  //   vm.return %0 : i32
  // }

  vm.func @func_res_ref() -> !vm.ref<?> {
    %0 = vm.const.ref.zero : !vm.ref<?>
    vm.return %0 : !vm.ref<?>
  }

  vm.func @func_res_ref_2(%arg0 : !vm.ref<?>) -> !vm.ref<?> {
    %0 = vm.const.ref.zero : !vm.ref<?>
    vm.return %0 : !vm.ref<?>
  }

  // vm.func @test_cond_br_int_arg(%arg0 : i32, %arg1 : i32) -> i32 {
  //   vm.cond_br %arg0, ^bb1(%arg1 : i32), ^bb2(%arg1 : i32)
  // ^bb1(%arg2 : i32):
  //   vm.return %arg2 : i32
  // ^bb2(%arg3 : i32):
  //   vm.return %arg3 : i32
  // }

  vm.func @test_cond_br_ref_arg(%arg0 : i32, %ref : !vm.ref<?>) -> !vm.ref<?> {
    vm.cond_br %arg0, ^bb1(%ref : !vm.ref<?>), ^bb2(%ref : !vm.ref<?>)
  ^bb1(%arg1 : !vm.ref<?>):
    vm.return %arg1 : !vm.ref<?>
  ^bb2(%arg2 : !vm.ref<?>):
    vm.return %arg2 : !vm.ref<?>
  }
}

// // -----

// // CHECK: vm.module public @func_arg_i32
// vm.module @func_arg_i32 {
//   vm.func @func(%arg0 : i32) {
//     vm.return
//   }
// }

// // -----

// // CHECK: vm.module public @func_arg_ref
// vm.module @func_arg_ref {
//   vm.func @func(%arg0 : !vm.ref<?>) {
//     vm.return
//   }
// }

// // -----

// // CHECK: vm.module public @func_res_i32
// vm.module @func_res_i32 {
//   vm.func @func() -> i32 {
//     %0 = vm.const.i32 0
//     vm.return %0 : i32
//   }
// }

// // -----

// // CHECK: vm.module public @func_res_ref
// vm.module @func_res_ref {
//   vm.func @func() -> !vm.ref<?> {
//     %0 = vm.const.ref.zero : !vm.ref<?>
//     vm.return %0 : !vm.ref<?>
//   }
// }
