vm.module @module_b {
  vm.func @square(%arg : i32) -> i32 {
    %0 = vm.mul.i32 %arg, %arg : i32
    vm.return %0 : i32
  }
  vm.export @square

  vm.func @ref_zero() -> !vm.ref<?> {
    %0 = vm.const.ref.zero : !vm.ref<?>
    vm.return %0 : !vm.ref<?>
  }
  vm.export @ref_zero

  vm.rodata private @buffer dense<[1, 2, 3]> : tensor<3xi8>
  vm.func @ref_nonzero() -> !vm.buffer {
    %0 = vm.const.ref.rodata @buffer : !vm.buffer
    vm.return %0 : !vm.buffer
  }
  vm.export @ref_nonzero

  vm.func @void() {
    vm.return
  }
  vm.export @void
}
