vm.module @module_a {
  vm.import @module_b.square(%arg : i32) -> i32
  vm.import @module_b.ref_nonzero() -> !vm.ref<?>
  vm.import @module_b.ref_zero() -> !vm.ref<?>
  vm.import @module_b.void() -> ()

  vm.func @test_square(%arg0: i32) -> i32 {
    %0 = vm.call @module_b.square(%arg0) : (i32) -> i32
    vm.return %0 : i32
  }
  vm.export @test_square

  vm.func @test_ref_nonzero() -> i32 {
    %0 = vm.call @module_b.ref_nonzero() : () -> !vm.ref<?>
    %1 = vm.call @module_b.ref_nonzero() : () -> !vm.ref<?>
    %2 = vm.cmp.eq.ref %0, %1 : !vm.ref<?> 
    vm.return %2 : i32
  }
  vm.export @test_ref_nonzero
  
  vm.func @test_ref_zero() -> i32 {
    %0 = vm.call @module_b.ref_zero() : () -> !vm.ref<?>
    %1 = vm.call @module_b.ref_zero() : () -> !vm.ref<?>
    // %1 = vm.const.ref.zero : !vm.ref<?>
    %2 = vm.cmp.eq.ref %0, %1 : !vm.ref<?> 
    vm.return %2 : i32
  }
  vm.export @test_ref_zero

  vm.func @test_void() -> i32 {
    %0 = vm.const.i32 0 : i32
    vm.call @module_b.void() : () -> ()
    vm.return %0 : i32
  }
  vm.export @test_void
}
