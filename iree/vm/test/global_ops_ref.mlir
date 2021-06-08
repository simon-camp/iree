vm.module @global_ops_ref {

  //===--------------------------------------------------------------------===//
  // global.ref
  //===--------------------------------------------------------------------===//
  
  vm.rodata @buffer dense<[0]> : tensor<1xi32>
  vm.global.ref @buffer_ref mutable : !vm.buffer
  
  vm.global.ref @ref : !vm.ref<?>
  // TODO(simon-camp): Add test for initializer

  vm.export @test_global_load_ref
  vm.func @test_global_load_ref() {
    %actual = vm.global.load.ref @ref : !vm.ref<?>
    %expected = vm.const.ref.zero : !vm.ref<?>
    vm.check.eq %actual, %expected, "@ref != null" : !vm.ref<?>
    vm.return
  }

  vm.export @test_global_store_ref
  vm.func @test_global_store_ref() {
    %expected = vm.const.ref.rodata @buffer : !vm.buffer
    vm.global.store.ref %expected, @buffer_ref : !vm.buffer
    %actual = vm.global.load.ref @buffer_ref : !vm.buffer
    vm.check.eq %actual, %expected, "@buffer_ref != @buffer" : !vm.buffer
    vm.return
  }

}
