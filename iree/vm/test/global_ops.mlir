vm.module @global_ops {

  //===--------------------------------------------------------------------===//
  // global.i32
  //===--------------------------------------------------------------------===//

  vm.global.i32 @c42 mutable 42 : i32
  vm.global.i32 @c42_init mutable init(@initializer) : i32
  vm.global.i32 @c107_mut mutable 107 : i32

  vm.func @initializer() -> i32 {
    %c42 = vm.const.i32 42 : i32
    // This ensures the initializer doesn't get folded into an initial_value attribute.
    %c42_dno = iree.do_not_optimize(%c42) : i32
    vm.return %c42_dno : i32
  }

  vm.export @test_global_load_i32
  vm.func @test_global_load_i32() {
    // The global is marked mutable to prevent folding.
    %actual = vm.global.load.i32 @c42 : i32
    %expected = vm.const.i32 42 : i32
    vm.check.eq %actual, %expected, "@c42 != 42" : i32
    vm.return
  }

  vm.export @test_global_load_i32_init
  vm.func @test_global_load_i32_init() {
    %actual = vm.global.load.i32 @c42_init : i32
    %expected = vm.const.i32 42 : i32
    vm.check.eq %actual, %expected, "@c42_init != 42" : i32
    vm.return
  }

  vm.export @test_global_store_i32
  vm.func @test_global_store_i32() {
    %c17 = vm.const.i32 17 : i32
    vm.global.store.i32 %c17, @c107_mut : i32
    %actual = vm.global.load.i32 @c107_mut : i32
    vm.check.eq %actual, %c17, "@c107_mut != 17" : i32
    vm.return
  }

}
