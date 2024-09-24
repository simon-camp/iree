vm.module @list_ops_f32 {

  //===--------------------------------------------------------------------===//
  // vm.list.* with F32 types
  //===--------------------------------------------------------------------===//

  vm.export @test_f32
  vm.func @test_f32() {
    %capacity = vm.const.i32 42
    %index = vm.const.i32 41
    %c123 = vm.const.f32 123.0
    %list = vm.list.alloc %capacity : (i32) -> !vm.list<f32>
    %sz = vm.list.size %list : (!vm.list<f32>) -> i32
    vm.list.resize %list, %capacity : (!vm.list<f32>, i32)
    vm.list.set.f32 %list, %index, %c123 : (!vm.list<f32>, i32, f32)
    %v = vm.list.get.f32 %list, %index : (!vm.list<f32>, i32) -> f32
    vm.check.eq %v, %c123, "list<f32>.empty.set(41, 123.0).get(41)=123.0" : f32
    vm.return
  }

}
