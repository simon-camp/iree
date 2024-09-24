vm.module @list_ops_f64 {

  //===--------------------------------------------------------------------===//
  // vm.list.* with F32 types
  //===--------------------------------------------------------------------===//

  vm.export @test_f64 attributes {emitc.exclude}
  vm.func private @test_f64() {
    %capacity = vm.const.i32 42
    %index = vm.const.i32 41
    %c54312 = vm.const.f64 54321.0
    %list = vm.list.alloc %capacity : (i32) -> !vm.list<f64>
    %sz = vm.list.size %list : (!vm.list<f64>) -> i32
    vm.list.resize %list, %capacity : (!vm.list<f64>, i32)
    vm.list.set.f64 %list, %index, %c54312 : (!vm.list<f64>, i32, f64)
    %v = vm.list.get.f64 %list, %index : (!vm.list<f64>, i32) -> f64
    vm.check.eq %v, %c54312, "list<f64>.empty.set(41, 54321.0).get(41)=54321.0" : f64
    vm.return
  }

}
