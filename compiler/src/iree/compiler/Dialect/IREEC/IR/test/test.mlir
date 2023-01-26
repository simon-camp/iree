// RUN: iree-opt --split-input-file %s | iree-opt --split-input-file | FileCheck %s

func.func @test_ref() {
  %0 = ireec.ref.null : !ireec.ref
  %1 = ireec.ref.null : !ireec.ref
  ireec.ref.release %0
  ireec.ref.move %0 -> %1
  ireec.ref.move.parallel (%0, %0) -> (%1, %1)
  ireec.ref.retain %0 -> %1
  ireec.ref.assign %0 -> %1
  return
}

func.func @test_status(%arg0 : !ireec.stack) -> !ireec.status {
  %0 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !ireec.status
  %1 = ireec.status.is_ok %0 : i1
  ireec.status.return_if_error %0
  return %0 : !ireec.status
}

ireec.global.struct @struct = [#ireec.struct_field<"a" : i32>]

ireec.func @struct_type(%arg0 : !ireec.struct_ref<@buffer1>) {
  ireec.return
}

ireec.func @destroy(%arg0 : !ireec.voidptr) {
  %0 = ireec.module.cast %arg0 : !ireec.module
  ireec.module.free %0
  ireec.return
}

ireec.global.byte_buffer alignas 16 @buffer1 = [0, 1, 2, 3]

ireec.func @alloc_state(
  %arg0 : !ireec.voidptr,
  %arg1 : !ireec.allocator,
  %arg2 : !ireec.ptr<!ireec.module_state>
) -> !ireec.status {
  %0 = ireec.module_state.derived.alloc %arg1 : !ireec.module_state.derived<type_name = "derived_t">
  
  // initialize buffers
  %1 = ireec.byte_span.from_buffer @buffer1 : !ireec.byte_span
  %2 = ireec.module_state.derived.buffer.get %0[0] : !ireec.module_state.derived<type_name = "derived_t"> -> !ireec.buffer
  %3 = ireec.allocator.null : !ireec.allocator
  ireec.buffer.init %2, %3, %1
  
  // zero out refs
  %4 = ireec.module_state.derived.ref.get %0[1] : !ireec.module_state.derived<type_name = "derived_t"> -> !ireec.ref
  ireec.ref.clear %4

  // write out result
  %5 = ireec.module_state.derived.cast.base %0 : !ireec.module_state.derived<type_name = "derived_t"> -> !ireec.module_state
  ireec.ptr.write %5, %arg2 : !ireec.ptr<!ireec.module_state>
  
  // return
  %6 = ireec.status.ok : !ireec.status
  ireec.return %6 : !ireec.status
}

// func.func @test_status_2(%arg0 : !ireec.stack) -> i32 {
//   %0 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !ireec.status
//   %1 = ireec.status.is_ok %0 : i1
//   ireec.status.return_if_error %0
//   %2 = arith.constant 0 : i32
//   return %2 : i32
// }
