# Example `transform` dialect module to run mlir pipeline end-to-end

Input IR:
```
#map = affine_map<(d0) -> (d0)>
func.func @workload(%arg0: tensor<64x720xf32>, %arg1: tensor<720x64xf32>) -> tensor<64x64xf32> {
    %0 = tensor.empty() : tensor<64x64xf32>
    %1 = linalg.matmul 
           ins(%arg0, %arg1 : tensor<64x720xf32>, tensor<720x64xf32>) 
         outs(%0 : tensor<64x64xf32>) -> tensor<64x64xf32>

    return %1 : tensor<64x64xf32>
}

```

Transform module to generate llvm ir for above function:

command: `mlir-opt -transfrom-interpreter <input-file>`

```
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op) {
    %0 = transform.structured.match ops{["linalg.matmul"]} in %arg1 : (!transform.any_op) -> !transform.any_op

    %padded, %pad, %copy_back = transform.structured.pad %0 pad_to_multiple_of [64, 64, 64] {
      padding_values=[0.0: f32, 0.0:f32, 0.0:f32],
      padding_dimensions=[0, 1, 2], nofold, copy_back_op="linalg.copy"
    } : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)

    %1:2 = transform.structured.tile_using_forall %padded tile_sizes [64, 64]
           : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    %func_val = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op

    transform.apply_cse to %func_val: !transform.any_op
    transform.apply_patterns to %func_val {
      transform.apply_patterns.canonicalization
    } : !transform.any_op

    // transform.structured.vectorize %1#0 vector_sizes [64, 64] : !transform.any_op

    %new_module = transform.bufferization.one_shot_bufferize %arg1 {
      bufferize_function_boundaries = true,
      function_boundary_type_conversion = 1 : i32, 
      memcpy_op="linalg.copy" }
      : (!transform.any_op) -> !transform.any_op

    %f2 = transform.structured.match ops{["func.func"]} in %new_module : (!transform.any_op) -> !transform.any_op
    %f3 = transform.apply_registered_pass "convert-linalg-to-affine-loops" to %f2 : (!transform.any_op) -> !transform.any_op
    %f4 = transform.apply_registered_pass "affine-loop-fusion" to %f3 : (!transform.any_op) -> !transform.any_op
    // %f5 = transform.apply_registered_pass "test-lower-to-llvm" to %f4 : (!transform.any_op) -> !transform.any_op
    %f5 = transform.apply_registered_pass "lower-affine" to %f4 : (!transform.any_op) -> !transform.any_op
    %f6 = transform.apply_registered_pass "convert-scf-to-cf" to %f5 : (!transform.any_op) -> !transform.any_op

    transform.apply_conversion_patterns to %f6 {
      transform.apply_conversion_patterns.dialect_to_llvm "math"
      transform.apply_conversion_patterns.vector.vector_to_llvm
      transform.apply_conversion_patterns.dialect_to_llvm "memref"
      transform.apply_conversion_patterns.func.func_to_llvm
      transform.apply_conversion_patterns.dialect_to_llvm "index"
      transform.apply_conversion_patterns.dialect_to_llvm "arith"
      transform.apply_conversion_patterns.dialect_to_llvm "cf"
    } with type_converter {
    transform.apply_conversion_patterns.memref.memref_to_llvm_type_converter
      {index_bitwidth = 64,
       use_bare_ptr = false,
       use_bare_ptr_memref_call_conv = false,
       use_opaque_pointers = true}
  } {
    legal_dialects = ["llvm"],
    partial_conversion
  } : !transform.any_op

  %f7 = transform.structured.match ops{["llvm.func"]} in %new_module 
    : (!transform.any_op) -> !transform.any_op
  %f8 = transform.apply_registered_pass "reconcile-unrealized-casts" to %f7
    : (!transform.any_op) -> !transform.any_op

    transform.yield
  }
}
```

Note that initial padding, tiling, etc are added as part of experiment. It can be removed for simple end-to-end run.
