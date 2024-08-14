# Exploring various transformations at Linalg on Tensors level

## Tiling linalg operations using `transform` dialect

Take a simple elementwise add operation to tile:

**Example 1:**
```
#map = affine_map<(d0) -> (d0)>
func.func @workload(%arg0: tensor<128xf32>, %arg1: tensor<128xf32>) -> tensor<128xf32> {
    %0 = tensor.empty() : tensor<128xf32>
    %1 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} 
           ins(%arg0, %arg1 : tensor<128xf32>, tensor<128xf32>) 
         outs(%0 : tensor<128xf32>) -> tensor<128xf32>
    return %1 : tensor<128xf32>
  }

// transform module
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.elemwise_binary"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1:2 = transform.structured.tile_using_forall %0 tile_sizes [16]
           : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}

``` 

The transform module when interpreted (using `mlir-opt -transform-interpreter`), tiles all the linalg.elemwise_binary operations by size "16" and represent the loop using `scf.forall` operation.

Output:

```
#map = affine_map<(d0) -> (d0 * 16)>
module {
  func.func @workload(%arg0: tensor<128xf32>, %arg1: tensor<128xf32>) -> tensor<128xf32> {
    %0 = tensor.empty() : tensor<128xf32>
    %1 = scf.forall (%arg2) in (8) shared_outs(%arg3 = %0) -> (tensor<128xf32>) {
      %2 = affine.apply #map(%arg2)
      %3 = affine.apply #map(%arg2)
      %4 = affine.apply #map(%arg2)
      %extracted_slice = tensor.extract_slice %arg0[%2] [16] [1] : tensor<128xf32> to tensor<16xf32>
      %extracted_slice_0 = tensor.extract_slice %arg1[%3] [16] [1] : tensor<128xf32> to tensor<16xf32>
      %extracted_slice_1 = tensor.extract_slice %arg3[%4] [16] [1] : tensor<128xf32> to tensor<16xf32>
      %5 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%extracted_slice, %extracted_slice_0 : 
             tensor<16xf32>, tensor<16xf32>) outs(%extracted_slice_1 : tensor<16xf32>) -> tensor<16xf32>
      %6 = affine.apply #map(%arg2)
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %5 into %arg3[%6] [16] [1] : tensor<16xf32> into tensor<128xf32>
      }
    }
    return %1 : tensor<128xf32>
  }

...<Ignored the transform module>...
}

```

More complicated cases like matmul can be found [here](https://github.com/llvm/llvm-project/blob/main/mlir/test/Dialect/Linalg/tile-to-forall.mlir#L10)


Whenever the size of the tensor dimension is not a multiple of the tiling factor, the codegen would be more conservative.

The example below tiles the initial example with tile size of `15` instead of `16`. The output will be as follows:

**Example 2**
```
#map = affine_map<(d0) -> (d0 * -15 + 128, 15)>
#map1 = affine_map<(d0) -> (d0 * 15)>
module {
  func.func @workload(%arg0: tensor<128xf32>, %arg1: tensor<128xf32>) -> tensor<128xf32> {
    %0 = tensor.empty() : tensor<128xf32>
    %1 = scf.forall (%arg2) in (9) shared_outs(%arg3 = %0) -> (tensor<128xf32>) {
      %2 = affine.min #map(%arg2)
      %3 = affine.apply #map1(%arg2)
      %4 = affine.apply #map1(%arg2)
      %5 = affine.apply #map1(%arg2)
      %extracted_slice = tensor.extract_slice %arg0[%3] [%2] [1] : tensor<128xf32> to tensor<?xf32>
      %extracted_slice_0 = tensor.extract_slice %arg1[%4] [%2] [1] : tensor<128xf32> to tensor<?xf32>
      %extracted_slice_1 = tensor.extract_slice %arg3[%5] [%2] [1] : tensor<128xf32> to tensor<?xf32>
      %6 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%extracted_slice, %extracted_slice_0 
            : tensor<?xf32>, tensor<?xf32>) outs(%extracted_slice_1 : tensor<?xf32>) -> tensor<?xf32>
      %7 = affine.apply #map1(%arg2)
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %6 into %arg3[%7] [%2] [1] : tensor<?xf32> into tensor<128xf32>
      }
    }
    return %1 : tensor<128xf32>
  }

...<Ignored the transform module>...
}
```

` %2 = affine.min #map(%arg2)` is the key here. Note that tensors are marked with `?` to indicate dynamic shape. It can either be `15`(for first 8 iterations) or `8` (For last iteration, 128 % 8)

You can also pad the tensor to the multiple of `15` and tile it to that factor. Below section introduces the tensor padding operation.

## Tensor Padding using `tensor.pad` operation


Formal definition of the op can be found [here](https://mlir.llvm.org/docs/Dialects/TensorOps/#tensorpad-tensorpadop)

Let us look at the following 1-D Tensor padding example:

**Example 3**
```
func.func private @printMemrefF32(tensor<*xf32>)

func.func @main() {
  %plain_tensor = arith.constant dense<11.0> : tensor<3xf32>
  
  %pad = arith.constant 0.0 : f32
  %padded_tensor = tensor.pad %plain_tensor low[2] high[3] {
    ^bb0(%arg1: index):
      tensor.yield %pad : f32
    } : tensor<3xf32> to tensor<8xf32>

  %res2 = tensor.cast %plain_tensor : tensor<3xf32> to tensor<*xf32>
  call @printMemrefF32(%res2) : (tensor<*xf32>) -> ()
  %res3 = tensor.cast %padded_tensor : tensor<8xf32> to tensor<*xf32>
  call @printMemrefF32(%res3) : (tensor<*xf32>) -> ()
  return
}

```

Running the [TOSA pipeline](./tosa-lowerings.md) produces the following output:

```
Unranked Memref base@ = 0x6338ebd7dc80 rank = 1 offset = 0 sizes = [3] strides = [1] data = 
[11,  11,  11]
Unranked Memref base@ = 0x6338ebdfc140 rank = 1 offset = 0 sizes = [8] strides = [1] data = 
[0,  0,  11,  11,  11,  0,  0,  0]

```

NOTE: First output shows content of `plain_tensor` and the second one corresponds to `padded_tensor`

`tensor.pad` operation is applied on `plain_tensor`. `tensor.pad` prepends 2 elements (`low`) to `dim 0` and appends 3 elements to `dim 0`. So, `padded_tensor` of size `2(low) + 3(original) + 3 (high) = 8` is generated.

**Example 4 (2-D)**

```
func.func private @printMemrefF32(tensor<*xf32>)

func.func @main() {
  %plain_tensor = arith.constant dense<11.0> : tensor<3x3xf32>
  
  %pad = arith.constant 0.0 : f32
  %padded_tensor = tensor.pad %plain_tensor low[1, 2] high[3, 4] {
    ^bb0(%arg1: index, %arg2: index):
      tensor.yield %pad : f32
    } : tensor<3x3xf32> to tensor<7x9xf32>

  %res2 = tensor.cast %plain_tensor : tensor<3x3xf32> to tensor<*xf32>
  call @printMemrefF32(%res2) : (tensor<*xf32>) -> ()
  %res3 = tensor.cast %padded_tensor : tensor<7x9xf32> to tensor<*xf32>
  call @printMemrefF32(%res3) : (tensor<*xf32>) -> ()
  return
}
```

Output:

``` 
Unranked Memref base@ = 0x62e9d1ca5b00 rank = 2 offset = 0 sizes = [3, 3] strides = [3, 1] data = 
[[11,   11,   11], 
 [11,   11,   11], 
 [11,   11,   11]]
Unranked Memref base@ = 0x62e9d1d967c0 rank = 2 offset = 0 sizes = [7, 9] strides = [9, 1] data = 
[[0,   0,   0,   0,   0,   0,   0,   0,   0], 
 [0,   0,   11,   11,   11,   0,   0,   0,   0], 
 [0,   0,   11,   11,   11,   0,   0,   0,   0], 
 [0,   0,   11,   11,   11,   0,   0,   0,   0], 
 [0,   0,   0,   0,   0,   0,   0,   0,   0], 
 [0,   0,   0,   0,   0,   0,   0,   0,   0], 
 [0,   0,   0,   0,   0,   0,   0,   0,   0]]
```

Here `padded_tensor` is prepended with `1` row and appended with `3` rows in `dim 0`. Leading to `dim 0` size of `7`.

Prepended with `2` columns and appended with `4` columns in `dim 1`. Leading to `dim 1` size of `9`.

## Padding the Linalg operations using `transform` dialect

Let us now use the transform based padding for linalg operations to pad the tensor in *Example 2* when tiling factor is `15`

**Example 5**

```
#map = affine_map<(d0) -> (d0)>
func.func @workload(%arg0: tensor<128xf32>, %arg1: tensor<128xf32>) -> tensor<128xf32> {
    %0 = tensor.empty() : tensor<128xf32>
    %1 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} 
           ins(%arg0, %arg1 : tensor<128xf32>, tensor<128xf32>) 
         outs(%0 : tensor<128xf32>) -> tensor<128xf32>

    return %1 : tensor<128xf32>
  }

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.elemwise_binary"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %padded, %pad, %copy_back = transform.structured.pad %0 pad_to_multiple_of [15] {
      padding_values=[0.0: f32, 0.0:f32, 0.0:f32],
      padding_dimensions=[0], nofold, copy_back_op="linalg.copy"
    } : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    transform.yield
  }
}
```

Note that we are padding all 3 tensors to multiple of `15` with `0.0` as the padding value. We are using the `linalg.copy` operation to *copy back* the original tensor.

Output:

```
module {
  func.func @workload(%arg0: tensor<128xf32>, %arg1: tensor<128xf32>) -> tensor<128xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<128xf32>
    %padded = tensor.pad %arg0 low[0] high[7] {
    ^bb0(%arg2: index):
      tensor.yield %cst : f32
    } : tensor<128xf32> to tensor<135xf32>
    %padded_0 = tensor.pad %arg1 low[0] high[7] {
    ^bb0(%arg2: index):
      tensor.yield %cst : f32
    } : tensor<128xf32> to tensor<135xf32>
    %padded_1 = tensor.pad %0 low[0] high[7] {
    ^bb0(%arg2: index):
      tensor.yield %cst : f32
    } : tensor<128xf32> to tensor<135xf32>
    %1 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%padded, %padded_0 : tensor<135xf32>, tensor<135xf32>) outs(%padded_1 : tensor<135xf32>) -> tensor<135xf32>
    %extracted_slice = tensor.extract_slice %1[0] [128] [1] : tensor<135xf32> to tensor<128xf32>
    %2 = linalg.copy ins(%extracted_slice : tensor<128xf32>) outs(%0 : tensor<128xf32>) -> tensor<128xf32>
    return %2 : tensor<128xf32>
  }

  ...<Ignored the transform module>...
}
```

## Combining both Tiling and Padding operations using `transform` dialect for linalg operations

Now, let us add tiling (factor = `15`) to the above padding example (*Example 5*). 

Transform module :

``` 
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {

    // Chose the op to work on.
    %0 = transform.structured.match ops{["linalg.elemwise_binary"]} in %arg1 : (!transform.any_op) -> !transform.any_op

    // Padding op
    %padded, %pad, %copy_back = transform.structured.pad %0 pad_to_multiple_of [15] {
      padding_values=[0.0: f32, 0.0:f32, 0.0:f32],
      padding_dimensions=[0], nofold, copy_back_op="linalg.copy"
    } : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)

    // Tiling op
    %1:2 = transform.structured.tile_using_forall %padded tile_sizes [15]
           : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}
```

Output of `mlir-opt -transform-interpreter -canonicalize -cse` is:

```
#map = affine_map<(d0) -> (d0 * 15)>
#map = affine_map<(d0) -> (d0 * 15)>
module {
  func.func @workload(%arg0: tensor<128xf32>, %arg1: tensor<128xf32>) -> tensor<128xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<128xf32>
    %padded = tensor.pad %arg0 low[0] high[7] {
    ^bb0(%arg2: index):
      tensor.yield %cst : f32
    } : tensor<128xf32> to tensor<135xf32>
    %padded_0 = tensor.pad %arg1 low[0] high[7] {
    ^bb0(%arg2: index):
      tensor.yield %cst : f32
    } : tensor<128xf32> to tensor<135xf32>
    %padded_1 = tensor.pad %0 low[0] high[7] {
    ^bb0(%arg2: index):
      tensor.yield %cst : f32
    } : tensor<128xf32> to tensor<135xf32>
    %1 = scf.forall (%arg2) in (9) shared_outs(%arg3 = %padded_1) -> (tensor<135xf32>) {
      %3 = affine.apply #map(%arg2)
      %extracted_slice_2 = tensor.extract_slice %padded[%3] [15] [1] : tensor<135xf32> to tensor<15xf32>
      %extracted_slice_3 = tensor.extract_slice %padded_0[%3] [15] [1] : tensor<135xf32> to tensor<15xf32>
      %extracted_slice_4 = tensor.extract_slice %arg3[%3] [15] [1] : tensor<135xf32> to tensor<15xf32>
      %4 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%extracted_slice_2, %extracted_slice_3 : tensor<15xf32>, tensor<15xf32>) outs(%extracted_slice_4 : tensor<15xf32>) -> tensor<15xf32>
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %4 into %arg3[%3] [15] [1] : tensor<15xf32> into tensor<135xf32>
      }
    }
    %extracted_slice = tensor.extract_slice %1[0] [128] [1] : tensor<135xf32> to tensor<128xf32>
    %2 = linalg.copy ins(%extracted_slice : tensor<128xf32>) outs(%0 : tensor<128xf32>) -> tensor<128xf32>
    return %2 : tensor<128xf32>
  }
  ...<Ignored the transform module>...
}

```

Now, all the tensor shapes are static. And the padded value is *extracted* and stored to the original tensor fo size `128`.

## Vectorizing the linalg operations using `transform` dialect

Let us vectorize the IR by `VF = 15`

Transform module:
```
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.elemwise_binary"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %padded, %pad, %copy_back = transform.structured.pad %0 pad_to_multiple_of [15] {
      padding_values=[0.0: f32, 0.0:f32, 0.0:f32],
      padding_dimensions=[0], nofold, copy_back_op="linalg.copy"
    } : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    %1:2 = transform.structured.tile_using_forall %padded tile_sizes [15]
           : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.structured.vectorize %1#0 vector_sizes [15] : !transform.any_op
    transform.yield
  }
}
```
Output of `mlir-opt -transform-interpreter -canonicalize -cse` is:

```
#map = affine_map<(d0) -> (d0 * 15)>
module {
  func.func @workload(%arg0: tensor<128xf32>, %arg1: tensor<128xf32>) -> tensor<128xf32> {
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<128xf32>
    %padded = tensor.pad %arg0 low[0] high[7] {
    ^bb0(%arg2: index):
      tensor.yield %cst : f32
    } : tensor<128xf32> to tensor<135xf32>
    %padded_0 = tensor.pad %arg1 low[0] high[7] {
    ^bb0(%arg2: index):
      tensor.yield %cst : f32
    } : tensor<128xf32> to tensor<135xf32>
    %padded_1 = tensor.pad %0 low[0] high[7] {
    ^bb0(%arg2: index):
      tensor.yield %cst : f32
    } : tensor<128xf32> to tensor<135xf32>
    %1 = scf.forall (%arg2) in (9) shared_outs(%arg3 = %padded_1) -> (tensor<135xf32>) {
      %3 = affine.apply #map(%arg2)
      %extracted_slice_2 = tensor.extract_slice %padded[%3] [15] [1] : tensor<135xf32> to tensor<15xf32>
      %extracted_slice_3 = tensor.extract_slice %padded_0[%3] [15] [1] : tensor<135xf32> to tensor<15xf32>
      %extracted_slice_4 = tensor.extract_slice %arg3[%3] [15] [1] : tensor<135xf32> to tensor<15xf32>
      %4 = vector.transfer_read %extracted_slice_2[%c0], %cst {in_bounds = [true]} : tensor<15xf32>, vector<15xf32>
      %5 = vector.transfer_read %extracted_slice_3[%c0], %cst {in_bounds = [true]} : tensor<15xf32>, vector<15xf32>
      %6 = arith.addf %4, %5 : vector<15xf32>
      %7 = vector.transfer_write %6, %extracted_slice_4[%c0] {in_bounds = [true]} : vector<15xf32>, tensor<15xf32>
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %7 into %arg3[%3] [15] [1] : tensor<15xf32> into tensor<135xf32>
      }
    }
    %extracted_slice = tensor.extract_slice %1[0] [128] [1] : tensor<135xf32> to tensor<128xf32>
    %2 = linalg.copy ins(%extracted_slice : tensor<128xf32>) outs(%0 : tensor<128xf32>) -> tensor<128xf32>
    return %2 : tensor<128xf32>
  }
... <Ignore the transform module>...
}
```

You can see that the `linalg.elemwise_binary` op is converted to `arith.addf` with vector of 15 elements.


## Lowering the above IR to loops

Lowering the above output using the following pipeline

```
mlir-opt -pass-pipeline="builtin.module(transform-interpreter, canonicalize, cse,\
          one-shot-bufferize{bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map},\
          func.func(finalizing-bufferize),\
          convert-linalg-to-affine-loops, fold-memref-alias-ops, memref-expand, canonicalize, cse, canonicalize, scf-forall-to-parallel, func.func(affine-loop-fusion),\ 
          func.func(affine-parallelize)
```

Above command yeilds:

```
func.func @workload(%arg0: memref<128xf32>, %arg1: memref<128xf32>) -> memref<128xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<128xf32>
  %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<135xf32>
  affine.parallel (%arg2) = (0) to (135) {
    affine.store %cst, %alloc_0[%arg2] : memref<135xf32>
  }
  %subview = memref.subview %alloc_0[0] [128] [1] : memref<135xf32> to memref<128xf32, strided<[1]>>
  memref.copy %arg0, %subview : memref<128xf32> to memref<128xf32, strided<[1]>>
  %alloc_1 = memref.alloc() {alignment = 64 : i64} : memref<135xf32>
  affine.parallel (%arg2) = (0) to (135) {
    affine.store %cst, %alloc_1[%arg2] : memref<135xf32>
  }
  %subview_2 = memref.subview %alloc_1[0] [128] [1] : memref<135xf32> to memref<128xf32, strided<[1]>>
  memref.copy %arg1, %subview_2 : memref<128xf32> to memref<128xf32, strided<[1]>>
  %alloc_3 = memref.alloc() {alignment = 64 : i64} : memref<135xf32>
  affine.parallel (%arg2) = (0) to (135) {
    affine.store %cst, %alloc_3[%arg2] : memref<135xf32>
  }
  %subview_4 = memref.subview %alloc_3[0] [128] [1] : memref<135xf32> to memref<128xf32, strided<[1]>>
  memref.copy %alloc, %subview_4 : memref<128xf32> to memref<128xf32, strided<[1]>>
  %c0 = arith.constant 0 : index
  %c9 = arith.constant 9 : index
  %c1 = arith.constant 1 : index
  scf.parallel (%arg2) = (%c0) to (%c9) step (%c1) {
    %0 = affine.apply affine_map<(d0) -> (d0 * 15)>(%arg2)
    %1 = vector.transfer_read %alloc_0[%0], %cst {in_bounds = [true]} : memref<135xf32>, vector<15xf32>
    %2 = vector.transfer_read %alloc_1[%0], %cst {in_bounds = [true]} : memref<135xf32>, vector<15xf32>
    %3 = arith.addf %1, %2 : vector<15xf32>
    vector.transfer_write %3, %alloc_3[%0] {in_bounds = [true]} : vector<15xf32>, memref<135xf32>
    scf.reduce 
  }
  affine.parallel (%arg2) = (0) to (128) {
    %0 = affine.load %alloc_3[%arg2] : memref<135xf32>
    affine.store %0, %alloc[%arg2] : memref<128xf32>
  }
  return %alloc : memref<128xf32>
}
```

There are still inefficiencies in the IR:

1. `affine.parallel (%arg2)` could be eliminated
2. `memref.copy %alloc, %subview_4` is un-necessary
3. Last two loops, i.e., `scf.parallel` and `affine.parallel` can be merged. so,  `alloc_3` can be totally eliminated.

TODO: Find passes, transformations to do the same.