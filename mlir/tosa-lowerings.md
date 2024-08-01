# TOSA Lowerings in lastest MLIR(July 2024)

MLIR build instructions:
```
git clone --depth 1 https://github.com/llvm/llvm-project.git
cd llvm-project
mkdir -p build
cd build
cmake -G "Ninja" -DCMAKE_BUILD_TYPE="Debug" -DLLVM_ENABLE_ASSERTIONS=ON \
                 -DLLVM_TARGETS_TO_BUILD="Native" ../llvm -DCMAKE_C_COMPILER=clang \
                 -DCMAKE_CXX_COMPILER=clang++ -DLLVM_USE_LINKER=lld \
                 -DLLVM_ENABLE_PROJECTS="mlir" -DLLVM_CCACHE_BUILD=ON

ninja mlir-opt mlir-cpu-runner mlir_runner_utils
```
## TOSA to Linalg lowering

TOSA 1D Add example (tosa_add.mlir):

```
func.func @example(%arg0: tensor<128xf32>, %arg1: tensor<128xf32>) -> tensor<128xf32> {
  %0 = tosa.add %arg0, %arg1 : (tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
  return %0 : tensor<128xf32>
}
``` 

MLIR pass pipeline:

```
<llvm-project>/build/bin/mlir-opt -pass-pipeline="builtin.module(func.func(tosa-to-linalg), \
          one-shot-bufferize{bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map},\
          func.func(finalizing-bufferize),\
          convert-linalg-to-affine-loops)" tosa_add.mlir -o out.mlir
```

Output for the example (out.mlir):

```
module {
  func.func @example(%arg0: memref<128xf32>, %arg1: memref<128xf32>) -> memref<128xf32> {
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<128xf32>
    affine.for %arg2 = 0 to 128 {
      %0 = affine.load %arg0[%arg2] : memref<128xf32>
      %1 = affine.load %arg1[%arg2] : memref<128xf32>
      %2 = arith.addf %0, %1 : f32
      affine.store %2, %alloc[%arg2] : memref<128xf32>
    }
    return %alloc : memref<128xf32>
  }
}
```
## Running TOSA workload in upstream MLIR using mlir-cpu-runner tool

TOSA Add example (tosa_add_run.mlir):

```
// Function to test the workload
func.func @workload(%arg0: tensor<128xf32>, %arg1: tensor<128xf32>) -> tensor<128xf32> {
  %0 = tosa.add %arg0, %arg1 : (tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
  return %0 : tensor<128xf32>
}

// Inbuilt function to print the memrefs. Contained in mlir_runner_utils
func.func private @printMemrefF32(tensor<*xf32>)

// Main function to test the workload()
func.func @main() {
  %0 = arith.constant dense<11.0> : tensor<128xf32>
  %1 = arith.constant dense<12.0> : tensor<128xf32>
  %res = func.call @workload(%0, %1) : (tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
  %res2 = tensor.cast %res : tensor<128xf32> to tensor<*xf32>
  call @printMemrefF32(%res2) : (tensor<*xf32>) -> ()
  return
}
```

MLIR pass pipeline:

```
<llvm-project>/build/bin/mlir-opt -pass-pipeline="builtin.module(func.func(tosa-to-linalg),\
          one-shot-bufferize{bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map},\
          func.func(finalizing-bufferize),\
          convert-linalg-to-affine-loops, \
          func.func(affine-parallelize), \
          test-lower-to-llvm)" tosa_add_run.mlir |
          mlir-cpu-runner -O3 -e main -entry-point-result=void \
          -shared-libs=<llvm-project>/build/lib/libmlir_runner_utils.so
```

Output should be like below:

```
Unranked Memref base@ = 0x5f36035b8780 rank = 1 offset = 0 sizes = [128] strides = [1] data = 
[23,  23,  23,  23,  23,  23,  23,  23,  23,  23,  23,  23,  23,  23,  23,  23,  23,  23,  23,  23,  23,  23,  23,  23,  23,  23,  23,  23,  23,  23,  23,  23,  23,  23,  23,  23,  23,  23,  23,  23,  23,  23,  23,  23,  23,  23,  23,  23,  23,  23,  23,  23,  23,  23,  23,  23,  23,  23,  23,  23,  23,  23,  23,  23,  23,  23,  23,  23,  23,  23,  23,  23,  23,  23,  23,  23,  23,  23,  23,  23,  23,  23,  23,  23,  23,  23,  23,  23,  23,  23,  23,  23,  23,  23,  23,  23,  23,  23,  23,  23,  23,  23,  23,  23,  23,  23,  23,  23,  23,  23,  23,  23,  23,  23,  23,  23,  23,  23,  23,  23,  23,  23,  23,  23,  23,  23,  23,  23]
```
