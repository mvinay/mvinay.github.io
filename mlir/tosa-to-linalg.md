# TOSA to Affine conversion in lastest MLIR(July 2024)

MLIR build instructions:
```
git clone --depth 1 https://github.com/llvm/llvm-project.git
cd llvm-project
mkdir -p build
cd build
cmake -G "Ninja" -DCMAKE_BUILD_TYPE="Debug" -DLLVM_ENABLE_ASSERTIONS=ON 
                 -DLLVM_TARGETS_TO_BUILD="Native" ../llvm -DCMAKE_C_COMPILER=clang 
                 -DCMAKE_CXX_COMPILER=clang++ -DLLVM_USE_LINKER=lld 
                 -DLLVM_ENABLE_PROJECTS="mlir" -DLLVM_CCACHE_BUILD=ON

ninja mlir-opt
```

TOSA 1D Add example (tosa_add.mlir):

```
func.func @example(%arg0: tensor<128xf32>, %arg1: tensor<128xf32>) -> tensor<128xf32> {
  %0 = tosa.add %arg0, %arg1 : (tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
  return %0 : tensor<128xf32>
}
``` 

MLIR pass pipeline:

```

<llvm-project>/build/bin/mlir-opt -pass-pipeline="builtin.module(func.func(tosa-to-linalg),
                         one-shot-bufferize{bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map},
                         func.func(finalizing-bufferize),
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
