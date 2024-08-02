# Makefile utility for LLVM projects

Following is a Makefile template for LLVM based project to build, compile and run the llvm tools. Store the below contents to a file named `Makefile`. 

It can be easily extended to create more targets and more MLIR pipelines.

```Makefile

BUILD_DIRECTORY=<****llvm-project-build-repo**>
BIN_DIRECTORY= $(BUILD_DIRECTORY)/bin

# MLIR tools
MLIR_OPT=$(BIN_DIRECTORY)/mlir-opt
MLIR_CPU_RUNNER=$(BIN_DIRECTORY)/mlir-cpu-runner

# Input file
# Can be provided as option. Example:  `make c FILE=newfile.mlir`
FILE=new.mlir

# Temporary files created
COMPILE_OUT= c_$(FILE)
PFA_OUT = pfa_$(FILE)

# Example MLIR pipeline
TOSA_TO_LLVM_PIPELINE=-pass-pipeline="builtin.module(func.func(tosa-to-linalg),\
          one-shot-bufferize{bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map},\
          func.func(finalizing-bufferize),\
          convert-linalg-to-affine-loops, \
          func.func(affine-parallelize), \
          test-lower-to-llvm)"

# BUILD Target
b:
	ninja -C $(BUILD_DIRECTORY) mlir-opt

# COMPILE target
c:
	$(MLIR_OPT) $(TOSA_TO_LLVM_PIPELINE) $(FILE) -o $(COMPILE_OUT)

# COMPILE With print-after-all.
ca:
	$(MLIR_OPT) $(TOSA_TO_LLVM_PIPELINE) $(FILE) -o $(COMPILE_OUT) --mlir-print-ir-after-all 2> $(PFA_OUT)

# RUN target
r:
	$(MLIR_CPU_RUNNER) $(COMPILE_OUT) -O3 -e main -entry-point-result=void \
          -shared-libs=$(BUILD_DIRECTORY)/lib/libmlir_runner_utils.so

# Clean all the temp files.
clean:
	rm -rf $(COMPILE_OUT) $(PFA_OUT)

```

Usage:
1. Set the right LLVM build directory path for `BUILD_DIRECTORY`
1. Enter `make b` to build the project using Ninja
2. Enter `make b c r` to build, compile and run
3. Enter `make c FILE=new_file.mlir` to compile the new_file.mlir