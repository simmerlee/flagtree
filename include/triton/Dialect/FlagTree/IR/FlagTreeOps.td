#ifndef FLAGTREE_OPS
#define FLAGTREE_OPS

include "mlir/Dialect/LLVMIR/LLVMTypes.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/CommonTypeConstraints.td"
include "triton/Dialect/FlagTree/IR/FlagTreeDialect.td"
include "triton/Dialect/FlagTree/IR/FlagTreeAttrDefs.td"
include "triton/Dialect/Triton/IR/TritonInterfaces.td"
include "triton/Dialect/Triton/IR/TritonTypes.td"

class FlagTree_Op<string mnemonic, list<Trait> traits = []> :
    Op<FlagTree_Dialect, mnemonic,
       !listconcat(traits, [VerifyTensorLayoutsTrait])> {
}

def FlagTree_DSLRegionOp : FlagTree_Op<"dsl_region", [IsolatedFromAbove, RecursiveMemoryEffects]> {
  let arguments = (ins Variadic<AnyType>:$inputs);
  let regions = (region SizedRegion<1>:$body);
}

def FlagTree_YieldOp : FlagTree_Op<"yield", [Pure, Terminator, ReturnLike, HasParent<"DSLRegionOp">]> {
  let assemblyFormat = " attr-dict ";
}

def FlagTree_ExtractAllocatedPtrOp : FlagTree_Op<"extract_allocated_ptr", [Pure, HasParent<"DSLRegionOp">]> {
  let arguments = (ins TT_Tensor:$tensor);
  let results = (outs LLVMPointerType:$ptr);
  let assemblyFormat = "$tensor attr-dict `:` type($tensor) `to` type($ptr)";
}

def FlagTree_ExtractAlignedPtrOp : FlagTree_Op<"extract_aligned_ptr", [Pure, HasParent<"DSLRegionOp">]> {
  let arguments = (ins TT_Tensor:$tensor);
  let results = (outs LLVMPointerType:$ptr);
  let assemblyFormat = "$tensor attr-dict `:` type($tensor) `to` type($ptr)";
}

def FlagTree_ExtractOffsetOp : FlagTree_Op<"extract_offset", [Pure, HasParent<"DSLRegionOp">]> {
  let arguments = (ins TT_Tensor:$tensor);
  let results = (outs I64:$ptr);
  let assemblyFormat = "$tensor attr-dict `:` type($tensor) `to` type($ptr)";
}

def FlagTree_ExtractSizesOp : FlagTree_Op<"extract_sizes", [Pure, HasParent<"DSLRegionOp">]> {
  let arguments = (ins TT_Tensor:$tensor);
  let results = (outs Variadic<I64>:$sizes);
  let assemblyFormat = "$tensor attr-dict `:` type($tensor) `to` type($sizes)";
}

def FlagTree_ExtractStridesOp : FlagTree_Op<"extract_strides", [Pure, HasParent<"DSLRegionOp">]> {
  let arguments = (ins TT_Tensor:$tensor);
  let results = (outs Variadic<I64>:$strides);
  let assemblyFormat = "$tensor attr-dict `:` type($tensor) `to` type($strides)";
}

#endif
