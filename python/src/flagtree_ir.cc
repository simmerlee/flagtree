#include "ir.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/Value.h"
#include "mlir/Parser/Parser.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/FlagTree/IR/Dialect.h"

namespace py = pybind11;

class FlagTreeOpBuilder : public TritonOpBuilder {
public:
  ModuleOp moveEdslFunc(std::string_view text, std::string_view fnname);
};

ModuleOp FlagTreeOpBuilder::moveEdslFunc(std::string_view text,
                                         std::string_view fnname) {
  ParserConfig config(getContext());
  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(text, config);
  LLVM::LLVMFuncOp func = module->lookupSymbol<LLVM::LLVMFuncOp>(fnname);
  return module->clone();
}

void init_flagtree_ir(py::module &&m) {
  using ret = py::return_value_policy;

  py::class_<flagtree::DSLRegionOp>(m, "DSLRegionOp", py::module_local(),
                                    py::dynamic_attr())
      .def("get_operation", &flagtree::DSLRegionOp::getOperation)
      .def("get_body", &flagtree::DSLRegionOp::getBody, ret::reference)
      .def("dump", &flagtree::DSLRegionOp::dump);

  py::class_<flagtree::YieldOp>(m, "YieldOp", py::module_local(),
                                py::dynamic_attr())
      .def("dump", &flagtree::YieldOp::dump);

  py::class_<FlagTreeOpBuilder, TritonOpBuilder>(
      m, "FlagTreeOpBuilder", py::module_local(), py::dynamic_attr())
      .def(py::init<MLIRContext *>())
      .def("get_op_builder", &FlagTreeOpBuilder::getBuilder, ret::reference)
      .def("move_edsl_func", &FlagTreeOpBuilder::moveEdslFunc,
           ret::take_ownership)
      .def("create_dsl_region_op",
           [](FlagTreeOpBuilder &self,
              const std::vector<Value> &inputs) -> flagtree::DSLRegionOp {
             return self.create<flagtree::DSLRegionOp>(inputs);
           })
      .def("create_yield_op",
           [](FlagTreeOpBuilder &self) -> flagtree::YieldOp {
             return self.create<flagtree::YieldOp>();
           })
      .def("create_extract_allocated_ptr_op",
           [](FlagTreeOpBuilder &self, Value tensor) -> Value {
             Type ptr = LLVM::LLVMPointerType::get(self.getContext());
             return self.create<flagtree::ExtractAllocatedPtrOp>(ptr, tensor);
           })
      .def("create_extract_aligned_ptr_op",
           [](FlagTreeOpBuilder &self, Value tensor) -> Value {
             Type ptr = LLVM::LLVMPointerType::get(self.getContext());
             return self.create<flagtree::ExtractAlignedPtrOp>(ptr, tensor);
           })
      .def("create_extract_offset_op",
           [](FlagTreeOpBuilder &self, Value tensor) -> Value {
             return self.create<flagtree::ExtractOffsetOp>(tensor);
           })
      .def("create_extract_sizes_op",
           [](FlagTreeOpBuilder &self, const std::vector<Type> &sizes,
              Value tensor) -> flagtree::ExtractSizesOp {
             return self.create<flagtree::ExtractSizesOp>(sizes, tensor);
           })
      .def("create_extract_strides_op",
           [](FlagTreeOpBuilder &self, const std::vector<Type> &strides,
              Value tensor) -> flagtree::ExtractStridesOp {
             return self.create<flagtree::ExtractStridesOp>(strides, tensor);
           });
}
