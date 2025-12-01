from typing import TypeVar

from triton._C.libtriton.flagtree_ir import FlagTreeOpBuilder
from triton.language.semantic import TritonSemantic

TensorTy = TypeVar("TensorTy")


class FlagTreeSemantic(TritonSemantic[TensorTy]):

    def __init__(self, builder: FlagTreeOpBuilder, *args, **kwargs) -> None:
        super().__init__(builder, *args, **kwargs)

    def call(self, func, operands):
        operands = [operand.handle for operand in operands]
        operand_tys = [operand.get_type() for operand in operands]
        dsl_region_op = self.builder.create_dsl_region_op(operands)
        pt = self.builder.get_insertion_point()
        region = dsl_region_op.get_body()
        block = self.builder.create_block_with_parent(region, operand_tys)
        self.builder.set_insertion_point_to_start(block)
        args = []
        for idx in range(block.get_num_arguments()):
            arg = block.arg(idx)
            args += [self.builder.create_extract_allocated_ptr_op(arg), self.builder.create_extract_aligned_ptr_op(arg)]
        self.builder.move_edsl_func(str(func.make_llir()), func.fnname).dump()
        self.builder.create_yield_op()
        self.builder.restore_insertion_point(pt)
