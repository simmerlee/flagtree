import triton.language as tl
from triton.language import cast
from triton.language.semantic import to_tensor, bitcast
from triton.language._utils import TRITON_MAX_TENSOR_NUMEL

def is_arange_check_power_of_two():
    return False

def check_arange_less_than_max_numel(range):
    if range > TRITON_MAX_TENSOR_NUMEL:
        raise ValueError(f"end - start must be less than or equal to TRITON_MAX_TENSOR_NUMEL = {TRITON_MAX_TENSOR_NUMEL}")

def is_cast_src_dst_scalar_type_equal(src_sca_ty, dst_sca_ty):
    if src_sca_ty == dst_sca_ty:
        return True
    return False

def check_unsupported_fp8_fp64(src_sca_ty, dst_sca_ty):
    if (src_sca_ty.is_fp8() or dst_sca_ty.is_fp8()) or (src_sca_ty.is_fp64() or dst_sca_ty.is_fp64()):
        raise ValueError("[fp8, fp64] is unsupported on Ascend for now."
                         "Source scalar type is " + str(src_sca_ty) + " and destination type is " + str(dst_sca_ty))

def ext_dot_lhs_supported_type():
    return (tl.int1)

def ext_dot_rhs_supported_type():
    return (tl.int1)

def dot_check_hf32_input_precision(input_precision, ir, lhs, rhs, ret_scalar_ty):
    if (input_precision == getattr(ir.INPUT_PRECISION, "HF32")):
        if (not lhs.dtype.is_fp32() or not rhs.dtype.is_fp32() or not ret_scalar_ty.is_fp32()):
            raise ValueError("input_precision = 'hf32' must be used with f32 * f32 = f32 on Ascend")

def is_dot_check_max_num_imprecise_acc():
    return False

def reset_dot_max_num_imprecise_acc(max_num_imprecise_acc):
    max_num_imprecise_acc = 0
    return max_num_imprecise_acc

def check_was_bool_to_int8_dtype(input):
    if hasattr(input, 'was_bool_to_int8'):
        if input.type.scalar.is_int8():
            raise TypeError(f"unexpected type bool")

def check_was_bool_to_int8_dtype_and_cast(input, builder):
    assert input.type.scalar.is_int8(), "input wat bool to int8. However, input.type is not int8."
    return cast(input, tl.int1, builder)

def check_unexpected_dtype_float(input):
    if input.type.scalar.is_floating():
        raise TypeError(f"unexpected type {input.type.scalar}")

def check_unexpected_dtype_bool(dtype):
    if dtype.is_bool():
        raise TypeError(f"Unexpected dtype {dtype}")

def set_load_legacy_other_input(builder):
    return to_tensor(0, builder)

def cast_back_when_load_legacy_ptr_is_bool():
    return False

def set_attr_was_bool_to_int8(ret, is_bool):
    if is_bool:
        ret.was_bool_to_int8 = True

def is_atomic_need_original_check():
    return False

def ext_atomic_element_typechecking(element_ty, op):
    # Add `tl.int64` restriction for NPU
    if element_ty in [tl.int1, tl.int64, tl.float16, tl.float32, tl.float64, tl.bfloat16] and op in ['or', 'xor']:
        raise ValueError(f"atomic_{op} does not support {str(element_ty)}. "
                         "All support dtypes are int8, int16, int32.")
    if element_ty in [tl.int1, tl.int64, tl.float64, tl.bfloat16] and op == 'xchg':
        raise ValueError(f"atomic_{op} does not support {str(element_ty)}. "
                         "All support dtypes are int8, int16, int32, float16, float32.")
    if element_ty in [tl.int1, tl.int64, tl.float64]:
        raise ValueError(f"atomic_{op} does not support {str(element_ty)}. "
                         "All support dtypes are int8, int16, int32, float16, float32, bfloat16.")

def is_atomic_cas_need_element_bitwidth_check():
    return False

def ext_atomic_cas_element_typechecking(element_ty):
    if element_ty in [tl.int1, tl.int8, tl.float64, tl.bfloat16]:
        raise ValueError(f"atomic_cas does not support {str(element_ty)}. "
                        "All support dtypes are int16, int32, int64, float16, float32.")

def is_atomic_max_no_bitcast():
    return True

def is_atomic_min_no_bitcast():
    return True

def atomic_max_returning_tensor(ir, ptr, val, mask, sem, scope, builder):
    return tl.tensor(
        builder.create_atomic_rmw(ir.ATOMIC_OP.MAX, ptr.handle, val.handle, mask.handle, sem, scope), val.type)

def atomic_min_returning_tensor(ir, ptr, val, mask, sem, scope, builder):
    return tl.tensor(
        builder.create_atomic_rmw(ir.ATOMIC_OP.MIN, ptr.handle, val.handle, mask.handle, sem, scope), val.type)

def is_float_format_support_bf16():
    return True

def is_float_format_support_fp16():
    return True

def ext_dot_scaled_validate_lhs_dtype(lhs):
    assert lhs.dtype == tl.bfloat16 or lhs.dtype == tl.float16, f"lhs matrix dtype must be bf16 or fp16"

def ext_dot_scaled_validate_rhs_dtype(rhs):
    assert rhs.dtype == tl.bfloat16 or rhs.dtype == tl.float16, f"rhs matrix dtype must be bf16 or fp16"

def ext_dot_scaled_check_same_dtype(lhs, rhs):
    assert lhs.dtype == rhs.dtype, f"lhs rhs matrix must get same dtype"

def is_dot_scaled_need_original_check():
    return False

def ext_dot_scaled_check_lhs_rhs_format(lhs_format, rhs_format):
    lhs_format: str = lhs_format.value
    rhs_format: str = rhs_format.value
    allowed_formats = {"bf16", "fp16"} # unsupported fp8/4 dtype: "e2m1", "e4m3", "e5m2"
    assert lhs_format in allowed_formats, f"NYI: lhs_format {lhs_format}"
    assert rhs_format in allowed_formats, f"NYI: rhs_format {rhs_format}"

def dot_scaled_recheck_rhs_scale_is_none(rhs_scale, rhs_scale_is_none):
    rhs_scale_is_none = rhs_scale is None or (isinstance(rhs_scale, tl.constexpr) and rhs_scale.value is None)
    return rhs_scale_is_none

def dot_scaled_check_lhs_scale_is_none(lhs_scale):
    lhs_scale_is_none = lhs_scale is None or (isinstance(lhs_scale, tl.constexpr) and lhs_scale.value is None)
    return lhs_scale_is_none

def is_dot_scaled_support_rhs_scale():
    return True

def check_dot_scaled_lhs_scale_dtype(lhs_scale):
    assert isinstance(lhs_scale, tl.tensor) and lhs_scale.dtype == tl.int8, f"lhs_scale must be int8 tensor"

def check_dot_scaled_rhs_scale_dtype(rhs_scale, rhs_scale_is_none):
    if not rhs_scale_is_none:
        assert isinstance(rhs_scale, tl.tensor) and rhs_scale.dtype == tl.int8, f"rhs_scale must be int8 tensor"

def _bitcast_to_fp_type(val, float_format, builder):
    triton_ty = {"e5m2": tl.float8e5, "e4m3": tl.float8e4nv, "bf16": tl.bfloat16, "fp16": tl.float16}.get(float_format)
    if triton_ty is None:
        assert float_format == "e2m1", f"Internal Error: Unexpected float format: {float_format}"
        assert val.dtype == tl.uint8, f"e2m1 format must be packed as uint8. Got {val.dtype}"
        return val
    if val.dtype == triton_ty:
        return val
    else:
        unsigned_ty = {"e5m2": tl.uint8, "e4m3": tl.uint8, "bf16": tl.uint16, "fp16": tl.uint16}[float_format]
        assert val.dtype == unsigned_ty, f"Unexpected dtype for {float_format}. Got {val.dtype}"
        return bitcast(val, triton_ty, builder)

def dot_scaled_lhs_bitcast_to_fp_type(lhs, lhs_format, builder):
    lhs_format: str = lhs_format.value
    lhs = _bitcast_to_fp_type(lhs, lhs_format, builder)
    return lhs

def dot_scaled_rhs_bitcast_to_fp_type(rhs, rhs_format, builder):
    rhs_format: str = rhs_format.value
    rhs = _bitcast_to_fp_type(rhs, rhs_format, builder)
    return rhs

def check_dot_scaled_dimension(lhs, rhs):
    assert lhs.type.shape[-1] == rhs.type.shape[-2], (
        f"lhs last dimension (columns) {lhs.shape[-1]} "
        f"must equal rhs penultimate dimension (rows) {rhs.shape[-2]}"
    )

def check_dot_scaled_pack_size(PACKED_A, K, lhs_format, lhs, rhs):
    lhs_format: str = lhs_format.value
    PACKED_B = 2 if lhs_format == "e2m1" else 1
    assert K * PACKED_B == PACKED_A * lhs.type.shape[
        -1], f"Reduction dimension should pack the same number of elements; (lhs: {lhs.shape} vs rhs: {rhs.shape})"

def set_dot_scaled_lhs_scale_handle(lhs_scale, lhs_scale_is_none):
    return None if lhs_scale_is_none else lhs_scale.handle
