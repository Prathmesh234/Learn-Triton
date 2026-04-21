
import torch, triton, triton.language as tl
from triton.compiler import ASTSource

DEVICE = torch.device("cuda:0")

@triton.jit
def _softmax_kernel(
    input_ptr, output_ptr,
    input_row_stride, output_row_stride,
    n_rows, n_cols,
    BLOCK_SIZE: tl.constexpr,
    num_stages: tl.constexpr,
):
    row_start = tl.program_id(0)
    row_step  = tl.num_programs(0)
    for row_idx in tl.range(row_start, n_rows, row_step, num_stages=num_stages):
        row_start_ptr  = input_ptr + row_idx * input_row_stride
        col_offsets    = tl.arange(0, BLOCK_SIZE)
        input_ptrs     = row_start_ptr + col_offsets
        mask           = col_offsets < n_cols
        row            = tl.load(input_ptrs, mask=mask, other=float('-inf'))
        row_minus_max  = row - tl.max(row, axis=0)
        numerator      = tl.exp(row_minus_max)
        denominator    = tl.sum(numerator, axis=0)
        softmax_output = numerator / denominator
        out_ptr        = output_ptr + row_idx * output_row_stride
        tl.store(out_ptr + col_offsets, softmax_output, mask=mask)

src = ASTSource(
    fn=_softmax_kernel,
    signature={"0":"*fp32", "1":"*fp32", "2":"i32", "3":"i32", "4":"i32", "5":"i32"},
    constants={"BLOCK_SIZE": 512, "num_stages": 2},
)
ccinfo = triton.compile(src, options={"num_warps": 8, "num_stages": 2})
print("COMPILE DONE")
