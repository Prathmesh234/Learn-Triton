"""
Practice: Fused Softmax Kernel
This is a scaffolding for practicing the fused softmax implementation.
The goal is to fill in the missing logic in the `_softmax_kernel` function.
"""

import torch
import triton
import triton.language as tl

DEVICE = torch.device(f'cuda:{torch.cuda.current_device()}')

# Helping to fetch the specifications of our GPU
properties = triton.runtime.driver.active.utils.get_device_properties(DEVICE.index)
NUM_SM = properties["multiprocessor_count"]
NUM_REGS = properties["max_num_regs"]
TOTAL_SRAM_PER_SM = properties["max_shared_mem"]
WARP_SIZE = properties["warpSize"]

######### Step 1: Naive Implementation #########
def naive_softmax(x):
    x_max = x.max(dim=1)[0]
    z = x - x_max[:, None]
    numerator = torch.exp(z)
    denominator = numerator.sum(dim=1)
    out = numerator / denominator[:, None]
    return out

######### Step 2: The Kernel (TODO) #########
"""
EXAMPLE WALKTHROUGH (3x4 matrix):
[1,2,3,4,5,6,7,8,9,10,11,12]
→ stored flat in DRAM at some address (e.g. 1000):

  Row 0: [1,  2,  3,  4 ]   row_start_ptr = 1000 + 0*4 = 1000
  Row 1: [5,  6,  7,  8 ]   row_start_ptr = 1000 + 1*4 = 1004
  Row 2: [9,  10, 11, 12]   row_start_ptr = 1000 + 2*4 = 1008

With 2 programs (PIDs) and BLOCK_SIZE=4:
  PID 0 handles row_idx = 0, 2  (step = num_programs = 2)
  PID 1 handles row_idx = 1

For PID 0, row_idx=0:
  row_start_ptr = 1000 + 0*4 = 1000
  col_offsets   = [0, 1, 2, 3]
  input_ptrs    = [1000, 1001, 1002, 1003]
  mask          = [T, T, T, T]  (all valid since n_cols=BLOCK_SIZE=4)
  row (SRAM)    = [1, 2, 3, 4]
  row - max(4)  = [-3, -2, -1, 0]
  exp(...)      = [0.050, 0.135, 0.368, 1.0]
  sum           = 1.553
  softmax_out   = [0.032, 0.087, 0.237, 0.644]
  → stored back to output at address 2000 + 0*4 = 2000

For PID 0, row_idx=2:
  row_start_ptr = 1000 + 2*4 = 1008  ← jumps over Row 1
  row (SRAM)    = [9, 10, 11, 12]
  → same softmax steps, stored to output at 2000 + 2*4 = 2008
"""
@triton.jit
def _softmax_kernel(
    input_ptr, output_ptr,
    input_row_stride, output_row_stride,
    n_rows, n_cols,
    BLOCK_SIZE: tl.constexpr,
    num_stages: tl.constexpr,
):
##understand difference between list and a single value - that is basically it
## a list of addresses and a single address 


   ##in a simple way softmax is basically softmax = e^(i - row_max)/sum(e^(i - row_max))
   ##first we have to get the program id 
   ##also because row_start is 0 in the beginng 
   row_start= tl.program_id(0)
   row_step = tl.num_programs(0) #this is because we will have 2 parallel  programs (in case program is 2)
   ## num_programs is just basically programs that can execute on a gpu 
   ##num_program is decided by taking the min between the max register count for the op and max shared mem possible 
   for row_idx in tl.range(row_start, n_rows, row_step, num_stages=num_stages):
    ##now we have to get the input ptr for the starting row 
    ## remember we have the input_ptr now for each row it will be input_ptr + row_idx*input_row_stride
    ## the input_ptr by default is an address, row_idx will be 0 making ( row_idx*input_row_stride == 0)
    ## in turn helping understand 
    input_row_ptr = input_ptr + row_idx*input_row_stride
    #now that we have address if the input_row_ptr we have to get the cols and the offsets 
    col_offsets = tl.arange(0, BLOCK_SIZE)
    row_list = col_offsets + input_row_ptr
    #now we have to create the mask 
    ## BLOCK_SIZE is a power-of-2 padded size (e.g. 1024) but n_cols is the REAL width (e.g. 781)
    ## col_offsets < BLOCK_SIZE is ALWAYS True → loads garbage memory past column 781!
    ## col_offsets < n_cols correctly marks positions 781-1023 as False → gets -inf instead
    mask = col_offsets < n_cols
    #now we load the entire row  - tl.load and tl.store expect the *entire list of addresses
    row = tl.load(row_list, mask=mask, other=float('-inf'))
    #now that we have loaded it 
    #we first compute the max (axis=0 = reduce across the 1D row)
    row_max = tl.max(row, axis=0)
    numerator = tl.exp(row - row_max)
    # reuse numerator (already computed above) instead of calling exp again
    denominator = tl.sum(numerator, axis=0)
    output  = numerator / denominator
    ##now we store it in the output_ptr 
    ## row_idx is a good proxy for shifts which are happening like it is a unified multiplier
    output_ptrs = output_ptr + row_idx*output_row_stride
    ##again tl.store expects the addresses, and values
    ##however our output_ptrs only has that single address as output_ptr (address - 1000), row_idx*input_row_stide is a single value 

    tl.store(output_ptrs + col_offsets, output, mask=mask)


######### Step 3: The Wrapper #########
def softmax(x):
    assert x.ndim == 2
    n_rows, n_cols = x.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols)

    # Heuristic for num_warps
    num_warps = 4
    if BLOCK_SIZE >= 2048:
        num_warps = 8
    if BLOCK_SIZE >= 4096:
        num_warps = 16

    num_stages = 4 if TOTAL_SRAM_PER_SM > 200_000 else 2
    y = torch.empty_like(x)

    # Pre-compile to get occupancy info
    kernel = _softmax_kernel.warmup(x, y,
                                    x.stride(0), y.stride(0),
                                    n_rows, n_cols,
                                    BLOCK_SIZE=BLOCK_SIZE,
                                    num_stages=num_stages,
                                    num_warps=num_warps,
                                    grid=(1,))
    
    kernel._init_handles()
    n_regs = kernel.n_regs
    sram_needed_per_program = kernel.metadata.shared
    ##this gets us the programs based on max number of registers 
    reg_occupancy = NUM_REGS // (n_regs * WARP_SIZE * num_warps)
    ##this gets us the sram based on the max amount of sram 
    sram_occupancy = TOTAL_SRAM_PER_SM // sram_needed_per_program
    programs_per_sm = min(reg_occupancy, sram_occupancy)
    num_programs = min(NUM_SM * programs_per_sm, n_rows)

    grid = (num_programs, 1, 1)

    # Note: BLOCK_SIZE and num_stages are tl.constexpr — they are already
    # baked into the compiled kernel by .warmup(). In Triton 3.x, do NOT
    # pass them again here or you'll get a "too many arguments" error.
    kernel[grid](
        x, y,
        x.stride(0), y.stride(0),
        n_rows, n_cols,
    )
    return y

######### Step 4: Unit Test #########
def test_softmax_kernel(size=(1823, 781), device=DEVICE):
    torch.manual_seed(0)
    x = torch.randn(size[0], size[1], device=DEVICE)
    
    # Run practice implementation
    try:
        z_tri = softmax(x)
        # Run reference implementation
        z_ref = torch.softmax(x, axis=1)
        # Compare
        torch.testing.assert_close(z_tri, z_ref, atol=1e-3, rtol=1e-3)
        print("TEST PASSED")
    except Exception as e:
        print(f"TEST FAILED: {e}")

######### Step 5: Benchmark #########
@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['N'],
        x_vals=[128 * i for i in range(2, 50)],
        line_arg='provider',
        line_vals=['triton', 'torch'],
        line_names=["Triton (Practice)", "Torch"],
        styles=[('blue', '-'), ('green', '-')],
        ylabel="GB/s",
        plot_name="softmax-performance-practice",
        args={'M': 4096}
    ))
def benchmark(M, N, provider):
    x = torch.randn(M, N, device=DEVICE, dtype=torch.float32)
    
    if provider == 'torch':
        ms = triton.testing.do_bench(lambda: torch.softmax(x, axis=-1))
    if provider == 'triton':
        ms = triton.testing.do_bench(lambda: softmax(x))
    
    gbps = lambda ms: 2 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
    return gbps(ms)

if __name__ == "__main__":
    test_softmax_kernel()
    
    # Uncomment to run benchmark
    # benchmark.run(save_path='.', print_data=False)
