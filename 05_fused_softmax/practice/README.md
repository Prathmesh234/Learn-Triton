# Fused Softmax Practice

In this exercise, you will implement a fused softmax kernel in Triton.

## Goal
The goal is to implement a kernel that reads a row from DRAM, computes the softmax in SRAM, and writes the result back to DRAM. This "fusion" reduces memory operations, which are often the bottleneck in GPU computations.

## Steps
1.  Open `softmax_practice.py`.
2.  Navigate to the `_softmax_kernel` function.
3.  Implement the logic following the `TODO` comments.
4.  Run the script with `python softmax_practice.py` to verify your implementation against PyTorch's native softmax.

## Implementation Details to Remember
-   **Numerical Stability**: Subtract the maximum value from each element before exponentiating.
-   **Padding**: Since Triton requires block sizes to be powers of 2, you'll need to use a mask when loading and storing if the number of columns is not a power of 2.
-   **Masking Values**: When finding the maximum, use `-inf` for the padded elements so they don't affect the result.
