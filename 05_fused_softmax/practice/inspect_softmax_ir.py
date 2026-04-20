"""
Inspect Softmax Kernel IR Stages
---------------------------------
Based on: https://www.kapilsharma.dev/posts/deep-dive-into-triton-internals/

Compilation pipeline:
  Python (Triton DSL)
    → Triton IR      (ttir)
    → Triton GPU IR  (ttgir)
    → LLVM IR        (llir)
    → PTX            (ptx)
    → CUBIN          (cubin)
    → SASS           (via nvdisasm / triton.tools.disasm)

Usage:
  # View everything in one shot:
  python inspect_softmax_ir.py

  # View a single stage:
  python inspect_softmax_ir.py ttir
  python inspect_softmax_ir.py ttgir
  python inspect_softmax_ir.py llir
  python inspect_softmax_ir.py ptx
  python inspect_softmax_ir.py cubin     # writes /tmp/softmax_cubin.o + runs readelf
  python inspect_softmax_ir.py sass
  python inspect_softmax_ir.py keys      # just show available asm keys
"""

import sys
import subprocess
import pathlib
import torch
import triton
import triton.language as tl

# Root dir where all IR output folders will be written
# Creates: practice/ir_output/ttir/, practice/ir_output/ptx/, etc.
OUTPUT_ROOT = pathlib.Path(__file__).parent / "ir_output"

# ── kernel (copy from softmax_practice.py) ────────────────────────────────────

DEVICE = torch.device(f'cuda:{torch.cuda.current_device()}')

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
        row_start_ptr = input_ptr + row_idx * input_row_stride
        col_offsets   = tl.arange(0, BLOCK_SIZE)
        input_ptrs    = row_start_ptr + col_offsets
        mask          = col_offsets < n_cols
        row           = tl.load(input_ptrs, mask=mask, other=float('-inf'))
        row_minus_max = row - tl.max(row, axis=0)
        numerator     = tl.exp(row_minus_max)
        denominator   = tl.sum(numerator, axis=0)
        softmax_output = numerator / denominator
        output_row_start_ptr = output_ptr + row_idx * output_row_stride
        tl.store(output_row_start_ptr + col_offsets, softmax_output, mask=mask)


# ── compile / run kernel to get the compiled object ───────────────────────────

def get_compiled_kernel():
    n_rows, n_cols = 128, 512          # small size just for inspection
    BLOCK_SIZE = triton.next_power_of_2(n_cols)   # 512
    num_warps  = 8
    num_stages = 2

    x = torch.randn(n_rows, n_cols, device=DEVICE)
    y = torch.empty_like(x)

    # Launch the kernel so Triton JIT-compiles it and caches the compiled obj
    grid = (4,)
    compiled = _softmax_kernel[grid](
        x, y,
        x.stride(0), y.stride(0),
        n_rows, n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
        num_stages=num_stages,
        num_warps=num_warps,
    )
    return compiled


# ── pretty printing helpers ────────────────────────────────────────────────────

SEPARATOR = "=" * 80

def section(title):
    print(f"\n{SEPARATOR}")
    print(f"  {title}")
    print(SEPARATOR)

# File extensions for each text stage
STAGE_EXT = {"ttir": ".ttir", "ttgir": ".ttgir", "llir": ".ll", "ptx": ".ptx"}

def save_text(key, content):
    """Save a text IR stage to ir_output/<key>/softmax<ext>."""
    ext  = STAGE_EXT.get(key, f".{key}")
    out_dir = OUTPUT_ROOT / key
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"softmax{ext}"
    out_path.write_text(content)
    print(f"  Saved → {out_path}")

def print_stage(key, compiled):
    """Print a text IR stage from compiled.asm and save it to disk."""
    section(f"[{key.upper()}]  –  compiled_kernel.asm['{key}']")
    # compiled.asm.get(key) fetches the IR text for that stage
    content = compiled.asm.get(key, None)
    if content is None:
        print(f"  !! Key '{key}' not found in asm dict.")
        print(f"  Available keys: {list(compiled.asm.keys())}")
        return
    if isinstance(content, bytes):
        print(f"  (binary, {len(content)} bytes — use 'cubin' or 'sass' mode)")
    else:
        save_text(key, content)
        print(content)

def print_cubin(compiled):
    """Dump cubin binary + readelf output, save both to ir_output/cubin/."""
    section("[CUBIN]  –  compiled_kernel.asm['cubin']")
    cubin_bytes = compiled.asm.get("cubin")
    if cubin_bytes is None:
        print("  !! 'cubin' not found in asm dict.")
        return

    out_dir = OUTPUT_ROOT / "cubin"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save the raw binary
    bin_path = out_dir / "softmax.cubin"
    bin_path.write_bytes(cubin_bytes)
    print(f"  Saved binary  → {bin_path}")

    # Run readelf and save its output
    result = subprocess.run(["readelf", "-a", str(bin_path)], capture_output=True, text=True)
    elf_path = out_dir / "softmax_readelf.txt"
    elf_path.write_text(result.stdout)
    print(f"  Saved readelf → {elf_path}")
    print(f"\n  Running:  readelf -a {bin_path}\n")
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)

def print_sass(compiled):
    """Decode CUBIN to SASS (native GPU assembly) and save to ir_output/sass/."""
    section("[SASS]  –  NVIDIA native GPU assembly (from CUBIN)")
    cubin_bytes = compiled.asm.get("cubin")
    if cubin_bytes is None:
        print("  !! 'cubin' not found in asm dict.")
        return

    sass = None

    # Method 1: triton.tools.disasm (preferred — no external tool needed)
    try:
        from triton.tools.disasm import get_sass
        sass = get_sass(cubin_bytes)
    except ImportError:
        pass

    # Method 2: nvdisasm (requires CUDA toolkit)
    if sass is None:
        tmp = "/tmp/softmax_cubin.o"
        with open(tmp, "wb") as f:
            f.write(cubin_bytes)
        result = subprocess.run(["nvdisasm", "-sf", tmp], capture_output=True, text=True)
        if result.returncode == 0:
            sass = result.stdout
        else:
            print("  nvdisasm failed:", result.stderr)
            print("  Install CUDA toolkit or use triton.tools.disasm.get_sass()")
            return

    # Save to disk
    out_dir = OUTPUT_ROOT / "sass"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "softmax.sass"
    out_path.write_text(sass)
    print(f"  Saved → {out_path}")
    print(sass)


# ── main ──────────────────────────────────────────────────────────────────────

ALL_STAGES = ["keys", "ttir", "ttgir", "llir", "ptx", "cubin", "sass"]

def main():
    mode = sys.argv[1].lower() if len(sys.argv) > 1 else "all"

    print("Compiling softmax kernel …")
    compiled = get_compiled_kernel()
    print(f"Done. Available ASM keys: {list(compiled.asm.keys())}\n")

    if mode == "keys":
        return

    if mode == "cubin":
        print_cubin(compiled)
    elif mode == "sass":
        print_sass(compiled)
    elif mode == "all":
        for stage in ["ttir", "ttgir", "llir", "ptx"]:
            print_stage(stage, compiled)
        print_cubin(compiled)
        print_sass(compiled)
    else:
        # Any single text-based stage (ttir, ttgir, llir, ptx).
        # cubin and sass need special handling (binary decode / external tools),
        # so they are handled above as explicit cases.
        print_stage(mode, compiled)


if __name__ == "__main__":
    main()
