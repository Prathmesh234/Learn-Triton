"""
Triton Internals Explorer — Fused Softmax Edition
===================================================
Inspired by: https://www.kapilsharma.dev/posts/deep-dive-into-triton-internals-2/

We do NOT reimplement the blog — instead, we USE the same internal hooks
the blog describes to peek inside Triton's compilation of OUR softmax kernel.

What this explores:
  1.  ASTSource  — the object Triton creates from a @triton.jit function
  2.  triton.compile  — calling the compiler directly (not via JIT dispatch)
  3.  CompiledKernel  — ccinfo.asm dict:  ttir → ttgir → llir → ptx → cubin
  4.  make_backend  — which backend (CUDA/AMD) gets picked and its options
  5.  Compiler stages/passes  — add_inliner, add_combine, add_cse, etc.
  6.  MLIR env vars  — MLIR_ENABLE_DUMP, MLIR_ENABLE_DIAGNOSTICS
  7.  JIT vs compile path  — same kernel, two compilation entry points
  8.  Constexpr specialisation  — how BLOCK_SIZE bakes into the IR

Run with:
  python triton_internals_explorer.py            # all sections
  python triton_internals_explorer.py astsource  # section 1 only
  python triton_internals_explorer.py compile    # section 2+3 (compile + asm)
  python triton_internals_explorer.py backend    # section 4
  python triton_internals_explorer.py passes     # section 5
  python triton_internals_explorer.py mlirdump   # section 6 (env-var dump)
  python triton_internals_explorer.py jit        # section 7
  python triton_internals_explorer.py constexpr  # section 8
  python triton_internals_explorer.py ttir       # just pretty-print TTIR
  python triton_internals_explorer.py ptx        # just pretty-print PTX
"""

import os
import sys
import pathlib
import textwrap
import pprint
import subprocess

import torch
import triton
import triton.language as tl
from triton.compiler import ASTSource, make_backend
from triton.runtime.driver import driver


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

DEVICE = torch.device(f"cuda:{torch.cuda.current_device()}")
SEP = "─" * 80
SECTION_WIDTH = 80

# ── Tee: write to stdout AND a markdown file simultaneously ───────────────────
class _Tee:
    """Wraps sys.stdout so every print() also appends to a file."""
    def __init__(self, fh):
        self._fh   = fh
        self._orig = sys.stdout
    def write(self, data):
        self._orig.write(data)
        self._fh.write(data)
    def flush(self):
        self._orig.flush()
        self._fh.flush()
    def __enter__(self):
        sys.stdout = self
        return self
    def __exit__(self, *_):
        sys.stdout = self._orig
        # Do NOT close _fh here — the outer `with open()` owns the lifecycle

def header(title: str):
    print(f"\n{'═' * SECTION_WIDTH}")
    pad = (SECTION_WIDTH - len(title) - 2) // 2
    print(f"{'═' * pad} {title} {'═' * (SECTION_WIDTH - pad - len(title) - 2)}")
    print(f"{'═' * SECTION_WIDTH}")

def subheader(title: str):
    print(f"\n{SEP}")
    print(f"  {title}")
    print(SEP)

def bullet(key: str, value):
    print(f"  ▸ {key:<30} {value}")

def cmd_banner(mode: str):
    """Print the exact CLI command that produced the following output."""
    print(f"\n  ┌─ CLI command to reproduce this section:")
    print(f"  │    python3 triton_internals_explorer.py {mode}")
    print(f"  └{'─' * 60}")

OUTPUT_DIR = pathlib.Path(__file__).parent / "ir_output" / "internals"


# ─────────────────────────────────────────────────────────────────────────────
# The Kernel (identical to softmax_practice.py so no dependency needed)
# ─────────────────────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────────────────────
# Shared compile helpers
# ─────────────────────────────────────────────────────────────────────────────

N_ROWS, N_COLS = 128, 512          # small, just for inspection
BLOCK_SIZE     = triton.next_power_of_2(N_COLS)   # 512
NUM_WARPS      = 8
NUM_STAGES     = 2


def make_ast_source() -> ASTSource:
    """
    Build an ASTSource for _softmax_kernel.

    ASTSource is the object that wraps the @triton.jit function *before*
    compilation.  The blog shows this being populated by compile.py (the CLI
    tool).  We create it directly here — same effect.

    Signature encoding (Triton 3.x style):
      *fp32  → pointer to float32 tensor
      i32    → int32 scalar
      64     → constexpr equal to 64  (used for constants baked at compile time)
    """
    # Map positional index → type string (or constexpr int)
    # _softmax_kernel args:
    #   0: input_ptr       → *fp32
    #   1: output_ptr      → *fp32
    #   2: input_row_stride→ i32
    #   3: output_row_stride→ i32
    #   4: n_rows          → i32
    #   5: n_cols          → i32
    #   6: BLOCK_SIZE      → constexpr (= BLOCK_SIZE)
    #   7: num_stages      → constexpr (= NUM_STAGES)
    # Triton 3.x: signature keys = positional strings ("0", "1", ...)
    # constants keys = parameter NAME strings ("BLOCK_SIZE", "num_stages")
    signature = {
        "0": "*fp32",   # input_ptr
        "1": "*fp32",   # output_ptr
        "2": "i32",     # input_row_stride
        "3": "i32",     # output_row_stride
        "4": "i32",     # n_rows
        "5": "i32",     # n_cols
        # constexpr args go into constants= keyed by their PARAMETER NAME
    }
    constants = {
        "BLOCK_SIZE": BLOCK_SIZE,
        "num_stages": NUM_STAGES,
    }

    src = ASTSource(
        fn=_softmax_kernel,
        signature=signature,
        constants=constants,
    )
    return src


def compile_kernel(src: ASTSource | None = None) -> "triton.compiler.compiler.CompiledKernel":
    """Run triton.compile() on the softmax ASTSource and return CompiledKernel."""
    if src is None:
        src = make_ast_source()
    opts = {"num_warps": NUM_WARPS, "num_stages": NUM_STAGES}
    ccinfo = triton.compile(src, options=opts)
    return ccinfo


# ─────────────────────────────────────────────────────────────────────────────
# Section 1: ASTSource Introspection
# ─────────────────────────────────────────────────────────────────────────────

def section_astsource():
    """
    The blog peeks inside ASTSource via ipdb.  We do the same thing
    programmatically — no debugger needed.

    Key fields (mirroring blog output):
      src.fn           → JITFunction wrapper around our Python kernel
      src.signature    → {arg_index: type_string}
      src.attrs        → AttrsDescriptor (alignment / constexpr hints)
      src.hash         → unique fingerprint of this specialisation
      src.fn.arg_names → names of all parameters
      src.fn.params    → KernelParam list (name, is_constexpr, ...)
      src.fn.src       → raw Python source of the kernel
    """
    header("SECTION 1 — ASTSource Introspection")

    src = make_ast_source()

    # ── What is ASTSource? ────────────────────────────────────────────────────
    subheader("What is ASTSource?")
    print(textwrap.dedent("""
      ASTSource is Triton's bridge between a @triton.jit function and the
      compiler.  When triton.compile() is called it receives an ASTSource.
      Internally it calls src.make_ir() which walks the Python AST and
      emits MLIR Triton IR (ttir).

      Blog ref:  src = triton.compiler.ASTSource(fn=kernel,
                         constants=constants, signature=signature, attrs=attrs)
    """))

    # ── Basic identity ────────────────────────────────────────────────────────
    subheader("src object — dir() snapshot (key attrs)")
    key_attrs = [a for a in dir(src) if not a.startswith("__")]
    print(f"  {key_attrs}")

    subheader("src.fn  (JITFunction)")
    bullet("repr",        repr(src.fn))
    bullet("type",        type(src.fn))

    subheader("src.signature  — {arg_idx: type_string}")
    pprint.pprint(src.signature, indent=4)
    print()
    print(textwrap.dedent("""
      *fp32  = pointer-to-float32  (a tensor passed by base address)
      i32    = a plain 32-bit integer scalar
      constexpr args are NOT in the signature dict; they go into constants={}
      and are specialised (baked) into BLOCK_SIZE / num_stages at compile time.
    """))

    subheader("src.attrs  — AttrsDescriptor")
    print(f"  {src.attrs}")
    print(textwrap.dedent("""
      AttrsDescriptor records which args are divisible-by-16 (pointer
      alignment optimisation) and which are equal-to-1.  Triton uses this
      to generate more efficient memory accesses.
    """))

    subheader("src.hash  — unique specialisation fingerprint")
    bullet("hash", src.hash)
    print(textwrap.dedent("""
      Every distinct (signature + constants + attrs) combo gets a unique hash.
      This is used as the cache key on disk  (~/.triton/cache/...).
      Changing BLOCK_SIZE from 512 → 1024 would produce a different hash
      because it's a different constexpr specialisation.
    """))

    subheader("src.fn.arg_names  — all parameter names")
    print(f"  {src.fn.arg_names}")

    subheader("src.fn.params  — KernelParam objects")
    for i, p in enumerate(src.fn.params):
        constexpr_flag = " [constexpr]" if p.is_constexpr else ""
        print(f"  [{i}]  {p.name}{constexpr_flag}")
    print(textwrap.dedent("""
      KernelParam.is_constexpr == True means the value is specialised at
      compile time and baked into the binary.  That is why BLOCK_SIZE shows
      up in the TTIR as a literal integer constant rather than a runtime arg.
    """))

    subheader("src.fn.src  — raw Python source of the kernel")
    print(src.fn.src)


# ─────────────────────────────────────────────────────────────────────────────
# Section 2+3: triton.compile + CompiledKernel
# ─────────────────────────────────────────────────────────────────────────────

def section_compile():
    """
    The blog shows:  ccinfo = triton.compile(src, options=opts)
    ccinfo.asm.keys()  →  dict_keys(['ttir', 'ttgir', 'llir', 'ptx', 'cubin'])

    We do exactly that and inspect the result.
    """
    header("SECTION 2+3 — triton.compile() → CompiledKernel")

    subheader("Calling triton.compile(src, options=opts)")
    src    = make_ast_source()
    opts   = {"num_warps": NUM_WARPS, "num_stages": NUM_STAGES}
    print(f"  src   = {src}")
    print(f"  opts  = {opts}")
    print()
    ccinfo = triton.compile(src, options=opts)
    print(f"  ccinfo = {ccinfo}")

    subheader("ccinfo.asm.keys()  — the compilation pipeline")
    asm_keys = list(ccinfo.asm.keys())
    print(f"  {asm_keys}")
    print(textwrap.dedent("""
      Each key is a stage of the compilation pipeline:

        ttir   – Triton IR          (MLIR dialect, human-readable)
        ttgir  – Triton GPU IR      (after GPU-specific lowering)
        llir   – LLVM IR            (after MLIR→LLVM lowering)
        ptx    – PTX assembly       (NVIDIA intermediate asm)
        cubin  – CUDA binary        (final GPU binary, ELF-like)

      Every stage is either a text string (ttir/ttgir/llir/ptx) or
      raw bytes (cubin).  We can print / save / diff them.
    """))

    subheader("ccinfo.metadata  — compilation metadata")
    meta = ccinfo.metadata
    print(textwrap.dedent(f"""
      num_warps      = {meta.num_warps}
      num_stages     = {meta.num_stages}
      shared (bytes) = {meta.shared}   ← SRAM used by this kernel
      name           = {meta.name}
    """))

    # Stage sizes
    subheader("Stage sizes (bytes / lines)")
    for key in asm_keys:
        val = ccinfo.asm[key]
        if isinstance(val, bytes):
            print(f"  {key:<8}  {len(val):>8} bytes  (binary)")
        else:
            lines = val.count('\n')
            print(f"  {key:<8}  {lines:>8} lines  (text)")

    # Save all text stages to disk
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for key in asm_keys:
        val = ccinfo.asm[key]
        ext = {"ttir": ".ttir", "ttgir": ".ttgir", "llir": ".ll",
               "ptx": ".ptx", "cubin": ".cubin"}.get(key, f".{key}")
        out = OUTPUT_DIR / f"softmax{ext}"
        mode = "wb" if isinstance(val, bytes) else "w"
        with open(out, mode) as f:
            f.write(val)
        print(f"  Saved {key:>6} → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Section 4: Backend Inspection
# ─────────────────────────────────────────────────────────────────────────────

def section_backend():
    """
    The blog traces:
      backend = make_backend(target)      → <nvidia.CUDABackend>
      options  (CUDAOptions)              → num_warps, num_stages, ...
      target   (GPUTarget)               → arch=89, warp_size=32
    """
    header("SECTION 4 — Backend Inspection (make_backend)")

    # Discover the GPU target Triton uses
    target = triton.runtime.driver.active.get_current_target()
    backend = make_backend(target)

    subheader("GPU Target")
    bullet("backend attr",  target.backend)
    bullet("arch (SM)",     target.arch)
    bullet("warp_size",     target.warp_size)
    print(textwrap.dedent(f"""
      target = {target}

      SM {target.arch} means your GPU is compute capability {target.arch // 10}.{target.arch % 10}.
      For example SM89 = Ada Lovelace (RTX 4090 / L4 / L40).
                   SM80 = Ampere      (A100 / A10G).
                   SM90 = Hopper      (H100).
    """))

    subheader("Backend object")
    bullet("repr",  repr(backend))
    bullet("type",  type(backend).__name__)

    subheader("Backend capabilities detected by Triton")
    try:
        props = triton.runtime.driver.active.utils.get_device_properties(DEVICE.index)
        for k, v in props.items():
            print(f"  {k:<35} {v}")
    except Exception as e:
        print(f"  (could not read device properties: {e})")

    subheader("Parsed CUDAOptions (from triton.compile)")
    src    = make_ast_source()
    opts   = {"num_warps": NUM_WARPS, "num_stages": NUM_STAGES}
    ccinfo = triton.compile(src, options=opts)
    # The options object is stored in metadata
    print(textwrap.dedent(f"""
      CUDAOptions influence how passes are configured and codegen decisions:
        num_warps    = {ccinfo.metadata.num_warps}   (threads/32 per program)
        num_stages   = {ccinfo.metadata.num_stages}   (software pipelining stages)
        shared bytes = {ccinfo.metadata.shared}  (SRAM per block)

      num_warps controls the number of hardware warp groups assigned to one
      Triton "program" (CTA).  More warps → more parallelism within one block,
      but also more register pressure.

      num_stages controls the software pipeline depth (see tl.range).
      Stage 1 = sequential.  Stage 2 = prefetch next while computing current.
    """))

    subheader("Compiler stages added by CUDABackend.add_stages()")
    print(textwrap.dedent("""
      From third_party/nvidia/backend/compiler.py: (blog ref)

        stages["ttir"]  = make_ttir    ← MLIR pass manager, optimization passes
        stages["ttgir"] = make_ttgir   ← Triton → TritonGPU lowering
        stages["llir"]  = make_llir    ← MLIR → LLVM IR, + libdevice linking
        stages["ptx"]   = make_ptx     ← LLVM → PTX via ptxas
        stages["cubin"] = make_cubin   ← PTX → binary (nvcc / ptxas)

      Each stage is a pure function:  fn(src_module, metadata) → new_module
      triton.compile drives them in order, threading the output of each
      stage as the input of the next.
    """))


# ─────────────────────────────────────────────────────────────────────────────
# Section 5: Compiler Pass Sequence
# ─────────────────────────────────────────────────────────────────────────────

def section_passes():
    """
    The blog examines make_ttir and lists each MLIR pass.
    We map each pass name to what it does (from the MLIR / Triton docs).
    We also show how to discover which passes run using MLIR_ENABLE_DUMP.
    """
    header("SECTION 5 — Compiler Pass Sequence")

    subheader("make_ttir  — Python-side MLIR pass manager setup")
    print(textwrap.dedent("""
      From nvidia/backend/compiler.py  (what the blog traces):

      def make_ttir(mod, metadata, opt):
          pm = ir.pass_manager(mod.context)   # ← mlir::PassManager (C++ side)
          pm.enable_debug()
          passes.common.add_inliner(pm)
          passes.ttir.add_rewrite_tensor_pointer(pm)
          passes.ttir.add_combine(pm)
          passes.common.add_canonicalizer(pm)
          passes.ttir.add_reorder_broadcast(pm)
          passes.common.add_cse(pm)
          passes.common.add_licm(pm)
          passes.common.add_symbol_dce(pm)
          pm.run(mod)
          return mod

      ir.pass_manager()  →  triton._C.libtriton.ir.pass_manager
                          →  mlir::PassManager  (C++ object via pybind11)

      All passes are wired through python/src/passes.cc  which contains
      ADD_PASS_WRAPPER_0 macros that bind C++ pass factories to Python.
    """))

    subheader("Pass-by-pass breakdown")

    passes = [
        ("add_inliner",
         "mlir::createInlinerPass",
         "Inlines all @triton.jit helper functions called inside the kernel.\n"
         "    After this, the kernel is one flat function with no call sites."),

        ("add_rewrite_tensor_pointer",
         "triton::createRewriteTensorPointerPass",
         "Rewrites tl.make_block_ptr / tl.advance (tensor pointer style) into\n"
         "    legacy scalar pointer arithmetic.  Our softmax uses col_offsets\n"
         "    directly so this pass is a no-op for us."),

        ("add_combine",
         "triton::createCombineOpsPass",
         "Peephole optimisations specific to Triton IR:\n"
         "    - Fuse consecutive loads/stores into wider ops\n"
         "    - Simplify broadcast / reshape chains\n"
         "    - Constant-fold trivial arithmetic (e.g. 0 + x → x)"),

        ("add_canonicalizer",
         "mlir::createCanonicalizerPass",
         "Standard MLIR canonicalization: applies all registered fold / rewrite\n"
         "    patterns. Tidies up the IR after earlier passes."),

        ("add_reorder_broadcast",
         "triton::createReorderBroadcastPass",
         "Moves broadcast ops earlier so other passes can simplify them.\n"
         "    Matters most for kernels with explicit tl.broadcast calls."),

        ("add_cse",
         "mlir::createCSEPass",
         "Common Subexpression Elimination: detects identical sub-computations\n"
         "    and replaces duplicates with a single value.  In softmax our\n"
         "    col_offsets + row_start_ptr appear once per iter — CSE keeps them\n"
         "    in registers rather than recomputing."),

        ("add_licm",
         "mlir::createLoopInvariantCodeMotionPass",
         "Loop Invariant Code Motion: hoists computations that don't change\n"
         "    across loop iterations OUT of the loop.  In our kernel the\n"
         "    BLOCK_SIZE constant and arange bounds are loop-invariant."),

        ("add_symbol_dce",
         "mlir::createSymbolDCEPass",
         "Dead Code Elimination at the symbol level: removes functions /\n"
         "    globals that are never referenced after inlining."),
    ]

    for i, (py_name, cpp_name, desc) in enumerate(passes, 1):
        print(f"\n  [{i}] passes.*.{py_name}(pm)")
        print(f"       C++: {cpp_name}")
        for line in desc.split('\n'):
            print(f"       {line}")

    subheader("make_ttgir  — key passes after ttir")
    print(textwrap.dedent("""
      make_ttgir converts plain Triton IR → Triton GPU IR.  Key passes:

        add_convert_triton_to_tritongpu
            Maps tt.func → ttg.func, assigns thread/warp layout to tensors,
            and introduces memory hierarchy annotations (shared / global).

        add_coalesce
            Reorders tensor dimensions so memory accesses are coalesced
            (consecutive threads touch consecutive addresses → fewer cache lines).

        add_optimize_thread_locality
            Moves shared-memory producers closer to their consumers.

        add_pipeline (software pipelining)
            Splits the loop body into stages so prefetch (load) overlaps
            with compute.  Controlled by num_stages constant.
            This is what tl.range(... num_stages=N) triggers.

      After make_ttgir the IR has explicit:
        - Thread block tile sizes
        - Shared memory allocation / synchronisation barriers
        - Warp layout on all tensor operands
    """))

    subheader("How to expose the pass sequence live (MLIR_ENABLE_DUMP)")
    print(textwrap.dedent(f"""
      The blog edits ir.cc to hardcode haveDump = true.
      The same effect is achievable with env vars (no C++ rebuild needed):

        # Dump IR printed before/after every MLIR pass:
        MLIR_ENABLE_DUMP=1 python triton_internals_explorer.py mlirdump

        # Also emit MLIR diagnostics (warnings / errors from passes):
        MLIR_ENABLE_DIAGNOSTICS=1 MLIR_ENABLE_DUMP=1 python ...

      The output is verbose (hundreds of lines) so we redirect it.
      Run `python triton_internals_explorer.py mlirdump` to see a live demo.
    """))


# ─────────────────────────────────────────────────────────────────────────────
# Section 6: MLIR_ENABLE_DUMP Demo
# ─────────────────────────────────────────────────────────────────────────────

def section_mlirdump():
    """
    Launches a subprocess with MLIR_ENABLE_DUMP=1 and MLIR_ENABLE_DIAGNOSTICS=1
    so we can see the IR evolution across passes without touching C++.
    Captures stderr (where MLIR writes dumps) and saves to disk.
    """
    header("SECTION 6 — MLIR_ENABLE_DUMP Live Demo")

    print(textwrap.dedent("""
      The blog modifies ir.cc (C++) to hardcode haveDump=true/haveDiagnostics=true.
      We achieve the same result without touching C++ by setting env vars:

        MLIR_ENABLE_DUMP=1           → prints IR before/after every pass
        MLIR_ENABLE_DIAGNOSTICS=1   → prints MLIR-level diagnostics

      We run a tiny compile in a subprocess so the huge dump goes to a file
      rather than flooding your terminal.  Watch the beginning and end of the
      captured output so you can see the IR at different points in the pipeline.
    """))

    dump_script = textwrap.dedent("""
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
""")

    # Write the helper script
    tmp_script = OUTPUT_DIR / "_mlir_dump_helper.py"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tmp_script.write_text(dump_script)

    dump_out = OUTPUT_DIR / "mlir_dump.txt"
    env = os.environ.copy()
    env["MLIR_ENABLE_DUMP"] = "1"
    env["MLIR_ENABLE_DIAGNOSTICS"] = "1"

    print(f"  Running compile with MLIR_ENABLE_DUMP=1 …")
    print(f"  Output → {dump_out}")
    print(f"  (This may take 10-30 s on first run while Triton JIT-compiles)\n")

    result = subprocess.run(
        [sys.executable, str(tmp_script)],
        capture_output=True, text=True, env=env, timeout=120
    )
    combined = result.stdout + result.stderr
    dump_out.write_text(combined)

    # Show summary stats
    lines = combined.splitlines()
    stage_markers = [l for l in lines if "IR Dump" in l]
    print(f"  Total output lines      : {len(lines)}")
    print(f"  'IR Dump' pass markers  : {len(stage_markers)}")
    print()
    print("  First few markers found:")
    for m in stage_markers[:12]:
        print(f"    {m.strip()}")
    if len(stage_markers) > 12:
        print(f"    ... and {len(stage_markers) - 12} more")

    print(f"\n  Full dump saved to → {dump_out}")
    print(textwrap.dedent("""
      Open that file and search for:
        "IR Dump Before"     → IR at pass input
        "IR Dump After"      → IR at pass output
        "tt.func"            → the top-level Triton function
        "tt.get_program_id"  → maps to tl.program_id(0)
        "tt.load"            → maps to tl.load(...)
        "tt.exp"             → maps to tl.exp(...)
        "arith.divf"         → maps to softmax_output = numerator / denominator
    """))


# ─────────────────────────────────────────────────────────────────────────────
# Section 7: JIT vs triton.compile
# ─────────────────────────────────────────────────────────────────────────────

def section_jit():
    """
    Shows the two ways to compile a Triton kernel and compares their outputs.

    Path A: JIT  → @triton.jit + kernel[grid](args...)
    Path B: API  → ASTSource + triton.compile(src)

    Both end up calling the same triton.compile internally.
    Path A is the normal user-facing path.
    Path B is what the CLI tool (triton/tools/compile.py) uses.
    """
    header("SECTION 7 — JIT vs triton.compile (two compilation paths)")

    subheader("Path A: Standard JIT dispatch")
    print(textwrap.dedent("""
      @triton.jit
      def _softmax_kernel(...):
          ...

      # Triton compiles on first call (then caches):
      _softmax_kernel[grid](x, y, ..., BLOCK_SIZE=512, num_stages=2, num_warps=8)

      The @triton.jit decorator wraps _softmax_kernel in a JITFunction.
      On the first call with a specific signature specialisation, JITFunction:
        1. Builds an ASTSource from the function + call-site arguments
        2. Calls triton.compile(src)
        3. Caches the CompiledKernel to ~/.triton/cache/<hash>/
        4. Launches the kernel via the CUDA driver API
    """))

    # Run path A
    x = torch.randn(N_ROWS, N_COLS, device=DEVICE)
    y = torch.empty_like(x)
    compiled_a = _softmax_kernel[(4,)](
        x, y,
        x.stride(0), y.stride(0),
        N_ROWS, N_COLS,
        BLOCK_SIZE=BLOCK_SIZE,
        num_stages=NUM_STAGES,
        num_warps=NUM_WARPS,
    )
    bullet("Path A CompiledKernel", repr(compiled_a))
    bullet("Path A asm keys",       list(compiled_a.asm.keys()))

    subheader("Path B: Direct triton.compile() via ASTSource")
    print(textwrap.dedent("""
      src    = ASTSource(fn=_softmax_kernel, signature={...}, constants={...})
      ccinfo = triton.compile(src, options={"num_warps": 8, "num_stages": 2})

      This is exactly what the CLI tool does.  It gives us a CompiledKernel
      without ever *running* the kernel on GPU.  Useful for:
        - Offline / AOT compilation (ship cubins without source)
        - IR inspection without a warm-up run
        - Comparing PTX across different BLOCK_SIZE specialisations
    """))
    src = make_ast_source()
    compiled_b = compile_kernel(src)
    bullet("Path B CompiledKernel", repr(compiled_b))
    bullet("Path B asm keys",       list(compiled_b.asm.keys()))

    subheader("Comparison — metadata")
    for attr in ("num_warps", "num_stages", "shared", "name"):
        val_a = getattr(compiled_a.metadata, attr, "N/A")
        val_b = getattr(compiled_b.metadata, attr, "N/A")
        match = "✓" if val_a == val_b else "✗ DIFFER"
        print(f"  {attr:<15} path-A={val_a!r:<20} path-B={val_b!r:<20}  {match}")

    subheader("Comparison — PTX hash (are they identical?)")
    ptx_a = compiled_a.asm.get("ptx", "")
    ptx_b = compiled_b.asm.get("ptx", "")
    if ptx_a == ptx_b:
        print("  PTX IS IDENTICAL ✓  — both paths produce exactly the same code")
    else:
        # They differ by source location annotations but are functionally the same
        # Strip location comments and compare
        def strip_loc(ptx): return "\n".join(
            l for l in ptx.splitlines() if not l.strip().startswith("//"))
        if strip_loc(ptx_a) == strip_loc(ptx_b):
            print("  PTX IS IDENTICAL after stripping comments ✓")
        else:
            print("  PTX DIFFERS — this is unusual, please investigate")


# ─────────────────────────────────────────────────────────────────────────────
# Section 8: constexpr specialisation
# ─────────────────────────────────────────────────────────────────────────────

def section_constexpr():
    """
    BLOCK_SIZE and num_stages are tl.constexpr parameters.
    We compile three specialisations (256, 512, 1024) and show how the
    generated TTIR and PTX differ: each produces a completely different binary.
    This is Triton's specialisation-based AOT compilation model.
    """
    print(textwrap.dedent("""
      tl.constexpr parameters are NOT runtime arguments.  They are part of
      the ASTSource signature via the constants={} dict.  Each distinct value
      triggers a fully independent compilation and produces a separate binary
      with different register usage, loop trip counts, etc.

      We compile three BLOCK_SIZE specialisations and compare:
    """))

    block_sizes = [256, 512, 1024]
    results = {}

    for bs in block_sizes:
        sig    = {"0":"*fp32", "1":"*fp32", "2":"i32", "3":"i32", "4":"i32", "5":"i32"}
        consts = {"BLOCK_SIZE": bs, "num_stages": NUM_STAGES}
        src = ASTSource(fn=_softmax_kernel, signature=sig, constants=consts)
        cc  = triton.compile(src, options={"num_warps": NUM_WARPS, "num_stages": NUM_STAGES})
        results[bs] = cc

    subheader("Metadata comparison across specialisations")
    print(f"  {'Attribute':<20} {'BLOCK=256':>12} {'BLOCK=512':>12} {'BLOCK=1024':>12}")
    print(f"  {'-'*60}")
    for attr in ("num_warps", "num_stages", "shared"):
        row = f"  {attr:<20}"
        for bs in block_sizes:
            val = getattr(results[bs].metadata, attr, "?")
            row += f" {str(val):>12}"
        print(row)

    subheader("CUBIN size comparison (larger BLOCK → more code)")
    print(f"  {'BLOCK_SIZE':<15} {'cubin bytes':>12} {'ptx lines':>12}")
    print(f"  {'-'*45}")
    for bs in block_sizes:
        cubin_sz  = len(results[bs].asm.get("cubin", b""))
        ptx_lines = results[bs].asm.get("ptx", "").count('\n')
        print(f"  {bs:<15} {cubin_sz:>12,} {ptx_lines:>12,}")

    subheader("TTIR: how BLOCK_SIZE appears as a literal constant")
    ttir_512 = results[512].asm.get("ttir", "")
    # Find lines mentioning arange / BLOCK_SIZE
    relevant = [l for l in ttir_512.splitlines() if "arange" in l or "512" in l][:10]
    print(f"  Lines from TTIR (BLOCK_SIZE=512) containing 'arange' or '512':")
    for l in relevant:
        print(f"    {l.rstrip()}")

    print(textwrap.dedent("""
      Notice that the TTIR contains literal constants like `tt.make_range`
      with a hard-coded upper-bound of 512.  If we had compiled BLOCK_SIZE=1024
      that same line would read 1024.  There is no runtime check — the value
      is literally baked into the IR at compile time.

      This is fundamentally different from a runtime `if BLOCK_SIZE == 512`
      branch.  The compiler produces three entirely separate kernels.
    """))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for bs in block_sizes:
        ttir = results[bs].asm.get("ttir", "")
        (OUTPUT_DIR / f"softmax_block{bs}.ttir").write_text(ttir)
    print(f"  TTIR files for all three specialisations saved to {OUTPUT_DIR}")


# ─────────────────────────────────────────────────────────────────────────────
# Bonus: Pretty-print TTIR or PTX on demand
# ─────────────────────────────────────────────────────────────────────────────

def section_print_stage(stage: str):
    header(f"ASM STAGE — {stage.upper()}")
    ccinfo = compile_kernel()
    content = ccinfo.asm.get(stage)
    if content is None:
        print(f"  Stage '{stage}' not found.  Available: {list(ccinfo.asm.keys())}")
        return
    if isinstance(content, bytes):
        print(f"  Binary stage: {len(content)} bytes.  Use 'cubin' mode in inspect_softmax_ir.py")
        return
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ext = {"ttir": ".ttir", "ttgir": ".ttgir", "llir": ".ll", "ptx": ".ptx"}.get(stage, f".{stage}")
    out = OUTPUT_DIR / f"softmax{ext}"
    out.write_text(content)
    print(f"  Saved to {out}\n")
    print(content)


# ─────────────────────────────────────────────────────────────────────────────
# Main dispatch
# ─────────────────────────────────────────────────────────────────────────────

# Maps section name → (function, short description)
SECTION_MAP = {
    "astsource": (section_astsource, "ASTSource introspection"),
    "compile":   (section_compile,   "triton.compile() + CompiledKernel"),
    "backend":   (section_backend,   "CUDABackend + GPU target"),
    "passes":    (section_passes,    "MLIR compiler pass sequence"),
    "mlirdump":  (section_mlirdump,  "MLIR_ENABLE_DUMP live demo"),
    "jit":       (section_jit,       "JIT vs triton.compile paths"),
    "constexpr": (section_constexpr, "constexpr / BLOCK_SIZE baking"),
}
TEXT_STAGES = {"ttir", "ttgir", "llir", "ptx"}


def _run_section(mode: str, fn):
    """Print the CLI command banner then run the section."""
    cmd_banner(mode)
    fn()


def main():
    mode = sys.argv[1].lower() if len(sys.argv) > 1 else "all"

    # Decide output markdown filename
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    md_path = OUTPUT_DIR / f"explorer_{mode}.md"

    with open(md_path, "w") as md_fh:
        md_fh.write(f"# Triton Internals Explorer — `{mode}`\n")
        md_fh.write(f"Generated by: `python3 triton_internals_explorer.py {mode}`\n\n")
        md_fh.write("```\n")   # open fenced code block — everything below goes in it

        with _Tee(md_fh):
            if mode in TEXT_STAGES:
                cmd_banner(mode)
                section_print_stage(mode)

            elif mode == "all":
                for name, (fn, _) in SECTION_MAP.items():
                    _run_section(name, fn)

            else:
                entry = SECTION_MAP.get(mode)
                if entry is None:
                    print(f"Unknown section '{mode}'.  Choose from:")
                    for k in list(SECTION_MAP.keys()) + list(TEXT_STAGES):
                        print(f"  {k}")
                    sys.exit(1)
                _run_section(mode, entry[0])

            # Print inside Tee → appears in BOTH terminal and .md
            print(f"\n  ✓ Output saved → {md_path}")

        # Close the fenced code block — written directly to file, NOT to terminal
        md_fh.write("\n```\n")



if __name__ == "__main__":
    main()
