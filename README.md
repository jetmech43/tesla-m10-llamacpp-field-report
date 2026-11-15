# Tesla M10 llama.cpp patch — production deployment field report

An independent deployment and validation of **duncandoestech's Maxwell token-generation
patch** for llama.cpp on **4× NVIDIA Tesla M10** cards, built on a different toolchain and
rolled into a **live multi-lane `llama-server` fleet** rather than a benchmark harness.

> **Credit.** All of the kernel work is duncandoestech's:
> **[duncandoestech/nvidia-tesla-m10-llm](https://github.com/duncandoestech/nvidia-tesla-m10-llm)**
> (`patches/m10-mmvq-tunables.patch`, MIT). This repo adds nothing to the patch — it just
> reports that it builds and works on a second toolchain and survives a real production
> deployment, with the numbers and the operational mechanics written down. If you're here
> for the patch, go there.

## TL;DR

- The patch **applies zero-fuzz at its stated base (`f477f795c`)**, builds clean **natively**
  (no container) with **CUDA 12.6 / gcc 13.3 / cmake 3.28.3** on a dual-Xeon server, and
  produces a working `llama-server`.
- **llama-bench, same commit stock vs patched, idle GM107 die:** Qwen3.5-2B Q4_K_M
  **18.99 → 32.67 t/s (+72%)**, a Qwen3.5-4B-class Q4_K_M finetune **10.32 → 17.63 t/s (+71%)**.
  Prompt processing flat (+1–3%). Consistent with the author's own tables.
- **Deployed to all five M10 worker lanes** of a running agent backend and measured on the
  **real request path** (`llama-server`, not `llama-bench`): **+65–72% token generation**,
  fully reversible, zero firmware/driver risk.
- **Maxwell-only, as documented.** `-DCMAKE_CUDA_ARCHITECTURES=50` reaches the M10 (sm_50)
  workers only; the Pascal **Tesla P40 (sm_61)** lane on the same box is intentionally left on
  stock mainline.

## Why this is a useful second data point

The author's test system is an **X79 desktop (i7-3930K, Sandy Bridge)** in a **Docker
container with CUDA 12.9 / gcc 11** — a box whose CPU lacks AVX2 (that's what stopped them
benchmarking ik_llama.cpp for comparison). This deployment is the opposite end:

| | Author's test system | This deployment |
|---|---|---|
| Host | X79 desktop, i7-3930K (AVX, no AVX2) | Dell PowerEdge R730, dual Xeon, 40 threads (AVX2) |
| Build | Docker, CUDA 12.9, gcc 11 | **Native**, CUDA 12.6, gcc 13.3, cmake 3.28.3 |
| Driver | 580.178.04 | 580.178.04 (same — last Maxwell branch) |
| GPUs | 1× Tesla M10 (4 GM107) + 2× P100 | 4× Tesla M10 (16 GM107) + 1× Tesla P40 |
| Workload | `llama-bench`, isolated | **`llama-server` systemd fleet, live request path** |

So this confirms the patch is not tied to the author's exact toolchain or host, and that it
holds up under a real serving workload, not just a bench.

## Build

llama.cpp checked out at **`f477f795c`** (the patch's stated base; 2026-11-08). The patch
fails a strict `git apply` on a ~3-week-newer mainline (`b11716` / `797de63b2`) and needs
`patch --fuzz=3` there, but applies **zero-fuzz** at `f477f795c`, so we built from that base.

```bash
cmake -B build-m10 \
  -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=50 -DCMAKE_BUILD_TYPE=Release \
  -DLLAMA_CURL=OFF -DGGML_NATIVE=ON \
  -DCMAKE_CUDA_FLAGS="-DM10_NW1=1 -DM10_RPB1=4 -DM10_NWB=1 -DM10_RPBB=4 -DM10_VMAD -DM10_MOE_NOSPILL -DM10_GRAPHS"
cmake --build build-m10 -j 24
```

That is the author's `build-best` flag set plus `-DM10_GRAPHS`. (`M10_GRAPHS` is a no-op for
the hybrid Qwen3.5 decode path, as the author notes, but it's harmless here and benefits any
dense model.)

> **A stock build from the identical commit and flags, minus the `-DM10_*` defines, is the
> control** for every "stock vs patched" number below — so the delta is the patch alone, not a
> version or hardware change.

## Benchmarks — `llama-bench`, idle GM107 die

Stock and patched **both built from `f477f795c`, `sm_50`, identical flags**. Run on an idle
die (`CUDA_DEVICE_ORDER=PCI_BUS_ID`), stable to ±0.04.

| Model (Q4_K_M) | test | stock | patched | Δ |
|---|---|---|---|---|
| Qwen3.5-4B-class finetune | tg128 | 10.32 | **17.63** | **+71%** |
| Qwen3.5-4B-class finetune | pp512 | 118.6 | 120.2 | +1.3% |
| Qwen3.5-2B | tg128 | 18.99 | **32.67** | **+72%** |
| Qwen3.5-2B | pp512 | 314.9 | 324.5 | +3.0% |

Decode up ~1.7×, prefill flat — exactly the shape the author reports (the patch targets the
quantized mat-vec **decode** kernel, not the batched GEMM used for prompt processing).

## Production deployment — `llama-server`, real request path

The patched binary was deployed to **all five Tesla M10 worker lanes** of a running local-AI
agent backend (one `llama-server` process per GM107 die, served as OpenAI-compatible
endpoints), each measured streaming real completions before and after:

| Lane (all Q4_K_M) | stock t/s | patched t/s | Δ |
|---|---|---|---|
| 4B worker A | 10.3 | 16.6–17.3 | +65–69% |
| 4B worker B | 9.54 | 15.85 | +66% |
| 4B worker C | 9.55 | 15.87 | +66% |
| 2B router | 18.89 | 32.39 | +71% |
| 4B vision (+ `--mmproj`) | 10.65 | 18.25 | +71% |

The Pascal **Tesla P40** lane serving a 35B-A3B MoE was left untouched — the patch is
Maxwell-only and does not apply to sm_61.

## Deploying to a systemd `llama-server` fleet (reversibly)

The patch is a compile-time change to one binary, so deployment is just *swap the binary,
keep everything else*. What made it clean and reversible across a fleet:

1. **Binary + its shared libs go to a path the service account can read** (here
   `/opt/llama-m10/bin`), with `LD_LIBRARY_PATH` pointing at that lib dir. The build tree
   under a user home is not readable by the service user.
2. **A systemd drop-in overrides only `ExecStart`** — cleared, then re-set to the patched
   binary with the lane's *existing* model and args preserved verbatim (including
   `--mmproj`). See [`deploy/zz-m10-patch.conf.example`](deploy/zz-m10-patch.conf.example).
3. **Name the drop-in so it sorts last.** systemd applies drop-ins in lexical order; a
   `zz-` prefix wins over any pre-existing per-lane drop-in. Deriving the new `ExecStart`
   from the lane's *live* `ExecStart` (swapping only argv[0]) is what preserves per-lane args.
4. **Rollback = delete the drop-in, `daemon-reload`, restart** → back to stock mainline.

### One gotcha worth writing down

An in-place `cp -a` refresh of `/opt/llama-m10/bin` throws **`cp: … 'llama-server': Text file
busy`** when a lane is currently executing that binary. It's harmless when the bytes are
unchanged (the file is already correct), but to *actually update* the binary later you must
**stop the lanes first, or copy to a fresh path and repoint** — you can't overwrite a running
executable in place.

### Verifying the swap stuck

A drift checker that only asserts each lane's **model** (`-m`) won't notice a binary that
silently reverted to stock. Assert **argv[0]** per lane too: M10 lanes → the patched binary
path, the Pascal lane → stock mainline. (`systemctl show -p ExecStart --value <unit>` exposes
the full argv.)

## Correctness

Correctness verification of the kernels themselves is the author's
([`test-backend-ops`, perplexity, byte-identical output in RESULTS.md](https://github.com/duncandoestech/nvidia-tesla-m10-llm/blob/main/RESULTS.md));
this deployment did not re-run that suite. What was checked here: the patched `llama-server`
builds, serves, and returns coherent completions on the real request path, and the measured
decode speedup matches the author's published magnitude.

## License

This field report (text, tables, the example drop-in) is MIT — see [LICENSE](LICENSE). The
llama.cpp patch it describes is **not** included here and belongs to its author under that
project's MIT license.
