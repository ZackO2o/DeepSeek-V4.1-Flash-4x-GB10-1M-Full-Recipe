# NOTICE — third-party components and attribution

This repository is a *recipe*: configuration, measurements, and a small number of
scripts/patches. It ships no model weights. The material here builds on other
people's work, and the attribution is not optional — please keep it intact if you
redistribute.

## Original licence headers must be preserved

`patches/gpu_worker_cachefix.py` is a **Python source file from vLLM**, modified.
It carries the original `SPDX-License-Identifier: Apache-2.0` /
`SPDX-FileCopyrightText: Copyright contributors to the vLLM project` headers at the
top of the file. **Do not strip those headers.** If you redistribute this file, you
are redistributing an Apache-2.0 derivative work of vLLM, in addition to the MIT
terms this repository applies to its own documentation and scripts.

## Components

| Component | Used for | Licence |
|---|---|---|
| **vLLM** (`vllm-project/vllm`) | serving engine; `patches/gpu_worker_cachefix.py` is a modified copy of `vllm/v1/worker/gpu_worker.py` | Apache-2.0 |
| **DeepSeek-V4.1-Flash** (`deepseek-ai/DeepSeek-V4.1-Flash`) | model, checkpoint, draft layers | see the model repository (the upstream card states MIT; DeepSeek's model licence applies to the weights) |
| **Tech2Wild/Kai** (`tonyd2wild/DeepSeek-V4.1-Flash-vLLM-DGX-Spark`) | foundational four-node DGX Spark recipe: patch set, disk-backed Engram staging, worker-first boot order, image chain, benchmark protocol | see that repository (MIT for its own material) |
| **0xTank** (`0xTank/DeepSeek-V4.1-Flash-vLLM-4x-GB10-Recipe`) | graph startup-state fix (skip the throwaway graph-memory profiling pass; clear startup state after capture) and the compact output-projection idea | Apache-2.0 / MIT (see that repository's `NOTICE.md`) |
| **FlashInfer**, **Triton**, **PyTorch**, **CUDA/cuDNN/NCCL** | kernels, compilation, communication | Apache-2.0 / BSD / NVIDIA EULA respectively |

## What is ours

* the parameterisation and measured configuration in `README.md` / `README.zh-CN.md`,
* the `gpu-memory-utilization` → KV-pool measurements and the failure boundary data,
* the needle / garble / vision-tool validation results,
* `tools/preflight.sh`, `tools/capture-allnode-logs.sh`, `tools/ctx_decode_bench.py`,
* the **re-derivation** of the graph startup-state changes for the vLLM tree this
  recipe pins (`patches/gpu_worker_cachefix.py` — the changes themselves are 0xTank's;
  the derivation is ours and is documented in `patches/README.md`).

## No secrets, no weights

This repository contains no credentials, no private addresses, no hostnames, no
model weights and no operational secrets. Addresses in examples are placeholders
(`<IP0>`, `<roce-iface>`, …). If you fork this and add your own numbers, check them
before publishing.

## Licence scope

The MIT licence in `LICENSE` covers the original material of this repository:
the documentation (`README.md`, `README.zh-CN.md`, `patches/README.md`,
`NOTICE.md`), the scripts under `tools/`, and the re-derivation work in
`patches/gpu_worker_cachefix.py` **excluding** the vLLM source it is based on.

Third-party material keeps its own licence:

* `patches/gpu_worker_cachefix.py` — modified vLLM source, **Apache-2.0**
  (original `SPDX` headers preserved at the top of the file; do not remove them).
* Model weights — DeepSeek's model licence (see the upstream model repository).
* Everything else listed in the table above — its own upstream licence.
