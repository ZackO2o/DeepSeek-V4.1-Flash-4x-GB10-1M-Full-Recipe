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
| **FujitsuPolycom/sparkring** (`FujitsuPolycom/sparkring`) | switchless-ring NCCL patch set and the prebuilt `libnccl.so.2.30.7` artifact referenced by the transport measurements in the results/ notes; dual-HCA channel configuration; Engram `BALANCED`/packed-shard approach (reported there as not transferring to this recipe) | Apache-2.0 |
| **yunwei37/dgx-spark-4-ring-no-switch** (`yunwei37/dgx-spark-4-ring-no-switch`) | out-of-tree per-node NVMe KV prefix tier (`dsv41_kv_nvme.py`): `tools/kv_persist_policy.py` extends that module and is useless without it. The tier itself is theirs; our file adds only the persistence layer | MIT |
| **bilikaz/qwen38-flash-next-cluster-recipe** (`bilikaz/qwen38-flash-next-cluster-recipe`) | two host-level findings we tested here (boot memory gate, RDMA-versus-TCP proof). Both reimplemented for this recipe in `tools/pool_boot_gate.sh` and `tools/rdma_proof.sh`; two other claims of theirs are reported as not transferring | MIT |
| **FlashInfer**, **Triton**, **PyTorch**, **CUDA/cuDNN/NCCL** | kernels, compilation, communication | Apache-2.0 / BSD / NVIDIA EULA respectively |

## What is ours

* the parameterisation and measured configuration in `README.md` / `README.zh-CN.md`,
* the `gpu-memory-utilization` → KV-pool measurements and the failure boundary data,
* the needle / garble / vision-tool validation results,
* the acceptance-gate protocol and the "baseline measured twice, candidate must beat
  both runs" rule, the transport A/B numbers, and the KV-pool vs gmu ranges in
  `results/` (measurements and method — the transport software itself is
  FujitsuPolycom's, see above),
* `tools/preflight.sh`, `tools/capture-allnode-logs.sh`, `tools/ctx_decode_bench.py`,
* `tools/stream_bench.py` (client-side probe, incl. the rule that throughput is computed
  from the server's `completion_tokens`, not from counted stream chunks),
* `tools/pool_boot_gate.sh` (memory gate + page-cache eviction before a boot),
* `tools/rdma_proof.sh` (RDMA-versus-TCP proof; the *method* is from the recipe credited
  above, the implementation is ours),
* `tools/kv_persist_policy.py` — the restart-survivable index for the NVMe prefix tier:
  the policy/spec subclasses, the allocator-state restore, the fingerprinting, and the
  periodic snapshot. It **subclasses and imports** the tier module credited above; that
  module is not vendored here and keeps its own MIT terms,
* the measurements in `results/2026-09-14-kv-prefix-tier-and-pool-pinning.md` /
  `.zh-CN.md`,
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
* `tools/kv_persist_policy.py` — our own MIT work, but it imports and subclasses
  `dsv41_kv_nvme` from `yunwei37/dgx-spark-4-ring-no-switch` (MIT), which is **not**
  vendored in this repository. Fetch that module under its own terms if you want to run
  the tier.
* Everything else listed in the table above — its own upstream licence.
