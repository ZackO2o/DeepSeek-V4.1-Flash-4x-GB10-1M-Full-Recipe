# `gpu_worker_cachefix.py` — the graph startup-state patch

**Target path inside the container:** `vllm/v1/worker/gpu_worker.py`
(where `vllm` is the site-packages install root of the pinned tree).

Mount it read-only over that file, e.g. with a mount list consumed by your launcher:

```
gpu_worker_cachefix.py v1/worker/gpu_worker.py
```

Both environment gates default to *off*; the file behaves exactly like the stock
module unless you set them.

---

## What it changes (three things)

### 1. `DSV41_SKIP_GRAPH_MEMORY_PROFILE=1` — skip the throwaway profiling pass

```diff
         if (
             current_platform.is_cuda_alike()
             and self.vllm_config.compilation_config.cudagraph_mode != CUDAGraphMode.NONE
+            and os.environ.get("DSV41_SKIP_GRAPH_MEMORY_PROFILE", "0") != "1"
         ):
             cudagraph_memory_estimate = self.model_runner.profile_cudagraph_memory()
```

The profiler runs a full dummy pass to *estimate* graph memory before the real
capture. That transient allocation scales badly with `--max-model-len`: at a 600K
window on a tight node it is the difference between "engine initialises" and
"engine core dies with an unhelpful message". When you have already sized the
KV pool with `--gpu-memory-utilization`, the estimate is not worth the memory.

### 2. Kernel warmup moves *before* capture

```diff
         kernel_warmup(self)
 
+        if self.use_v2_model_runner:
+            # A workspace resize after capture frees what the graphs point at.
+            warmup_kernels(self.model_runner, self.execute_model, self.sample_tokens)
+
         cuda_graph_memory_bytes = 0
         if not self.model_config.enforce_eager:
             cuda_graph_memory_bytes = self.model_runner.capture_model()
```

### 3. `DSV41_CLEAR_STATE_AFTER_CAPTURE=1` — clear dummy-warmup state in place

```diff
+        if os.environ.get("DSV41_CLEAR_STATE_AFTER_CAPTURE", "0") == "1":
+            if not self.use_v2_model_runner:
+                raise RuntimeError("DSV41 cache cleanup requires the V2 runner")
+            runner = self.model_runner
+            with torch.inference_mode():
+                for cache in runner.kv_caches:
+                    cache.zero_()
+                for module in runner.get_model().modules():
+                    if type(module).__name__ == "NgramHashState":
+                        cache = getattr(module, "_cache", None)
+                        if cache is not None:
+                            cache.zero_()
+                    rows = getattr(module, "staged_rows", None)
+                    if isinstance(rows, torch.Tensor):
+                        rows.zero_()
+                window = runner.model_state.lookback_token_ids
+                if window is not None:
+                    window.fill_(-1)
+                torch.cuda.synchronize()
+            logger.info("DSV41 startup cache cleanup complete: ...")
```

Two separate problems, one fix: dummy warmup/capture can leave **NaNs in KV slots**
that masked attention later reads, and a resize after capture **frees memory the
captured graphs still point at**. Zeroing in place (instead of reallocating) keeps
the graph buffer addresses valid. Requires the V2 model runner; the patch raises
rather than silently doing nothing.

On a healthy boot you see it in the log:

```
(Worker_TP0 pid=...) INFO [gpu_worker.py:920] DSV41 startup cache cleanup complete: 54 KV tensors; graph buffers retained
```

---

## Why this file is *re-derived* rather than copied

The published version of these changes (0xTank's `dsv41-gpu-worker-cache-fix.py`,
Apache-2.0/MIT — see `../NOTICE.md`) is a whole-file replacement authored against a
**newer** vLLM tree than the one this recipe pins. Mounting it over an older tree
fails loudly (`ImportError: cannot import name 'get_ec_transfer'` — the newer tree
has distributed EC-transfer APIs the older one lacks).

So we transplanted **only the three changes above** into the `gpu_worker.py` of the
tree we build against. Verification we used: `diff` between this file and the
upstream one must show **only** the extra/tree-specific hunks (in our case 7 lines of
EC-transfer import/call), i.e. every functional change made it across and nothing
else did.

**Rule of thumb:** whole-file patches belong to a tree. If you pin a different vLLM
commit, re-derive — do not mount a file that was written for another tree. A patch
file that is *older* than your tree is the dangerous direction: the engine can boot,
pass profiling and graph capture, and then emit repeated garbage tokens from the
first position while the speculative drafter accepts nothing.

---

## Preflight requirement

Patch files are bind-mounted **per node**. Verify the file exists on **every rank**
before booting. If it exists only on the head, the workers' launch step exits early,
the head blocks in the process-group store, and after ~601 s you get:

```
torch.distributed.DistStoreError: Timed out after 601 seconds waiting for clients. 1/4 clients joined.
```

That message points at the network; the real cause is a missing file. `../tools/preflight.sh`
checks exactly this.

---

## Licence of this file

`patches/gpu_worker_cachefix.py` is a modified copy of a file from **vLLM**
(Apache-2.0); the original `SPDX` headers are preserved at the top of the file and
must stay there if you redistribute it. The three functional changes originate from
**0xTank**'s published patch (see `../NOTICE.md`); the re-derivation for this tree,
and everything else in this repository, is MIT (see `../LICENSE`).
