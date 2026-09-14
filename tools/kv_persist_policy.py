# SPDX-License-Identifier: MIT
"""
Restart-survivable NVMe KV tier for DeepSeek-V4.1-Flash on 4x DGX Spark (TP4).

Built on top of the (MIT) per-node NVMe tier in `dsv41_kv_nvme.py`
(https://github.com/yunwei37/dgx-spark-4-ring-no-switch). That tier parks cacheable
KV groups in a per-node NVMe slot file and serves them back by block hash, which is
worth ~92x on a resent long prompt -- but its block table (hash -> slot) lives in
process memory, so every engine restart orphans the gigabytes of valid KV still on
disk. This module adds a durable index for it, without touching the vLLM source tree.

Adds:
  * PersistentLRUCachePolicy    - vLLM's LRU policy + a durable key -> slot index
  * PersistentNVMeOffloadingSpec - restores the manager's allocator state and the
    policy's index at startup, then snapshots the index periodically (atomic rename)

Wiring (three places, no source edits). 1) mount this file next to the tier module:
    -v ./kv_persist_policy.py:/usr/local/lib/python3.12/dist-packages/kv_persist_policy.py:ro
2) point the connector at the subclasses by name:
    "kv_connector_extra_config": {
        "spec_name": "PersistentNVMeOffloadingSpec",
        "spec_module_path": "kv_persist_policy",
        "cache_policy": "PersistentLRUCachePolicy",
        "cache_policy_module_path": "kv_persist_policy",
        "nvme_dir": "/kv-offload", "nvme_bytes": 68719476736, "offload_prompt_only": false
    }
3) environment (optional; the defaults are derived from nvme_dir + tag):
    KV_TIER_STATE_PATH   where the index is written (default <nvme_dir>/kv-index-<tag>.json)
    KV_TIER_TAG          per-rank namespace, e.g. r0..r3
    KV_TIER_FINGERPRINT  checkpoint/config id; any change starts the cache cold

Design notes / safety
---------------------
* Slot addressing is `block_id * slot_bytes`, deterministic; live slots are never
  handed out again because the allocator state (`_num_allocated_blocks`, `_free_list`,
  `_num_evictable_cache_blocks`) is restored together with the index. Restoring the
  key -> slot table alone is NOT enough: the manager would hand a live slot to a new
  request and that request would read another request's KV (silently wrong output,
  no error anywhere).
* Only blocks whose store COMPLETED are snapshotted -> a half-written slot can never
  be looked up. Entries stored after the last snapshot are simply misses.
* A fingerprint (model, layout, slot geometry, dtype, user tag) guards against
  reusing an index after a weight or config change, which would serve KV computed by
  a different model.
* Any failure while restoring degrades to a cold cache - it never crashes the engine.

Compatibility: written against the vLLM 0.29-era tree this recipe pins, where the
offloading manager takes (cache_policy, cache_policy_module_path, store_threshold).
Check that signature before mounting it on another build.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time

from vllm.logger import init_logger
from vllm.v1.kv_offload.cpu.policies.base import BlockStatus
from vllm.v1.kv_offload.cpu.policies.lru import LRUCachePolicy

import dsv41_kv_nvme as _base

logger = init_logger("vllm.dsv41_kv_persist")

_MAGIC = "DSV41KVS1"
_ENV_STATE = "KV_TIER_STATE_PATH"
_ENV_TAG = "KV_TIER_TAG"
_ENV_FP = "KV_TIER_FINGERPRINT"
_SAVE_EVERY_S = 20.0


class PersistentLRUCachePolicy(LRUCachePolicy):
    """LRU policy that mirrors its index to disk so a restart keeps the cache warm."""

    def __init__(self, cache_capacity: int):
        super().__init__(cache_capacity)
        self._persist = False          # enabled by the spec once paths are known
        self._dirty = 0
        self._lock = threading.RLock()

    # ---- mutation hooks (the actual journaling is the spec's periodic snapshot) ----
    def insert(self, key, block) -> None:
        with self._lock:
            super().insert(key, block)
            self._dirty += 1

    def remove(self, key) -> None:
        with self._lock:
            super().remove(key)
            self._dirty += 1

    # ---- used by the spec ----
    def restore_entries(self, entries) -> int:
        """entries: iterable of (key_bytes, block_id). Bypasses journaling."""
        n = 0
        with self._lock:
            for key, bid in entries:
                blk = BlockStatus(int(bid))
                blk.ref_cnt = 0                      # ready, evictable
                LRUCachePolicy.insert(self, key, blk)
                n += 1
        return n

    def snapshot_entries(self):
        with self._lock:
            return [(k, b.block_id) for k, b in self.blocks.items() if b.is_ready]


class PersistentNVMeOffloadingSpec(_base.NVMeOffloadingSpec):
    """NVMe tier + a restart-survivable index for it."""

    def get_manager(self):
        mgr = self._build_manager()
        try:
            self._state_path = os.environ.get(_ENV_STATE) or os.path.join(
                self.nvme_dir, "kv-index-%s.json" % os.environ.get(_ENV_TAG, "r0")
            )
            pol = mgr._policy
            if isinstance(pol, PersistentLRUCachePolicy):
                pol._persist = True
                n = self._restore(mgr, pol)
                if n:
                    logger.info(
                        "NVMe KV tier: restored %d cached blocks from %s (restart-survivable index)",
                        n, self._state_path,
                    )
            self._start_saver(mgr, pol)
        except Exception as exc:  # never break serving because of the index
            logger.warning("NVMe KV tier: persistent index disabled (%s: %s)",
                           type(exc).__name__, exc)
        return mgr

    # ------------------------------------------------------------------ helpers
    def _build_manager(self):
        """Build the manager the way the base class does, but pass the policy through.

        Trap: the base `get_manager()` reads its policy from a DIFFERENT config key
        (`eviction_policy`) and never passes `cache_policy_module_path`. Going through
        it therefore always ends up with the built-in LRU policy while the snapshot
        thread keeps running against it, logging every 20 s:
            index snapshot failed ('LRUCachePolicy' object has no attribute '_dirty')
        The subclass has to construct the manager itself.
        """
        if getattr(self, "_manager", None) is None:
            from vllm.v1.kv_offload.cpu.manager import CPUOffloadingManager
            logger.info("NVMe KV tier: %d slots x %d bytes per worker (%.1f GiB)",
                        self.num_slots, self.slot_bytes,
                        self.num_slots * self.slot_bytes / 2 ** 30)
            self._manager = CPUOffloadingManager(
                num_blocks=self.num_slots,
                cache_policy=self.extra_config.get("cache_policy", "lru"),
                cache_policy_module_path=self.extra_config.get("cache_policy_module_path"),
                enable_events=self.kv_events_config.enable_kv_cache_events,
                store_threshold=0,
            )
        return self._manager

    def _fingerprint(self, mgr) -> str:
        cfg = self.config
        parts = [
            _MAGIC,
            os.environ.get(_ENV_FP, ""),                   # user-supplied (model dir/revision)
            str(getattr(getattr(cfg, "model", None), "name", "")),
            str(getattr(getattr(cfg, "model", None), "dtype", "")),
            str(getattr(getattr(cfg, "cache", None), "tokens_per_hash", "")),
            str(getattr(getattr(cfg, "cache", None), "blocks_per_chunk", "")),
            str(getattr(cfg, "worker_kv_bytes_per_block", "")),
            str(getattr(mgr, "_num_blocks", "")),
        ]
        return hashlib.sha256("|".join(parts).encode()).hexdigest()[:32]

    def _restore(self, mgr, pol) -> int:
        path = getattr(self, "_state_path", None)
        if not path or not os.path.exists(path):
            return 0
        with open(path, "r") as fh:
            st = json.load(fh)
        if st.get("magic") != _MAGIC:
            return 0
        if st.get("fingerprint") != self._fingerprint(mgr):
            logger.info("NVMe KV tier: index fingerprint mismatch -> starting cold")
            return 0
        if int(st.get("num_blocks", -1)) != int(mgr._num_blocks):
            logger.info("NVMe KV tier: slot count changed -> starting cold")
            return 0
        entries = [(bytes.fromhex(k), bid) for k, bid in st.get("blocks", [])]
        n = pol.restore_entries(entries)
        free = [int(x) for x in st.get("free_list", [])]
        free = [x for x in free if x not in {b for _, b in entries}]
        mgr._free_list = free
        mgr._num_allocated_blocks = int(st.get("num_allocated", len(entries)))
        mgr._num_evictable_cache_blocks = n
        mgr._num_write_pending_blocks = 0
        return n

    def _start_saver(self, mgr, pol) -> None:
        if getattr(self, "_saver", None) is not None:
            return
        path = getattr(self, "_state_path", None)
        if not path:
            return
        stop = threading.Event()

        def loop():
            while not stop.wait(_SAVE_EVERY_S):
                try:
                    if not pol._dirty:
                        continue
                    pol._dirty = 0
                    payload = {
                        "magic": _MAGIC,
                        "fingerprint": self._fingerprint(mgr),
                        "num_blocks": int(mgr._num_blocks),
                        "num_allocated": int(mgr._num_allocated_blocks),
                        "free_list": list(mgr._free_list),
                        "blocks": [[k.hex(), int(b)] for k, b in pol.snapshot_entries()],
                        "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    }
                    tmp = path + ".tmp"
                    with open(tmp, "w") as fh:
                        json.dump(payload, fh)
                    os.replace(tmp, path)
                except Exception as exc:      # pragma: no cover - defensive
                    logger.warning("NVMe KV tier: index snapshot failed (%s)", exc)

        t = threading.Thread(target=loop, name="kv-tier-index-saver", daemon=True)
        t.start()
        self._saver = t
