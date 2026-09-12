---
license: mit
base_model: deepseek-ai/DeepSeek-V4.1-Flash
tags:
  - deepseek
  - deepseek-v4.1-flash
  - vllm
  - dgx-spark
  - gb10
  - sm121
  - tensor-parallel
  - cuda-graphs
  - dspark
  - speculative-decoding
  - engram
  - 1m-context
  - vision
  - tool-calling
language:
  - zh
  - en
pipeline_tag: text-generation
---

# 四台 NVIDIA DGX Spark(GB10)上的 DeepSeek-V4.1-Flash
### 一套配置同时做到:**1M 上下文 · CUDA graphs · 视觉 · 工具调用 · DSpark 投机解码**

**English version: [README.md](README.md) · 英文档见 [README.md](README.md)**

本仓库公开一份**实测过的**服务配方:在**四台 NVIDIA GB10("DGX Spark",SM 12.1,每台 128GB 统一内存)**
上用 vLLM 服务 `deepseek-ai/DeepSeek-V4.1-Flash`(官方 FP8 权重,48 分片,4 路张量并行)。

这里既是一份工程记录,也只是原始数字。真正有价值的部分不是"它能跑起来"——能做到的配方很多——
而是:**到底哪些能力能在 4×128GB 上同时装下**,以及 **KV 缓存池作为 `--gpu-memory-utilization`
的函数究竟怎么变**。下表中每一个数字都是我们在自己机器上量出来的,没有从单机 extrapolate 出来的值。

| | |
|---|---|
| **模型** | `deepseek-ai/DeepSeek-V4.1-Flash`(官方 FP8 权重,未做任何改动) |
| **权重** | 48 个 safetensors 分片,共约 510GB(其中两个约 101GB 的分片是 **Engram** 表) |
| **硬件** | 4 × GB10 / SM 12.1,128GB 统一内存,TP4 over RoCE |
| **引擎** | vLLM(`vllm/vllm-openai:nightly-…` 基础镜像)+ **锁定的 Python 树** + 7 个上游补丁 + 1 个我们自己的补丁 |
| **上下文** | **1,048,576 tokens**(`--max-model-len 1048576`)—— 实测可用,不是宣传值 |
| **1M 下同时开启** | ✅ CUDA graphs(`FULL_AND_PIECEWISE`)✅ 视觉(`--limit-mm-per-prompt {"image":4}`)✅ 工具调用 + 推理解析器 ✅ DSpark 投机解码(k=5)✅ 磁盘 Engram + 每节点本地行 |
| **1M 时的 KV 池** | **1,716,692 tokens = 1M 请求的 1.64 倍**(gmu 0.83) |
| **解码速度** | 单流综合 **50.9 tok/s**;coding **67.7**;计数类天花板 **84.2**;C6 聚合 **125 tok/s** |
| **质量闸门** | needle 在 280K / 300K / 600K / 900K / 1M 全部 PASS · garble gate 30/30 · 视觉+工具 7/7 |
| **协议** | 本仓库 MIT;模型权重遵循 DeepSeek 自身协议 |

---

## 一、这套配方补了什么

相对于它所依托的优秀公开工作(见[第十二节](#十二致谢)),本仓库的增量是:

1. **四个能力同时开、且都在 1M 上。** 公开的四机 GB10 配置普遍要做取舍:要么有 graphs 没有视觉,
   要么 300K 换速度,要么纯文本换来工具。本配方让 **1M max-model-len + FULL_AND_PIECEWISE
   CUDA graphs + 视觉 + 工具/推理解析器 + DSpark k=5 + 磁盘 Engram(每节点本地行)** 跑在同一个进程里。
   原因在 KV 池:是**容量**决定可行性,而不是能力之间互相冲突。
2. **一条实测的 `gpu-memory-utilization` → KV 池曲线**,含失败边界:600K 窗口在 `0.80` 起不来、
   在 `0.82` 能起;以及一个重要观察 —— **池子不是这个 flag 的纯函数**,它随宿主内存状态浮动,
   所以要调的是**余量**,不是一个神奇的数字。
3. **锁定树上的 graph 启动态补丁**(`v1/worker/gpu_worker.py` 两处、由环境变量开关):跳过一次性
   graph 显存 profiling;capture 之后**原地**清理 dummy warmup 留下的状态,保住 graph 缓冲区地址。
   见 [`patches/`](patches/README.md) —— 里面也写了"把为**更新**的树写的补丁挂上来"这个坑。
4. **两次救命的运维铁律**,并直接公开成脚本:起服前的**四台 preflight**,以及失败时的
   **全节点日志抢收集**(赶在下一次起服把证据删掉之前)。
5. **吞吐与质量成对发布**:性能表旁边必须有 needle / garble / 视觉工具闸门,
   这样"快"就不能脱离"没坏"单独发布。

---

## 二、硬件与网络

| | |
|---|---|
| 节点 | 4 × GB10(`SM 12.1`),每台 128GB 统一内存 |
| 互联 | 直连 RoCE 环,逐链路点对点,每链路 `/30`,MTU 9000 |
| NCCL | `NCCL_IB_HCA` 指向 RoCE HCA,`NCCL_IB_GID_INDEX=3`,`NCCL_NET=IB`,RoCE v2 |
| 管理网 | 仅用于 `--master-addr` 会合与权重/NFS 的流量控制 |

两条用代价换来的经验:

* **要用的每个口,GID index 3 必须非零。** 若某口的 GID 全为零,NCCL 会以 `errno 61` /
  `ibv_modify_qp` 超时失败。重新初始化(或冷断电)可以恢复。
* **权重走高速 fabric,别走管理网。** 510GB 权重通过 1GbE 管理网读取要数小时,走 RoCE 环是几分钟。
  我们的布局里,头机只读导出权重,每台 worker 另外保留**自己那份节点本地 Engram 行**(见第五节)。

---

## 三、权重布局与 Engram 表

```
DeepSeek-V4.1-Flash/
├── config.json                 # architectures: ["DeepseekV41ForCausalLM"]
├── model.safetensors.index.json
├── tokenizer.json / tokenizer_config.json
├── model-00001-of-00048.safetensors   ...  model-00046-of-00048.safetensors
├── model-00047-of-00048.safetensors   # 约 101GB —— Engram 表
└── model-00048-of-00048.safetensors   # 约 101GB —— Engram 表
```

那两个 101GB 分片**不是**稠密权重,而是 Engram n-gram 记忆表;服务时可以不常驻内存、从磁盘读,
而且每个 rank **只需要自己那一段行**。我们的做法:

* 头机把权重以只读方式经 fabric 提供给 worker;
* 每台 worker 在本地 NVMe 上放一份约 48GB 的**本 rank 行区间稀疏副本**,并从它读取
  (`DSV41_ENGRAM_DIR`),逐行抽样与源校验;
* 这样模型本身每 rank 只占约 82GB,剩下的才是 KV 缓存的空间。

**这是整份配方里最关键的内存决策**:没有"磁盘 Engram + 本地行",就没有 1M 的 KV 池可言。

---

## 四、启动配置(配方本体)

产生第六节数字的环境变量与参数如下。地址/网卡名用占位符,替换成你自己的。

```bash
# 每个节点,rank 0..3
NODE_IPS=(<IP0> <IP1> <IP2> <IP3>)
FABRIC_IFACE=<roce-iface>        # 例如那条 200G 链路的名字
IB_HCA=<roce-hca>
MODEL_PATH=/srv/models/DeepSeek-V4.1-Flash
ENGRAM_DIR=/srv/engram-local/DeepSeek-V4.1-Flash   # 仅 worker
```

```bash
docker run --gpus all --network host --ipc host \
  --shm-size 32g --memory 112g --memory-swap 112g \
  --ulimit memlock=-1:-1 --cap-add IPC_LOCK \
  --device /dev/infiniband:/dev/infiniband --oom-score-adj 500 \
  -v "$MODEL_PATH:/models/DeepSeek-V4.1-Flash:ro" \
  -v "$ENGRAM_DIR:/engram-local:ro" \
  # ... 以及把补丁文件 bind-mount 覆盖到 vLLM 的 Python 树上 ...
  -e DSV41_ENGRAM_DISK=1 -e DSV41_ENGRAM_DISK_THREADS=32 -e DSV41_ENGRAM_DISK_CHUNK=16 \
  -e DSV41_ENGRAM_DIR=/engram-local \
  -e DSV41_SKIP_GRAPH_MEMORY_PROFILE=1 \
  -e DSV41_CLEAR_STATE_AFTER_CAPTURE=1 \
  -e VLLM_USE_BREAKABLE_CUDAGRAPH=1 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e VLLM_ENGINE_READY_TIMEOUT_S=3600 \
  -e NCCL_NET=IB -e NCCL_IB_HCA="$IB_HCA" -e NCCL_IB_GID_INDEX=3 \
  -e NCCL_SOCKET_IFNAME="$FABRIC_IFACE" -e TORCH_CUDA_ARCH_LIST=12.1a \
  "$IMAGE" /models/DeepSeek-V4.1-Flash \
    --served-model-name deepseek-v4.1-flash \
    --host 0.0.0.0 --port 8888 \
    --tensor-parallel-size 4 \
    --gpu-memory-utilization 0.83 \
    --max-model-len 1048576 \
    --max-num-seqs 8 \
    --max-num-batched-tokens 8192 \
    --block-size 128 \
    --engram-config '{"cpu_offload": false}' \
    --default-chat-template-kwargs '{"thinking": false}' \
    --limit-mm-per-prompt '{"image":4}' --mm-processor-cache-gb 1 \
    --tool-call-parser deepseek_v41 --enable-auto-tool-choice \
    --reasoning-parser deepseek_v41 \
    --speculative-config '{"method":"dspark","num_speculative_tokens":5,
                           "draft_sample_method":"probabilistic",
                           "rejection_sample_method":"block",
                           "enable_adaptive_verification":false}' \
    --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE",
                           "cudagraph_capture_sizes":[...]}' \
    --distributed-executor-backend mp --nnodes 4 --node-rank "$RANK" \
    --master-addr "${NODE_IPS[0]}" --master-port <PORT>
```

几个真正要紧的点:

* **capture sizes 必须与投机配置匹配。** 投机 `k` token、`max-num-seqs S` 时,调度器可能产生的
  batch 是 `k, 2k, … Sk` 以及 `k+1, 2(k+1), … S(k+1)`。capture 列表要按这个生成;
  直接抄一份别的 `k` 的列表,在稀疏 MLA 路径上表现为**挂死**,而不是报错。
* **`--language-model-only` 与视觉二选一**:不写它=保留视觉塔;写了=纯文本。
  开视觉会吃掉 KV 池容量(见 6.4),因为多模态缓冲与 processor cache 都出自同一块统一内存。
* **`--default-chat-template-kwargs '{"thinking": false}'`**:若默认开思考,短请求会把预算全花在
  推理字段上,`content` 返回空。客户端仍可按请求单独打开。
* **启动顺序**:先起 worker(rank 从高到低),再起头机,并把 engine-ready 超时设大。
  头机负责会合;若某个 worker 没加入,头机是卡在 store 上**超时**,不会给出清晰错误(见第九节)。

---

## 五、graph 启动态补丁

在大 `max-model-len` 下,仅有"基础镜像 + 锁定树"还不够。启动期有两件事我们做了补丁:

| 改动 | 原因 |
|---|---|
| 跳过一次性 **graph 显存 profiling**(`DSV41_SKIP_GRAPH_MEMORY_PROFILE=1`) | profiler 会在真正 capture 之前跑一遍完整 dummy pass 来**估算** graph 显存。窗口一大,这次瞬时分配就会把内存吃紧的节点顶出边界,引擎在初始化阶段直接死掉,而且报错信息毫无指向。 |
| 把 V2 **kernel warmup 挪到 capture 之前**,并在 capture 之后**原地**清理 dummy warmup 状态(`DSV41_CLEAR_STATE_AFTER_CAPTURE=1`) | capture 之后再做 workspace resize,会释放掉已捕获图仍然指向的内存;而 dummy warmup 可能在 KV slot 里留下 NaN,随后被 masked attention 读到。**原地清零**而不是重新分配,才能保住 graph 缓冲区地址。 |

`patches/gpu_worker_cachefix.py` 是这两处改动**针对我们所构建的那棵树**的重新实现(目标是
`v1/worker/gpu_worker.py`)。完整的差异理由与许可/署名说明见
[`patches/README.md`](patches/README.md) —— 其中包括**为什么绝对不能**挂一个为更新版本
vLLM 树写的同名文件。

---

## 六、实测结果

以下全部来自我们自己的四机 GB10 部署。关闭思考、温度 0、固定题集,token 数取服务端 usage 字段。

### 6.1 1M 配置下的吞吐(gmu 0.83,k=5,视觉+工具+graphs)

| 并发 | 聚合 tok/s | 单流 tok/s | 平均 TTFT (s) |
|---|---|---|---|
| C1 | 44.7 | **50.9** | 0.38 |
| C2 | 72.4 | 42.0 | 0.42 |
| C3 | 84.8 | 32.4 | 0.45 |
| C6 | **125.1** | 24.2 | 0.55 |

各分类单流 tok/s(C1 … C6):

| 类别 | C1 | C2 | C3 | C6 |
|---|---|---|---|---|
| coding | 67.7 | 60.9 | 52.1 | 34.7 |
| json | 44.1 | 49.3 | 29.5 | 23.7 |
| narrative | 27.5 | 19.4 | 16.1 | 11.7 |
| prose | 31.1 | 22.5 | 17.9 | 13.1 |
| math | **74.0** | 51.1 | 42.5 | 32.3 |
| reasoning | 56.9 | 41.5 | 31.9 | 22.2 |
| summary | 33.0 | 26.7 | 23.2 | 14.4 |
| format | 73.1 | 64.7 | 45.7 | 41.8 |
| **计数天花板** | **84.2** | 75.1 | 45.3 | 43.0 |

计数类不计入聚合;它是本配置的实际解码天花板。

### 6.2 冷 prefill(唯一前缀,无缓存复用)

| 提示词 token | TTFT (s) | prefill tok/s |
|---|---|---|
| 2,950 | 2.0 | 1,449.9 |
| 46,810 | 32.4 | 1,445.8 |

### 6.3 长上下文:needle 检索与 prefill 速率

50% 深度单针、精确匹配、冷前缀:

| 上下文 | 配置 | 提示词 token | TTFT (s) | prefill tok/s | 结果 |
|---|---|---|---|---|---|
| 280K | 600K 窗口 + graphs | 278,118 | 204.0 | 1,363.2 | PASS |
| 300K | 1M 窗口 + eager | 298,172 | 293.9 | 1,014.4 | PASS |
| 600K | 600K 窗口 + graphs | 596,196 | 529.5 | 1,125.9 | PASS |
| 600K | 1M 窗口 + eager | 609,017 | 714.8 | 852.0 | PASS |
| 900K | **1M 窗口 + graphs** | 894,548 | 1,012.0 | 884.0 | PASS |
| 1M | 1M 窗口 + eager | 993,435 | 1,243.3 | 799.1 | PASS |

每一种情况都精确检索到目标。注意 **prefill 速率随配置变化**:同一个 600K 提示词,带 graphs 约
1,126 tok/s,eager 只有约 852 tok/s。**如果客户端超时短于 TTFT,再好的模型也拿不到答案** ——
客户端超时应按 `上下文 / prefill速率` 来定。

### 6.4 KV 缓存池 vs `--gpu-memory-utilization`(别人不公开的那部分)

统一内存机器在这里与独立显卡不同。以下是各配置起服瞬间量到的池子大小:

| gmu | max-model-len | 视觉/工具 | graphs | KV 池(token) | 池/窗口 |
|---|---|---|---|---|---|
| 0.80 | 300,000 | 开 | 是 | 648,717 | 2.16× |
| 0.80 | 600,000 | 开 | **起服失败** | — | — |
| 0.82 | 600,000 | 开 | 是 | 780,110 … 1,149,256 | 1.30× … 1.92× |
| 0.83 | 1,048,576 | 开 | 是 | **1,716,692** | **1.64×** |
| 0.83 | 1,048,576 | 纯文本 + eager | 否 | 1,529,317 … 1,627,978 | 1.46× … 1.55× |

两条"靠推理一定会弄错、只能靠实测"的结论:

* **池子不是这个 flag 的纯函数。** 同为 `0.82`,不同次起服量到的池子相差约 37 万 token ——
  因为池子是从**初始化那一刻的空闲内存**算出来的,而它取决于宿主状态。所以要调的是**余量**
  (我们要求 ≥1.5× 窗口),不是一个神奇数字。
* **600K 窗口在 `0.80` 起不来、`0.82` 能起** —— 差距约 2.4GB(≈50 万 token 的 KV)。
  配合第五节的 graph 补丁,这就是"带视觉做 600K 不可能"与"带视觉做 1M 可行"之间的全部距离。

### 6.5 深上下文解码(1M 窗口 + eager,生成 1024 token)

| 任务 | 提示词 token | TTFT (s) | prefill tok/s | **decode tok/s** |
|---|---|---|---|---|
| code | 1,015,402 | 1,129.4 | 899.1 | **52.5** |
| prose | 1,015,382 | 995.2 | 1,020.3 | **24.3** |

深上下文解码与内容相关(投机接受率在代码与散文间差一倍)——两个都要发,否则数字无法被检验。

### 6.6 质量闸门

| 闸门 | 结果 |
|---|---|
| garble gate(30 次结构化生成,温度 0 / 0.7 / 1.0,6 路并发) | **30/30 干净** |
| 视觉任务(色序、双图、网格象限) | **3/3 PASS** |
| 工具调用(单次调用、结果回灌、并行双调用、强制 `tool_choice`) | **4/4 PASS** |
| NaN 探针(温度 0:算术/事实/自我介绍) | 无 NaN,正确 |
| needle-in-haystack | 6.3 节每一个上下文均 PASS |

为什么要这些闸门:开发期我们遇到过一种配置 —— **起服干净、profiling 与 graph capture 全过,
然后从第一个 token 起吐重复垃圾**。吞吐表永远发现不了这种问题,闸门可以。

---

## 七、复现测量

```bash
# 吞吐网格:并发 1/2/3/6 × 8 类 + 冷 prefill
python3 bench/v41bench.py --base http://127.0.0.1:8888/v1 \
        --model deepseek-v4.1-flash --levels 1,2,3,6 \
        --prefill 2000,8000,32000 --out ./results

# 深上下文解码(code + prose,生成 1024 token,4 段观测窗口)
python3 tools/ctx_decode_bench.py --base http://127.0.0.1:8888/v1 \
        --model deepseek-v4.1-flash --targets 1000000 --maxtok 1024
```

本仓库 `tools/` 里只有我们自己的脚本:深上下文解码基准、四机 preflight、失败证据收集器。

---

## 八、运维要点

* **每次起服前 preflight**(`tools/preflight.sh`):在**每一台**上核验镜像存在、NCCL 库存在、
  权重与本 rank 的 Engram 行已挂载,以及挂载清单里**每一个补丁文件**本地都存在。
  补丁文件是按节点 bind-mount 的:只在头机存在的补丁目录,会让三个 worker 永远起不来,
  而头机在等它们(见第九节)。
* **先 worker 后头机**,rank 从高到低,`VLLM_ENGINE_READY_TIMEOUT_S` 要设大(我们 3600)。
  权重加载每 rank 约 4–5 分钟,再加 graph capture,到 API 就绪通常 **10–15 分钟**。
* **本地 Engram 副本必须与权重 revision 对齐**;行区间与 revision 绑定,跨 revision 别复用。
* **一套节点只跑一个部署。** 四台是一个 TP4 组;在同样的节点上再跑第二个模型,会争同一块统一内存。

---

## 九、坑(每一条都真实消耗过我们的时间)

1. **挂了为更新树写的补丁。** 整文件替换型补丁必须与它所覆盖的树匹配。若补丁期望的符号你的树没有,
   会得到 import 错误 —— 烦人但诚实。**更危险的是相反方向**:树比补丁锚点更新,可能得到一个
   *能起服*、profiling 全过、然后从第一个位置吐重复垃圾 token、投机草稿一个都不接受的引擎。
   请把 **Python 树与编译出来的扩展一起钉在补丁集对应的同一个 commit** 上。
2. **补丁目录只在头机。** 头机侧症状:容器起来了、权重加载完了,然后卡在 process-group store 上,
   最后以 `DistStoreError: Timed out after 601 seconds waiting for clients. 1/4 clients joined` 结束。
   真正原因是 worker 的启动步骤以 `PATCH MISSING …` 退出,根本没人来连。**先看 worker 的启动输出,
   再怀疑内存。**
3. **证据丢失。** 最有价值的日志在 worker 上,而下一次起服会把它连容器一起删掉。失败后**立刻**
   收集每一台的容器日志(`tools/capture-allnode-logs.sh`)。
4. **只看 KV 容量来定窗口。** `池 ≥ 窗口` 是必要条件而非充分条件:启动期还需要瞬时内存
   (profiling、capture、warmup)。要么留余量,要么让这些瞬时开销变小(第五节)。
5. **重启时的会合端口。** 连续起服应使用不同的 `--master-port`(或确认旧的 store 已消失),
   否则可能出现旧 store 占着地址、新组试图在其上会合的情况。
6. **只信一个吞吐数字。** 深上下文下代码与散文的解码差约 2 倍(投机接受率不同)。两个都要报,
   并且每张吞吐表都必须配一个质量闸门。
7. **以为模型知道自己被怎么配的。** 问它"你的最大上下文是多少",它可能凭训练数据回答。
   请从 API 读 `max_model_len`,不要问模型。

---

## 十、常见问题:300K / 600K / 1M 怎么选

用本配方时,这是一个**容量**问题,不是功能取舍问题:

| 窗口 | KV 池(实测) | 池/窗口 | 单流 C1 | 备注 |
|---|---|---|---|---|
| 300K | 648,717(gmu 0.80,视觉+graphs) | 2.16× | 48.3 | 并发长请求最多 |
| 600K | 780,110 – 1,149,256(gmu 0.82) | 1.30× – 1.92× | ~50 | gmu 0.80 下**起不来** |
| 1M | **1,716,692(gmu 0.83)** | **1.64×** | **50.9** | 视觉 + 工具 + graphs 全开 |

池子在这三个窗口下是同一量级,而窗口本身相差 3.5 倍 —— 所以**你能服务的窗口由池子决定,
池子由你留给它多少内存决定**,与"要不要视觉/工具"无关。这也是我们到处都发池子这一列的原因。

---

## 十一、我们的测试环境

| 组件 | 版本/备注 |
|---|---|
| 节点 | 4 × GB10,每台 128GB 统一内存 |
| CUDA 架构 | `12.1a` |
| 基础镜像 | `vllm/vllm-openai:nightly-<pin>`(配方中以精确 digest 固定) |
| 构建 | 为 `sm121a` 编译扩展,`MAX_JOBS=2`、`FLASHINFER_NVCC_THREADS=1`、`VLLM_USE_FLASHINFER_SAMPLER=0` |
| 容器限制 | `--memory 112g --memory-swap 112g --shm-size 32g --oom-score-adj 500` |
| 主机调优 | 提高 `vm.min_free_kbytes`、提高 watermark scale factor、提高 NFS 服务线程、解锁 GPU 频率、把容器缓存落到 NVMe |

---

## 十二、致谢

站在别人的工作上,把话说明白:

* **DeepSeek** —— 模型、权重与草稿层(`deepseek-ai/DeepSeek-V4.1-Flash`)。
* **Tech2Wild/Kai(tonyd2wild)** —— 四机 Spark 的基础配方:补丁集、磁盘 Engram staging、
  worker-first 启动顺序、镜像链与基准协议。我们跑的绝大部分是他们的骨架;
  我们改的是"如何参数化"以及"在 1M 上能装下多少"。
* **0xTank** —— graph 启动态修复(跳过 profiling pass、capture 后清理状态)与紧凑输出投影思路。
  `patches/` 里的补丁是这些改动针对我们所构建之树的**重新实现**。
* **vLLM、FlashInfer、Triton、PyTorch、NVIDIA** 及其贡献者 —— 引擎、内核与工具链。
* **整个 DGX Spark 社区** —— 持续公开这块硬件上的量化与配方;6.4 节的池子/窗口数据,
  正是因为看到别人的数字后我们开始怀疑自己,才去量的。

如果你用了这份配方,请先引用上面这些人;我们添加的价值是**测量**与**运维纪律**。

---

## 十三、许可

MIT —— 见 [LICENSE](LICENSE)。模型权重与任何上游源码文件保留其自身许可,见 [NOTICE.md](NOTICE.md)。

本仓库**不含**模型权重、**不含**任何凭据、**不含**私有地址、**不含**运维机密。
这里的一切要么是一份配置,要么是一个测量结果,要么是一个脚本。
