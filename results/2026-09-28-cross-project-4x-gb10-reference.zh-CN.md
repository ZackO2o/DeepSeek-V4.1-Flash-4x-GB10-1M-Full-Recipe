# 2026-09-28 —— 4 × GB10 无交换机环网:跨项目参考表

**请把它当书单读,不要当排行榜。** 下表每一行都由各自项目发布,用的是那个项目自己的
harness、prompt 集与检出版本。换一套 harness 会让同一个指标动 20–25 %,所以这些数字
**不能互换**,本页也不给它们排名。它的用处是:找到硬件与引擎和你最接近的那个项目,
并知道它 README 里哪个数字该被第一个复现。

我们自己的同口径对照(两侧同一套 harness)在
[`2026-09-28-sglang-switchless-ring-tp4.zh-CN.md`](2026-09-28-sglang-switchless-ring-tp4.zh-CN.md);
本页是它外面更大的语境。

## 一、4 × GB10 上的 DeepSeek-V4.1-Flash

| 项目 | 引擎 / 网络 | 已发布头条(按项目原文) |
|---|---|---|
| **MiaAI-Lab/DeepSeek-v4.1-Flash-DGX-Sparks**(本仓所基于的配方) | SGLang,带交换机生产档 **以及** 无交换机环网变体 | 带交换机:prose C1 **87.7** / code C1 **124.8** tok/s,C1–C16 聚合 87.7 / 120.4 / 163.6 / 237.6 / 342.7;冷 prefill 4,059–5,925 tok/s(4k–262k);1,011,084 token needle PASS;qeval 72/75。环网变体(用他们自己的 `decode_window.py`):C1 **74.03**、C8 **224.84** 聚合;8 × 6,000 token 长流 → 墙钟 215.81 tok/s,引擎 batch 峰值 258.63 |
| **luxingcom/LuZ-0.1.7-DeepSeek-v4.1-Flash-DGXspark-TP4-Ring** | vLLM(W4A4 MoE 路径),无交换机环网 | 纯 prefill 峰值 **5,823.1 t/s**(65,536 × C2),512 K 单流 5,042.2;解码聚合峰值 **572.6 t/s**(code,C16),C1 峰值 89.19 |
| **nero-/deepseek-v41-flash-gb10-ring** | SGLang,无交换机环网 | 解码 81.8 / 76.7 / 78.8 tok/s;coding 峰值 **97.5**、中位 80.4;qeval 72/75;并公开了单步拆解(rank 0 profiler) |
| **ChrisLou-bioinfo/dsv41-4x-spark-tutorial** | SGLang,环网(教程形态) | 单流 **45**、4 流聚合 **103**、128 K prefill 3.2 K tok/s、1 M 可用。参数集与本仓环网 lane 相同(`0.80` / `MAX_RUNNING 8` / 定容 `8,000,000` / chunk 1024 / `TP_PAD 0`) |
| **yunwei37/dgx-spark-4-ring-no-switch** | vLLM,无交换机环网,多模型日志 | V4.1-Flash,DSpark k=5,gmu 0.83:**993,435 token needle 在 1,048,576 下正确**,KV 池 2,058,026;**C1 均值 50.14 tok/s,code 68.06,8 并发聚合 144.71**;生产前缀缓存命中率 87–92 %。同仓还记录了 GLM-5.3 与 Qwen3.8 两条 lane |
| **我们**(本仓) | SGLang,无交换机环网,`canary-roce` 配置 + sparkring NCCL | 用上游 harness:C1 **104.10** / C2 147.56 / C4 190.00 / C8 **245.98** 聚合;KV 池 3,253,248 token;方法与细节见同目录另一篇 |

27 B–30 B 量级:其中几个项目同样在四节点环网上服务 `GLM-5.3-Flash` 与
`Qwen3.8-Flash-Next`,这正好可以交叉验证**这张网本身值多少**:

| 项目 | 模型 | 已发布 |
|---|---|---|
| ntxf31415/glm-5.3-flash-nvfp4-4x-dgx-spark-switchless | GLM-5.3-Flash **NVFP4** | 单流 **94.7 t/s 峰值 / 75.9 均值**(开思考);400 K 冷 prefill TTFT 191.4 s / 2,096.9 tps |
| yunwei37/dgx-spark-4-ring-no-switch | GLM-5.3-Flash FP8(zai-org) | 240,000 token 检索通过;**20.16 tok/s** 单流;MTP=5 变体 25.57 tok/s |
| yunwei37/dgx-spark-4-ring-no-switch | Qwen3.8-Flash-Next NVFP4,TP2 | 单流 **40.20 tok/s**,4 并发 **102.19**(262,144 窗口) |
| yunwei37/dgx-spark-4-ring-no-switch | GLM-5.3 Int4/Int8Mix | 8,192 有界基线:均值 12.90 tok/s(近 200 K prefill 撞上内存墙) |

## 二、我们在跑的几条 lane

同一档机器,前面一个网关,按负载选模型。下表数字是我们的,各自用**那条 lane 自己的
快速 harness** 量得,描述的是我们的部署,不与上面两张表直接可比:

| Lane | 模型 | 引擎 / 硬件 | 用途 |
|---|---|---|---|
| 旗舰长上下文 | `deepseek-ai/DeepSeek-V4.1-Flash`(官方 FP8) | SGLang TP4,4 × GB10 无交换机环网 | 1 M 上下文、视觉 + 工具、编码;即本仓主题 |
| 第二条长上下文 lane | `deepseek-ai/DeepSeek-V4.1-Flash` | vLLM TP4,4 × GB10 | 与 SGLang lane 做 A/B;主 README 里的配方跑在这条上 |
| 快速 27 B 级 | `Qwen3.8-Flash-Next` | SGLang TP2,2 × GB10 环网 | 较短上下文的批量工作 |
| 成本 lane | 27 B 级 FP8,2 × 2080 Ti | vLLM fork(FP8 KV、MTP;反量化路径 —— SM 7.5 无原生 FP8) | 旧硬件上的单卡吞吐 |
| 备份 lane | `GLM-5.3-Flash` | EXL3 跑在 2 × GB10 | 第二家厂商路径,长上下文 |

公开这张表的意义在于:**可复用的是那张网**。环网一旦跑通,同样四台机器可以分别承载
1 M 的 DeepSeek lane、GLM lane 或 Qwen lane,引擎可以不同 —— 而上面那些跨项目行,
就是每种组合该去查的配置来源。

## 三、怎么读别人的数字

1. **先找 harness。** 若是 `decode_window.py` 这类,单流值是按字符缩放的估计;
   能经得起对照的是聚合列(服务端 usage ÷ 墙钟)。若项目写的是"引擎 batch 峰值",
   那是内部计数器,不是端到端吞吐 —— 它读数一定高于任何客户端能观测到的值。
2. **再找检出版本。** `chunk prefill 1024` vs `4096`、`max_running_requests 8` vs
   `16`、投机解码 `k=3` vs `k=5`,每一条对结果的移动都比多数硬件差异更大。
   两个项目对**同一个模型、同一档机器**分别发布 45 和 105 tok/s,可能都是诚实的。
3. **再找时钟策略。** GB10 的默认值不等于锁频后的值;一台实测 2,177–2,190 MHz 的机器
   对一台 2,392–2,398 MHz 的机器,在比任何东西之前就先差 ~9.5 %。
4. **再找 prompt。** 256 token 的生成由 TTFT/prefill 主导,6,000 token 的不是。
   把两者混在一张跨项目表里,那就不算表。
5. **然后复现其中一行。** 挑与你配置最接近的那个数字,在你自己的机器上跑他们的
   harness。这一个动作比本页所有表格(包括我们的)都更值钱。

## 四、出处

以上全部数值读自各项目自己的 README / 结果文档,读取日期 **2026-09-28**。
仓库名会变 —— 若链接 404,请按项目标题搜索,不要相信过期路径。

- MiaAI-Lab/DeepSeek-v4.1-Flash-DGX-Sparks — `README.md`、`docs/tp4-switchless-ring-results.md`
- luxingcom/LuZ-0.1.7-DeepSeek-v4.1-Flash-DGXspark-TP4-Ring — `README.md`
- nero-/deepseek-v41-flash-gb10-ring — `README.md`、`RESULTS.md`
- ChrisLou-bioinfo/dsv41-4x-spark-tutorial — `README.md`
- yunwei37/dgx-spark-4-ring-no-switch — `README.md`
- ntxf31415/glm-5.3-flash-nvfp4-4x-dgx-spark-switchless — `README.md`
