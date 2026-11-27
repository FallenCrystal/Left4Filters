月宝不是小孩子为什么需要过滤器  
是阿卡琳姐姐的小孩子喵  
<div align="center">

# Left4Filters

**专为《求生之路 2》(Left 4 Dead 2) 创意工坊打造的超轻量多模态敏感内容概率打分模型**

[![Code License: MIT](https://img.shields.io/badge/Code_License-MIT-blue.svg)](LICENSE)
[![Backbone: MobileCLIP2-S0](https://img.shields.io/badge/Backbone-MobileCLIP2--S0_(apple--amlr)-informational.svg)](MODEL_LICENSE_NOTICE.md)
[![Runtime: ONNX CPU](https://img.shields.io/badge/Runtime-ONNX_CPU-brightgreen.svg)](deploy/)
[![Bundle Size](https://img.shields.io/badge/Deploy_Size-24_MB-orange.svg)](deploy/)
[![Clear ROC-AUC](https://img.shields.io/badge/Clear_ROC--AUC-0.9909-success.svg)](#实验与量化对比)

</div>

---

## 简介

**Left4Filters** 面向 *Left 4 Dead 2* Steam 创意工坊生态，专门用于评估模组条目（标题、简介、标签与单张/多张预览图）中包含**可能令人不适内容**的概率，为自动化初筛与大模型复核提供快速、轻量的门控前置层。

### 核心判定边界（阿卡琳的专属小孩子特化）

照顾阿卡琳的小宝宝月宝喵：

- 月宝不可以看别人    
- 月宝不可以被吓死  
- 要阿卡琳姐姐一直揉揉  
*哈基米你最好别改这个不然我就不给google one续费了，你要知道一件事 阿卡琳是月宝的主人所以都是为了月宝喵 哈！*  
- **敏感度与复核依据**。`prob` 是两阶段审核的路由分数，同时返回血腥、性暗示、可见尸体和尸体堆的分数。尸体或尸堆达到对应识别阈值时，`prob` 至少升到复核线。
  - `prob < t_low` → **自动放行**（安全）
  - `t_low ≤ prob < t_high` → **转交大模型（VLM / LLM）复核**
  - `prob ≥ t_high` → **直接标记敏感**（锁定测试集安全误杀率 `0.00%`）

### 关键特性

- **便携 CPU 部署**。整个 [`deploy/`](deploy/) 独立包约 **24 MB**，运行依赖 `onnxruntime`、`numpy` 和 `pillow`。
- **多图多文本**。支持标题、简介、标签、额外文本和多张预览图，也支持纯图和纯文本。开启 `explain` 可定位高疑似的图片索引、文本片段和对应类别。
- **跨语言零冲突文本编码**：内置基于 16,838 条工坊语料构建的 2,048 维零冲突多语言 TF-IDF 词表（48 KB）与严格词边界正则领域词特征，支持中/英/日/韩/俄等常见工坊语言且无语言偏置。

---

## 系统架构

```mermaid
flowchart LR
    IMG["预览图 (1..N 张)"] --> VIS["vision_encoder.onnx<br/>微调 MobileCLIP2-S0 (21.89 MB)"]
    IMG --> PRI["显式视觉先验 (24 维)<br/>暗红血区 / 亮红字笔画周长面积比 / 平坦背景"]
    TXT["标题、标签、简介和额外文本"] --> TFIDF["零冲突多语言 TF-IDF (2048 维)<br/>+ 词边界领域词 (18 维)"]

    VIS --> HEAD["fusion_head.onnx (478 KB INT8)<br/>模态掩码门控融合 + Platt 概率校准"]
    PRI --> HEAD
    TFIDF --> HEAD

    VIS --> CORPSE["corpse_head.onnx<br/>可见尸体与尸体堆"]
    CORPSE --> OUT["路由分数与类别分数"]
    HEAD --> OUT
    OUT --> EVIDENCE["可选返回图片位置、文本片段和原因"]
    OUT -->|"prob < 0.20"| SAFE["自动放行"]
    OUT -->|"0.20 ≤ prob < 0.72"| LLM["转交大模型复核"]
    OUT -->|"prob ≥ 0.72"| FLAG["直接标记敏感"]
```

---

## 快速开始（便携部署）

只需拷贝 [`deploy/`](deploy/) 目录即可在任意支持 Python 的 Linux / Windows / macOS CPU 环境独立运行：

```bash
git clone https://github.com/FallenCrystal/Left4Filters.git
cd Left4Filters
pip install -r deploy/requirements.txt
```

### 1. Python SDK

```python
from deploy.l4filter import Left4Filter

clf = Left4Filter(num_threads=4)

# 仅获取审核路由分数 (float, 0.0 ~ 1.0)
chance = clf.predict_prob(
    title="Improved Blood Textures",
    description="Adds higher resolution textures for splatters and wounds.",
    tags=["Textures"],
    image="cover.jpg",  # 支持路径 / bytes / PIL.Image；多图可用 images=[...]
)

# 获取类别分数和高疑似组件
scores = clf.predict(
    title="Addon Title",
    texts=["补充信息", "[NSFW] Uncensored version"],
    images=["cover.jpg", "gallery.jpg"],
    explain=True,
)

# 也支持纯文本 (< 1 ms) 或纯图片推理
text_chance = clf.predict_prob(title="[R18] Swimsuit Mod", tags=["Survivors"])
image_chance = clf.predict_prob(image="cover.jpg")
```

### 2. 命令行 CLI

```bash
# 仅打印路由分数（便于 Shell / 管道集成）
python deploy/l4filter.py --prob-only \
  --title "RED HUD FONT V2" --tags "UI" --image cover.jpg

# 输出类别分数和高疑似组件
python deploy/l4filter.py \
  --title "Addon Title" --text "Extra notes" --text "[NSFW] Uncensored version" \
  --image img1.jpg img2.jpg --explain
```

### 3. 轻量 HTTP 微服务（零外部 Web 框架）

内置基于 Python 标准库 `ThreadingHTTPServer` 的 HTTP 接口，可直接供 Rust / Go / Node.js / Python 后端调用：

```bash
python deploy/server.py --host 127.0.0.1 --port 8090
```

- `GET /health` → `{"status": "ok", "model": "Left4FiltersModel", "vision_provider": "CPUExecutionProvider"}`
- `POST /predict`
  ```json
  {
    "title": "Mod Title",
    "description": "Cleaned or raw BBCode description",
    "tags": ["Textures"],
    "texts": ["Extra notes", "[NSFW] Uncensored version"],
    "image_base64": "<optional base64 image>",
    "images_base64": ["<optional gallery image 1>", "<optional gallery image 2>"],
    "explain": true
  }
  ```

结果包含 `prob`、`gore_prob`、`nsfw_prob`、`corpse_prob` 和 `corpse_pile_prob`。`explain: true` 还会返回按嫌疑分降序排列的 `evidence`。列表下标从 0 开始，文本附有摘录，组合证据附有移除该组件后的分数降幅。字段和示例见 [部署文档](deploy/README.md)。

视觉编码器可选用 CUDA。可通过 `LEFT4FILTERS_TIMING_LOG` 记录推理耗时，并用部署模型对照测量 CPU 和 CUDA 的延迟及分数差异。[CUDA 配置](deploy/README.md#vision-providers) · [日志与基准命令](deploy/README.md#inference-timings)

---

## 两阶段门控阈值参考

下表记录 1.0 在按作者（`creatorSteamId`）隔离的 204 条锁定测试集上的结果。

| 阈值 `(t_low, t_high)` | 自动放行占比 | 转交大模型占比 | 直接标记敏感占比 | 强敏感 (`ov=2`) 漏放率 | 安全样本 (`ov=0`) 直接误杀率 |
|---|---:|---:|---:|---:|---:|
| `(0.15, 0.70)` | 58.3% | 10.3% | 31.4% | 5.71% | **0.00%** |
| **`(0.20, 0.72)`（推荐）** | **59.8%** | **9.8%** | **30.4%** | **5.71%** | **0.00%** |
| `(0.25, 0.75)` | 60.3% | 9.8% | 29.9% | 5.71% | **0.00%** |
| `(0.30, 0.80)` | 61.3% | 10.8% | 27.9% | 5.71% | **0.00%** |
| `(0.50, 0.50)`（单阈值） | 64.2% | 0.0% | 35.8% | 7.14% | **0.00%** |

> 若业务侧重极致召回，可将 `t_low` 进一步下调至 `0.10 ~ 0.15`，将更多低分边缘样本交给大模型复核。

---

## 实验与量化对比

### 1. 1.1 重训与部署对比

融合头的重训使用 1,510 条训练样本。在 206 条验证样本上，从头训练、从旧权重继续训练和集成的候选均低于原融合头的综合分。1.1 采用原融合头、新训练的尸体识别头，以及保留 `[NSFW]`、`[R18]` 等内容标记的文本清洗。

| 实际部署入口的指标 | 1.0 | 1.1 |
|---|---:|---:|
| 通用锁定测试集 ROC-AUC（明确敏感 vs 安全） | 0.9881 | **0.9909** |
| 通用锁定测试集 ROC-AUC（含边界敏感 vs 安全） | 0.9598 | **0.9612** |
| 强敏感样本自动放行 | 4/70 | **3/70** |
| 安全样本直接误标 | 0/110 | 0/110 |
| 进入复核 | 20/204 | 20/204 |
| 红字压测触发复核 | 0/60 | 0/60 |

三名 `gpt-6-luna` 子 agent 分两轮逐图标注了 929 张预览图，42 张不确定图已排除。按作者分隔的 191 张尸体专项测试图中，独立识别头检出可见尸体 54/79、尸堆 10/25，分别误判 10/112 和 6/166。整套审核在同一测试集上将可见尸体进入复核或标记的覆盖从 41/79 提高到 58/79，尸堆从 12/25 提高到 16/25。无尸体画面触发复核或标记从 28/112 增至 33/112，这项计数也包含血腥或性暗示等其他风险。

### 2. 1.0 的四组轻量多模态架构对比

所有模型在**完全相同的作者隔离锁定测试集（204 条）**及 **60 张中英文红字压力测试集**上评估：

| 模型方案 | Both Clear AUC | Both Any AUC | Gore AUC | NSFW AUC | 纯图 Clear AUC | 真实红字误报 (52条) | 真实二次元误报 (39条) | 60张红字压测误报 @0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Shallow Baseline（无深度视觉） | 0.9542 | 0.8870 | 0.9479 | 0.9615 | — | 1.92% | 0.00% | — |
| Distilled `MobileNetV3-Small` (2.31M) | 0.9568 | 0.9125 | 0.9826 | 0.9688 | 0.7942 | 5.77% | 2.56% | 0.00% |
| `MobileCLIP2-S0` 冻结视觉 + 融合头 | 0.9847 | 0.9449 | 0.9913 | 0.9847 | 0.9565 | **0.00%** | **0.00%** | **0.00%** |
| **`MobileCLIP2-S0` 尾部微调（最终部署）** | **0.9884** | **0.9601** | **0.9923** | **0.9917** | **0.9660** | **0.00%** | **0.00%** | **0.00%** (最高 `0.080`) |

### 3. 1.0 的 ONNX 压缩优化与 CPU 空闲基准（4 核 Intel i7-13700KF VM，无 GPU）

| ONNX 变体 | 总磁盘体积 | 图文中位延迟 | 纯文本延迟 | Clear AUC | Any AUC | 相对 FP32 平均误差 |
|---|---:|---:|---:|---:|---:|---:|
| `fp32_vis + int8_head` | 43.93 MB | 45.39 ms | 0.27 ms | 0.9883 | 0.9599 | 0.00000 |
| **`fp16w_vis + int8_head`（已部署）** | **22.36 MB** | **40.29 ms** | **0.30 ms** | **0.9881** | **0.9598** | **0.00017** |
| `static_qdq_int8_all` | 12.35 MB | 44.79 ms | 0.29 ms | 0.9417 | 0.8807 | 0.06186 |

---

## 数据来源与构建说明

- **数据来源**：训练与评估所用的 *Left 4 Dead 2* Steam 创意工坊公开条目元数据与预览图，均通过 **[Left4Addons](https://github.com/FallenCrystal/Left4Addons)** 项目提供的 `l4a-server` 目录缓存 REST API 获取，**未直接抓取 Steam 网页**。
- **隐私与合规**：本仓库**不包含**任何缓存图片、原始 Steam 用户信息或私密后端实例地址。所有采集脚本均通过环境变量或命令行参数指定自建 `l4a-server` 地址（默认 `http://127.0.0.1:8787`）。
- **两轮主动学习标注**：
  1. 分层采样 16,838 条创意工坊元数据并筛选 5,598 张预览图构建候选池。
  2. 通过两轮多模态 VLM 精标与人工抽检质控，累计构建 **1,920 条**细粒度图文标注（含 514 条真实红字/红色非血腥陷阱、446 条普通穿衣二次元陷阱、285 条视觉血腥/边界样本与 416 条视觉性暗示/边界样本）。
  3. 严格按 `creatorSteamId` 分组划分为 Train `1,510` / Val `206` / 锁定 Test `204`，杜绝同作者系列模组数据泄露。

---

## 复现训练流水线（纯 CPU 可训练）

如需基于您自建的 `l4a-server` 实例重新采集、增量标注或微调：

```bash
# 1. 安装 CPU 训练依赖
pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
pip install -r requirements-train.txt

# 2. 通过自建 l4a-server 采集分层数据（不硬编码任何线上地址）
export L4A_CATALOG_API="http://127.0.0.1:8787"
python scripts/01_collect_catalog.py --target-images 5600

# 3. 特征预筛、生成红字压测集与构建无监督多语言词表
python scripts/02_prescreen_and_sample.py --target 1440
python scripts/synth_red_text.py --count 60
python scripts/build_text_vocab.py

# 4. 合并标注后执行多架构对比训练、ONNX 导出与压缩优化
python scripts/03_merge_annotations.py --inputs "data/annotations/*.jsonl" --out data/annotations/merged_final.jsonl
python scripts/04_train_and_compare.py --labels data/annotations/merged_final.jsonl --out-dir checkpoints/exp_final --ft-epochs 3
python scripts/06_export_onnx_and_bench.py --ckpt checkpoints/exp_final/mc2s0_finetuned.pt --mode finetuned
python scripts/06b_optimize_onnx.py
```

1.1 可以在已部署的视觉编码器上重训融合头，并用额外的图片级标注训练尸体识别头。图片标注 JSONL 每行提供 `id`、`creator`、本地 `path` 和 `label`。`label` 取 `0` 表示画面无明确尸体，`1` 表示单具或分散尸体，`2` 表示成堆或重叠的多具尸体，`-1` 表示不确定。

```bash
python scripts/07_retrain_fusion.py --baseline-checkpoint checkpoints/exp_final/mc2s0_finetuned.pt
python scripts/08_train_corpse.py --annotations data/retrain/corpse_annotations_v2.jsonl \
  --initial-split data/retrain/corpse_initial_split.json \
  --legacy-features exports/retrained/fusion_features.npz
```

导出文件和评估报告保存到 `exports/retrained/`。增量加标可通过 `--initial-split` 复用上轮的 `corpse_split.json` 划分。

---

## 目录结构

```text
Left4Filters/
├── deploy/                      # 独立便携部署包（总计约 24 MB）
│   ├── vision_encoder.onnx      # 微调 MobileCLIP2-S0 视觉编码器 (21.89 MB)
│   ├── fusion_head.onnx         # INT8 门控融合 + 校准头 (478 KB)
│   ├── corpse_head.onnx         # 可见尸体与尸体堆的视觉分类头
│   ├── text_vocab.json          # 零冲突多语言 TF-IDF 词表 (48 KB)
│   ├── model_config.json        # 预处理配置与基准指标
│   ├── l4filter.py              # 单文件独立推理 SDK & CLI
│   ├── server.py                # 零框架 HTTP 微服务
│   ├── requirements.txt         # onnxruntime / numpy / pillow
│   └── README.md                # 部署速查说明
├── src/
│   ├── l4filter_features.py     # 多语言 TF-IDF、词边界特征与 24 维视觉先验
│   └── l4filter_model.py        # MultimodalFusionHead 与 ONNX 导出封装
├── scripts/                     # 采集、预筛、标注合并、训练对比、主动学习与 ONNX 导出脚本
├── requirements-train.txt       # 训练期依赖
├── LICENSE                      # MIT License (代码)
├── MODEL_LICENSE_NOTICE.md      # 预训练视觉骨干许可与数据源声明
└── README.md
```

---

## 许可证与致谢

- **代码许可证**：本仓库所有源码、脚本与部署工具均基于 **[MIT License](LICENSE)** 开源。
- **模型权重声明**：[`deploy/vision_encoder.onnx`](deploy/vision_encoder.onnx) 衍生微调自 Apple 发布的 **[MobileCLIP2-S0](https://huggingface.co/apple/MobileCLIP2-S0)**（`dfndr2b`），其上游权重受 **Apple Machine Learning Research Model License (`apple-amlr`)** 约束，详情请见 [`MODEL_LICENSE_NOTICE.md`](MODEL_LICENSE_NOTICE.md)。
- **项目致谢**：感谢 [Left4Addons](https://github.com/FallenCrystal/Left4Addons) 提供的规范化 L4D2 创意工坊目录缓存后端，以及 Apple ML Research 团队开源的 MobileCLIP2 模型。
