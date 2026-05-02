# RaCoT

AAAI 2026 oral 论文官方实现：
["RaCoT: Plug-and-Play Contrastive Example Generation Mechanism for Enhanced LLM Reasoning Reliability"](https://arxiv.org/abs/2510.22710)

联系方式：`lanqinb@gmail.com`

## 项目简介

本仓库将 RaCoT 封装为一个独立可运行的、类 FlashRAG 风格的检索增强推理流水线。

当前仓库包含：

- 集成了 RaCoT 的 `SequentialPipeline`
- 检索器、生成器、评估器、配置系统等基础代码
- 本地工具命令，如仓库检查、模型下载等

当前仓库不包含：

- 预训练检索模型权重
- 检索语料库
- 已构建好的检索索引
- 反事实生成模型权重

因此，这个仓库可以直接跑一个最小 demo，但如果你想跑真实的 retrieval + generation 流程，还需要自己准备模型、语料和索引。

## 安装

最小安装：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

开发安装：

```bash
pip install -e ".[dev]"
```

## 当前已经验证可运行的入口

下面这几个命令已经实际验证通过：

```bash
python examples/minimal_racot_demo.py
PYTHONPATH=src python -m RaCoT.tools --json
python -m build --no-isolation
```

## 快速开始

运行最小端到端 demo：

```bash
python examples/minimal_racot_demo.py
```

这个 demo 使用的是：

- 一个假的 retriever
- 一个假的 generator
- 真正的 `SequentialPipeline`
- 开启了 RaCoT

它的作用是验证：在不依赖外部模型和索引的前提下，核心 RaCoT 流水线已经能跑通。

预期输出类似：

```text
Question: What is the capital city of Australia?
Prediction: Canberra
RaCoT Delta: {Australia} vs {China}
Retrieved Titles: ['Canberra', 'Sydney']
```

## 可选的本地反事实模型

你可以选择下载一个本地模型，用于 RaCoT 的反事实生成步骤：

```bash
make download-model
```

自定义下载目录：

```bash
PYTHONPATH=src python -m RaCoT.tools download-model --target-dir /path/to/model_dir
```

RaCoT 本地模型的选择优先级为：

1. 配置中的 `racot_model_name_or_path`
2. 环境变量 `RACOT_LOCAL_MODEL_PATH`
3. `models/Qwen_1.5b`（如果存在）
4. Hugging Face 上的 `Qwen/Qwen2.5-1.5B-Instruct`

## 如何运行真实流程

如果你想从最小 demo 切到真实的 retrieval + RaCoT 流程，至少要准备以下内容：

1. 一个数据集目录，例如 `dataset/nq/test.jsonl`
2. 一个检索语料文件，对应 `corpus_path`
3. 一个已构建好的检索索引，对应 `index_path`
4. 一个检索模型路径或模型名，对应 `retrieval_model_path`
5. 一个可用的生成后端配置
6. 可选的本地 RaCoT 反事实生成模型

一个最容易落地的起步方案是：

- 检索：`e5` + FAISS
- 生成：`framework: openai`
- RaCoT：`open_racot: true`

### 第一步：准备数据

默认情况下，配置会按下面的目录结构寻找数据：

```text
dataset/<dataset_name>/<split>.jsonl
```

例如：

```text
dataset/nq/test.jsonl
```

每一行数据至少应包含类似字段：

```json
{"id": "1", "question": "What is the capital of Australia?", "golden_answers": ["Canberra"]}
```

### 第二步：准备检索资产

至少需要提供：

- `corpus_path`
- `index_path`
- `retrieval_model_path`

需要特别注意：

- 包里有检索代码，但没有附带现成索引和语料
- 默认依赖里已经包含 `faiss-cpu` 和 `bm25s`
- 代码里虽然支持 `pyserini` 和 `seismic`，但它们不在默认安装依赖中

### 第三步：选择生成后端

如果你想最简单地先跑起来，建议先用 OpenAI：

- `framework: openai`
- `generator_model: gpt-4o-mini` 或其他兼容模型
- `openai_setting.api_key: ...`

如果你想本地生成，则需要自己提供可用的 `generator_model_path`。

### 第四步：开启 RaCoT

RaCoT 相关核心配置位于：

`src/RaCoT/config/basic_config.yaml`

其中最关键的开关包括：

- `open_racot`
- `racot_num_candidates`
- `racot_similarity_min`
- `racot_similarity_max`
- `racot_filter_threshold`
- `racot_enable_filtering`
- `racot_inject_delta_to_generation`
- `racot_use_llm`
- `racot_model_name_or_path`
- `racot_max_new_tokens`

### 配置示例

```yaml
data_dir: "dataset/"
dataset_name: "nq"
split: ["test"]

retrieval_method: "e5"
retrieval_model_path: "intfloat/e5-base-v2"
corpus_path: "/path/to/corpus.jsonl"
index_path: "/path/to/e5.index"
retrieval_topk: 5

framework: "openai"
generator_model: "gpt-4o-mini"
openai_setting:
  api_key: "YOUR_API_KEY"
  base_url: ~

open_racot: true
racot_use_llm: true
racot_model_name_or_path: ~
racot_enable_filtering: true
racot_filter_threshold: 0.7
```

### 程序化运行示例

```python
from RaCoT.config import Config
from RaCoT.pipeline import SequentialPipeline
from RaCoT.utils import get_dataset

config = Config(config_file_path="my_config.yaml")
all_split = get_dataset(config)
test_data = all_split["test"]

pipeline = SequentialPipeline(config)
output_dataset = pipeline.run(test_data, do_eval=True)
```

## 仓库结构

- `src/RaCoT/`：包根目录
- `src/RaCoT/RaCoT/`：RaCoT 专属 prompts 与对比推理逻辑
- `src/RaCoT/pipeline/`：端到端流水线编排
- `src/RaCoT/retriever/`：检索与 rerank
- `src/RaCoT/generator/`：生成后端
- `src/RaCoT/tools/`：工具命令
- `examples/`：最小可运行示例

## 开发命令

```bash
make quality-gate
make preflight
make verify
make release-check
```

- `make quality-gate`：执行注释策略与仓库卫生检查
- `make preflight`：编译、demo、质量检查、可选 pytest
- `make verify`：在 `preflight` 基础上追加格式检查和打包
- `make release-check`：在 `verify` 基础上追加 `twine check dist/*`

## 当前限制

- 仓库不自带检索模型权重、检索语料或索引文件。
- 最小 demo 是当前最可靠的本地验证入口。
- 某些可选分支，尤其是少用的 refiner 或外部检索后端，可能还需要额外依赖或额外资源。

## 引用

```bibtex
@misc{cai2025racotplugandplaycontrastiveexample,
  title={RaCoT: Plug-and-Play Contrastive Example Generation Mechanism for Enhanced LLM Reasoning Reliability},
  author={Kaitong Cai and Jusheng Zhang and Yijia Fan and Jing Yang and Keze Wang},
  year={2025},
  eprint={2510.22710},
  archivePrefix={arXiv},
  primaryClass={cs.AI},
  url={https://arxiv.org/abs/2510.22710}
}
```
