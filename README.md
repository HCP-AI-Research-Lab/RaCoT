# RaCoT

Official implementation for AAAI 2026 oral paper:
["RaCoT: Plug-and-Play Contrastive Example Generation Mechanism for Enhanced LLM Reasoning Reliability"](https://arxiv.org/abs/2510.22710)

Contact: `lanqinb@gmail.com`

## Overview

This repository packages RaCoT as a standalone FlashRAG-style pipeline.

It includes:

- the `SequentialPipeline` integration for RaCoT
- retriever, generator, evaluator, and config glue code
- local tooling for repository hygiene and model download

It does not include:

- pretrained retrieval model weights
- retrieval corpora
- built retrieval indexes
- bundled counterfactual generation model weights

That means the repository can run a minimal demo out of the box, but a real retrieval run still requires you to provide your own model, corpus, and index assets.

## Installation

Minimal install:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

Developer install:

```bash
pip install -e ".[dev]"
```

## What Works Right Now

These commands are currently the simplest verified entry points:

```bash
python examples/minimal_racot_demo.py
PYTHONPATH=src python -m RaCoT.tools --json
python -m build --no-isolation
```

## Quick Start

Run the minimal end-to-end demo:

```bash
python examples/minimal_racot_demo.py
```

This demo uses:

- a dummy retriever
- a dummy generator
- a real `SequentialPipeline`
- RaCoT enabled

So it verifies that the core RaCoT pipeline wiring works without requiring external models or indexes.

Expected output looks like:

```text
Question: What is the capital city of Australia?
Prediction: Canberra
RaCoT Delta: {Australia} vs {China}
Retrieved Titles: ['Canberra']
```

## Optional Local Counterfactual Model

You can optionally download a local model for the RaCoT counterfactual generation step:

```bash
make download-model
```

Custom target:

```bash
PYTHONPATH=src python -m RaCoT.tools download-model --target-dir /path/to/model_dir
```

RaCoT local model selection priority:

1. `racot_model_name_or_path` in config
2. `RACOT_LOCAL_MODEL_PATH` environment variable
3. `models/Qwen_1.5b` if it exists
4. `Qwen/Qwen2.5-1.5B-Instruct` from Hugging Face

## How To Run A Real Pipeline

To move from the minimal demo to a real run, prepare the following first:

1. A dataset directory such as `dataset/nq/test.jsonl`
2. A retrieval corpus file for `corpus_path`
3. A built retrieval index for `index_path`
4. A retrieval model path or model name for `retrieval_model_path`
5. A generation backend configuration
6. Optionally, a local RaCoT counterfactual model

The easiest practical setup is:

- retrieval: dense `e5` + FAISS index
- generation: `framework: openai`
- RaCoT: `open_racot: true`

### Step 1: Prepare Data

By default the config expects datasets under:

```text
dataset/<dataset_name>/<split>.jsonl
```

Example:

```text
dataset/nq/test.jsonl
```

Each row should contain fields such as:

```json
{"id": "1", "question": "What is the capital of Australia?", "golden_answers": ["Canberra"]}
```

### Step 2: Prepare Retriever Assets

At minimum you need to provide:

- `corpus_path`
- `index_path`
- `retrieval_model_path`

Important:

- the package contains retriever code, but does not ship an index or corpus
- `faiss-cpu` and `bm25s` are included in dependencies
- `pyserini` and `seismic` are referenced in code, but are not installed by default

### Step 3: Choose A Generator Backend

For a simple first run, use OpenAI:

- `framework: openai`
- `generator_model: gpt-4o-mini` or another compatible model
- `openai_setting.api_key: ...`

For local generation, you need to provide a working local model path in `generator_model_path`.

### Step 4: Enable RaCoT

The main RaCoT keys live in `src/RaCoT/config/basic_config.yaml`:

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

### Example Config

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

### Example Run

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

## Repository Layout

- `src/RaCoT/`: package root
- `src/RaCoT/RaCoT/`: RaCoT-specific prompts and contrastive reasoning logic
- `src/RaCoT/pipeline/`: end-to-end orchestration
- `src/RaCoT/retriever/`: retrieval and rerank
- `src/RaCoT/generator/`: generation backends
- `src/RaCoT/tools/`: quality and utility CLI
- `examples/`: minimal runnable example

## Developer Commands

```bash
make quality-gate
make preflight
make verify
make release-check
```

- `make quality-gate`: comment policy + repository hygiene
- `make preflight`: compile + demo + quality gate + optional pytest
- `make verify`: preflight + format check + package build
- `make release-check`: verify + `twine check dist/*`

## Known Limits

- This repository does not ship pretrained retrieval weights, corpora, or indexes.
- The minimal demo is the safest way to verify the package locally.
- Some optional branches, especially rarely used refiners and external retrieval backends, may require extra assets or dependencies beyond the default install.

## Citation

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
