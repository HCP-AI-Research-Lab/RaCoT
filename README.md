# RaCoT: Plug-and-Play Contrastive Example Generation Mechanism for Enhanced LLM Reasoning Reliability
> Official code for the AAAI 2026 oral paper [“RaCoT: Plug-and-Play Contrastive Example Generation Mechanism for Enhanced LLM Reasoning Reliability”](https://arxiv.org/abs/2510.22710).  
<div align="center">
<img src="img/434981.png" width="1000">
</div>

[English](README.md)

## Abstract
Retrieval-Augmented Generation (RAG) faces a core bottleneck with knowledge-sparse and semantically ambiguous long-tail queries, where retrieval noise distorts reasoning and necessitates costly post-processing. To tackle this, we propose RaCoT (Retrieval-aware Contrastive-of-Thought), a novel framework that shifts contrastive thinking to the pre-retrieval stage. By automatically generating a semantically adjacent yet differently answered contrastive question and extracting a Δ-Prompt to capture their key differences, RaCoT guides the model to proactively focus on the "critical details that determine answer divergence." This approach allows it to suppress semantic interference within a single retrieval pass, overcoming the theoretical bottleneck of single-vector queries that struggle to simultaneously encode signals for what to attend to and what to ignore. On six authoritative benchmarks, including PopQA and TriviaQA-unfiltered, RaCoT outperforms strong baselines like RankRAG and Self-RAG by 0.9-2.4 percentage points. It exhibits superior robustness, with a performance drop of only 8.6% in adversarial tests, far surpassing the over 15\% degradation in other methods. Furthermore, its low latency (3.12s) and token overhead (11.54) place it on the accuracy-efficiency Pareto frontier, while ablation studies validate the necessity of each component. Ultimately, RaCoT reframes the RAG paradigm from ``post-hoc context cleaning" to "a priori shaping of discriminative reasoning", offering an efficient and robust path toward reliable AI systems for real-time, resource-constrained deployments.

## :mag_right: Roadmap
We have ported the RaCoT framework to the widely recognized and followed [FlashRAG](https://github.com/RUC-NLPIR/FlashRAG/tree/main) framework for ease of evaluation! RaCoT is still under active development—please contact me (Kaitong Cai) for any code-related issues!
- [ ] Adapted to more RAG methodology frameworks
- [ ] Counterfactual-generated question fine-tuning data
- [ ] Adapted to more evaluation datasets


## :wrench: Installation
1.Please first download [Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/tree/main) from the official Hugging Face (HF) website and place it in the RaCoT folder.This step is to download and deploy the counterfactual generation model, and you may also freely choose any model you prefer.
   ```bash
    uvx hf download Qwen/Qwen2.5-1.5B-Instruct
   ```

2.We follow the procedure of FlashRAG to set up the environment, and you may:

```base
pip install flashrag-dev --pre
```
or

```bash
git clone https://github.com/RUC-NLPIR/FlashRAG.git
cd FlashRAG
pip install -e .
```

3.To use RaCoT, you need to download the additional dependencies:
```bash
# Install sentence-transformers
pip install sentence-transformers
```：

## :rocket: Quick Start
1.Please follow the official documentation of FlashRAG to complete the setup of the retrieval dataset.

2.With RaCoT, you can choose whether to enable RaCoT in the file flashrag/config/basic_config.yaml.

```bash
open_racot: False
```

4.Using the ready-made pipeline

We use the official pipeline (specifically SequentialPipeline) to implement the RAG workflow. In this case, you only need to configure the config and load the corresponding pipeline. Additionally, we would like to emphasize that SequentialPipeline is a basic pipeline, and you can easily adapt it to different methods through simple migration.
First, load the configuration for the entire workflow, which records various hyperparameters required for the RAG process. You can input a YAML file as a parameter, or directly input it as a variable.

```python
from flashrag.config import Config

# hybrid load configs
config_dict = {'data_dir': 'dataset/'}
my_config = Config(
    config_file_path = 'my_config.yaml',
    config_dict = config_dict
```

Next, load the corresponding dataset and initialize the pipeline, where all required components will be automatically instantiated and configured.

```python
from flashrag.utils import get_dataset
from flashrag.pipeline import SequentialPipeline
from flashrag.prompt import PromptTemplate
from flashrag.config import Config

config_dict = {'data_dir': 'dataset/'}
my_config = Config(
    config_file_path = 'my_config.yaml',
    config_dict = config_dict
)
all_split = get_dataset(my_config)
test_data = all_split['test']

pipeline = SequentialPipeline(my_config)
```

You can specify your own input prompt using `PromptTemplete`:

```python
prompt_templete = PromptTemplate(
    config,
    system_prompt = "Answer the question based on the given document. Only give me the answer and do not output any other words.\nThe following are given documents.\n\n{reference}",
    user_prompt = "Question: {question}\nAnswer:"
)
pipeline = SequentialPipeline(
  my_config,
  prompt_template = prompt_templete
)
```

Finally, execute `pipeline.run` to obtain the final result.
```python
output_dataset = pipeline.run(test_data, do_eval=True)
```

The output_dataset stores per-item intermediate results and metric scores. If save_intermediate_data and save_metric_score are enabled, the intermediate outputs and overall evaluation score are also saved to disk.


## Citation

If you find this project useful, please cite:
```bibtex
@misc{cai2025racotplugandplaycontrastiveexample,
      title={RaCoT: Plug-and-Play Contrastive Example Generation Mechanism for Enhanced LLM Reasoning Reliability}, 
      author={Kaitong Cai and Jusheng Zhang and Yijia Fan and Jing Yang and Keze Wang},
      year={2025},
      eprint={2510.22710},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2510.22710}, 
}
