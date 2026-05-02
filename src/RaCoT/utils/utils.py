import logging
import os
import re
import json
import importlib
import warnings
from transformers import AutoConfig
from RaCoT.dataset.dataset import Dataset
import torch

SUPPORTED_DATASET_FILES = ("jsonl", "json", "parquet")
SPLITS_WITH_SAMPLING = {"test", "val", "dev", "train"}
LOGGER = logging.getLogger(__name__)


def _resolve_split_path(dataset_path: str, split: str):
    for file_postfix in SUPPORTED_DATASET_FILES:
        candidate_path = os.path.join(dataset_path, f"{split}.{file_postfix}")
        if os.path.exists(candidate_path):
            return candidate_path
    return None


def get_dataset(config):
    """Load dataset from config."""
    dataset_path = config["dataset_path"]
    all_split = config["split"]

    split_dict = {split: None for split in all_split}

    for split in all_split:
        split_path = _resolve_split_path(dataset_path, split)
        if split_path is None:
            continue
        LOGGER.info("Loading %s dataset from: %s", split, split_path)
        if split in SPLITS_WITH_SAMPLING:
            split_dict[split] = Dataset(
                config,
                split_path,
                sample_num=config["test_sample_num"],
                random_sample=config["random_sample"],
            )
        else:
            split_dict[split] = Dataset(config, split_path)

    return split_dict


def get_generator(config, **params):
    """Automatically select generator class based on config."""

    if config["framework"] == "openai":
        return getattr(importlib.import_module("RaCoT.generator"), "OpenaiGenerator")(
            config, **params
        )

    with open(os.path.join(config["generator_model_path"], "config.json"), "r") as f:
        model_config = json.load(f)
    arch = model_config["architectures"][0]
    if all(["vision" not in key for key in model_config.keys()]):
        is_mm = False
    else:
        is_mm = True

    if is_mm:
        return getattr(importlib.import_module("RaCoT.generator"), "HFMultiModalGenerator")(
            config, **params
        )
    else:
        if config["framework"] == "vllm":
            return getattr(importlib.import_module("RaCoT.generator"), "VLLMGenerator")(
                config, **params
            )
        elif config["framework"] == "fschat":
            return getattr(importlib.import_module("RaCoT.generator"), "FastChatGenerator")(
                config, **params
            )
        elif config["framework"] == "hf":
            if "t5" in arch.lower() or "bart" in arch.lower():
                return getattr(
                    importlib.import_module("RaCoT.generator"),
                    "EncoderDecoderGenerator",
                )(config, **params)
            else:
                return getattr(
                    importlib.import_module("RaCoT.generator"), "HFCausalLMGenerator"
                )(config, **params)
        else:
            raise NotImplementedError


def get_retriever(config):
    r"""Automatically select retriever class based on config's retrieval method

    Args:
        config (dict): configuration with 'retrieval_method' key

    Returns:
        Retriever: retriever instance
    """
    if config["use_multi_retriever"]:
        return getattr(importlib.import_module("RaCoT.retriever"), "MultiRetrieverRouter")(
            config
        )

    if config["retrieval_method"] == "bm25":
        return getattr(importlib.import_module("RaCoT.retriever"), "BM25Retriever")(config)
    elif config["retrieval_method"] == "splade":
        return getattr(importlib.import_module("RaCoT.retriever"), "SparseRetriever")(config)
    else:
        try:
            model_config = AutoConfig.from_pretrained(config["retrieval_model_path"])
            arch = model_config.architectures[0]
            if "clip" in arch.lower():
                return getattr(
                    importlib.import_module("RaCoT.retriever"), "MultiModalRetriever"
                )(config)
            else:
                return getattr(importlib.import_module("RaCoT.retriever"), "DenseRetriever")(
                    config
                )
        except Exception:
            return getattr(importlib.import_module("RaCoT.retriever"), "DenseRetriever")(config)


def get_reranker(config):
    model_path = config["rerank_model_path"]
    model_config = AutoConfig.from_pretrained(model_path)
    arch = model_config.architectures[0]
    if "forsequenceclassification" in arch.lower():
        return getattr(importlib.import_module("RaCoT.retriever"), "CrossReranker")(config)
    else:
        return getattr(importlib.import_module("RaCoT.retriever"), "BiReranker")(config)


def get_judger(config):
    judger_name = config["judger_name"]
    if "skr" in judger_name.lower():
        return getattr(importlib.import_module("RaCoT.judger"), "SKRJudger")(config)
    elif "adaptive" in judger_name.lower():
        return getattr(importlib.import_module("RaCoT.judger"), "AdaptiveJudger")(config)
    else:
        raise ValueError(f"No judger implementation found for `{judger_name}`.")


def get_refiner(config, retriever=None, generator=None):
    DEFAULT_PATH_DICT = {
        "recomp_abstractive_nq": "fangyuan/nq_abstractive_compressor",
        "recomp:abstractive_tqa": "fangyuan/tqa_abstractive_compressor",
        "recomp:abstractive_hotpotqa": "fangyuan/hotpotqa_abstractive",
    }
    REFINER_MODULE = importlib.import_module("RaCoT.refiner")

    refiner_name = config["refiner_name"]
    if refiner_name is None:
        raise ValueError("`refiner_name` must not be None.")
    refiner_path = (
        config["refiner_model_path"]
        if config["refiner_model_path"] is not None
        else DEFAULT_PATH_DICT.get(refiner_name, None)
    )

    try:
        model_config = AutoConfig.from_pretrained(refiner_path)
        arch = model_config.architectures[0].lower()
    except Exception as e:
        warnings.warn(f"Failed to load refiner config from `{refiner_path}`: {e}")
        model_config, arch = "", ""

    if "recomp" in refiner_name:
        if model_config.model_type == "t5":
            refiner_class = "AbstractiveRecompRefiner"
        else:
            refiner_class = "ExtractiveRefiner"
    elif "bert" in arch:
        refiner_class = "ExtractiveRefiner"
    elif "t5" in arch or "bart" in arch:
        refiner_class = "AbstractiveRecompRefiner"
    elif "lingua" in refiner_name:
        refiner_class = "LLMLinguaRefiner"
    elif "selective-context" in refiner_name or "sc" in refiner_name:
        refiner_class = "SelectiveContextRefiner"
    elif "kg-trace" in refiner_name:
        return getattr(REFINER_MODULE, "KGTraceRefiner")(config, retriever, generator)
    else:
        raise ValueError("No implementation!")

    return getattr(REFINER_MODULE, refiner_class)(config)


def hash_object(o) -> str:
    """Returns a character hash code of arbitrary Python objects."""
    import hashlib
    import io
    import dill
    import base58

    m = hashlib.blake2b()
    with io.BytesIO() as buffer:
        dill.dump(o, buffer)
        m.update(buffer.getbuffer())
        return base58.b58encode(m.digest()).decode()


def extract_between(text: str, start_tag: str, end_tag: str):
    pattern = re.escape(start_tag) + r"(.*?)" + re.escape(end_tag)
    matches = re.findall(pattern, text, flags=re.DOTALL)
    if matches:
        return matches[-1].strip()
    return None


def extract_between_all(text: str, start_tag: str, end_tag: str):
    pattern = re.escape(start_tag) + r"(.*?)" + re.escape(end_tag)
    matches = re.findall(pattern, text, flags=re.DOTALL)
    if matches:
        return matches
    return None


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"
