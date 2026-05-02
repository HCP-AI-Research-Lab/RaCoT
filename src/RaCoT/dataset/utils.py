from typing import Any, List
import numpy as np
from RaCoT.dataset import Dataset


def convert_numpy(data: Any) -> Any:
    if isinstance(data, dict):
        return {key: convert_numpy(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [convert_numpy(element) for element in data]
    elif isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, (np.integer,)):
        return int(data)
    elif isinstance(data, (np.floating,)):
        return float(data)
    elif isinstance(data, (np.bool_)):
        return bool(data)
    elif isinstance(data, (np.str_)):
        return str(data)
    else:
        return data


def filter_dataset(dataset: Dataset, filter_func=None):
    if filter_func is None:
        return dataset
    filtered_data = [item for item in dataset.data if filter_func(item)]
    return Dataset(config=dataset.config, data=filtered_data)


def split_dataset(dataset: Dataset, split_symbol: list):
    if len(split_symbol) != len(dataset):
        raise ValueError("`split_symbol` length must match dataset length.")

    data = dataset.data
    symbol_order = list(dict.fromkeys(split_symbol))
    data_split = {}
    for symbol in symbol_order:
        symbol_data = [item for item, item_symbol in zip(data, split_symbol) if item_symbol == symbol]
        data_split[symbol] = Dataset(config=dataset.config, data=symbol_data)

    return data_split


def merge_dataset(dataset_split: dict, split_symbol: list):
    if not dataset_split:
        raise ValueError("`dataset_split` must not be empty.")
    if len(split_symbol) != sum(len(data) for data in dataset_split.values()):
        raise ValueError("`split_symbol` length does not match total split dataset size.")
    dataset_split_iter = {symbol: iter(dataset.data) for symbol, dataset in dataset_split.items()}

    final_data = []
    for item_symbol in split_symbol:
        if item_symbol not in dataset_split_iter:
            raise KeyError(f"Unknown split symbol: {item_symbol}")
        final_data.append(next(dataset_split_iter[item_symbol]))
    final_dataset = Dataset(config=list(dataset_split.values())[0].config, data=final_data)

    return final_dataset


def get_batch_dataset(dataset: Dataset, batch_size=16):
    if batch_size <= 0:
        raise ValueError("`batch_size` must be positive.")
    data = dataset.data
    for idx in range(0, len(data), batch_size):
        batched_data = data[idx : idx + batch_size]
        batch_dataset = Dataset(config=dataset.config, data=batched_data)
        yield batch_dataset


def merge_batch_dataset(dataset_list: List[Dataset]):
    if len(dataset_list) == 0:
        raise ValueError("`dataset_list` must not be empty.")
    dataset = dataset_list[0]
    total_data = []
    for batch_dataset in dataset_list:
        total_data.extend(batch_dataset.data)
    dataset = Dataset(config=dataset.config, data=total_data)
    return dataset


def remove_images(data: Any) -> Any:
    from PIL import Image
    from typing import Any

    if isinstance(data, dict):
        return {
            key: remove_images(value)
            for key, value in data.items()
            if not isinstance(value, Image.Image)
        }
    elif isinstance(data, list):
        return [remove_images(element) for element in data if not isinstance(element, Image.Image)]
    elif isinstance(data, tuple):
        return tuple(
            remove_images(element) for element in data if not isinstance(element, Image.Image)
        )
    elif isinstance(data, set):
        return {remove_images(element) for element in data if not isinstance(element, Image.Image)}
    else:
        return data


def clean_prompt_image(prompt):
    if not isinstance(prompt, list):
        return prompt

    for message in prompt:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if isinstance(content, list):
            message["content"] = [
                item
                for item in content
                if not (isinstance(item, dict) and item.get("type") == "image")
            ]
    return prompt
