import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_ROOT = os.path.join(PROJECT_ROOT, "src")
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from RaCoT.dataset.dataset import Dataset
from RaCoT.pipeline import SequentialPipeline


class DummyRetriever:
    def batch_search(self, queries):
        docs = [
            {
                "id": "d1",
                "title": "Canberra",
                "text": "Canberra is the capital city of Australia.",
                "contents": "Canberra\nCanberra is the capital city of Australia.",
            },
            {
                "id": "d2",
                "title": "Beijing",
                "text": "Beijing is the capital city of China.",
                "contents": "Beijing\nBeijing is the capital city of China.",
            },
            {
                "id": "d3",
                "title": "Sydney",
                "text": "Sydney is the largest city in Australia.",
                "contents": "Sydney\nSydney is the largest city in Australia.",
            },
        ]
        return [docs for _ in queries]


class DummyGenerator:
    def generate(self, prompts):
        outputs = []
        for prompt in prompts:
            text = prompt if isinstance(prompt, str) else str(prompt)
            outputs.append("Canberra" if "capital city of Australia" in text else "Unknown")
        return outputs


def main():
    config = {
        "device": "cpu",
        "save_retrieval_cache": False,
        "save_dir": "/tmp/racot_minimal",
        "save_metric_score": False,
        "save_intermediate_data": False,
        "metrics": [],
        "use_fid": False,
        "refiner_name": None,
        "framework": "openai",
        "generator_model": "gpt-4o-mini",
        "generator_max_input_len": 1024,
        "open_racot": True,
        "racot_use_llm": False,
        "racot_enable_filtering": True,
        "racot_filter_threshold": 0.7,
        "racot_num_candidates": 3,
        "racot_similarity_min": 0.8,
        "racot_similarity_max": 0.95,
        "racot_inject_delta_to_generation": True,
        "racot_max_new_tokens": 64,
    }

    pipeline = SequentialPipeline(config, retriever=DummyRetriever(), generator=DummyGenerator())
    data = Dataset(
        config=config,
        data=[
            {
                "id": "1",
                "question": "What is the capital city of Australia?",
                "golden_answers": ["Canberra"],
            }
        ],
    )

    output = pipeline.run(data, do_eval=False)
    item = output[0]
    print("Question:", item.question)
    print("Prediction:", item.pred)
    print("RaCoT Delta:", item.racot.get("delta", ""))
    print("Retrieved Titles:", [doc.get("title", "") for doc in item.retrieval_result])


if __name__ == "__main__":
    main()
