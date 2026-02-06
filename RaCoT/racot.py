from sentence_transformers import SentenceTransformer
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict
from prompts import (
    RACOT_SYSTEM_PROMPT,
    RACOT_USER_TEMPLATE,
    RACOT_DELTA_TEMPLATE
)

_similarity_model = SentenceTransformer("all-MiniLM-L6-v2")

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def text_similarity(text_a: str, text_b: str) -> float:
    embeddings = _similarity_model.encode(
        [text_a, text_b],
        convert_to_tensor=True
    )
    sim = F.cosine_similarity(
        embeddings[0].unsqueeze(0),
        embeddings[1].unsqueeze(0)
    ).item()
    return sim

def qwen_generate(
    prompt: str,
    system_prompt: str = RACOT_SYSTEM_PROMPT,
    max_new_tokens: int = 512
) -> str:

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer(
        [text],
        return_tensors="pt"
    ).to(model.device)
    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens
        )
    output_ids = generated_ids[0][model_inputs.input_ids.shape[1]:]
    response = tokenizer.decode(
        output_ids,
        skip_special_tokens=True
    ).strip()

    return response

def build_racot_delta_prompt(
    original_question: str,
    counterfactual_question: str,
    delta_passage: str
) -> str:

    filled_prompt = RACOT_DELTA_TEMPLATE.format(
        original_question=original_question,
        counterfactual_question=counterfactual_question,
        delta_passage=delta_passage
    )

    return filled_prompt

def RaCoT(
    original_question: str
) -> str:

    candidates: List[Dict] = []
    for _ in range(3):
        output = qwen_generate(original_question)
        if ";" not in output:
            continue
        parts = output.split(";")
        contrastive_q = parts[0].strip()
        delta_word = parts[1].strip()
        candidates.append({
            "contrastive_question": contrastive_q,
            "delta": delta_word
        })
    if len(candidates) == 0:
        return original_question
    for item in candidates:
        sim = text_similarity(
            original_question,
            item["contrastive_question"]
        )
        item["similarity"] = sim
    candidates.sort(
        key=lambda x: x["similarity"],
        reverse=True
    )
    best = candidates[0]
    filled_prompt = build_racot_delta_prompt(
        original_question=original_question,
        counterfactual_question=best["contrastive_question"],
        delta_passage=best["delta"]
    )
    return filled_prompt

def batch_run_racot(input_queries):
    new_queries = []

    for q in input_queries:
        racot_prompt = RaCoT(q)   # ← 你已经实现的函数
        new_queries.append(racot_prompt)

    return new_queries

