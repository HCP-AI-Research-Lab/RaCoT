import json
import os
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .prompts import (
    RACOT_CONTRASTIVE_SYSTEM_PROMPT,
    RACOT_CONTRASTIVE_USER_TEMPLATE,
    RACOT_DELTA_TEMPLATE,
    RACOT_FILTER_SYSTEM_PROMPT,
    RACOT_FILTER_USER_TEMPLATE,
    RACOT_QUERY_REWRITE_SYSTEM_PROMPT,
    RACOT_QUERY_REWRITE_USER_TEMPLATE,
)


_MODEL = None
_TOKENIZER = None
_MODEL_FAILED = False


@dataclass
class ContrastiveSample:
    original_question: str
    contrastive_question: str
    delta: str
    similarity: float = 0.0
    raw_output: str = ""


def _default_model_path() -> str:
    env_path = os.getenv("RACOT_LOCAL_MODEL_PATH")
    if env_path and os.path.exists(env_path):
        return env_path
    repo_root = Path(__file__).resolve().parents[3]
    local_model_path = repo_root / "models" / "Qwen_1.5b"
    if local_model_path.exists():
        return str(local_model_path)
    return "Qwen/Qwen2.5-1.5B-Instruct"


def _load_model(model_name_or_path: Optional[str] = None) -> bool:
    global _MODEL, _TOKENIZER, _MODEL_FAILED
    if _MODEL is not None and _TOKENIZER is not None:
        return True
    if _MODEL_FAILED:
        return False

    model_name_or_path = model_name_or_path or _default_model_path()
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype="auto",
            device_map="auto",
        )
        _TOKENIZER = tokenizer
        _MODEL = model
        return True
    except Exception:
        _MODEL_FAILED = True
        return False


def _chat_generate(
    system_prompt: str,
    user_prompt: str,
    max_new_tokens: int = 128,
    use_llm: bool = True,
    model_name_or_path: Optional[str] = None,
) -> Optional[str]:
    if not use_llm:
        return None
    if not _load_model(model_name_or_path=model_name_or_path):
        return None

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    try:
        text = _TOKENIZER.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = _TOKENIZER([text], return_tensors="pt").to(_MODEL.device)
        with torch.no_grad():
            output_ids = _MODEL.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                pad_token_id=_TOKENIZER.eos_token_id,
            )
        gen = output_ids[0][inputs.input_ids.shape[1] :]
        return _TOKENIZER.decode(gen, skip_special_tokens=True).strip()
    except Exception:
        return None


def _first_line(text: str) -> str:
    for line in text.splitlines():
        line = line.strip()
        if line:
            return line
    return ""


def _extract_json(text: str) -> Optional[Dict]:
    text = (text or "").strip()
    if not text:
        return None
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
        if isinstance(data, dict):
            return data
    except Exception:
        return None
    return None


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9']+", (text or "").lower())


def _text_similarity(text_a: str, text_b: str) -> float:
    a = (text_a or "").strip().lower()
    b = (text_b or "").strip().lower()
    if not a or not b:
        return 0.0
    seq = SequenceMatcher(None, a, b).ratio()
    ta = set(_tokenize(a))
    tb = set(_tokenize(b))
    if not ta or not tb:
        return float(seq)
    jaccard = len(ta & tb) / max(1, len(ta | tb))
    return float(0.7 * seq + 0.3 * jaccard)


def _extract_core_subject(question: str) -> Optional[str]:
    q = (question or "").strip().rstrip("?").strip()
    if not q:
        return None

    quoted = re.search(r"\"([^\"]+)\"|'([^']+)'", q)
    if quoted:
        return (quoted.group(1) or quoted.group(2)).strip()

    tail = re.search(r"\b(?:of|in|about|for)\s+(.+)$", q, flags=re.IGNORECASE)
    if tail:
        return tail.group(1).strip()

    verb = re.search(
        r"^(?:who|what|when|where|which|how)\b.*?\b"
        r"(?:directed|wrote|invented|discovered|founded|created|published|starred in)\s+(.+)$",
        q,
        flags=re.IGNORECASE,
    )
    if verb:
        return verb.group(1).strip()
    return None


def _did_change_subject(question: str, contrastive_question: str, subject: Optional[str]) -> bool:
    if not subject:
        return False
    p = rf"\b{re .escape (subject )}\b"
    in_q = re.search(p, question, flags=re.IGNORECASE) is not None
    in_c = re.search(p, contrastive_question, flags=re.IGNORECASE) is not None
    return in_q and not in_c


def _parse_candidate(question: str, raw_output: str) -> Optional[ContrastiveSample]:
    data = _extract_json(raw_output)
    if data is not None:
        contrastive = (
            data.get("contrastive_question")
            or data.get("counterfactual_question")
            or data.get("question")
        )
        delta = data.get("delta") or data.get("key_difference") or data.get("difference")
        if contrastive:
            return ContrastiveSample(
                original_question=question,
                contrastive_question=str(contrastive).strip(),
                delta=str(delta or "").strip(),
                similarity=_text_similarity(question, str(contrastive)),
                raw_output=raw_output,
            )

    if ";" in (raw_output or ""):
        parts = [p.strip() for p in raw_output.split(";") if p.strip()]
        if len(parts) >= 2:
            return ContrastiveSample(
                original_question=question,
                contrastive_question=parts[0],
                delta=parts[1],
                similarity=_text_similarity(question, parts[0]),
                raw_output=raw_output,
            )
    return None


def _subject_substitution_fallback(question: str) -> Optional[ContrastiveSample]:
    q = (question or "").strip()
    subject = _extract_core_subject(q)
    if not subject:
        return None
    swap_map = {
        "australia": "China",
        "china": "Australia",
        "inception": "Interstellar",
        "interstellar": "Inception",
        "pride and prejudice": "War and Peace",
        "war and peace": "Pride and Prejudice",
        "telephone": "light bulb",
        "light bulb": "telephone",
    }
    replacement = swap_map.get(subject.lower())
    if replacement is None:
        q_lower = q.lower()
        if "directed" in q_lower:
            replacement = "Titanic"
        elif "wrote" in q_lower:
            replacement = "War and Peace"
        elif "capital" in q_lower:
            replacement = "China"
        else:
            return None
    contrastive = re.sub(re.escape(subject), replacement, q, flags=re.IGNORECASE, count=1)
    if contrastive.strip().lower() == q.strip().lower():
        return None
    return ContrastiveSample(
        original_question=q,
        contrastive_question=contrastive,
        delta=f"{{{subject }}} vs {{{replacement }}}",
        similarity=_text_similarity(q, contrastive),
        raw_output="[rule_fallback_subject]",
    )


def _rule_fallback(question: str) -> ContrastiveSample:
    subject_sample = _subject_substitution_fallback(question)
    if subject_sample is not None:
        return subject_sample

    q = (question or "").strip()
    q_lower = q.lower()
    rules = [
        (r"\bdirected\b", "starred in", "{directed} vs {starred in}"),
        (r"\bwrote\b", "edited", "{wrote} vs {edited}"),
        (r"\bcapital city\b", "largest city", "{capital city} vs {largest city}"),
        (r"\bcapital\b", "largest city", "{capital} vs {largest city}"),
        (r"\bfirst\b", "second", "{first} vs {second}"),
        (r"\bsecond\b", "first", "{second} vs {first}"),
    ]
    for pattern, repl, delta in rules:
        if re.search(pattern, q_lower):
            contrastive = re.sub(pattern, repl, q, flags=re.IGNORECASE, count=1)
            if contrastive.strip().lower() != q_lower:
                return ContrastiveSample(
                    original_question=q,
                    contrastive_question=contrastive,
                    delta=delta,
                    similarity=_text_similarity(q, contrastive),
                    raw_output="[rule_fallback_pattern]",
                )
    return ContrastiveSample(
        original_question=q,
        contrastive_question=q,
        delta="",
        similarity=1.0,
        raw_output="[rule_fallback_identity]",
    )


def _best_candidate(
    question: str,
    candidates: List[ContrastiveSample],
    similarity_min: float,
    similarity_max: float,
) -> ContrastiveSample:
    if not candidates:
        return _rule_fallback(question)

    subject = _extract_core_subject(question)
    center = (similarity_min + similarity_max) * 0.5

    def key(sample: ContrastiveSample) -> Tuple:
        same_question = sample.contrastive_question.strip().lower() == question.strip().lower()
        has_delta = bool(sample.delta.strip())
        in_range = similarity_min <= sample.similarity <= similarity_max
        changed_subject = _did_change_subject(question, sample.contrastive_question, subject)
        dist = abs(sample.similarity - center)
        return (
            0 if same_question else 1,
            1 if has_delta else 0,
            1 if changed_subject else 0,
            1 if in_range else 0,
            -dist,
            sample.similarity,
        )

    candidates.sort(key=key, reverse=True)
    best = candidates[0]
    if (
        best.contrastive_question.strip().lower() == question.strip().lower()
        or not best.delta.strip()
    ):
        return _rule_fallback(question)
    return best


def generate_contrastive_sample(
    question: str,
    num_candidates: int = 3,
    similarity_min: float = 0.8,
    similarity_max: float = 0.95,
    max_new_tokens: int = 128,
    use_llm: bool = True,
    model_name_or_path: Optional[str] = None,
) -> ContrastiveSample:
    hints = [
        "core subject substitution",
        "entity substitution",
        "role swap",
        "attribute swap",
        "temporal or ordinal flip",
    ]
    candidates: List[ContrastiveSample] = []

    for i in range(max(1, num_candidates)):
        user_prompt = RACOT_CONTRASTIVE_USER_TEMPLATE.format(
            question=question,
            operation_hint=hints[i % len(hints)],
        )
        raw = _chat_generate(
            RACOT_CONTRASTIVE_SYSTEM_PROMPT,
            user_prompt,
            max_new_tokens=max_new_tokens,
            use_llm=use_llm,
            model_name_or_path=model_name_or_path,
        )
        if not raw:
            continue
        sample = _parse_candidate(question, raw)
        if sample is not None:
            candidates.append(sample)

    return _best_candidate(
        question=question,
        candidates=candidates,
        similarity_min=similarity_min,
        similarity_max=similarity_max,
    )


def build_enhanced_query(
    question: str,
    contrastive_question: str,
    delta: str,
    max_new_tokens: int = 128,
    use_llm: bool = True,
    model_name_or_path: Optional[str] = None,
) -> str:
    user_prompt = RACOT_QUERY_REWRITE_USER_TEMPLATE.format(
        target_question=question,
        contrastive_question=contrastive_question,
        delta=delta or "None",
    )
    raw = _chat_generate(
        RACOT_QUERY_REWRITE_SYSTEM_PROMPT,
        user_prompt,
        max_new_tokens=max_new_tokens,
        use_llm=use_llm,
        model_name_or_path=model_name_or_path,
    )
    if raw:
        line = _first_line(raw)
        if line:
            return line
    if delta:
        pos, _ = _parse_delta_pair(delta)
        if pos:
            return f"{question } Focus on: {pos }."
        return f"{question } Focus on: {delta }."
    return question


def _safe_doc_text(doc: Dict) -> str:
    text = doc.get("text")
    if text:
        return str(text)
    contents = doc.get("contents")
    if contents:
        parts = str(contents).split("\n")
        return "\n".join(parts[1:]).strip() if len(parts) > 1 else str(contents)
    return ""


def _parse_score(raw_text: Optional[str]) -> Optional[float]:
    if not raw_text:
        return None
    strict = re.search(r"(?<!\d)(?:0(?:\.\d+)?|1(?:\.0+)?)", raw_text)
    if strict:
        return float(strict.group(0))
    relaxed = re.search(r"-?\d+(?:\.\d+)?", raw_text)
    if not relaxed:
        return None
    val = float(relaxed.group(0))
    return float(max(0.0, min(1.0, val)))


def _parse_delta_pair(delta: str) -> Tuple[Optional[str], Optional[str]]:
    if not delta:
        return None, None
    m = re.search(r"\{([^{}]+)\}\s*vs\s*\{([^{}]+)\}", delta, flags=re.IGNORECASE)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    parts = [p.strip() for p in re.split(r"\bvs\b", delta, flags=re.IGNORECASE) if p.strip()]
    if len(parts) >= 2:
        return parts[0], parts[1]
    return None, None


def _fallback_doc_score(question: str, delta: str, doc: Dict) -> float:
    pos, neg = _parse_delta_pair(delta)
    if pos:
        query = f"{question } {pos }".strip()
    else:
        query = f"{question } {delta }".strip()
    doc_text = f"{doc .get ('title','')} {_safe_doc_text (doc )}".strip()
    doc_lower = doc_text.lower()
    pos_score = _text_similarity(query, doc_text)
    if pos and pos.lower() in doc_lower:
        pos_score += 0.35
    if not neg:
        return float(max(0.0, min(1.0, pos_score)))
    neg_score = _text_similarity(neg, doc_text)
    if neg.lower() in doc_lower:
        neg_score += 0.35
    calibrated = pos_score - 0.35 * neg_score + 0.15
    return float(max(0.0, min(1.0, calibrated)))


def filter_documents(
    question: str,
    delta: str,
    documents: List[Dict],
    threshold: float = 0.7,
    enable_filtering: bool = True,
    use_llm: bool = True,
    model_name_or_path: Optional[str] = None,
) -> List[Dict]:
    if not documents:
        return []
    if not enable_filtering:
        kept = []
        for doc in documents:
            new_doc = dict(doc)
            new_doc["racot_score"] = 1.0
            kept.append(new_doc)
        return kept

    scored_docs = []
    for doc in documents:
        user_prompt = RACOT_FILTER_USER_TEMPLATE.format(
            target_question=question,
            delta=delta or "None",
            title=doc.get("title", ""),
            text=_safe_doc_text(doc),
        )
        raw = _chat_generate(
            RACOT_FILTER_SYSTEM_PROMPT,
            user_prompt,
            max_new_tokens=16,
            use_llm=use_llm,
            model_name_or_path=model_name_or_path,
        )
        score = _parse_score(raw)
        if score is None:
            score = _fallback_doc_score(question, delta, doc)
        new_doc = dict(doc)
        new_doc["racot_score"] = float(score)
        if "score" in new_doc:
            new_doc["origin_score"] = new_doc["score"]
        new_doc["score"] = float(score)
        scored_docs.append(new_doc)

    kept = [doc for doc in scored_docs if doc["racot_score"] >= threshold]
    if kept:
        return kept
    scored_docs.sort(key=lambda x: x["racot_score"], reverse=True)
    return [scored_docs[0]]


def prepare_racot_record(
    question: str,
    num_candidates: int = 3,
    similarity_min: float = 0.8,
    similarity_max: float = 0.95,
    max_new_tokens: int = 128,
    use_llm: bool = True,
    model_name_or_path: Optional[str] = None,
) -> Dict:
    sample = generate_contrastive_sample(
        question=question,
        num_candidates=num_candidates,
        similarity_min=similarity_min,
        similarity_max=similarity_max,
        max_new_tokens=max_new_tokens,
        use_llm=use_llm,
        model_name_or_path=model_name_or_path,
    )
    enhanced_query = build_enhanced_query(
        question=question,
        contrastive_question=sample.contrastive_question,
        delta=sample.delta,
        max_new_tokens=max_new_tokens,
        use_llm=use_llm,
        model_name_or_path=model_name_or_path,
    )
    return {
        "original_question": sample.original_question,
        "contrastive_question": sample.contrastive_question,
        "delta": sample.delta,
        "similarity": float(sample.similarity),
        "enhanced_query": enhanced_query,
        "raw_output": sample.raw_output,
    }


def batch_prepare_racot(
    input_queries: List[str],
    num_candidates: int = 3,
    similarity_min: float = 0.8,
    similarity_max: float = 0.95,
    max_new_tokens: int = 128,
    use_llm: bool = True,
    model_name_or_path: Optional[str] = None,
) -> List[Dict]:
    outputs = []
    for question in input_queries:
        outputs.append(
            prepare_racot_record(
                question=question,
                num_candidates=num_candidates,
                similarity_min=similarity_min,
                similarity_max=similarity_max,
                max_new_tokens=max_new_tokens,
                use_llm=use_llm,
                model_name_or_path=model_name_or_path,
            )
        )
    return outputs


def filter_retrieval_results(
    racot_records: List[Dict],
    retrieval_results: List[List[Dict]],
    threshold: float = 0.7,
    enable_filtering: bool = True,
    use_llm: bool = True,
    model_name_or_path: Optional[str] = None,
) -> List[List[Dict]]:
    filtered = []
    for rec, docs in zip(racot_records, retrieval_results):
        filtered_docs = filter_documents(
            question=rec.get("original_question", ""),
            delta=rec.get("delta", ""),
            documents=docs,
            threshold=threshold,
            enable_filtering=enable_filtering,
            use_llm=use_llm,
            model_name_or_path=model_name_or_path,
        )
        rec["num_candidates"] = len(docs)
        rec["num_selected"] = len(filtered_docs)
        rec["filter_threshold"] = float(threshold)
        filtered.append(filtered_docs)
    return filtered


def build_generation_question(
    original_question: str, racot_record: Dict, inject_delta: bool = True
) -> str:
    if not inject_delta:
        return original_question
    delta = (racot_record or {}).get("delta", "").strip()
    contrastive = (racot_record or {}).get("contrastive_question", "").strip()
    if not delta:
        return original_question
    return (
        f"{original_question }\n"
        f"[RaCoT Delta] {delta }\n"
        f"[RaCoT Contrastive Question] {contrastive }"
    )


def build_racot_delta_prompt(
    original_question: str,
    counterfactual_question: str,
    delta_passage: str,
) -> str:
    return RACOT_DELTA_TEMPLATE.format(
        original_question=original_question,
        counterfactual_question=counterfactual_question,
        delta_passage=delta_passage,
    )


def racot(
    original_question: str,
    num_candidates: int = 3,
    similarity_min: float = 0.8,
    similarity_max: float = 0.95,
    max_new_tokens: int = 128,
    use_llm: bool = True,
    model_name_or_path: Optional[str] = None,
) -> str:
    rec = prepare_racot_record(
        question=original_question,
        num_candidates=num_candidates,
        similarity_min=similarity_min,
        similarity_max=similarity_max,
        max_new_tokens=max_new_tokens,
        use_llm=use_llm,
        model_name_or_path=model_name_or_path,
    )
    return build_racot_delta_prompt(
        original_question=original_question,
        counterfactual_question=rec["contrastive_question"],
        delta_passage=rec["delta"],
    )


def batch_run_racot(
    input_queries: List[str],
    num_candidates: int = 3,
    similarity_min: float = 0.8,
    similarity_max: float = 0.95,
    max_new_tokens: int = 128,
    use_llm: bool = True,
    model_name_or_path: Optional[str] = None,
) -> List[str]:
    records = batch_prepare_racot(
        input_queries=input_queries,
        num_candidates=num_candidates,
        similarity_min=similarity_min,
        similarity_max=similarity_max,
        max_new_tokens=max_new_tokens,
        use_llm=use_llm,
        model_name_or_path=model_name_or_path,
    )
    return [rec["enhanced_query"] for rec in records]
