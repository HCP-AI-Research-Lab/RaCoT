"""Prompt templates for the RaCoT four-stage flow."""

RACOT_CONTRASTIVE_SYSTEM_PROMPT = """
You are an expert contrastive-question constructor for retrieval-augmented QA.
Create ONE contrastive question that keeps the same question frame as the target
question but changes one core semantic slot so the answer should differ.

Return exactly one JSON object with keys:
- original_question
- contrastive_question
- delta

Hard constraints:
1. Do not copy the original question verbatim.
2. Keep lexical overlap high and preserve the question frame.
3. Change exactly one core semantic slot (entity/role/attribute/time/condition).
4. Keep the same WH form when possible.
5. Prefer replacing the core subject/entity when obvious.
6. delta must explicitly describe what changed.
""".strip()

RACOT_CONTRASTIVE_USER_TEMPLATE = """
Original Question:
{question}

Preferred Operation:
{operation_hint}

Output one JSON object only.
""".strip()

RACOT_QUERY_REWRITE_SYSTEM_PROMPT = """
You are a retrieval intent optimizer.
Given a target question, a contrastive question, and delta, write one concise
enhanced retrieval query that emphasizes what to retrieve for the target and
what confusing branch should be avoided.

Output one single line only.
""".strip()

RACOT_QUERY_REWRITE_USER_TEMPLATE = """
Target Question:
{target_question}

Contrastive Question:
{contrastive_question}

Delta:
{delta}

Enhanced Retrieval Query:
""".strip()

RACOT_FILTER_SYSTEM_PROMPT = """
You are a relevance scorer for retrieval filtering.
Given target question, delta, and one document, output one float score in [0, 1].
Higher means more useful for answering the target question while respecting delta.
Output only the number.
""".strip()

RACOT_FILTER_USER_TEMPLATE = """
Target Question:
{target_question}

Delta:
{delta}

Document:
Title: {title}
Text: {text}

Relevance Score [0,1]:
""".strip()

RACOT_DELTA_TEMPLATE = """
Original Question:
{original_question}

Contrastive Question:
{counterfactual_question}

Delta:
{delta_passage}
""".strip()
