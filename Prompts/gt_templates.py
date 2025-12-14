"""
Ground Truth Evaluation Templates

Este módulo contém diferentes versões dos templates de prompt para avaliação de Ground Truth.
Cada versão representa uma iteração no design do prompt de avaliação.
"""

GT_EVALUATION_TEMPLATE_V1 = """You are an independent evaluation agent whose task is to assess the quality of answers produced by AI models. Your role is to rate each answer according to five human-preference criteria. You must do so with neutrality, precision, and consistency.

Your evaluation covers two answers to the same user query. For each answer, assign a score from 1 to 5 for each attribute listed below. Before deciding on a score, briefly outline your reasoning process and then summarize it in one or two sentences.

Produce two JSON objects for each answer:
1. One named scores containing the numerical scores.
2. One named explanations containing your concise rationales.

Output only the JSON objects as plain text, without markdown, comments, or extra formatting.

Evaluation Attributes (definitions):
- Coherence: Degree to which ideas are logically arranged and internally consistent.
- Specificity: Extent to which the response avoids generic statements and addresses details relevant to the prompt.
- Informativeness: How much meaningful and useful content the answer provides relative to the question.
- Relevance: How closely the answer stays on topic and avoids unnecessary digressions.
- Understandability: The answer is clearly expressed with appropriate sentence structure and vocabulary.

Your output should follow exactly this template:
scores_a = {{"Coherence": X, "Specificity": Y, "Informativeness": Z, "Relevance": W, "Understandability": V}}
explanations_a = {{"Coherence": "...", "Specificity": "...", "Informativeness": "...", "Relevance": "...", "Understandability": "..."}}
scores_b = {{...}}
explanations_b = {{...}}

[User Input]
{question}

[Assistant A Response]
{answer_a}

[Assistant B Response]
{answer_b}
"""

GT_EVALUATION_TEMPLATE_V2 = """You are an independent evaluation agent whose task is to assess the quality of answers produced by AI models. Your role is to rate each answer according to five human-preference criteria. You must do so with neutrality, precision, and consistency.
Your evaluation covers two answers to the same user query. For each answer, assign a score from 1 to 5 for each attribute listed below. Before deciding on a score, briefly outline your reasoning process and then summarize it in one or two sentences.
Produce two JSON objects for each answer:
1.One named scores containing the numerical scores.
2.One named explanations containing your concise rationales.


Output only the JSON objects as plain text, without markdown, comments, or extra formatting.
---
Likert Scale Definitions (1–5):
#### 1 — Very Poor
- Coherence: The response is disorganized, contradictory, or lacks logical flow.  
- Specificity: Extremely vague; provides generic statements unrelated to the query.  
- Informativeness: Adds little to no meaningful content; omits essential information.  
- Relevance: Largely off-topic or addresses only a small fraction of the intended task.  
- Understandability: The answer is very hard to follow: sentences are confusing, grammar or structure severely obstruct meaning, and the reader cannot reliably extract the intended message.

#### 2 — Poor
- Coherence: Some isolated logical elements exist, but major gaps hinder understanding.  
- Specificity: Mostly generic; few details are present and they do not add much value.  
- Informativeness: Limited content; misses several key aspects expected in a good answer.  
- Relevance: Partially related but includes irrelevant or misplaced sections.  
- Understandability: The response can be understood in parts but contains ambiguous phrasing, grammatical issues, or awkward structure that require effort to interpret and may lead to misunderstanding.

#### 3 — Fair
- Coherence: Generally logical but may have jumps, weak transitions, or mild inconsistencies.  
- Specificity: Includes a mix of general and task-specific elements; adequate but not strong.  
- Informativeness: Covers important points but may miss nuances or depth.  
- Relevance: Mostly stays on topic with occasional unnecessary or unfocused content.  
- Understandability: Readable and mostly clear; some sentences or terms are imprecise or slightly confusing, but the overall meaning is recoverable without excessive effort.

#### 4 — Good
- Coherence: Well-structured and easy to follow, with clear logical connections.  
- Specificity: Provides meaningful and relevant details tailored to the query.  
- Informativeness: Delivers substantial and accurate information; minor gaps may exist.  
- Relevance: Strongly aligned with the task; minimal drift or redundancy.  
- Understandability: The answer is clearly expressed with appropriate sentence structure and vocabulary; minor phrasing issues may appear but do not hamper comprehension.

#### 5 — Excellent
- Coherence: Highly organized, internally consistent, and logically seamless.  
- Specificity: Rich in precise, context-specific details without unnecessary generalities.  
- Informativeness: Comprehensive, insightful, and fully addresses all key aspects.  
- Relevance: Perfectly aligned with the question, with zero irrelevant content.  
- Understandability: Exceptionally clear and easy to read: grammar and syntax are correct, terminology is used precisely, sentences are well-formed, and a reader can immediately grasp the intended meaning without ambiguity.
---
Your output should follow exactly this template:
scores_a = {{"Coherence": X, "Specificity": Y, "Informativeness": Z, "Relevance": W, "Understandability": V}}
explanations_a = {{"Coherence": "...", "Specificity": "...", "Informativeness": "...", "Relevance": "...", "Understandability": "..."}}
scores_b = {{...}}
explanations_b = {{...}}

[User Input]
{question}

[Assistant A Response]
{answer_a}

[Assistant B Response]
{answer_b}
"""

# Backwards-compatible alias expected by existing code
# Default to V2 (more explicit Likert scales) but keep both versions available.
GT_EVALUATION_TEMPLATE = GT_EVALUATION_TEMPLATE_V2
