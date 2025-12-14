# Template de prompt para avaliação
GT_EVALUATION_TEMPLATE_V1 = """
Please act as an impartial judge and evaluate 
the quality of the responses provided by two AI assistants to the user 
question displayed below. Given the following user question and answers, 
please assign a score from 1(worst) to 5(best) following the Likert scale 
for each attribute listed below. 
For each attribute, begin your evaluation thinking step by step and providing a short explanation. 
The evaluation should be generated for the two provided responses. 
Avoid any position biases and ensure that the order in 
which the responses were presented does not influence your decision. 
Do not allow the length of the responses to influence your evaluation. Do not favor certain names of the assistants.

For each provided response, please return your answer in two JSON:

- One called scores where each key is the name of the attribute and the value is the score (a int between 1 and 5).  
- One called explanations where each key is the name of the attribute and the value is your explanation (1-2 sentences) for that score.

Please output the JSONs as plain text only, do not include code blocks, markdown, or any extra formatting.

Here are the attributes and their definitions:

Coherence:
How much does the generated text make sense?

Specificity:
Is the generated text generic or specific to the source text?

Informativeness:
How well does the generated text capture the key ideas of its source text?

Relevance:
How well is the generated text relevant to its source text?

Understandability:
Is the generated text understandable?

Example output format:  

scores_a = {{ 
  "Coherence": 5, 
  "Specificity": 3, 
  "Informativeness": 4, 
  "Relevance": 2, 
  "Understandability": 5 
}}  

explanations_a = {{ 
  "Coherence": "The text flows logically, with ideas that connect naturally and transitions that make sense throughout.", 
  "Specificity": "The response includes some details tied to the source text, but still contains a few general or vague statements.", 
  "Informativeness": "The text captures most of the key ideas from the source, though a few secondary points are missing.", 
  "Relevance": "Some parts of the generated text deviate from the main topic, introducing information that isn’t directly related to the source.", 
  "Understandability": "The language is clear and easy to read, with no grammatical or structural issues that hinder comprehension." 
}}}

scores_b = {{
  "Coherence": 2,
  "Specificity": 4,
  "Informativeness": 3,
  "Relevance": 1,
  "Understandability": 4
}}


explanations_b = {{
  "Coherence": "The response shows some logical ordering, but several ideas feel disconnected and the flow is inconsistent.",
  "Specificity": "The answer incorporates meaningful details related to the source text, demonstrating a solid level of specificity.",
  "Informativeness": "While the response includes a few important points, it overlooks several key elements that would provide a fuller understanding.",
  "Relevance": "A significant portion of the content strays from the original topic, introducing information that does not align with the source material.",
  "Understandability": "The text is mostly clear and readable, with only minor phrasing issues that slightly disrupt comprehension."
}}



[User Question]
{{question}}

[The Start of Assistant A’s Answer]
{{answer_a}} 
[The End of Assistant A’s Answer]
  
[The Start of Assistant B’s Answer] 
{{answer_b}} 
[The End of Assistant B’s Answer]
"""

GT_EVALUATION_TEMPLATE_V2 = """
[System]
Act as a neutral and unbiased evaluator. 
You will evaluate two AI-generated answers to the same user query. For each answer, follow this evaluation procedure:

1. Analyze the answer step-by-step.
2. Assign a score from 1 to 5 following the Likert scale for each attribute:
   - Coherence
   - Specificity
   - Informativeness
   - Relevance
   - Understandability
3. Write a short (1–2 sentence) explanation for each score.

Fairness constraints:
- Avoid any position bias: the order in which the answers appear must not influence your evaluation.
- Do not allow response length to affect your judgment.
- Do not favor any assistant based on its name or label.

---
Likert Scale Definitions (1–5)

1 — Very Poor
- Coherence: The answer lacks internal structure; ideas conflict or appear in a confusing, disjointed way.  
- Specificity: Almost entirely generic, offering no concrete details tied to the prompt or task.  
- Informativeness: Provides little to nothing beyond surface-level text; essential content is missing.  
- Relevance: Largely unrelated to the task, deviating significantly from what was asked.  
- Understandability: Difficult to read or interpret; wording and grammar obstruct comprehension.

2 — Poor
- Coherence: Some elements make sense individually, but the overall flow is weak or inconsistent.  
- Specificity: Contains minimal detail; most statements remain broad or only loosely connected to the prompt.  
- Informativeness: Only partially informative; several expected points are absent or underdeveloped.  
- Relevance: Tangentially related to the task but includes noticeable irrelevant or misplaced content.  
- Understandability: Parts can be understood, but ambiguity, awkward phrasing, or structural issues create friction for the reader.

3 — Fair
- Coherence: Mostly logical with reasonable progression, though transitions or clarity may falter.  
- Specificity: Offers some pertinent details, but the level of precision or grounding in the task is moderate.  
- Informativeness: Covers the core ideas but lacks depth or nuance.  
- Relevance: Stays generally aligned with the task, despite some extraneous or unfocused elements.  
- Understandability: Overall clear enough, though occasional vagueness or imprecise wording may appear.

4 — Good
- Coherence: Well-organized and easy to follow, with clear connections among ideas.  
- Specificity: Provides relevant, meaningful details tailored to the prompt.  
- Informativeness: Offers substantial and accurate information; only minor aspects may be missing.  
- Relevance: Strong adherence to the task with little to no unnecessary content.  
- Understandability: Clearly written and accessible; small stylistic issues may exist but do not hinder understanding.

5 — Excellent
- Coherence: Exceptionally clear, logically structured, and internally consistent throughout.  
- Specificity: Highly precise and fully anchored in the task context, with no generic filler.  
- Informativeness: Thorough, comprehensive, and insightful; all key aspects are addressed.  
- Relevance: Perfectly aligned with the task, with zero irrelevant content or digression.  
- Understandability: Extremely clear and well-articulated; language is precise, fluid, and immediately comprehensible.
---

For each answer, output:
- A JSON object (scores_a / scores_b) containing the numerical scores.
- A JSON object (explanations_a / explanations_b) containing the explanations.
- Output only the JSON objects as plain text, with no extra formatting.

Your output should follow exactly this template:
scores_a = {{ "Coherence": X, "Specificity": Y, "Informativeness": Z, "Relevance": W, "Clarity": V }}
explanations_a = {{ "Coherence": "...", "Specificity": "...", "Informativeness": "...", "Relevance": "...", "Clarity": "..." }}
scores_b = {{ ... }}
explanations_b = {{ ... }}

[User Question]
{question}

[Assistant A Response]
{answer_a}

[Assistant B Response]
{answer_b}
"""

# Backwards-compatible alias expected by existing code
# Default to V2
GT_EVALUATION_TEMPLATE = GT_EVALUATION_TEMPLATE_V2
