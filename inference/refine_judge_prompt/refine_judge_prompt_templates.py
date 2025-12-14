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
}}

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

GT_EVALUATION_TEMPLATE_V3 = """
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

For each answer, output:
- A JSON object (scores_a / scores_b) containing the numerical scores.
- A JSON object (explanations_a / explanations_b) containing the explanations.
- Output only the JSON objects as plain text, with no extra formatting.

Your output should follow exactly this template:
scores_a = {{ "Coherence": 4, "Specific": 3, "Informativeness": 4, "Relevance": 3, "Understandability": 5 }}  
explanations_a = {{
    "Coherence": "The response is well-structured and follows a clear logical flow, with only minor issues in transitions or organization.",
    "Specificity": "The content provides adequate task-related details but still mixes general and specific elements.",
    "Informativeness": "The response delivers solid and useful information, covering the main points well, though it may miss some finer nuances.",
    "Relevance": "The response stays mostly on topic, with only occasional unnecessary or unfocused content that slightly reduces precision.",
    "Understandability": "The response is exceptionally clear and easy to read; its structure and language allow immediate comprehension without ambiguity."
}}
scores_b = {{ ... }}
explanations_b = {{ ... }}

[User Question]
{question}

[Assistant A Response]
{answer_a}

[Assistant B Response]
{answer_b}
"""

GT_EVALUATION_TEMPLATE_V4 = """
[System]
Act as a neutral and unbiased evaluator. You will evaluate two AI-generated answers to the same user query. For each answer, follow this evaluation procedure:

1. Before choosing a score, briefly outline your reasoning process and then summarize it in a short (1–2 sentence) explanation for each score.
2. Assign a score from 1 to 5 following the Likert scale for each attribute:
   - Coherence
   - Specificity
   - Informativeness
   - Relevance
   - Understandability

Fairness constraints:
- Avoid any position bias: the order in which the answers appear must not influence your evaluation.
- Do not allow response length to affect your judgment.
- Do not favor any assistant based on its name or label.

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

For each answer, output:
- A JSON object (scores_a / scores_b) containing the numerical scores.
- A JSON object (explanations_a / explanations_b) containing the explanations.
- Output only the JSON objects as plain text, with no extra formatting.

Your output should follow exactly this template:
scores_a = {{ "Coherence": 4, "Specific": 3, "Informativeness": 4, "Relevance": 3, "Understandability": 5 }}  
explanations_a = {{
    "Coherence": "The response is well-structured and follows a clear logical flow, with only minor issues in transitions or organization.",
    "Specificity": "The content provides adequate task-related details but still mixes general and specific elements.",
    "Informativeness": "The response delivers solid and useful information, covering the main points well, though it may miss some finer nuances.",
    "Relevance": "The response stays mostly on topic, with only occasional unnecessary or unfocused content that slightly reduces precision.",
    "Understandability": "The response is exceptionally clear and easy to read; its structure and language allow immediate comprehension without ambiguity."
}}
scores_b = {{ ... }}
explanations_b = {{ ... }}

[User Question]
{question}

[Assistant A Response]
{answer_a}

[Assistant B Response]
{answer_b}
"""

GT_EVALUATION_TEMPLATE_V5 = """
You are a neutral and unbiased evaluator. 
Your must rate each answer according to five human-preference criteria. 
You should act with neutrality, precision, and consistency.
You will evaluate two AI-generated answers to the same user query. For each answer, follow this evaluation procedure:

1. Before choosing a score, briefly outline your reasoning process and then summarize it in a short (1–2 sentence) explanation for each score.
2. Assign a score from 1 to 5 following the Likert scale for each attribute:
   - Coherence
   - Specificity
   - Informativeness
   - Relevance
   - Understandability

Fairness constraints:
- Avoid any position bias: the order in which the answers appear must not influence your evaluation.
- Do not allow response length to affect your judgment.
- Do not favor any assistant based on its name or label.

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

For each answer, output:
- A JSON object (scores_a / scores_b) containing the numerical scores.
- A JSON object (explanations_a / explanations_b) containing the explanations.
- Output only the JSON objects as plain text, with no extra formatting.

Your output should follow exactly this template:
scores_a = {{ "Coherence": A, "Specific": B, "Informativeness": C, "Relevance": D, "Understandability": E }}  
explanations_a = {{
    "Coherence": [EXPLANATION OF THE SCORE A],
    "Specificity": [EXPLANATION OF THE SCORE B],
    "Informativeness": [EXPLANATION OF THE SCORE C],
    "Relevance": [EXPLANATION OF THE SCORE D],
    "Understandability": [EXPLANATION OF THE SCORE E]
}}
scores_b = {{ ... }}
explanations_b = {{ ... }}

[User Question]
{question}

[Assistant A Response]
{answer_a}

[Assistant B Response]
{answer_b}
"""

GT_EVALUATION_TEMPLATE_GT = """
You are an independent evaluation agent whose task is to assess the quality of answers produced by AI models. Your role is to rate each answer according to five human-preference criteria. You must do so with neutrality, precision, and consistency.
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
—

### Example Evaluation

[User Input]
Read the following text and perform the two steps:
#1: Translate the text into Spanish.
#2: Provide a one-sentence summary of the translated text.

###Text
The cat jumped onto the windowsill and watched the birds outside. It remained still for several minutes before curling up to sleep.

[Assistant A Response]
El gato saltó al alféizar de la ventana y observó a los pájaros afuera.
El gato observó los pájaros y luego se durmió.

[Assistant B Response]
Gato salt ventana pájaros. Luego dormir.

scores_a = {{ 
  "Coherence": 5,
  "Specificity": 5,
  "Informativeness": 5,
  "Relevance": 5,
  "Understandability": 5
}}

explanations_a = {{
  "Coherence": "The response follows the required steps in a clear and logically structured manner.",
  "Specificity": "It directly addresses the given text with accurate detail.",
  "Informativeness": "It provides both a correct translation and a complete summary.",
  "Relevance": "All content corresponds exactly to the input task.",
  "Understandability": "The language is precise and easy to read."
}}

scores_b = {{
  "Coherence": 1,
  "Specificity": 1,
  "Informativeness": 1,
  "Relevance": 2,
  "Understandability": 1
}}

explanations_b = {{
  "Coherence": "The response lacks structure and does not follow the required two-step process.",
  "Specificity": "Important details from the original text are missing or distorted.",
  "Informativeness": "The translation is incomplete and the summary is not meaningful.",
  "Relevance": "There is minimal relation to the task, although a few words are loosely related to the text.",
  "Understandability": "The output is fragmented and difficult to interpret."
}}

[User Input]
Perform the two steps below:
#1: Translate the sentence into French.
#2: Summarize the translated sentence in one short phrase.

###Text
The old bridge collapsed during the storm, but no one was injured.

[Assistant A Response]
Le vieux pont s'est effondré pendant la tempête, mais personne n'a été blessé.
Résumé: Aucun blessé lors de l'effondrement du pont.

[Assistant B Response]
Le pont est tombé. Personne blessé. Tempête.

scores_a = {{ 
  "Coherence": 4,
  "Specificity": 4,
  "Informativeness": 4,
  "Relevance": 5,
  "Understandability": 4
}}

explanations_a = {{
  "Coherence": "The answer follows the steps clearly with minor awkward phrasing.",
  "Specificity": "The translation and summary reflect key details from the text.",
  "Informativeness": "It includes both required components, though the summary is slightly generic.",
  "Relevance": "Fully aligned with the requested tasks.",
  "Understandability": "Mostly clear and well-formed despite small stylistic issues."
}}

scores_b = {{
  "Coherence": 2,
  "Specificity": 2,
  "Informativeness": 2,
  "Relevance": 3,
  "Understandability": 2
}}

explanations_b = {{
  "Coherence": "The response follows the overall topic but lacks clear step separation.",
  "Specificity": "Contains only partial details from the original text.",
  "Informativeness": "Provides fragments rather than a full translation and summary.",
  "Relevance": "Stays loosely related but fails to complete the required steps.",
  "Understandability": "Choppy and lacking structure, requiring effort to infer meaning."
}}

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

GT_EVALUATION_TEMPLATE = GT_EVALUATION_TEMPLATE_V5