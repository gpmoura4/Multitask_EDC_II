"""
Script para geração de Ground Truth (GT) a partir de experimentos STI e MTI.

Este script:
1. Seleciona pares de experimentos STI/MTI com o mesmo model_name
2. Amostra instâncias para avaliação (modo teste: 24 instâncias/12 tasks, modo real: 240 instâncias/12 tasks)
3. Constrói prompts de avaliação combinando respostas STI e MTI
4. Envia para LLM (GPT-4o-mini) para avaliação de qualidade
5. Salva resultados na collection 'ground_truth_results'
"""

from __future__ import annotations

import json
import os
import random
import argparse
from datetime import datetime
from bson import ObjectId
from pymongo import MongoClient
from typing import List, Dict, Tuple
from tqdm import tqdm
import csv
from pathlib import Path

from evaluate.utils import load_hf_lm_and_tokenizer, generate_completions

# Conexão MongoDB
MONGO_URI = os.getenv("MONGODB_URI")

mongo_client = MongoClient(MONGO_URI)
db = mongo_client["experiments_db"]
experiment_collection = db["experiment_results"]
answer_collection = db["answer_results"]
gt_collection = db["ground_truth_results"]


# Template de prompt para avaliação
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

GT_EVALUATION_TEMPLATE_V3 = """You are an independent evaluation agent whose task is to assess the quality of answers produced by AI models. Your role is to rate each answer according to five human-preference criteria. You must do so with neutrality, precision, and consistency.
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

GT_EVALUATION_TEMPLATE = GT_EVALUATION_TEMPLATE_V3


def find_experiment_pairs(model_name: str | None = None) -> List[Tuple[ObjectId, ObjectId]]:
    """
    Encontra pares de experimentos (STI, MTI) com o mesmo model_name.
    
    Args:
        model_name: Nome do modelo para filtrar. Se None, busca todos os modelos.
    
    Returns:
        Lista de tuplas (sti_experiment_id, mti_experiment_id)
    """
    query = {"experimentIsOver": True}
    if model_name:
        query["model_name"] = model_name
    
    # Buscar experimentos STI
    sti_experiments = list(experiment_collection.find({
        **query,
        "inference_type": "STI"
    }))
    
    # Buscar experimentos MTI
    mti_experiments = list(experiment_collection.find({
        **query,
        "inference_type": "MTI"
    }))
    
    pairs = []
    for sti_exp in sti_experiments:
        for mti_exp in mti_experiments:
            if sti_exp["model_name"] == mti_exp["model_name"]:
                pairs.append((sti_exp["_id"], mti_exp["_id"]))
    
    return pairs


def get_common_instances(sti_experiment_id: ObjectId, mti_experiment_id: ObjectId) -> Dict[str, List[str]]:
    """
    Encontra instâncias comuns entre experimentos STI e MTI, agrupadas por task_id.
    
    Returns:
        Dict[task_id, List[instance_ids]]
    """
    # Buscar todas as instâncias do experimento STI
    sti_answers = answer_collection.find({"experiment_id": sti_experiment_id})
    sti_instances = {}
    for doc in sti_answers:
        task_id = doc["task_id"]
        instance_id = doc["instance_id"]
        if task_id not in sti_instances:
            sti_instances[task_id] = set()
        sti_instances[task_id].add(instance_id)
    
    # Buscar todas as instâncias do experimento MTI
    mti_answers = answer_collection.find({"experiment_id": mti_experiment_id})
    mti_instances = {}
    for doc in mti_answers:
        task_id = doc["task_id"]
        instance_id = doc["instance_id"]
        if task_id not in mti_instances:
            mti_instances[task_id] = set()
        mti_instances[task_id].add(instance_id)
    
    # Encontrar interseção
    common_instances = {}
    for task_id in sti_instances:
        if task_id in mti_instances:
            common = list(sti_instances[task_id].intersection(mti_instances[task_id]))
            if common:
                common_instances[task_id] = common
    
    return common_instances


def sample_instances(common_instances: Dict[str, List[str]], 
                     instances_per_task: int = 2) -> Dict[str, List[str]]:
    """
    Amostra um número específico de instâncias por task.
    
    Args:
        common_instances: Dict[task_id, List[instance_ids]]
        instances_per_task: Número de instâncias a amostrar por task
    
    Returns:
        Dict[task_id, List[instance_ids]] com instâncias amostradas
    """
    sampled = {}
    for task_id, instance_list in common_instances.items():
        # Amostrar instâncias (ou pegar todas se houver menos que o solicitado)
        n_samples = min(instances_per_task, len(instance_list))
        sampled[task_id] = random.sample(instance_list, n_samples)
    
    return sampled


def save_sampled_instances_to_csv(sampled_instances: Dict[str, List[str]], filename: str) -> None:
    """
    Salva o dict sampled_instances em um CSV com colunas: task_id,instance_id
    """
    p = Path(filename)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["task_id", "instance_id"])
        for task_id, instance_list in sampled_instances.items():
            for instance_id in instance_list:
                writer.writerow([task_id, instance_id])


def load_sampled_instances_from_csv(filename: str) -> Dict[str, List[str]]:
    """
    Carrega um CSV (task_id,instance_id) e retorna Dict[task_id, List[instance_id]]
    """
    sampled: Dict[str, List[str]] = {}
    p = Path(filename)
    if not p.exists():
        raise FileNotFoundError(f"Sample CSV file not found: {filename}")
    with p.open("r", newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            task_id = row.get("task_id")
            instance_id = row.get("instance_id")
            if task_id is None or instance_id is None:
                continue
            sampled.setdefault(task_id, []).append(instance_id)
    return sampled


def concatenate_sti_answers(llm_answer: Dict[str, str]) -> str:
    """
    Concatena as respostas STI (s1, s2, s3) em um formato legível.
    
    Args:
        llm_answer: Dict com chaves s1, s2, s3
    
    Returns:
        String concatenada formatada
    """
    result = ""
    stages = ["s1", "s2", "s3"]
    
    for i, stage in enumerate(stages, 1):
        if stage in llm_answer and llm_answer[stage]:
            answer = llm_answer[stage].strip()
            
            # Verificar se a resposta já começa com o prefixo ### Instruction
            if answer.startswith("### Instruction"):
                # Resposta já tem o prefixo, não adicionar novamente
                result += f"{answer}\n\n"
            else:
                # Resposta não tem o prefixo, adicionar
                result += f"### Instruction{i}:\n\n{answer}\n\n"
    
    return result.strip()


def parse_llm_evaluation(evaluation_text: str) -> Tuple[Dict, Dict]:
    """
    Parseia a resposta do LLM de avaliação para extrair scores e explanations.
    
    Args:
        evaluation_text: Texto de resposta do LLM
    
    Returns:
        Tupla (scores_dict, explanations_dict) onde cada dict tem chaves 'a' e 'b'
    """
    import re
    
    scores = {"a": {}, "b": {}}
    explanations = {"a": {}, "b": {}}
    
    # Tentar extrair scores_a
    score_a_match = re.search(r'scores_a\s*=\s*(\{[^}]+\})', evaluation_text)
    if score_a_match:
        try:
            scores["a"] = eval(score_a_match.group(1))
        except:
            print("⚠️ Erro ao parsear scores_a")
    
    # Tentar extrair scores_b
    score_b_match = re.search(r'scores_b\s*=\s*(\{[^}]+\})', evaluation_text)
    if score_b_match:
        try:
            scores["b"] = eval(score_b_match.group(1))
        except:
            print("⚠️ Erro ao parsear scores_b")
    
    # Tentar extrair explanations_a
    expl_a_match = re.search(r'explanations_a\s*=\s*(\{[^}]+\})', evaluation_text, re.DOTALL)
    if expl_a_match:
        try:
            explanations["a"] = eval(expl_a_match.group(1))
        except:
            print("⚠️ Erro ao parsear explanations_a")
    
    # Tentar extrair explanations_b
    expl_b_match = re.search(r'explanations_b\s*=\s*(\{[^}]+\})', evaluation_text, re.DOTALL)
    if expl_b_match:
        try:
            explanations["b"] = eval(expl_b_match.group(1))
        except:
            print("⚠️ Erro ao parsear explanations_b")
    
    return scores, explanations


def create_gt_document(
    sti_experiment_id: ObjectId,
    mti_experiment_id: ObjectId,
    task_id: str,
    instance_id: str,
    sti_answer_doc: Dict,
    mti_answer_doc: Dict,
    evaluator_model: str,
    experiment_gt_name: str
) -> Dict | None:
    """
    Cria um documento de ground truth avaliando respostas STI e MTI.
    
    Args:
        sti_experiment_id: ID do experimento STI
        mti_experiment_id: ID do experimento MTI
        task_id: ID da task
        instance_id: ID da instância
        sti_answer_doc: Documento de resposta STI do MongoDB
        mti_answer_doc: Documento de resposta MTI do MongoDB
        evaluator_model: Nome do modelo usado para avaliação (ex: "gpt-4o-mini-2024-07-18")
    
    Returns:
        Documento pronto para inserir na collection ground_truth_results
    """
    # Concatenar respostas STI
    sti_concatenated = concatenate_sti_answers(sti_answer_doc["llm_answer"])
    
    # Resposta MTI (armazenada em s1)
    mti_answer = mti_answer_doc["llm_answer"].get("s1", "")
    
    # Randomizar posição das respostas
    position_mti = random.choice(["A", "B"])
    position_sti = "B" if position_mti == "A" else "A"
    
    answer_a = mti_answer if position_mti == "A" else sti_concatenated
    answer_b = sti_concatenated if position_mti == "A" else mti_answer
    
    # Construir prompt de avaliação
    # Usar o prompt do MTI como question (mais completo)
    question = mti_answer_doc.get("prompt", "")
    
    gt_prompt = GT_EVALUATION_TEMPLATE.format(
        question=question,
        answer_a=answer_a,
        answer_b=answer_b
    )
    
    # Carregar modelo avaliador e gerar avaliação
    model, tokenizer = load_hf_lm_and_tokenizer(evaluator_model)
    
    print(f"  Avaliando instância {instance_id} da task {task_id}...")
    
    generation_time, generated_texts = generate_completions(
        model,
        tokenizer,
        [gt_prompt],
        batch_size=1,
        max_new_tokens=1024,
        min_new_tokens=100,
        do_sample=True,
        temperature=0.3,  # Temperatura baixa para avaliação mais consistente
        top_p=0.95
    )
    
    evaluation_response = generated_texts[0]
    
    # Parsear resposta do avaliador
    scores, explanations = parse_llm_evaluation(evaluation_response)
    
    # Mapear scores e explanations para MTI e STI
    if position_mti == "A":
        mti_scores = scores["a"]
        mti_explanations = explanations["a"]
        sti_scores = scores["b"]
        sti_explanations = explanations["b"]
    else:
        mti_scores = scores["b"]
        mti_explanations = explanations["b"]
        sti_scores = scores["a"]
        sti_explanations = explanations["a"]
    
    # Construir documento
    doc = {
        "experiment_gt_name": experiment_gt_name,
        "sti_experiment_id": sti_experiment_id,
        "mti_experiment_id": mti_experiment_id,
        "task_id": task_id,
        "instance_id": instance_id,
        "evaluator_model": evaluator_model,
        "evaluation_timestamp": datetime.now(),
        
        # Dados MTI
        "prompt_MTI": mti_answer_doc.get("prompt", ""),
        "llm_answer_MTI": {
            "s1": mti_answer,
            "position_answer_to_llm": position_mti
        },
        
        # Dados STI
        "prompt_STI": sti_answer_doc.get("prompt", ""),
        "llm_answer_STI": {
            **sti_answer_doc["llm_answer"],
            "position_answer_to_llm": position_sti
        },
        
        # Prompt e resposta de avaliação
        "gt_prompt": gt_prompt,
        "gt_raw_response": evaluation_response,
        
        # Avaliações MTI
        "gt_MTI_answer": {
            "coherence": {
                "score": mti_scores.get("Coherence", 0),
                "explanation": mti_explanations.get("Coherence", "")
            },
            "specificity": {
                "score": mti_scores.get("Specificity", 0),
                "explanation": mti_explanations.get("Specificity", "")
            },
            "informativeness": {
                "score": mti_scores.get("Informativeness", 0),
                "explanation": mti_explanations.get("Informativeness", "")
            },
            "relevance": {
                "score": mti_scores.get("Relevance", 0),
                "explanation": mti_explanations.get("Relevance", "")
            },
            "Understandability": {
                "score": mti_scores.get("Understandability", 0),
                "explanation": mti_explanations.get("Understandability", "")
            }
        },
        
        # Avaliações STI
        "gt_STI_answer": {
            "coherence": {
                "score": sti_scores.get("Coherence", 0),
                "explanation": sti_explanations.get("Coherence", "")
            },
            "specificity": {
                "score": sti_scores.get("Specificity", 0),
                "explanation": sti_explanations.get("Specificity", "")
            },
            "informativeness": {
                "score": sti_scores.get("Informativeness", 0),
                "explanation": sti_explanations.get("Informativeness", "")
            },
            "relevance": {
                "score": sti_scores.get("Relevance", 0),
                "explanation": sti_explanations.get("Relevance", "")
            },
            "Understandability": {
                "score": sti_scores.get("Understandability", 0),
                "explanation": sti_explanations.get("Understandability", "")
            }
        }
    }
    
    return doc


def main():
    parser = argparse.ArgumentParser(
        description="Gera ground truth comparando experimentos STI e MTI"
    )
    parser.add_argument(
        "--sti_experiment_id",
        type=str,
        required=True,
        help="ID do experimento STI (ObjectId do MongoDB)"
    )
    parser.add_argument(
        "--mti_experiment_id",
        type=str,
        required=True,
        help="ID do experimento MTI (ObjectId do MongoDB)"
    )
    parser.add_argument(
        "--evaluator_model",
        type=str,
        default="gpt-4o-mini-2024-07-18",
        help="Modelo usado para avaliação"
    )
    parser.add_argument(
        "--experiment_gt_name",
        type=str,
        required=True,
        help="Nome do experimento de ground truth"
    )
    parser.add_argument(
        "--is_test",
        action="store_true",
        help="Modo teste: 2 instâncias por task (total 24). Modo real: 20 por task (total 240)"
    )
    parser.add_argument(
        "--predefined_test",
        action="store_true",
        help=(
            "Quando usado junto com --is_test, seleciona instâncias pré-definidas "
            "(sempre as primeiras N instâncias por task) ao invés de amostragem aleatória"
        )
    )
    parser.add_argument(
        "--first_experiment",
        action="store_true",
        help=(
            "Indica que este é o primeiro experimento: a amostra será selecionada aleatoriamente "
            "e (opcionalmente) salva em CSV. Se você quiser reutilizar a mesma amostra em execuções posteriores, "
            "execute com --first_experiment --sample_csv_file <file.csv> to save the file."
        )
    )
    parser.add_argument(
        "--sample_csv_file",
        type=str,
        default=None,
        help=(
            "Caminho para um CSV de instâncias (task_id,instance_id). "
            "Se fornecido e existir, será usado como amostra em vez de sortear. "
            "Se não existir e --first_experiment for usado, o arquivo será criado com a amostra gerada."
        )
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed para reprodutibilidade"
    )
    
    args = parser.parse_args()
    
    # Definir seed
    random.seed(args.seed)
    
    # Converter IDs para ObjectId
    try:
        sti_exp_id = ObjectId(args.sti_experiment_id)
        mti_exp_id = ObjectId(args.mti_experiment_id)
    except Exception as e:
        print(f"❌ Erro ao converter IDs: {e}")
        return
    
    # Verificar se experimentos existem
    sti_exp = experiment_collection.find_one({"_id": sti_exp_id})
    mti_exp = experiment_collection.find_one({"_id": mti_exp_id})
    
    if not sti_exp:
        print(f"❌ Experimento STI {args.sti_experiment_id} não encontrado")
        return
    
    if not mti_exp:
        print(f"❌ Experimento MTI {args.mti_experiment_id} não encontrado")
        return
    
    print(f"\n📊 Iniciando geração de Ground Truth")
    print(f"Experiment GT Name: {args.experiment_gt_name}")
    print(f"STI Experiment: {sti_exp['model_name']} (ID: {args.sti_experiment_id})")
    print(f"MTI Experiment: {mti_exp['model_name']} (ID: {args.mti_experiment_id})")
    print(f"Evaluator Model: {args.evaluator_model}")
    print(f"Modo: {'TESTE (2 instâncias/task)' if args.is_test else 'COMPLETO (20 instâncias/task)'}\n")
    
    # Encontrar instâncias comuns
    common_instances = get_common_instances(sti_exp_id, mti_exp_id)
    print(f"✅ Encontradas {len(common_instances)} tasks com instâncias comuns")
    
    # Amostrar instâncias
    instances_per_task = 2 if args.is_test else 20
    # If a sample CSV file is provided, prefer that (load if exists).
    sampled_instances = None
    if args.sample_csv_file:
        csv_path = Path(args.sample_csv_file)
        # Se for uma pasta, cria o arquivo com nome do experimento
        if csv_path.is_dir() or (not csv_path.suffix and not csv_path.exists()):
            csv_path = csv_path / f"{args.experiment_gt_name}.csv"
        # Se for um arquivo sem extensão, adiciona .csv
        if not csv_path.suffix:
            csv_path = csv_path.with_suffix(".csv")
        if csv_path.exists():
            print(f"🔁 Loading sampled instances from CSV: {csv_path}")
            sampled_instances = load_sampled_instances_from_csv(str(csv_path))
        else:
            # If file doesn't exist but user indicated this is the first experiment,
            # we will sample and save to the provided path.
            if args.first_experiment:
                print(f"🆕 Sample CSV not found. Will sample and save to: {csv_path}")
                # determine sampled_instances below
            else:
                print(f"❌ Sample CSV file not found: {csv_path}. Use --first_experiment to create it or provide an existing file.")
                return

    # If CSV didn't provide samples, perform sampling (either predefined test or random)
    if sampled_instances is None:
        # Se estiver em modo teste e o usuário requisitou instâncias pré-definidas,
        # selecionamos de forma determinística as primeiras N instâncias por task.
        if args.is_test and getattr(args, "predefined_test", False):
            sampled_instances = {}
            for task_id, instance_list in common_instances.items():
                # ordem determinística
                try:
                    sorted_list = sorted(instance_list)
                except Exception:
                    # fallback para a lista original caso não seja ordenável
                    sorted_list = list(instance_list)

                n = min(instances_per_task, len(sorted_list))
                sampled_instances[task_id] = sorted_list[:n]
        else:
            sampled_instances = sample_instances(common_instances, instances_per_task)

        # Salvar CSV imediatamente após definir a amostra
        if args.first_experiment and args.sample_csv_file:
            try:
                save_sampled_instances_to_csv(sampled_instances, str(csv_path))
                print(f"💾 Sampled instances saved to: {csv_path}")
            except Exception as e:
                print(f"⚠️ Erro ao salvar sample CSV: {e}")
    
    total_instances = sum(len(instances) for instances in sampled_instances.values())
    print(f"✅ Amostradas {total_instances} instâncias para avaliação\n")
    
    # Processar cada instância
    gt_documents = []
    
    for task_id, instance_list in tqdm(sampled_instances.items(), desc="Processando tasks"):
        for instance_id in instance_list:
            # Buscar documentos de resposta
            sti_answer_doc = answer_collection.find_one({
                "experiment_id": sti_exp_id,
                "task_id": task_id,
                "instance_id": instance_id
            })
            
            mti_answer_doc = answer_collection.find_one({
                "experiment_id": mti_exp_id,
                "task_id": task_id,
                "instance_id": instance_id
            })
            
            if not sti_answer_doc or not mti_answer_doc:
                print(f"⚠️ Instância {instance_id} da task {task_id} não encontrada em ambos experimentos")
                continue
            
            # Criar documento GT
            try:
                gt_doc = create_gt_document(
                    sti_exp_id,
                    mti_exp_id,
                    task_id,
                    instance_id,
                    sti_answer_doc,
                    mti_answer_doc,
                    args.evaluator_model,
                    args.experiment_gt_name
                )
                
                if gt_doc:
                    gt_documents.append(gt_doc)
            
            except Exception as e:
                print(f"❌ Erro ao processar instância {instance_id} da task {task_id}: {e}")
                continue
    
    # Salvar documentos no MongoDB
    if gt_documents:
        print(f"\n💾 Salvando {len(gt_documents)} documentos na collection 'ground_truth_results'...")
        result = gt_collection.insert_many(gt_documents)
        print(f"✅ {len(result.inserted_ids)} documentos inseridos com sucesso!")
    else:
        print("\n⚠️ Nenhum documento GT foi gerado")
    
    print("\n✅ Processo de geração de Ground Truth concluído!")


if __name__ == "__main__":
    main()


