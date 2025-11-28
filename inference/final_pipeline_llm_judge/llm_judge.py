from __future__ import annotations

"""
Script para geração de LLM Judge a partir de experimentos STI e MTI.

Este script:
1. Seleciona pares de experimentos STI/MTI com o mesmo model_name
2. Para cada instância, constrói prompts de avaliação combinando respostas STI e MTI
3. Envia para o mesmo LLM usado no experimento para auto-avaliação
4. Salva resultados na collection 'llm_judge_results'
"""

"""
experiment result collection document example:

{
  "_id": {
    "$oid": "691e5cdb47cb353cce0b14b0"
  },
  "inference_type": "STI",
  "experiment_name": "STI_gpt-4o-mini-2024-07-18_Experiment",
  "batch_size": 20,
  "save_every": 10,
  "model_name": "gpt-4o-mini-2024-07-18",
  "llm_params": {
    "tokenizer": "NoneType",
    "model_name": "gpt-4o-mini-2024-07-18",
    "stop_id_sequences": null,
    "add_special_tokens": true,
    "disable_tqdm": false,
    "max_new_tokens": 2048,
    "min_new_tokens": 32,
    "do_sample": true,
    "temperature": 0.7,
    "top_p": 1
  },
  "initial_time": {
    "$date": "2025-11-19T21:12:11.872Z"
  },
  "final_time": {
    "$date": "2025-11-20T01:27:13.989Z"
  },
  "total_time": 15302.11725,
  "experimentIsOver": true
}


answer_results collection document example:

{
  "_id": {
    "$oid": "691e711c985ec6b7ca0527cb"
  },
  "experiment_id": {
    "$oid": "691e5cdb47cb353cce0b14b0"
  },
  "task_id": "030",
  "instance_id": "030_1002",
  "prompt": "<|user|> ### Example:\n\n### (1) Instruction: Transate the given text to German.\n\n###Text\n\"Around the 15th century, northern Estonia was under great cultural influence of Germany. Some German monks wanted to bring God closer to the native people, so they invented the Estonian literal language. It was based on the German alphabet and one character \"\"Õ/õ\"\" was added. As time passed, many words that were borrowed from German coalesced. This was the beginning of enlightenment.\"\n\n###Answer:\n\nWenn die Kapsel um etwa 5 Uhr morgens (Eastern Time) die Erde erreicht und in die Atmosphäre eintritt, wird sie für die Leute in Nordkalifornien, Oregon, Nevada und Utah wahrscheinlich eine beeindruckende Lightshow abgeben. Die Kapsel wird ähnlich wie eine Sternschuppe aussehen, die am Himmel fliegt. Die Kapsel wird sich mit etwa 12,8 km oder 8 Meilen pro Sekunde fortbewegen – schnell genug, um in einer Minute von San Francisco nach Los Angeles zu gelangen. Stardust wird einen neuen Allzeitrekord als das  schnellste Raumfahrzeug aufstellen, das je zur Erde zurückgekehrt ist. Damit bricht es den bisherigen Rekord, der im Mai 1969 bei der Rückkehr des Apollo X-Kommandomoduls aufgestellt wurde. „Es wird sich über die Westküste Nordkaliforniens bewegen und den Himmel von Kalifornien bis Zentraloregon und weiter über Nevada und Idaho bis nach Utah erhellen“, sagte Tom Duxbury, Projektmanager von Stardust.\n\n### (2) Instruction: Based on the text you have translated in step#1 and solve the question. Return the answer in <task2>N<task2/> format.\n\n###Question:\nWofür wird die Kapsel dem Abschnitt zufolge einen neuen Rekord aufstellen?\n###Options:\n1. Die weitläufigste Erhellung des Himmels\n2. Sichtbarkeit von den meisten Städten aus\n3. Höchste Geschwindigkeit bei der Rückkehr zur Erde\n4. Schnellstes Zurücklegen der Strecke zwischen San Francisco und Los Angeles\n\n###Answer:\n\nDie Frage lautet: \"Wofür wird die Kapsel dem Abschnitt zufolge einen neuen Rekord aufstellen?\".  Gemäß dem übersetzten Text wird die Kapsel einen neuen Allzeitrekord als das schnellste Raumfahrzeug aufstellen, das je zur Erde zurückgekehrt ist. Daher lautet die Antwort auf die Frage:\n\n<task2>3<task2/>\n\n### Task:\n\n### (1) Instruction: Transate the given text to German.\n\n###Text\n\"One of the most common methods used to illustrate the importance of socialization is to draw upon the few unfortunate cases of children who were, through neglect, misfortune, or wilful abuse, not socialized by adults while they were growing up. Such children are called \"\"feral\"\" or wild. Some feral children have been confined by people (usually their own parents); in some cases this child abandonment was due to the parents' rejection of a child's severe intellectual or physical impairment. Feral children may have experienced severe child abuse or trauma before being abandoned or running away. Others are alleged to have been brought up by animals; some are said to have lived in the wild on their own. When completely brought up by non-human animals, the feral child exhibits behaviors (within physical limits) almost entirely like those of the particular care-animal, such as its fear of or indifference to humans.\"\n\n### Answer:\n\n\"Eine der häufigsten Methoden, um die Bedeutung der Sozialisierung zu veranschaulichen, besteht darin, auf die wenigen unglücklichen Fälle von Kindern hinzuweisen, die durch Vernachlässigung, Unglück oder absichtlichen Missbrauch während ihres Aufwachsens nicht von Erwachsenen sozialisiert wurden. Solche Kinder werden als \"\"feral\"\" oder wild bezeichnet. Einige wilde Kinder wurden von Menschen (in der Regel ihren eigenen Eltern) eingesperrt; in einigen Fällen war diese Kindesentziehung auf die Ablehnung der Eltern gegenüber einer schweren intellektuellen oder körperlichen Beeinträchtigung des Kindes zurückzuführen. Wilde Kinder haben möglicherweise schweren Kindesmissbrauch oder Trauma erlebt, bevor sie verlassen wurden oder wegliefen. Andere sollen von Tieren aufgezogen worden sein; einige sollen alleine in der Wildnis gelebt haben. Wenn sie vollständig von nicht-menschlichen Tieren aufgezogen wurden, zeigt das wilde Kind Verhaltensweisen (innerhalb physischer Grenzen), die fast ausschließlich denen des jeweiligen Pflege-Tieres ähneln, wie zum Beispiel seine Angst vor oder Gleichgültigkeit gegenüber Menschen.\"\n\n### (2) Instruction: Based on the text you have translated in step#1 and solve the question. Return the answer in <task2>N<task2/> format.\n\n###Question:\nWelche der folgenden Aussagen würde dem Abschnitt zufolge nicht auf die Erziehung eines verwilderten Kindes zutreffen?\n###Options:\n1. Unabhängig leben\n2. Starke Konversationsfähigkeiten\n3. Tendenz, sich zurückzuziehen\n4. Soziale Ängste\n\n### Answer:\n<|assistant|>",
  "llm_answer": {
    "s1": "\"Eine der häufigsten Methoden, um die Bedeutung der Sozialisierung zu veranschaulichen, besteht darin, auf die wenigen unglücklichen Fälle von Kindern hinzuweisen, die durch Vernachlässigung, Unglück oder absichtlichen Missbrauch während ihres Aufwachsens nicht von Erwachsenen sozialisiert wurden. Solche Kinder werden als \"\"feral\"\" oder wild bezeichnet. Einige wilde Kinder wurden von Menschen (in der Regel ihren eigenen Eltern) eingesperrt; in einigen Fällen war diese Kindesentziehung auf die Ablehnung der Eltern gegenüber einer schweren intellektuellen oder körperlichen Beeinträchtigung des Kindes zurückzuführen. Wilde Kinder haben möglicherweise schweren Kindesmissbrauch oder Trauma erlebt, bevor sie verlassen wurden oder wegliefen. Andere sollen von Tieren aufgezogen worden sein; einige sollen alleine in der Wildnis gelebt haben. Wenn sie vollständig von nicht-menschlichen Tieren aufgezogen wurden, zeigt das wilde Kind Verhaltensweisen (innerhalb physischer Grenzen), die fast ausschließlich denen des jeweiligen Pflege-Tieres ähneln, wie zum Beispiel seine Angst vor oder Gleichgültigkeit gegenüber Menschen.\"",
    "s2": "Die Frage lautet: \"Welche der folgenden Aussagen würde dem Abschnitt zufolge nicht auf die Erziehung eines verwilderten Kindes zutreffen?\". Gemäß dem übersetzten Text zeigen wilde Kinder Verhaltensweisen, die fast ausschließlich denen des jeweiligen Pflege-Tieres ähneln, und sie haben möglicherweise soziale Ängste sowie eine Tendenz, sich zurückzuziehen. Es wird jedoch nicht erwähnt, dass sie starke Konversationsfähigkeiten entwickeln. Daher lautet die Antwort auf die Frage:\n\n<task2>2<task2/>"
  },
  "generation_time": 3.5295636653900146,
  "consumed_tokens": 206
}

"""

import os
import sys
import argparse
import random
import json
from datetime import datetime
from bson import ObjectId
from pymongo import MongoClient
from typing import List, Dict, Tuple
from tqdm import tqdm
from dotenv import load_dotenv
from pathlib import Path

# Adicionar path para importar evaluate.utils
sys.path.append(str(Path(__file__).parent.parent.parent))

from evaluate.utils import load_hf_lm_and_tokenizer, generate_completions

load_dotenv()

# Conexão MongoDB
MONGO_URI = os.getenv("MONGODB_URI")

mongo_client = MongoClient(MONGO_URI)
db = mongo_client["experiments_db"]
experiment_collection = db["experiment_results"]
answer_collection = db["answer_results"]
llm_judge_collection = db["llm_judge_results"]
experiments_llm_judge_collection = db["experiments_llm_judge"]


# Template de prompt para avaliação
LLM_JUDGE_EVALUATION_TEMPLATE ="""
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


def create_llm_judge_experiment_record(
    experiment_name: str,
    sti_experiment_id: ObjectId,
    mti_experiment_id: ObjectId,
    evaluator_model: str,
    batch_size: int,
    seed: int,
    is_test: bool = False,
    experiment_id: str | None = None
) -> ObjectId:
    """
    Cria o documento inicial do experimento LLM Judge.
    Se experiment_id for fornecido, tenta retornar o experimento existente.
    """
    # Se 'experiment_id' foi fornecido, tentar buscar experimento existente
    if experiment_id is not None:
        try:
            object_id = ObjectId(experiment_id)
            existing_doc = experiments_llm_judge_collection.find_one({"_id": object_id})
        except Exception:
            print(f"❌ ID '{experiment_id}' não é um ObjectId válido. Criando novo experimento.")
            existing_doc = None

        if existing_doc:
            print(f"⚠️  Experimento LLM Judge encontrado com _id='{experiment_id}'. Retornando documento existente.")
            return existing_doc["_id"]
        else:
            print(f"⚠️  Nenhum experimento LLM Judge encontrado com _id='{experiment_id}'. Criando novo experimento.")

    initial_time = datetime.now()

    doc = {
        "inference_type": "LLM_JUDGE",
        "experiment_name": experiment_name,
        "sti_experiment_id": sti_experiment_id,
        "mti_experiment_id": mti_experiment_id,
        "evaluator_model": evaluator_model,
        "batch_size": batch_size,
        "seed": seed,
        "is_test": is_test,
        "llm_params": {
            "model_name": evaluator_model,
            "batch_size": 1,
            "max_new_tokens": 1024,
            "min_new_tokens": 100,
            "do_sample": True,
            "temperature": 0.3,
            "top_p": 0.95
        },
        "initial_time": initial_time,
        "final_time": None,
        "total_time": None,
        "experimentIsOver": False
    }

    result = experiments_llm_judge_collection.insert_one(doc)
    print(f"✅ Experimento LLM Judge criado com _id='{result.inserted_id}'")
    return result.inserted_id


def finalize_llm_judge_experiment(experiment_id: ObjectId):
    """
    Atualiza o documento do experimento LLM Judge para marcar como concluído.
    Preenche também o final_time e total_time.
    """
    final_time = datetime.now()

    # Buscar o documento original para calcular total_time
    doc = experiments_llm_judge_collection.find_one({"_id": experiment_id})
    if not doc:
        print(f"⚠️  Documento de experimento LLM Judge {experiment_id} não encontrado.")
        return
    
    initial_time = doc["initial_time"]
    total_time = (final_time - initial_time).total_seconds()

    update_fields = {
        "experimentIsOver": True,
        "final_time": final_time,
        "total_time": total_time
    }

    experiments_llm_judge_collection.update_one(
        {"_id": experiment_id},
        {"$set": update_fields}
    )
    print(f"✅ Experimento LLM Judge {experiment_id} finalizado. Tempo total: {total_time:.2f}s")


def save_error_checkpoint(experiment_id: ObjectId, instance_id: str, error_msg: str):
    """
    Salva um checkpoint de erro em arquivo txt para recuperação posterior.
    """
    checkpoint_file = f"error_checkpoint_{experiment_id}.txt"
    with open(checkpoint_file, 'w') as f:
        f.write(f"experiment_id: {experiment_id}\n")
        f.write(f"instance_id: {instance_id}\n")
        f.write(f"error: {error_msg}\n")
        f.write(f"timestamp: {datetime.now().isoformat()}\n")
    print(f"💾 Checkpoint de erro salvo em {checkpoint_file}")


def check_instance_already_evaluated(llm_judge_experiment_id: ObjectId, instance_id: str) -> bool:
    """
    Verifica se uma instância específica já foi avaliada neste experimento LLM Judge.
    """
    doc = llm_judge_collection.find_one({
        "llm_judge_experiment_id": llm_judge_experiment_id,
        "instance_id": instance_id
    })
    return doc is not None


def concatenate_sti_answers(llm_answer: Dict) -> str:
    """
    Concatena as respostas STI (s1, s2, s3) em uma única string.
    
    Args:
        llm_answer: Dict com campos s1, s2, s3 da resposta STI
    
    Returns:
        String concatenada
    """
    parts = []
    for key in ["s1", "s2", "s3"]:
        if key in llm_answer and llm_answer[key]:
            parts.append(llm_answer[key])
    
    return "\n".join(parts)


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


def get_all_instances_for_experiments(
    sti_experiment_id: ObjectId,
    mti_experiment_id: ObjectId,
    is_test: bool = False
) -> List[Dict]:
    """
    Obtém todas as instâncias comuns entre experimentos STI e MTI.
    
    Args:
        sti_experiment_id: ID do experimento STI
        mti_experiment_id: ID do experimento MTI
        is_test: Se True, retorna apenas 10 instâncias para teste
    
    Returns:
        Lista de dicts com instance_id, task_id, sti_doc, mti_doc
    """
    # Buscar todas as respostas STI
    sti_answers = list(answer_collection.find({"experiment_id": sti_experiment_id}))
    
    # Criar mapa de instance_id para documento STI
    sti_map = {doc["instance_id"]: doc for doc in sti_answers}
    
    # Buscar todas as respostas MTI
    mti_answers = list(answer_collection.find({"experiment_id": mti_experiment_id}))
    
    # Encontrar instâncias comuns
    instances = []
    for mti_doc in mti_answers:
        instance_id = mti_doc["instance_id"]
        if instance_id in sti_map:
            instances.append({
                "instance_id": instance_id,
                "task_id": mti_doc["task_id"],
                "sti_doc": sti_map[instance_id],
                "mti_doc": mti_doc
            })
    
    # Se modo teste, limitar a 10 instâncias
    if is_test and len(instances) > 10:
        instances = instances[:5]
    
    return instances


def create_llm_judge_document(
    llm_judge_experiment_id: ObjectId,
    sti_experiment_id: ObjectId,
    mti_experiment_id: ObjectId,
    task_id: str,
    instance_id: str,
    sti_answer_doc: Dict,
    mti_answer_doc: Dict,
    evaluator_model: str,
    experiment_name: str
) -> Dict | None:
    """
    Cria um documento de LLM Judge avaliando respostas STI e MTI.
    
    Args:
        sti_experiment_id: ID do experimento STI
        mti_experiment_id: ID do experimento MTI
        task_id: ID da task
        instance_id: ID da instância
        sti_answer_doc: Documento de resposta STI do MongoDB
        mti_answer_doc: Documento de resposta MTI do MongoDB
        evaluator_model: Nome do modelo usado para avaliação
        experiment_name: Nome do experimento LLM Judge
    
    Returns:
        Documento pronto para inserir na collection llm_judge_results
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
    question = mti_answer_doc.get("prompt", "")
    
    llm_judge_prompt = LLM_JUDGE_EVALUATION_TEMPLATE.format(
        question=question,
        answer_a=answer_a,
        answer_b=answer_b
    )
    
    # Carregar modelo avaliador e gerar avaliação
    try:
        model, tokenizer = load_hf_lm_and_tokenizer(evaluator_model)
        
        print(f"  Avaliando instância {instance_id} da task {task_id}...")
        # Utilizar os mesmos parâmetros de geração do PAPER ORIGINAL ???
        # PARAMETROS UTILIZADOS NO GT
        generation_time, generated_texts = generate_completions(
        model,
        tokenizer,
        [llm_judge_prompt],
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
            "llm_judge_experiment_id": llm_judge_experiment_id,
            "experiment_name": experiment_name,
            "sti_experiment_id": sti_experiment_id,
            "mti_experiment_id": mti_experiment_id,
            "task_id": task_id,
            "instance_id": instance_id,
            "evaluator_model": evaluator_model,
            "evaluation_timestamp": datetime.now(),
            
            # Prompts utilizados
            "llm_judge_MTI_prompt": mti_answer_doc.get("prompt", ""),
            "llm_judge_STI_prompt": sti_answer_doc.get("prompt", ""),
            "llm_judge_prompt": llm_judge_prompt,
            
            # Dados MTI
            "llm_answer_MTI": {
                "s1": mti_answer,
                "position_answer_to_llm": position_mti
            },
            
            # Dados STI
            "llm_answer_STI": {
                **sti_answer_doc["llm_answer"],
                "position_answer_to_llm": position_sti
            },
            
            # Resposta de avaliação
            "llm_judge_raw_response": evaluation_response,
            
            # Avaliações MTI
            "llm_judge_MTI_answer": {
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
            "llm_judge_STI_answer": {
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
    
    except Exception as e:
        print(f"❌ Erro ao avaliar instância {instance_id}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Gera LLM Judge comparando experimentos STI e MTI usando o mesmo modelo do experimento"
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
        "--experiment_name",
        type=str,
        required=True,
        help="Nome do experimento LLM Judge"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed para reprodutibilidade"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=10,
        help="Número de documentos processados antes de salvar no MongoDB"
    )
    parser.add_argument(
        "--is_test",
        action="store_true",
        help="Modo teste: processa apenas 10 instâncias para validação rápida"
    )
    parser.add_argument(
        "--experiment_id",
        type=str,
        default=None,
        help="ID de um experimento LLM Judge existente para retomar (opcional)"
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
    
    # Verificar se os modelos são iguais
    if sti_exp["model_name"] != mti_exp["model_name"]:
        print(f"⚠️ Aviso: Experimentos têm modelos diferentes!")
        print(f"  STI: {sti_exp['model_name']}")
        print(f"  MTI: {mti_exp['model_name']}")
    
    #  Juiz mockado
    evaluator_model = "gpt-4o-mini-2024-07-18"
    
    # Criar registro do experimento LLM Judge
    llm_judge_experiment_id = create_llm_judge_experiment_record(
        experiment_name=args.experiment_name,
        sti_experiment_id=sti_exp_id,
        mti_experiment_id=mti_exp_id,
        evaluator_model=evaluator_model,
        batch_size=args.batch_size,
        seed=args.seed,
        is_test=args.is_test,
        experiment_id=args.experiment_id
    )
    
    print("\n" + "="*70)
    print(f"🚀 Iniciando LLM Judge")
    print(f"🆔 Experimento LLM Judge ID: {llm_judge_experiment_id}")
    print(f"📊 Experimento STI: {sti_exp['experiment_name']} (ID: {sti_exp_id})")
    print(f"📊 Experimento MTI: {mti_exp['experiment_name']} (ID: {mti_exp_id})")
    print(f"🤖 Modelo Avaliador: {evaluator_model}")
    print(f"📝 Nome do Experimento: {args.experiment_name}")
    print(f"🔧 Modo: {'TESTE (10 instâncias)' if args.is_test else 'COMPLETO (todas as instâncias)'}")
    print("="*70 + "\n")
    
    # Obter todas as instâncias comuns
    instances = get_all_instances_for_experiments(sti_exp_id, mti_exp_id, args.is_test)
    
    print(f"✅ Encontradas {len(instances)} instâncias comuns\n")
    
    if len(instances) == 0:
        print("❌ Nenhuma instância comum encontrada entre os experimentos. Encerrando.")
        finalize_llm_judge_experiment(llm_judge_experiment_id)
        return
    
    # Filtrar instâncias já processadas
    pending_instances = []
    already_processed = 0
    
    for instance in instances:
        if check_instance_already_evaluated(llm_judge_experiment_id, instance["instance_id"]):
            already_processed += 1
        else:
            pending_instances.append(instance)
    
    if already_processed > 0:
        print(f"⚙️  {already_processed} instâncias já processadas anteriormente. Pulando...")
    
    if len(pending_instances) == 0:
        print("✅ Todas as instâncias já foram processadas. Nada a fazer.")
        finalize_llm_judge_experiment(llm_judge_experiment_id)
        return
    
    print(f"📝 Processando {len(pending_instances)} instâncias restantes\n")
    
    # Processar instâncias
    documents_to_insert = []
    successful_evaluations = 0
    failed_evaluations = 0
    
    for i, instance in enumerate(tqdm(pending_instances, desc="Processando instâncias")):
        try:
            doc = create_llm_judge_document(
                llm_judge_experiment_id=llm_judge_experiment_id,
                sti_experiment_id=sti_exp_id,
                mti_experiment_id=mti_exp_id,
                task_id=instance["task_id"],
                instance_id=instance["instance_id"],
                sti_answer_doc=instance["sti_doc"],
                mti_answer_doc=instance["mti_doc"],
                evaluator_model=evaluator_model,
                experiment_name=args.experiment_name
            )
            
            if doc:
                documents_to_insert.append(doc)
                successful_evaluations += 1
            else:
                failed_evaluations += 1
        
        except Exception as e:
            print(f"\n❌ Erro ao processar instância {instance['instance_id']}: {e}")
            save_error_checkpoint(llm_judge_experiment_id, instance["instance_id"], str(e))
            failed_evaluations += 1
        
        # Salvar em lotes
        if len(documents_to_insert) >= args.batch_size:
            llm_judge_collection.insert_many(documents_to_insert)
            print(f"\n💾 {len(documents_to_insert)} documentos salvos no MongoDB")
            documents_to_insert = []
    
    # Salvar documentos restantes
    if documents_to_insert:
        llm_judge_collection.insert_many(documents_to_insert)
        print(f"\n💾 {len(documents_to_insert)} documentos finais salvos no MongoDB")
    
    # Finalizar experimento
    finalize_llm_judge_experiment(llm_judge_experiment_id)
    
    # Resumo
    print("\n" + "="*70)
    print("📊 Resumo da Execução")
    print("="*70)
    print(f"🆔 Experimento ID: {llm_judge_experiment_id}")
    print(f"✅ Avaliações bem-sucedidas: {successful_evaluations}")
    print(f"❌ Avaliações falhadas: {failed_evaluations}")
    print(f"📊 Total processado: {successful_evaluations + failed_evaluations}")
    print(f"💾 Collection: llm_judge_results")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()