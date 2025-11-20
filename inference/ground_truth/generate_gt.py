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

from evaluate.utils import load_hf_lm_and_tokenizer, generate_completions

# Conexão MongoDB
MONGO_URI = os.getenv("MONGODB_URI")

mongo_client = MongoClient(MONGO_URI)
db = mongo_client["experiments_db"]
experiment_collection = db["experiment_results"]
answer_collection = db["answer_results"]
gt_collection = db["ground_truth_results"]


# Template de prompt para avaliação
GT_EVALUATION_TEMPLATE = """You are an independent evaluation agent whose task is to assess the quality of answers produced by AI models. Your role is to rate each answer according to five human-preference criteria. You must do so with neutrality, precision, and consistency.

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
- Clarity: How easy the answer is to understand in terms of phrasing, structure, and readability.

Your output should follow exactly this template:
scores_a = {{"Coherence": X, "Specificity": Y, "Informativeness": Z, "Relevance": W, "Clarity": V}}
explanations_a = {{"Coherence": "...", "Specificity": "...", "Informativeness": "...", "Relevance": "...", "Clarity": "..."}}
scores_b = {{...}}
explanations_b = {{...}}

[User Input]
{question}

[Assistant A Response]
{answer_a}

[Assistant B Response]
{answer_b}
"""


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
            result += f"### Instruction{i}:\n\n{llm_answer[stage]}\n\n"
    
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
            "clarity": {
                "score": mti_scores.get("Clarity", 0),
                "explanation": mti_explanations.get("Clarity", "")
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
            "clarity": {
                "score": sti_scores.get("Clarity", 0),
                "explanation": sti_explanations.get("Clarity", "")
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
    sampled_instances = sample_instances(common_instances, instances_per_task)
    
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


