"""
Script para geração de LLM Judge a partir de experimentos STI e MTI.

Este script:
1. Seleciona pares de experimentos STI/MTI com o mesmo model_name
2. Para cada instância, constrói prompts de avaliação combinando respostas STI e MTI
3. Envia para o mesmo LLM usado no experimento para auto-avaliação
4. Salva resultados na collection 'llm_judge_results'
"""

from __future__ import annotations
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


# Template de prompt para avaliação
LLM_JUDGE_EVALUATION_TEMPLATE = "teste12345"


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
        instances = instances[:10]
    
    return instances


def create_llm_judge_document(
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
        
        generation_time, generated_texts = generate_completions(
            model,
            tokenizer,
            [llm_judge_prompt],
            batch_size=1,
            max_new_tokens=2048,
            min_new_tokens=100,
            do_sample=True,
            temperature=0.7,
            top_p=1
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
    
    evaluator_model = mti_exp["model_name"]
    
    print("\n" + "="*70)
    print(f"🚀 Iniciando LLM Judge")
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
        print("⚠️ Nenhuma instância comum encontrada entre os experimentos")
        return
    
    # Processar instâncias
    documents_to_insert = []
    successful_evaluations = 0
    failed_evaluations = 0
    
    for i, instance in enumerate(tqdm(instances, desc="Processando instâncias")):
        doc = create_llm_judge_document(
            sti_exp_id,
            mti_exp_id,
            instance["task_id"],
            instance["instance_id"],
            instance["sti_doc"],
            instance["mti_doc"],
            evaluator_model,
            args.experiment_name
        )
        
        if doc:
            documents_to_insert.append(doc)
            successful_evaluations += 1
        else:
            failed_evaluations += 1
        
        # Salvar em lotes
        if len(documents_to_insert) >= args.batch_size:
            llm_judge_collection.insert_many(documents_to_insert)
            print(f"\n✅ Salvos {len(documents_to_insert)} documentos no MongoDB")
            documents_to_insert = []
    
    # Salvar documentos restantes
    if documents_to_insert:
        llm_judge_collection.insert_many(documents_to_insert)
        print(f"\n✅ Salvos {len(documents_to_insert)} documentos finais no MongoDB")
    
    # Resumo
    print("\n" + "="*70)
    print("📊 Resumo da Execução")
    print("="*70)
    print(f"✅ Avaliações bem-sucedidas: {successful_evaluations}")
    print(f"❌ Avaliações falhadas: {failed_evaluations}")
    print(f"📝 Total de documentos criados: {successful_evaluations}")
    print(f"💾 Collection: llm_judge_results")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()