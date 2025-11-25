"""
Script para criação da collection final_llm_judge agregando múltiplos experimentos de avaliação LLM Judge.

Este script:
1. Recebe uma lista de nomes de experimentos LLM Judge
2. Busca os metadados dos experimentos em experiments_llm_judge
3. Para cada par (task_id, instance_id), agrega as avaliações de todos os experimentos
4. Calcula a mediana dos scores para cada métrica
5. Cria registros consolidados na collection final_llm_judge
"""

import os
import argparse
from pymongo import MongoClient
from bson import ObjectId
from typing import List, Dict
from dotenv import load_dotenv
import statistics

load_dotenv()

# Conexão MongoDB
MONGO_URI = os.getenv("MONGODB_URI")
mongo_client = MongoClient(MONGO_URI)
db = mongo_client["experiments_db"]
llm_judge_results_collection = db["llm_judge_results"]
llm_judge_experiments_collection = db["experiments_llm_judge"]
final_llm_judge_collection = db["final_llm_judge"]


def get_experiment_models(experiment_names: List[str]) -> Dict[str, str]:
    """
    Busca os modelos utilizados em cada experimento.
    
    Args:
        experiment_names: Lista de nomes de experimentos LLM Judge
    
    Returns:
        Dict mapeando experiment_name para evaluator_model
    """
    experiments = llm_judge_experiments_collection.find({
        "experiment_name": {"$in": experiment_names}
    })
    
    model_mapping = {}
    for exp in experiments:
        model_mapping[exp["experiment_name"]] = exp["evaluator_model"]
    
    return model_mapping


def get_unique_instances(experiment_names: List[str]) -> List[Dict[str, str]]:
    """
    Obtém todas as combinações únicas de (task_id, instance_id) dos experimentos.
    
    Args:
        experiment_names: Lista de nomes de experimentos LLM Judge
    
    Returns:
        Lista de dicts com task_id e instance_id únicos
    """
    pipeline = [
        {"$match": {"experiment_name": {"$in": experiment_names}}},
        {"$group": {
            "_id": {
                "task_id": "$task_id",
                "instance_id": "$instance_id"
            }
        }},
        {"$project": {
            "_id": 0,
            "task_id": "$_id.task_id",
            "instance_id": "$_id.instance_id"
        }}
    ]
    
    return list(llm_judge_results_collection.aggregate(pipeline))


def aggregate_metric_scores(evaluations: List[Dict], metric_name: str, models_map: Dict[str, str]) -> Dict:
    """
    Agrega scores e explanations de uma métrica específica de múltiplos experimentos.
    
    Args:
        evaluations: Lista de avaliações de diferentes experimentos
        metric_name: Nome da métrica (coherence, specificity, etc)
        models_map: Mapeamento de experiment_name para model_name
    
    Returns:
        Dict com scores e explanations por modelo, incluindo median_score
    """
    result = {}
    scores = []
    
    for eval_doc in evaluations:
        experiment_name = eval_doc["experiment_name"]
        model_name = models_map.get(experiment_name, "unknown")
        
        metric_data = eval_doc.get(metric_name, {})
        score = metric_data.get("score", 0)
        explanation = metric_data.get("explanation", "")
        
        result[f"score_{model_name}"] = score
        result[f"explanation_{model_name}"] = explanation
        
        if score > 0:
            scores.append(score)
    
    # Calcular mediana dos scores
    if scores:
        result["median_score"] = statistics.median(scores)
    
    return result


def aggregate_answer_evaluations(evaluations: List[Dict], answer_type: str, models_map: Dict[str, str]) -> Dict:
    """
    Agrega todas as métricas de avaliação (llm_judge_MTI_answer ou llm_judge_STI_answer).
    
    Args:
        evaluations: Lista de avaliações de diferentes experimentos
        answer_type: "llm_judge_MTI_answer" ou "llm_judge_STI_answer"
        models_map: Mapeamento de experiment_name para model_name
    
    Returns:
        Dict com todas as métricas agregadas
    """
    # Métricas a serem agregadas
    metrics = ["coherence", "specificity", "informativeness", "relevance", "Understandability"]
    
    aggregated = {}
    
    for metric in metrics:
        # Extrair dados da métrica de cada avaliação
        metric_evaluations = []
        for eval_doc in evaluations:
            answer_data = eval_doc.get(answer_type, {})
            if metric in answer_data:
                metric_evaluations.append({
                    "experiment_name": eval_doc["experiment_name"],
                    metric: answer_data[metric]
                })
        
        # Agregar scores e explanations
        aggregated[metric] = aggregate_metric_scores(metric_evaluations, metric, models_map)
    
    return aggregated


def create_final_llm_judge_document(
    task_id: str,
    instance_id: str,
    experiment_names: List[str],
    models_map: Dict[str, str]
) -> Dict | None:
    """
    Cria um documento consolidado para a collection final_llm_judge.
    
    Args:
        task_id: ID da task
        instance_id: ID da instância
        experiment_names: Lista de nomes de experimentos a agregar
        models_map: Mapeamento de experiment_name para model_name
    
    Returns:
        Documento pronto para inserir na collection final_llm_judge
    """
    # Buscar todas as avaliações desta instância nos experimentos especificados
    evaluations = list(llm_judge_results_collection.find({
        "experiment_name": {"$in": experiment_names},
        "task_id": task_id,
        "instance_id": instance_id
    }))
    
    if not evaluations:
        return None
    
    # Usar o primeiro documento como base
    base_doc = evaluations[0]
    
    # Agregar avaliações MTI
    llm_judge_mti_aggregated = aggregate_answer_evaluations(
        evaluations,
        "llm_judge_MTI_answer",
        models_map
    )
    
    # Agregar avaliações STI
    llm_judge_sti_aggregated = aggregate_answer_evaluations(
        evaluations,
        "llm_judge_STI_answer",
        models_map
    )
    
    # Construir documento final
    doc = {
        "experiments": experiment_names,
        "models": [models_map[exp_name] for exp_name in experiment_names if exp_name in models_map],
        "task_id": task_id,
        "instance_id": instance_id,
        "llm_judge_MTI_prompt": base_doc.get("llm_judge_MTI_prompt", ""),
        "llm_judge_STI_prompt": base_doc.get("llm_judge_STI_prompt", ""),
        "llm_judge_prompt": base_doc.get("llm_judge_prompt", ""),
        "llm_judge_MTI_answer": llm_judge_mti_aggregated,
        "llm_judge_STI_answer": llm_judge_sti_aggregated
    }
    
    return doc


def generate_final_llm_judge(experiment_names: List[str]) -> int:
    """
    Gera a collection final_llm_judge agregando múltiplos experimentos.
    
    Args:
        experiment_names: Lista de nomes de experimentos LLM Judge a agregar
    
    Returns:
        Número de documentos criados
    """
    print("\n" + "="*70)
    print("🚀 Iniciando geração de final_llm_judge")
    print("="*70)
    print(f"📊 Experimentos a agregar: {len(experiment_names)}")
    for exp_name in experiment_names:
        print(f"  - {exp_name}")
    print()
    
    # Buscar modelos dos experimentos
    models_map = get_experiment_models(experiment_names)
    
    if len(models_map) != len(experiment_names):
        print("⚠️ Aviso: Alguns experimentos não foram encontrados em experiments_llm_judge")
        print(f"  Esperados: {len(experiment_names)}")
        print(f"  Encontrados: {len(models_map)}")
    
    print(f"🤖 Modelos utilizados:")
    for exp_name, model in models_map.items():
        print(f"  - {exp_name}: {model}")
    print()
    
    # Obter todas as instâncias únicas
    instances = get_unique_instances(experiment_names)
    
    print(f"✅ Encontradas {len(instances)} instâncias únicas\n")
    
    if len(instances) == 0:
        print("⚠️ Nenhuma instância encontrada para os experimentos especificados")
        return 0
    
    # Processar cada instância
    documents_created = 0
    
    for instance in instances:
        doc = create_final_llm_judge_document(
            instance["task_id"],
            instance["instance_id"],
            experiment_names,
            models_map
        )
        
        if doc:
            # Verificar se documento já existe
            existing = final_llm_judge_collection.find_one({
                "task_id": instance["task_id"],
                "instance_id": instance["instance_id"],
                "experiments": {"$all": experiment_names}
            })
            
            if existing:
                # Atualizar documento existente
                final_llm_judge_collection.replace_one(
                    {"_id": existing["_id"]},
                    doc
                )
            else:
                # Inserir novo documento
                final_llm_judge_collection.insert_one(doc)
            
            documents_created += 1
    
    # Resumo
    print("\n" + "="*70)
    print("📊 Resumo da Geração")
    print("="*70)
    print(f"✅ Documentos criados/atualizados: {documents_created}")
    print(f"💾 Collection: final_llm_judge")
    print("="*70 + "\n")
    
    return documents_created


def main():
    parser = argparse.ArgumentParser(
        description="Agrega múltiplos experimentos LLM Judge em uma collection final"
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        required=True,
        help="Lista de nomes de experimentos LLM Judge a agregar"
    )
    
    args = parser.parse_args()
    
    # Gerar final_llm_judge
    documents_created = generate_final_llm_judge(args.experiments)
    
    if documents_created > 0:
        print(f"✅ Processo concluído com sucesso!")
        print(f"📝 {documents_created} documentos criados na collection final_llm_judge")
    else:
        print("⚠️ Nenhum documento foi criado")


if __name__ == "__main__":
    main()
