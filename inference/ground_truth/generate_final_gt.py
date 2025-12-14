
"""
Script para criação da collection final_ground_truth agregando múltiplos experimentos de avaliação.

Este script:
1. Recebe uma lista de nomes de experimentos GT
2. Busca os metadados dos experimentos em experiments_ground_truth
3. Para cada par (task_id, instance_id), agrega as avaliações de todos os experimentos
4. Calcula a mediana dos scores para cada métrica
5. Cria registros consolidados na collection final_ground_truth

"""

import os
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
gt_results_collection = db["ground_truth_results"]
gt_experiments_collection = db["experiments_ground_truth"]
final_gt_collection = db["final_ground_truth"]



def get_experiment_models(experiment_names: List[str]) -> Dict[str, str]:
    """
    Busca os modelos utilizados em cada experimento.
    
    Args:
        experiment_names: Lista de nomes de experimentos GT
    
    Returns:
        Dict mapeando experiment_gt_name para evaluator_model
    """
    experiments = gt_experiments_collection.find({
        "experiment_gt_name": {"$in": experiment_names}
    })
    
    model_mapping = {}
    for exp in experiments:
        model_mapping[exp["experiment_gt_name"]] = exp["evaluator_model"]
    
    return model_mapping


def get_unique_instances(experiment_names: List[str]) -> List[Dict[str, str]]:
    """
    Obtém todas as combinações únicas de (task_id, instance_id) dos experimentos.
    
    Args:
        experiment_names: Lista de nomes de experimentos GT
    
    Returns:
        Lista de dicts com task_id e instance_id únicos
    """
    pipeline = [
        {"$match": {"experiment_gt_name": {"$in": experiment_names}}},
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
    
    return list(gt_results_collection.aggregate(pipeline))


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
        exp_name = eval_doc["experiment_gt_name"]
        model_name = models_map.get(exp_name, exp_name)
        
        metric_data = eval_doc.get(metric_name, {})
        
        if "score" in metric_data:
            score = metric_data["score"]
            result[f"score_{model_name}"] = score
            scores.append(score)
        
        if "explanation" in metric_data:
            result[f"explanation_{model_name}"] = metric_data["explanation"]
    
    # Calcular mediana dos scores
    if scores:
        result["median_score"] = statistics.median(scores)
    
    return result


def aggregate_answer_evaluations(evaluations: List[Dict], answer_type: str, models_map: Dict[str, str]) -> Dict:
    """
    Agrega todas as métricas de avaliação (gt_MTI_answer ou gt_STI_answer).
    
    Args:
        evaluations: Lista de avaliações de diferentes experimentos
        answer_type: "gt_MTI_answer" ou "gt_STI_answer"
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
                    "experiment_gt_name": eval_doc["experiment_gt_name"],
                    metric: answer_data[metric]
                })
        
        if metric_evaluations:
            aggregated[metric] = aggregate_metric_scores(metric_evaluations, metric, models_map)
    
    return aggregated


def create_final_gt_document(
    task_id: str,
    instance_id: str,
    experiment_names: List[str],
    models_map: Dict[str, str]
) -> Dict | None:
    """
    Cria um documento consolidado para a collection final_ground_truth.
    
    Args:
        task_id: ID da task
        instance_id: ID da instância
        experiment_names: Lista de nomes de experimentos a agregar
        models_map: Mapeamento de experiment_name para model_name
    
    Returns:
        Documento pronto para inserir na collection final_ground_truth
    """
    # Buscar todas as avaliações desta instância nos experimentos especificados
    evaluations = list(gt_results_collection.find({
        "experiment_gt_name": {"$in": experiment_names},
        "task_id": task_id,
        "instance_id": instance_id
    }))
    
    if not evaluations:
        return None
    
    # Pegar informações comuns do primeiro documento
    first_eval = evaluations[0]
    
    # Agregar avaliações MTI
    gt_mti_aggregated = aggregate_answer_evaluations(evaluations, "gt_MTI_answer", models_map)
    
    # Agregar avaliações STI
    gt_sti_aggregated = aggregate_answer_evaluations(evaluations, "gt_STI_answer", models_map)
    
    # Construir documento final
    final_doc = {
        "experiments": experiment_names,
        "models": list(models_map.values()),
        "task_id": task_id,
        "instance_id": instance_id,
        "prompt_MTI": first_eval.get("prompt_MTI", ""),
        "gt_prompt": first_eval.get("prompt_MTI", ""),  # Assumindo que gt_prompt é o mesmo que prompt_MTI
        "gt_MTI_answer": gt_mti_aggregated,
        "gt_STI_answer": gt_sti_aggregated,
        "sti_experiment_id": first_eval.get("sti_experiment_id"),
        "mti_experiment_id": first_eval.get("mti_experiment_id"),
        "gt_experiment_ids": [eval_doc.get("gt_experiment_id") for eval_doc in evaluations if eval_doc.get("gt_experiment_id")]
    }
    
    return final_doc


def generate_final_ground_truth(experiment_names: List[str]) -> int:
    """
    Processa todos os pares (task_id, instance_id) e cria a collection final_ground_truth.
    
    Args:
        experiment_names: Lista de nomes de experimentos GT a agregar
    
    Returns:
        Número de documentos criados
    """
    print(f"\n🚀 Iniciando geração da collection final_ground_truth")
    print(f"📝 Experimentos: {experiment_names}\n")
    
    # Buscar modelos de cada experimento
    models_map = get_experiment_models(experiment_names)
    print(f"🤖 Modelos encontrados:")
    for exp_name, model in models_map.items():
        print(f"  - {exp_name}: {model}")
    print()
    
    # Verificar se todos os experimentos foram encontrados
    missing = set(experiment_names) - set(models_map.keys())
    if missing:
        print(f"⚠️  AVISO: Experimentos não encontrados: {missing}")
        print(f"   Continuando apenas com: {list(models_map.keys())}\n")
        experiment_names = list(models_map.keys())
    
    # Obter todas as combinações únicas de (task_id, instance_id)
    unique_instances = get_unique_instances(experiment_names)
    print(f"📊 Encontradas {len(unique_instances)} combinações únicas de (task_id, instance_id)\n")
    
    # Processar cada combinação
    documents_created = 0
    documents_to_insert = []
    
    print("🔄 Processando instâncias...")
    for instance_info in unique_instances:
        task_id = instance_info["task_id"]
        instance_id = instance_info["instance_id"]
        
        # Criar documento agregado
        final_doc = create_final_gt_document(
            task_id=task_id,
            instance_id=instance_id,
            experiment_names=experiment_names,
            models_map=models_map
        )
        
        if final_doc:
            documents_to_insert.append(final_doc)
            print(f"  ✅ {instance_id} (task {task_id})")
    
    # Inserir documentos em lote
    if documents_to_insert:
        print(f"\n💾 Inserindo {len(documents_to_insert)} documentos na collection final_ground_truth...")
        result = final_gt_collection.insert_many(documents_to_insert)
        documents_created = len(result.inserted_ids)
        print(f"✅ {documents_created} documentos criados com sucesso!")
    else:
        print("⚠️  Nenhum documento foi criado")
    
    return documents_created


def main():
    """
    Função principal para execução do script.
    """
    # Exemplo de uso - pode ser modificado ou parametrizado via argparse
    experiment_names = [
        "GT_gpt-4o-mini-2024-07-18_V3",
        "GT_llama-3.3-70b-versatile_V3",
        "GT_gpt-4o-2024-08-06_V3"
    ]
    
    print("\n" + "="*70)
    print("GERADOR DE FINAL GROUND TRUTH")
    print("="*70)
    
    # Gerar collection final_ground_truth
    num_docs = generate_final_ground_truth(experiment_names)
    
    print("\n" + "="*70)
    print("RESUMO:")
    print(f"  - Documentos criados: {num_docs}")
    print(f"  - Collection: final_ground_truth")
    print(f"  - Total de documentos na collection: {final_gt_collection.count_documents({})}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()

