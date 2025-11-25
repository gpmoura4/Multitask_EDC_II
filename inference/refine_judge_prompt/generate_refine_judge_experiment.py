"""
Script para centralizar metadados de experimentos de Refine Judge.

Este script:
1. Extrai valores únicos de experiment_gt_name da collection refine_judge_results
2. Cria documentos centralizados na collection experiments_refine_judge
3. Vincula os documentos de refine_judge_results aos experimentos criados via refine_judge_experiment_id
"""

import os
from pymongo import MongoClient
from bson import ObjectId
from datetime import datetime
from typing import List, Dict
from dotenv import load_dotenv
load_dotenv()

# Conexão MongoDB
MONGO_URI = os.getenv("MONGODB_URI")
mongo_client = MongoClient(MONGO_URI)
db = mongo_client["experiments_db"]
refine_judge_results_collection = db["refine_judge_results"]
refine_judge_experiments_collection = db["experiments_refine_judge"]


def get_unique_experiment_names() -> List[str]:
    """
    Extrai valores únicos de experiment_gt_name da collection refine_judge_results.
    
    Returns:
        Lista de nomes únicos de experimentos GT
    """
    experiment_names = refine_judge_results_collection.distinct("experiment_gt_name")
    return [name for name in experiment_names if name]  # Filtrar valores nulos


def create_experiment_gt_document(experiment_gt_name: str) -> ObjectId:
    """
    Cria um documento na collection experiments_refine_judge com metadados únicos
    extraídos da collection refine_judge_results. 
    
    Args:
        experiment_gt_name: Nome do experimento GT
    
    Returns:
        ObjectId do documento criado
    """
    # Buscar um documento representativo deste experimento para extrair metadados
    sample_doc = refine_judge_results_collection.find_one({"experiment_gt_name": experiment_gt_name})
    
    if not sample_doc:
        raise ValueError(f"Nenhum documento encontrado para experiment_gt_name: {experiment_gt_name}")
    
    # Extrair metadados únicos do experimento
    experiment_doc = {
        "experiment_gt_name": experiment_gt_name,
        "evaluator_model": sample_doc.get("evaluator_model"),
        "evaluation_timestamp": sample_doc.get("evaluation_timestamp"),
        "sti_experiment_id": sample_doc.get("sti_experiment_id"),
        "mti_experiment_id": sample_doc.get("mti_experiment_id"),
        "created_at": datetime.now()
    }
    
    # Verificar se já existe um documento para este experimento
    existing = refine_judge_experiments_collection.find_one({"experiment_gt_name": experiment_gt_name})
    
    if existing:
        print(f"⚠️  Experimento '{experiment_gt_name}' já existe com ID: {existing['_id']}")
        return existing["_id"]
    
    # Inserir documento
    result = refine_judge_experiments_collection.insert_one(experiment_doc)
    print(f"✅ Experimento '{experiment_gt_name}' criado com ID: {result.inserted_id}")
    
    return result.inserted_id


def link_gt_results_to_experiment(experiment_gt_name: str, refine_judge_experiment_id: ObjectId) -> int:
    """
    Adiciona o campo refine_judge_experiment_id a todos os documentos da collection
    refine_judge_results que possuem o experiment_gt_name especificado.
    
    Args:
        experiment_gt_name: Nome do experimento GT
        refine_judge_experiment_id: ObjectId do documento na collection experiments_refine_judge
    
    Returns:
        Número de documentos atualizados
    """
    result = refine_judge_results_collection.update_many(
        {"experiment_gt_name": experiment_gt_name},
        {"$set": {"refine_judge_experiment_id": refine_judge_experiment_id}}
    )
    
    print(f"✅ {result.modified_count} documentos vinculados ao experimento '{experiment_gt_name}'")
    
    return result.modified_count


def process_all_experiments() -> Dict[str, ObjectId]:
    """
    Processa todos os experimentos únicos:
    1. Cria documentos na collection experiments_refine_judge
    2. Vincula documentos de refine_judge_results aos experimentos
    
    Returns:
        Dict mapeando experiment_gt_name para refine_judge_experiment_id
    """
    experiment_names = get_unique_experiment_names()
    
    if not experiment_names:
        print("⚠️  Nenhum experimento encontrado na collection refine_judge_results")
        return {}
    
    print(f"\n📊 Encontrados {len(experiment_names)} experimentos únicos:")
    for name in experiment_names:
        print(f"  - {name}")
    print()
    
    experiment_mapping = {}
    
    for experiment_name in experiment_names:
        try:
            # Criar documento de experimento
            refine_judge_experiment_id = create_experiment_gt_document(experiment_name)
            
            # Vincular resultados ao experimento
            link_gt_results_to_experiment(experiment_name, refine_judge_experiment_id)
            
            experiment_mapping[experiment_name] = refine_judge_experiment_id
            
        except Exception as e:
            print(f"❌ Erro ao processar experimento '{experiment_name}': {e}")
    
    return experiment_mapping


def main():
    """
    Função principal que orquestra o processo de centralização de experimentos GT.
    """
    print("\n🚀 Iniciando centralização de metadados de experimentos GT\n")
    
    # Processar todos os experimentos
    experiment_mapping = process_all_experiments()
    
    if experiment_mapping:
        print(f"\n✅ Processo concluído com sucesso!")
        print(f"📝 {len(experiment_mapping)} experimentos processados:")
        for exp_name, exp_id in experiment_mapping.items():
            # Contar documentos vinculados
            count = refine_judge_results_collection.count_documents({"experiment_gt_name": exp_name})
            print(f"  - {exp_name}: {count} documentos (ID: {exp_id})")
    else:
        print("\n⚠️  Nenhum experimento foi processado")
    
    print("\n" + "="*70)
    print("Collections atualizadas:")
    print(f"  - experiments_refine_judge: {refine_judge_experiments_collection.count_documents({})} documentos")
    print(f"  - refine_judge_results: {refine_judge_results_collection.count_documents({'refine_judge_experiment_id': {'$exists': True}})} documentos com refine_judge_experiment_id")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()

