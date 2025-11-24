"""
Script para centralizar metadados de experimentos de LLM Judge.

Este script:
1. Extrai valores únicos de experiment_name da collection llm_judge_results
2. Cria documentos centralizados na collection experiments_llm_judge
3. Vincula os documentos de llm_judge_results aos experimentos criados via llm_judge_experiment_id
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
llm_judge_results_collection = db["llm_judge_results"]
llm_judge_experiments_collection = db["experiments_llm_judge"]


def get_unique_experiment_names() -> List[str]:
    """
    Extrai valores únicos de experiment_name da collection llm_judge_results.
    
    Returns:
        Lista de nomes únicos de experimentos LLM Judge
    """
    experiment_names = llm_judge_results_collection.distinct("experiment_name")
    return [name for name in experiment_names if name]  # Filtrar valores nulos


def create_experiment_llm_judge_document(experiment_name: str) -> ObjectId:
    """
    Cria um documento na collection experiments_llm_judge com metadados únicos
    extraídos da collection llm_judge_results.
    
    Args:
        experiment_name: Nome do experimento LLM Judge
    
    Returns:
        ObjectId do documento criado
    """
    # Buscar um documento representativo deste experimento para extrair metadados
    sample_doc = llm_judge_results_collection.find_one({"experiment_name": experiment_name})
    
    if not sample_doc:
        raise ValueError(f"Nenhum documento encontrado para experiment_name: {experiment_name}")
    
    # Extrair metadados únicos do experimento
    experiment_doc = {
        "experiment_name": experiment_name,
        "evaluator_model": sample_doc.get("evaluator_model"),
        "evaluation_timestamp": sample_doc.get("evaluation_timestamp"),
        "sti_experiment_id": sample_doc.get("sti_experiment_id"),
        "mti_experiment_id": sample_doc.get("mti_experiment_id"),
        "created_at": datetime.now()
    }
    
    # Verificar se já existe um documento para este experimento
    existing = llm_judge_experiments_collection.find_one({"experiment_name": experiment_name})
    
    if existing:
        print(f"⚠️  Experimento '{experiment_name}' já existe com ID: {existing['_id']}")
        return existing["_id"]
    
    # Inserir documento
    result = llm_judge_experiments_collection.insert_one(experiment_doc)
    print(f"✅ Experimento '{experiment_name}' criado com ID: {result.inserted_id}")
    
    return result.inserted_id


def link_llm_judge_results_to_experiment(experiment_name: str, llm_judge_experiment_id: ObjectId) -> int:
    """
    Adiciona o campo llm_judge_experiment_id a todos os documentos da collection
    llm_judge_results que possuem o experiment_name especificado.
    
    Args:
        experiment_name: Nome do experimento LLM Judge
        llm_judge_experiment_id: ObjectId do documento na collection experiments_llm_judge
    
    Returns:
        Número de documentos atualizados
    """
    result = llm_judge_results_collection.update_many(
        {"experiment_name": experiment_name},
        {"$set": {"llm_judge_experiment_id": llm_judge_experiment_id}}
    )
    
    print(f"✅ {result.modified_count} documentos vinculados ao experimento '{experiment_name}'")
    
    return result.modified_count


def process_all_experiments() -> Dict[str, ObjectId]:
    """
    Processa todos os experimentos únicos:
    1. Cria documentos na collection experiments_llm_judge
    2. Vincula documentos de llm_judge_results aos experimentos
    
    Returns:
        Dict mapeando experiment_name para llm_judge_experiment_id
    """
    experiment_names = get_unique_experiment_names()
    
    if not experiment_names:
        print("⚠️  Nenhum experimento encontrado na collection llm_judge_results")
        return {}
    
    print(f"\n📊 Encontrados {len(experiment_names)} experimentos únicos:")
    for name in experiment_names:
        print(f"  - {name}")
    print()
    
    experiment_mapping = {}
    
    for experiment_name in experiment_names:
        try:
            # Criar documento de experimento
            llm_judge_experiment_id = create_experiment_llm_judge_document(experiment_name)
            
            # Vincular resultados ao experimento
            link_llm_judge_results_to_experiment(experiment_name, llm_judge_experiment_id)
            
            experiment_mapping[experiment_name] = llm_judge_experiment_id
            
        except Exception as e:
            print(f"❌ Erro ao processar experimento '{experiment_name}': {e}")
    
    return experiment_mapping


def main():
    """
    Função principal que orquestra o processo de centralização de experimentos LLM Judge.
    """
    print("\n🚀 Iniciando centralização de metadados de experimentos LLM Judge\n")
    
    # Processar todos os experimentos
    experiment_mapping = process_all_experiments()
    
    if experiment_mapping:
        print(f"\n✅ Processo concluído com sucesso!")
        print(f"📝 {len(experiment_mapping)} experimentos processados:")
        for exp_name, exp_id in experiment_mapping.items():
            # Contar documentos vinculados
            count = llm_judge_results_collection.count_documents({"experiment_name": exp_name})
            print(f"  - {exp_name}: {count} documentos (ID: {exp_id})")
    else:
        print("\n⚠️  Nenhum experimento foi processado")
    
    print("\n" + "="*70)
    print("Collections atualizadas:")
    print(f"  - experiments_llm_judge: {llm_judge_experiments_collection.count_documents({})} documentos")
    print(f"  - llm_judge_results: {llm_judge_results_collection.count_documents({'llm_judge_experiment_id': {'$exists': True}})} documentos com llm_judge_experiment_id")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
