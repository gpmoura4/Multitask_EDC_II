"""
Script para deletar registros de ground truth baseado no ID de um experimento STI.

Remove todos os documentos da collection 'ground_truth_results' onde o campo
'sti_experiment_id' corresponde ao ID fornecido.
"""

from bson import ObjectId
from pymongo import MongoClient
import os
from dotenv import load_dotenv

load_dotenv()

# Conexão MongoDB
MONGO_URI = os.getenv("MONGODB_URI")

print('MongoDB URI:', MONGO_URI)

# Criar conexão
mongo_client = MongoClient(MONGO_URI)
db = mongo_client["experiments_db"]
gt_collection = db["ground_truth_results"]


def delete_ground_truth_by_sti_experiment(sti_experiment_id: str) -> dict:
    """
    Remove todos os documentos de ground truth associados a um experimento STI.
    
    Args:
        sti_experiment_id: ID do experimento STI (string do ObjectId)
    
    Returns:
        Dict com informações sobre a remoção
    """
    
    # Converter o id recebido para ObjectId
    try:
        sti_exp_obj_id = ObjectId(sti_experiment_id)
    except Exception as e:
        raise ValueError(f"sti_experiment_id inválido: {sti_experiment_id}") from e
    
    # Buscar quantos documentos existem antes de deletar
    count_before = gt_collection.count_documents({"sti_experiment_id": sti_exp_obj_id})
    
    print(f"📊 Encontrados {count_before} documentos GT vinculados ao experimento STI {sti_experiment_id}")
    
    if count_before == 0:
        print("⚠️  Nenhum documento encontrado para deletar.")
        return {
            "sti_experiment_id": sti_experiment_id,
            "gt_documents_deleted": 0,
            "message": "Nenhum documento encontrado"
        }
    
    # Deletar todos os documentos relacionados
    delete_result = gt_collection.delete_many({"sti_experiment_id": sti_exp_obj_id})
    
    print(f"✅ {delete_result.deleted_count} documentos GT deletados com sucesso!")
    
    return {
        "sti_experiment_id": sti_experiment_id,
        "gt_documents_deleted": delete_result.deleted_count,
        "message": f"Deletados {delete_result.deleted_count} documentos"
    }


def delete_ground_truth_by_experiment_name(experiment_gt_name: str) -> dict:
    """
    Remove todos os documentos de ground truth associados a um nome de experimento GT.
    
    Args:
        experiment_gt_name: Nome do experimento GT
    
    Returns:
        Dict com informações sobre a remoção
    """
    
    # Buscar quantos documentos existem antes de deletar
    count_before = gt_collection.count_documents({"experiment_gt_name": experiment_gt_name})
    
    print(f"📊 Encontrados {count_before} documentos GT com nome '{experiment_gt_name}'")
    
    if count_before == 0:
        print("⚠️  Nenhum documento encontrado para deletar.")
        return {
            "experiment_gt_name": experiment_gt_name,
            "gt_documents_deleted": 0,
            "message": "Nenhum documento encontrado"
        }
    
    # Deletar todos os documentos relacionados
    delete_result = gt_collection.delete_many({"experiment_gt_name": experiment_gt_name})
    
    print(f"✅ {delete_result.deleted_count} documentos GT deletados com sucesso!")
    
    return {
        "experiment_gt_name": experiment_gt_name,
        "gt_documents_deleted": delete_result.deleted_count,
        "message": f"Deletados {delete_result.deleted_count} documentos"
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Deleta documentos de ground truth por experimento STI ou nome do experimento GT"
    )
    parser.add_argument(
        "--sti_experiment_id",
        type=str,
        help="ID do experimento STI (ObjectId)"
    )
    parser.add_argument(
        "--experiment_gt_name",
        type=str,
        help="Nome do experimento GT"
    )
    
    args = parser.parse_args()
    
    # Validar que pelo menos um argumento foi fornecido
    if not args.sti_experiment_id and not args.experiment_gt_name:
        print("❌ Erro: Forneça --sti_experiment_id ou --experiment_gt_name")
        parser.print_help()
        exit(1)
    
    # Deletar baseado no critério fornecido
    if args.sti_experiment_id:
        print(f"\n🗑️  Deletando documentos GT do experimento STI: {args.sti_experiment_id}\n")
        result = delete_ground_truth_by_sti_experiment(args.sti_experiment_id)
    else:
        print(f"\n🗑️  Deletando documentos GT com nome: {args.experiment_gt_name}\n")
        result = delete_ground_truth_by_experiment_name(args.experiment_gt_name)
    
    print("\n📋 Resultado:")
    print(result)
