from bson import ObjectId
from pymongo import MongoClient
from datetime import datetime
import os
from dotenv import load_dotenv
load_dotenv()

# Conexão MongoDB (exemplo para teste, substitua pela uri real)
MONGO_URI = os.getenv("MONGODB_URI")

print('mongo uri:', MONGO_URI)

# Cria conexão global (melhor que abrir várias conexões dentro da função)
mongo_client = MongoClient(MONGO_URI)
db = mongo_client["experiments_db"]
collection = db["experiment_results"]
answer_collection = db["answer_results"]

def delete_experiment_and_answers(experiment_id: str) -> dict:
    """
    Remove:
      1. O documento da collection 'experiment_results' com o _id fornecido
      2. Todos os documentos da collection 'answer_results' cujo experiment_id == _id
    """

    # Converte o id recebido para ObjectId
    try:
        exp_obj_id = ObjectId(experiment_id)
    except Exception as e:
        raise ValueError(f"experiment_id inválido: {experiment_id}") from e

    # 1. Apagar o experimento da experiment_results
    exp_delete_result = collection.delete_one({"_id": exp_obj_id})

    # 2. Apagar todas as respostas relacionadas na answer_results
    ans_delete_result = answer_collection.delete_many({"experiment_id": exp_obj_id})

    # Retorna quantos documentos foram removidos
    return {
        "experiment_deleted": exp_delete_result.deleted_count,
        "answer_results_deleted": ans_delete_result.deleted_count
    }


result = delete_experiment_and_answers("691e599d5ee53d485ddb1b7a")
print(result)
