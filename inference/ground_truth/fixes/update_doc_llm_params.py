import os
import numpy as np
from pymongo import MongoClient
from bson import ObjectId
from datetime import datetime
from typing import List, Dict, Tuple
from dotenv import load_dotenv

# carregar variáveis de ambiente
load_dotenv()

# conexão com MongoDB
MONGO_URI = os.getenv("MONGODB_URI")
mongo_client = MongoClient(MONGO_URI)
db = mongo_client["experiments_db"]
collection = db["experiments_ground_truth"]

# objeto llm_params a ser inserido
llm_params = {
    "batch_size": 1,
    "max_new_tokens": 1024,
    "min_new_tokens": 100,
    "do_sample": True,
    "temperature": 0.3,
    "top_p": 0.95
}

# atualizar todos os documentos que NÃO têm llm_params
result = collection.update_many(
    {"llm_params": {"$exists": False}},     # condição
    {"$set": {"llm_params": llm_params}}    # operação
)

print(f"Documentos atualizados: {result.modified_count}")
