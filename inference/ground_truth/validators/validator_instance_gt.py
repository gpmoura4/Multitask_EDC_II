from __future__ import annotations


import os

from datetime import datetime
from bson import ObjectId
from pymongo import MongoClient
from typing import List, Dict, Tuple
from tqdm import tqdm
from dotenv import load_dotenv
load_dotenv()

# from evaluate.utils import load_hf_lm_and_tokenizer, generate_completions


# ----------------------------
# CONFIGURAÇÕES
# ----------------------------

MONGO_URI = os.getenv("MONGODB_URI")

# nomes dos dois grupos a comparar
GROUP1_GT = "GT_gpt-4o-mini-2024-07-18_V3"
GROUP2_GT = "GT_llama-3.3-70b-versatile_V3"

# ----------------------------
# CONEXÃO
# ----------------------------

mongo_client = MongoClient(MONGO_URI)
db = mongo_client["experiments_db"]
col = db["ground_truth_results"]

print("mongo db uri", MONGO_URI)
# ----------------------------
# OBTÉM TODAS AS task_id
# ----------------------------

task_ids = col.distinct("task_id")

print(f"Encontradas {len(task_ids)} task_ids.")
print(task_ids)
print("-" * 60)

# ----------------------------
# PROCESSAR CADA task_id
# ----------------------------

for task in task_ids:
    # instance_id do grupo 1
    group1_instances = set(col.distinct(
        "instance_id",
        {
            "task_id": task,
            "experiment_gt_name": GROUP1_GT
        }
    ))

    # instance_id do grupo 2
    group2_instances = set(col.distinct(
        "instance_id",
        {
            "task_id": task,
            "experiment_gt_name": GROUP2_GT
        }
    ))

    # diferença: o que existe no grupo 1 e não existe no 2
    missing = sorted(list(group1_instances - group2_instances))

    print(f"TASK: {task}")
    print(f" - instâncias no grupo 1: {len(group1_instances)}")
    print(f" - instâncias no grupo 2: {len(group2_instances)}")

    if missing:
        print(" ❌ instance_id que existem no doc 1 mas NÃO existem no doc 2:")
        for inst in missing:
            print("    →", inst)
    else:
        print(" ✅ Todos os instance_id estão presentes nos dois grupos.")

    print("-" * 60)
