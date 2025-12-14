import os
from pymongo import MongoClient
from bson import ObjectId
from dotenv import load_dotenv

load_dotenv()

MONGO_URI = os.getenv("MONGODB_URI")
mongo_client = MongoClient(MONGO_URI)
db = mongo_client["experiments_db"]
final_gt_collection = db["final_ground_truth"]
gt_results_collection = db["ground_truth_results"]

def fix_gt_prompt_in_final_ground_truth():
    print("🔄 Corrigindo campo gt_prompt em final_ground_truth por instance_id e task_id...")
    count = 0
    for doc in final_gt_collection.find({}):
        instance_id = doc.get("instance_id")
        task_id = doc.get("task_id")
        if not instance_id or not task_id:
            continue
        # Busca o documento na ground_truth_results com o mesmo instance_id e task_id
        gt_result_doc = gt_results_collection.find_one({
            "instance_id": instance_id,
            "task_id": task_id
        })
        if gt_result_doc and "gt_prompt" in gt_result_doc:
            new_gt_prompt = gt_result_doc["gt_prompt"]
            # Atualiza o campo gt_prompt no documento final_ground_truth
            final_gt_collection.update_one(
                {"_id": doc["_id"]},
                {"$set": {"gt_prompt": new_gt_prompt}}
            )
            count += 1
    print(f"✅ Corrigidos {count} documentos.")

if __name__ == "__main__":
    fix_gt_prompt_in_final_ground_truth()