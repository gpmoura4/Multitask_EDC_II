import os
import json
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

MONGO_URI = os.getenv("MONGODB_URI")
mongo_client = MongoClient(MONGO_URI)
db = mongo_client["experiments_db"]
final_gt_collection = db["final_ground_truth"]

# Path to dataset (Windows path in repo)
DATASET_PATH = os.path.join("data", "Free_Form_Generation.json")


def load_free_form_generation(dataset_path: str):
    """Load Free_Form_Generation.json into memory."""
    if not os.path.exists(dataset_path):
        # Try absolute path from workspace root if running elsewhere
        alt_path = os.path.join(os.getcwd(), dataset_path)
        dataset_path = alt_path
    with open(dataset_path, "r", encoding="utf-8") as f:
        return json.load(f)


def fix_instance_data():
    print("🔄 Atualizando campo instance_data em final_ground_truth com base no Free_Form_Generation.json...")

    try:
        ff_data = load_free_form_generation(DATASET_PATH)
    except Exception as e:
        print(f"❌ Erro ao carregar dataset '{DATASET_PATH}': {e}")
        return 0

    updated = 0
    not_found = 0

    cursor = final_gt_collection.find({}, {"task_id": 1, "instance_id": 1})
    for doc in cursor:
        task_id = doc.get("task_id")
        instance_id = doc.get("instance_id")
        if not task_id or not instance_id:
            continue

        # The dataset uses task_id as top-level keys (likely strings)
        task_block = ff_data.get(str(task_id))
        if not task_block:
            not_found += 1
            continue

        instances = task_block.get("instance", {})
        instance_data = instances.get(str(instance_id))
        if not instance_data:
            not_found += 1
            continue

        # Update document with instance_data
        res = final_gt_collection.update_one(
            {"_id": doc["_id"]},
            {"$set": {"instance_data": instance_data}}
        )
        if res.modified_count:
            updated += 1

    print(f"✅ Atualização concluída. Documentos modificados: {updated}. Não encontrados: {not_found}.")
    return updated


if __name__ == "__main__":
    fix_instance_data()
