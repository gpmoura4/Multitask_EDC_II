from __future__ import annotations

import json
import torch
import pandas as pd
from src import reconstruct_instruction
from evaluate.utils import load_hf_lm_and_tokenizer, generate_completions
from evaluate.templates import create_prompt_with_tulu_chat_format 
import argparse
import os
from bson import ObjectId
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()
from bson import ObjectId
from pymongo import MongoClient
from datetime import datetime
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


# Conexão MongoDB (exemplo para teste, substitua pela uri real)
MONGO_URI = os.getenv("MONGODB_URI")

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


def save_error_checkpoint(experiment_id, task_id, stage_name, error_msg):
    """
    Salva um checkpoint de erro em arquivo txt para recuperação posterior.
    """
    checkpoint_file = f"error_checkpoint_{experiment_id}.txt"
    with open(checkpoint_file, 'w') as f:
        f.write(f"experiment_id: {experiment_id}\n")
        f.write(f"task_id: {task_id}\n")
        f.write(f"stage: {stage_name}\n")
        f.write(f"error: {error_msg}\n")
        f.write(f"timestamp: {datetime.now().isoformat()}\n")
    print(f"💾 Checkpoint de erro salvo em {checkpoint_file}")


def finalize_experiment(experiment_id):
    """
    Atualiza o documento inicial de um experimento para marcar como concluído.
    Preenche também o final_time e total_time.
    """

    final_time = datetime.now()

    # Buscar o documento original para calcular total_time
    doc = collection.find_one({"_id": experiment_id})
    initial_time = doc["initial_time"]
    total_time = (final_time - initial_time).total_seconds()

    update_fields = {
        "experimentIsOver": True,
        "final_time": final_time,
        "total_time": total_time
    }

    collection.update_one(
        {"_id": experiment_id},
        {"$set": update_fields}
    )



def create_or_update_answer_document(experiment_id, task_id, instance_id, stage_name, prompt, llm_answer, generation_time, consumed_tokens):
    existing_doc = answer_collection.find_one({
        "experiment_id": experiment_id,
        "instance_id": instance_id
    })
    
    if existing_doc:
        update_data = {
            f"llm_answer.{stage_name}": llm_answer,
            f"generation_time": existing_doc.get("generation_time", 0) + generation_time,
            "prompt": prompt 
        }

        if consumed_tokens is not None:
            update_data["consumed_tokens"] = existing_doc.get("consumed_tokens", 0) + consumed_tokens
        
        answer_collection.update_one(
            {"_id": existing_doc["_id"]},
            {"$set": update_data}
        )
        return existing_doc["_id"]
    else:
        doc = {
            "experiment_id": experiment_id,
            "task_id": task_id,
            "instance_id": instance_id,
            "prompt": prompt,
            "llm_answer": {stage_name: llm_answer},
            "generation_time": generation_time,
            "consumed_tokens": consumed_tokens if consumed_tokens is not None else 0
        }
        result = answer_collection.insert_one(doc)
        return result.inserted_id


def check_instance_processed(experiment_id, instance_id, stage_name):
    doc = answer_collection.find_one({
        "experiment_id": experiment_id,
        "instance_id": instance_id
    })
    
    if doc and "llm_answer" in doc:
        return stage_name in doc["llm_answer"]
    return False


def create_experiment_record(
    inference_type: str,
    experiment_name: str,
    batch_size: int,
    save_every: int,
    model_name: str,
    llm_params: dict,
    id: str | None = None,
):

    if id is not None:
        try:
            object_id = ObjectId(id)  
            existing_doc = collection.find_one({"_id": object_id})
        except Exception:
            print(f"ID '{id}' não é um ObjectId válido. Criando novo experimento.")
            existing_doc = None

        if existing_doc:
            print(f"⚠️  Experimento encontrado com _id='{id}'. Retornando documento existente.")
            return existing_doc["_id"]
        else:
            print(f"⚠️  Nenhum experimento encontrado com _id='{id}'. Criando novo experimento.")

    initial_time = datetime.now()

    doc = {
        "inference_type": inference_type,
        "experiment_name": experiment_name,
        "batch_size": batch_size,
        "save_every": save_every,
        "model_name": model_name,
        "llm_params": llm_params,
        "initial_time": initial_time,
        "final_time": None,
        "total_time": None,
        "experimentIsOver": False
    }

    result = collection.insert_one(doc)
    return result.inserted_id


parser = argparse.ArgumentParser(description="Run the script with a specified model and batch size.")
parser.add_argument("--model_name", type=str, required=True, help="Name of the model to load")
parser.add_argument("--batch_size", type=int, required=True, help="Batch size for generation")
parser.add_argument("--output_dir", type=str, required=True, help="Directory to save output results")
parser.add_argument("--save_every", type=int, default=10, help="Number of generations before saving to CSV")
parser.add_argument("--is_test", action="store_true", help="Run in test mode (process only specified task and instances)")
parser.add_argument("--test_task_id", type=str, default=None, help="Specific task ID to process in test mode (e.g., '034')")
parser.add_argument("--test_num_instances", type=int, default=None, help="Number of instances to process in test mode")


args = parser.parse_args()

# ------------------------------
# Criação de diretórios
# ------------------------------
# Caminho base: {output_dir}/{model_name}/STI
base_output_path = os.path.join(args.output_dir, args.model_name, "STI")
os.makedirs(base_output_path, exist_ok=True)

# ------------------------------
# Carregar dados e modelo
# ------------------------------
with open("data/Free_Form_Generation.json", 'r') as file:
    data = json.load(file)

model, tokenizer = load_hf_lm_and_tokenizer(
    args.model_name,
    torch_dtype=torch.float16
)

CoT = pd.read_excel("data/cot_breakdown.xlsx")

# -------------------------------
# Função auxiliar: gera e salva só o que falta (agora com salvamento incremental)
# -------------------------------
def generate_missing_instances(stage_name, k_output_dir, k, input_builder_fn, experiment_id):
    csv_path = os.path.join(k_output_dir, f"free-form-{args.model_name}-STI-{k}-{stage_name}.csv")

    done_uids = set()
    if os.path.exists(csv_path):
        existing_df = pd.read_csv(csv_path)
        done_uids = set(existing_df["uid"].astype(str))
        print(f"⚙️  {stage_name}: {len(done_uids)} instâncias já processadas para {k} (CSV).")
    else:
        existing_df = pd.DataFrame(columns=["uid", "generation", "generation_time"])
    
    mongo_done_uids = set()
    for uid in data[k]["instance"].keys():
        if check_instance_processed(experiment_id, str(uid), stage_name):
            mongo_done_uids.add(str(uid))
    
    done_uids = done_uids.union(mongo_done_uids)
    if mongo_done_uids:
        print(f"⚙️  {stage_name}: {len(mongo_done_uids)} instâncias já processadas para {k} (MongoDB).")

    pending = [(uid, instance) for uid, instance in data[k]["instance"].items() if str(uid) not in done_uids]
    

    if args.is_test and args.test_num_instances is not None:
        pending = pending[:args.test_num_instances]
        print(f" MODO TESTE: Limitando a {args.test_num_instances} instâncias")
    
    if not pending:
        print(f"✅ Nenhuma instância restante para {stage_name} de {k}. Pulando...")
        return existing_df["generation"].tolist() if stage_name != "s3" else None

    print(f"Gerando {len(pending)} instâncias restantes em {stage_name} para {k}...")

    all_new = []
    uids_done = []

    try:
        for i in range(0, len(pending), args.batch_size):
            batch = pending[i:i + args.batch_size]
            uids_batch = [uid for uid, _ in batch]
            instances_batch = [instance for uid, instance in batch]
            
            # Construir inputs e manter mapeamento com UIDs
            inputs = []
            valid_uids = []
            valid_instances = []
            for uid, instance in batch:
                input_prompt = input_builder_fn(uid, instance)
                if input_prompt is not None:
                    inputs.append(input_prompt)
                    valid_uids.append(uid)
                    valid_instances.append(instance)

            if not inputs:
                continue

            generation_time, generated_texts = generate_completions(
                model,
                tokenizer,
                inputs,
                batch_size=args.batch_size,
                stop_id_sequences=None,
                add_special_tokens=True,
                disable_tqdm=False,
                max_new_tokens=2048,
                min_new_tokens=32,
                do_sample=True,
                temperature=0.7,
                top_p=1.0
            )


            for idx, (uid, instance, gen_text, gen_time) in enumerate(zip(valid_uids, valid_instances, generated_texts, generation_time)):
                # Reconstruir o prompt para salvar no MongoDB
                prompt = input_builder_fn(uid, instance)
                # Calcular tokens consumidos (aproximação baseada no texto gerado)
                consumed_tokens = len(gen_text.split())  # Aproximação simples
                
                try:
                    create_or_update_answer_document(
                        experiment_id=experiment_id,
                        task_id=str(k),
                        instance_id=str(uid),
                        stage_name=stage_name,
                        prompt=prompt,
                        llm_answer=gen_text,
                        generation_time=gen_time,
                        consumed_tokens=consumed_tokens
                    )
                except Exception as mongo_error:
                    print(f"⚠️  Erro ao salvar no MongoDB para {uid}: {mongo_error}")

            batch_df = pd.DataFrame({
                "uid": valid_uids,
                "generation": generated_texts,
                "generation_time": generation_time
            })

            all_new.append(batch_df)
            uids_done.extend(valid_uids)

            if len(all_new) * args.batch_size >= args.save_every or i + args.batch_size >= len(pending):
                new_df = pd.concat(all_new, ignore_index=True)
                final_df = pd.concat([existing_df, new_df], ignore_index=True)
                final_df.to_csv(csv_path, index=False)
                existing_df = final_df  # atualizar referência
                all_new = []
                print(f"💾 Progresso salvo ({len(uids_done)}/{len(pending)}) para {stage_name} de {k}")

            torch.cuda.empty_cache()

        print(f"✅ Etapa {stage_name} concluída para {k}. Total de {len(existing_df)} instâncias salvas.")
        return existing_df["generation"].tolist() if stage_name != "s3" else None

    except Exception as e:
        print(f"Erro durante {stage_name} para {k}: {e}")
        save_error_checkpoint(experiment_id, str(k), stage_name, str(e))
        torch.cuda.empty_cache()
        return None


# Alterar conforme necessidade para futuros experimentos
experiment_id = create_experiment_record(
    inference_type="STI",
    experiment_name="STI_ gpt-3.5-turbo-0125_Experiment_1120",
    batch_size=args.batch_size,
    save_every=args.save_every,
    model_name=args.model_name,
    llm_params={
        "tokenizer": str(tokenizer.__class__.__name__),
        "model_name": args.model_name,
        "stop_id_sequences": None,
        "add_special_tokens": True,
        "disable_tqdm": False,
        "max_new_tokens": 2048,
        "min_new_tokens": 32,
        "do_sample": True,
        "temperature": 0.7,
        "top_p": 1.0
    },
    id="691f6c99d40086eaa5c7a322"
)


if args.is_test:
    if args.test_task_id:
        if args.test_task_id in data:
            tasks_to_process = [(args.test_task_id, data[args.test_task_id])]
            print(f"Task selecionada: {args.test_task_id}")
        else:
            available_tasks = list(data.keys())[:5]  # Mostrar primeiras 5 tasks
            print(f"\nERRO: Task '{args.test_task_id}' não encontrada no dataset.")
            print(f"Tasks disponíveis (primeiras 5): {', '.join(available_tasks)}")
            exit(1)
    else:
        # Se não especificou task_id, usar a primeira
        tasks_to_process = list(data.items())[:1]
        print(f"\n MODO DE TESTE ATIVADO (primeira task)")
        print(f"Task selecionada: {tasks_to_process[0][0]}")
    
    total_instances = len(tasks_to_process[0][1]['instance'])
    instances_to_process = args.test_num_instances if args.test_num_instances else total_instances
    instances_to_process = min(instances_to_process, total_instances)  # Não exceder o total disponível
    
    print(f"Número de instâncias a processar: {instances_to_process} de {total_instances}")
    print(f"Batch size: {args.batch_size}\n")
else:
    # Modo normal: processar todas as tasks
    tasks_to_process = list(data.items())
    print(f"\n📊 MODO COMPLETO: Processando todas as {len(tasks_to_process)} tasks\n")

for k, v in tasks_to_process:
    print(f"\n=== Processando tuid {k} ===")

    k_output_dir = os.path.join(base_output_path, str(k))
    os.makedirs(k_output_dir, exist_ok=True)

    _, s1, s2, s3 = CoT[CoT["tuid"] == int(k)].values[0]
    cot = data[k]["sample"]

    # ---------- Etapa 1 ----------
    def build_input_s1(uid, instance):
        return create_prompt_with_tulu_chat_format([{
            "role": "user",
            "content": (
                "### Example:\n\n"
                + "### Instruction: " + cot
                + "\n\n### Task:\n\n"
                + "### (1) Instruction: " + reconstruct_instruction(instance, 1, False)
                + "\n\n### Answer [Respond only the Instruction (1)]:\n\n"
            )
        }])
    

    
    generated_texts_s1 = generate_missing_instances("s1", k_output_dir, k, build_input_s1, experiment_id)
    if generated_texts_s1 is None:
        continue

    torch.cuda.empty_cache()

    # ---------- Etapa 2 ----------
    def build_input_s2(uid, instance):
        idx = list(data[k]["instance"].keys()).index(uid)
        if idx >= len(generated_texts_s1):
            return None
        gen1 = generated_texts_s1[idx]
        query1 = reconstruct_instruction(instance, 1, False)
        query2 = reconstruct_instruction(instance, 2, False)
        content = (
            "### Example:\n\n"
            + "### (1) Instruction: " + s1
            + "\n\n### (2) Instruction: " + s2
            + "\n\n### Task:\n\n"
            + "### (1) Instruction: " + query1
            + "\n\n### Answer:\n\n" + str(gen1)
            + "\n\n### (2) Instruction: " + query2
            + "\n\n### Answer:"
        )
        return create_prompt_with_tulu_chat_format([{"role": "user", "content": content}])

    generated_texts_s2 = generate_missing_instances("s2", k_output_dir, k, build_input_s2, experiment_id)
    if generated_texts_s2 is None:
        continue

    torch.cuda.empty_cache()

    # ---------- Etapa 3 ----------
    def build_input_s3(uid, instance):
        num_instructions = len(instance.get('instruction', {}))
        if num_instructions < 3:
            return None

        idx = list(data[k]["instance"].keys()).index(uid)
        if idx >= len(generated_texts_s1) or idx >= len(generated_texts_s2):
            return None

        gen1 = generated_texts_s1[idx]
        gen2 = generated_texts_s2[idx]

        query1 = reconstruct_instruction(instance, 1, False)
        query2 = reconstruct_instruction(instance, 2, False)
        query3 = reconstruct_instruction(instance, 3, False)

        content = (
            "### Example:\n\n"
            + "### (1) Instruction: " + s1
            + "\n\n### (2) Instruction: " + s2
            + "\n\n### (3) Instruction: " + s3
            + "\n\n### Task:\n\n"
            + "### (1) Instruction: " + query1
            + "\n\n### Answer:\n\n" + str(gen1)
            + "\n\n### (2) Instruction: " + query2
            + "\n\n### Answer:\n\n" + str(gen2)
            + "\n\n### (3) Instruction: " + query3
            + "\n\n### Answer:"
        )
        return create_prompt_with_tulu_chat_format([{"role": "user", "content": content}])

    generate_missing_instances("s3", k_output_dir, k, build_input_s3, experiment_id)
    torch.cuda.empty_cache()


finalize_experiment(experiment_id)

print("\n✅ Experimento finalizado e registrado no MongoDB!")
print("\n✅ Processo concluído com sucesso!")