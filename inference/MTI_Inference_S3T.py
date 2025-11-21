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
from pymongo import MongoClient

# Conexão MongoDB
MONGO_URI = os.getenv("MONGODB_URI")

# Cria conexão global
mongo_client = MongoClient(MONGO_URI)
db = mongo_client["experiments_db"]
collection = db["experiment_results"]
answer_collection = db["answer_results"]


def save_error_checkpoint(experiment_id, task_id, error_msg):
    """
    Salva um checkpoint de erro em arquivo txt para recuperação posterior.
    """
    checkpoint_file = f"error_checkpoint_{experiment_id}.txt"
    with open(checkpoint_file, 'w') as f:
        f.write(f"experiment_id: {experiment_id}\n")
        f.write(f"task_id: {task_id}\n")
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
    if not doc:
        print(f"⚠️  Documento de experimento {experiment_id} não encontrado.")
        return
    
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


def create_or_update_answer_document(experiment_id, task_id, instance_id, prompt, llm_answer, generation_time, consumed_tokens):
    """
    Cria ou atualiza um documento de resposta na collection answer_results.
    Para MTI, há apenas uma resposta única (não tem stages s1, s2, s3 como no STI).
    """
    # Buscar documento existente
    existing_doc = answer_collection.find_one({
        "experiment_id": experiment_id,
        "instance_id": instance_id
    })
    
    if existing_doc:
        # Documento existe - fazer update
        update_data = {
            "llm_answer.s1": llm_answer,  # MTI armazena resposta única em s1
            "prompt": prompt,
            "generation_time": generation_time,
            "consumed_tokens": consumed_tokens if consumed_tokens is not None else 0
        }
        
        answer_collection.update_one(
            {"_id": existing_doc["_id"]},
            {"$set": update_data}
        )
        return existing_doc["_id"]
    else:
        # Documento não existe - criar novo
        doc = {
            "experiment_id": experiment_id,
            "task_id": task_id,
            "instance_id": instance_id,
            "prompt": prompt,
            "llm_answer": {"s1": llm_answer},  # MTI: resposta única armazenada em s1
            "generation_time": generation_time,
            "consumed_tokens": consumed_tokens if consumed_tokens is not None else 0
        }
        result = answer_collection.insert_one(doc)
        return result.inserted_id


def check_instance_processed(experiment_id, instance_id):
    """
    Verifica se uma instância específica já foi processada.
    Para MTI, verificamos se existe llm_answer.s1.
    """
    doc = answer_collection.find_one({
        "experiment_id": experiment_id,
        "instance_id": instance_id
    })
    
    if doc and "llm_answer" in doc and "s1" in doc["llm_answer"]:
        return True
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
    """
    Cria o documento inicial do experimento.
    """
    # Se 'id' foi fornecido, tentar buscar experimento existente
    if id is not None:
        try:
            object_id = ObjectId(id)
            existing_doc = collection.find_one({"_id": object_id})
        except Exception:
            print(f"❌ ID '{id}' não é um ObjectId válido. Criando novo experimento.")
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
# Caminho base: {output_dir}/{model_name}/MTI
base_output_path = os.path.join(args.output_dir, args.model_name, "MTI")
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

print("starting evaluation")

# Criar registro do experimento
experiment_id = create_experiment_record(
    inference_type="MTI",
    experiment_name="MTI_gpt-4o-mini-2024-07-18_Experiment",
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
    id="691f07ff79c79aad7e95f16d"
)

# -------------------------------
# Loop principal
# -------------------------------

# Determinar quais tasks processar baseado no modo de teste
if args.is_test:
    # Modo teste: processar task específica ou primeira disponível
    if args.test_task_id:
        # Verificar se a task existe
        if args.test_task_id in data:
            tasks_to_process = [(args.test_task_id, data[args.test_task_id])]
            print(f"\nMODO DE TESTE ATIVADO")
            print(f"Task selecionada: {args.test_task_id}")
        else:
            available_tasks = list(data.keys())[:5]  # Mostrar primeiras 5 tasks
            print(f"\n❌ ERRO: Task '{args.test_task_id}' não encontrada no dataset.")
            print(f"Tasks disponíveis (primeiras 5): {', '.join(available_tasks)}")
            exit(1)
    else:
        # Se não especificou task_id, usar a primeira
        tasks_to_process = list(data.items())[:1]
        print(f"\nMODO DE TESTE ATIVADO (primeira task)")
        print(f"Task selecionada: {tasks_to_process[0][0]}")
    
    total_instances = len(tasks_to_process[0][1]['instance'])
    instances_to_process = args.test_num_instances if args.test_num_instances else total_instances
    instances_to_process = min(instances_to_process, total_instances)
    
    print(f"Número de instâncias a processar: {instances_to_process} de {total_instances}")
    print(f"Batch size: {args.batch_size}\n")
else:
    # Modo normal: processar todas as tasks
    tasks_to_process = list(data.items())
    print(f"\n📊 MODO COMPLETO: Processando todas as {len(tasks_to_process)} tasks\n")

for k, v in tasks_to_process:
    print(f"\n=== Processando tuid {k} ===")
    
    # Criar diretório para a task
    k_output_dir = os.path.join(base_output_path, str(k))
    os.makedirs(k_output_dir, exist_ok=True)
    
    _, s1, s2, s3 = CoT[CoT["tuid"] == int(k)].values[0]
    cot = data[k]["sample"]
    
    csv_path = os.path.join(k_output_dir, f"free-form-{args.model_name}-MTI-{k}.csv")
    
    # Ler progresso existente do CSV
    done_uids = set()
    if os.path.exists(csv_path):
        existing_df = pd.read_csv(csv_path)
        done_uids = set(existing_df["uid"].astype(str))
        print(f"⚙️  {len(done_uids)} instâncias já processadas para {k} (CSV).")
    else:
        existing_df = pd.DataFrame(columns=["uid", "generation", "generation_time"])
    
    # Verificar também no MongoDB quais instâncias já foram processadas
    mongo_done_uids = set()
    for uid in data[k]["instance"].keys():
        if check_instance_processed(experiment_id, str(uid)):
            mongo_done_uids.add(str(uid))
    
    # Combinar ambos os conjuntos
    done_uids = done_uids.union(mongo_done_uids)
    if mongo_done_uids:
        print(f"⚙️  {len(mongo_done_uids)} instâncias já processadas para {k} (MongoDB).")
    
    # Filtrar instâncias pendentes
    pending_items = [(uid, instance) for uid, instance in data[k]["instance"].items() if str(uid) not in done_uids]
    
    # Se estiver em modo de teste e test_num_instances for especificado, limitar o número de instâncias
    if args.is_test and args.test_num_instances is not None:
        pending_items = pending_items[:args.test_num_instances]
        print(f"MODO TESTE: Limitando a {args.test_num_instances} instâncias")
    
    if not pending_items:
        print(f"✅ Nenhuma instância restante para {k}. Pulando...")
        continue
    
    print(f" Gerando {len(pending_items)} instâncias restantes para {k}...")
    
    all_new = []
    uids_done = []
    
    try:
        # Processar em batches
        for i in range(0, len(pending_items), args.batch_size):
            batch = pending_items[i:i + args.batch_size]
            
            # Construir lista de inputs para o batch
            input_list = []
            valid_uids = []
            valid_instances = []
            
            for uid, instance in batch:
                prompt_content = create_prompt_with_tulu_chat_format([{
                    "role": "user", 
                    "content": (
                        "### Example:\n\n" + "### Instruction: " + cot + 
                        "\n\n### Task:\n\n" + "### Instruction: " + reconstruct_instruction(instance, -1) + 
                        "\n\n" + "### Answer:\n\n"
                    )
                }])
                input_list.append(prompt_content)
                valid_uids.append(uid)
                valid_instances.append(instance)
            
            if not input_list:
                continue
            
            # Gerar respostas
            generation_time, generated_texts = generate_completions(
                model,
                tokenizer,
                input_list,
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
            
            # Salvar cada resposta no MongoDB
            for idx, (uid, instance, gen_text, gen_time) in enumerate(zip(valid_uids, valid_instances, generated_texts, generation_time)):
                # Reconstruir o prompt para salvar no MongoDB
                prompt = create_prompt_with_tulu_chat_format([{
                    "role": "user", 
                    "content": (
                        "### Example:\n\n" + "### Instruction: " + cot + 
                        "\n\n### Task:\n\n" + "### Instruction: " + reconstruct_instruction(instance, -1) + 
                        "\n\n" + "### Answer:\n\n"
                    )
                }])
                
                # Calcular tokens consumidos (aproximação)
                consumed_tokens = len(gen_text.split())
                
                try:
                    # Salvar/atualizar no MongoDB
                    create_or_update_answer_document(
                        experiment_id=experiment_id,
                        task_id=str(k),
                        instance_id=str(uid),
                        prompt=prompt,
                        llm_answer=gen_text,
                        generation_time=gen_time,
                        consumed_tokens=consumed_tokens
                    )
                except Exception as mongo_error:
                    print(f"⚠️  Erro ao salvar no MongoDB para {uid}: {mongo_error}")
            
            # Salvar também no CSV (backup)
            batch_df = pd.DataFrame({
                "uid": valid_uids,
                "generation": generated_texts,
                "generation_time": generation_time
            })
            
            all_new.append(batch_df)
            uids_done.extend(valid_uids)
            
            # Salvar incrementalmente a cada N gerações
            if len(all_new) * args.batch_size >= args.save_every or i + args.batch_size >= len(pending_items):
                new_df = pd.concat(all_new, ignore_index=True)
                final_df = pd.concat([existing_df, new_df], ignore_index=True)
                final_df.to_csv(csv_path, index=False)
                existing_df = final_df
                all_new = []
                print(f"💾 Progresso salvo ({len(uids_done)}/{len(pending_items)}) para {k}")
            
            torch.cuda.empty_cache()
        
        print(f"✅ Task {k} concluída. Total de {len(existing_df)} instâncias salvas.")
        
    except Exception as e:
        print(f"❌ Erro durante processamento de {k}: {e}")
        # Salvar checkpoint de erro
        save_error_checkpoint(experiment_id, str(k), str(e))
        torch.cuda.empty_cache()
        continue

finalize_experiment(experiment_id)

print("\n✅ Experimento finalizado e registrado no MongoDB!")
print("\n✅ Processo concluído com sucesso!")

