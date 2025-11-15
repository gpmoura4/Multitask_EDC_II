import json
import torch
import pandas as pd
from src import reconstruct_instruction
from evaluate.utils import load_hf_lm_and_tokenizer, generate_completions
from evaluate.templates import create_prompt_with_tulu_chat_format 
import argparse
import os


from pymongo import MongoClient
from datetime import datetime

# Conexão MongoDB (exemplo para teste, substitua pela uri real)
MONGO_URI = os.getenv("MONGODB_URI")

# Cria conexão global (melhor que abrir várias conexões dentro da função)
mongo_client = MongoClient(MONGO_URI)
db = mongo_client["experiments_db"]
collection = db["experiment_results"]


def insert_experiment_results(
    date: datetime,
    model_name: str,
    uuid: str,
    prompt: str,
    batch_size: int,
    params: dict | str,
    consumed_tokens: int,
    generation_time: float
):
    """
    Insere um documento na coleção 'experiment_results' contendo
    o estado do experimento em um dado momento.
    """

    # Monta o documento a ser inserido
    doc = {
        "date": date,
        "model_name": model_name,
        "uuid": uuid,
        "prompt": prompt,
        "batch_size": batch_size,
        "params": params,
        "consumed_tokens": consumed_tokens,
        "generation_time": generation_time
    }

    # Insere no MongoDB
    result = collection.insert_one(doc)

    return result.inserted_id


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
    Este é chamado APENAS antes do loop principal.
    
    Se o parâmetro opcional 'id' for fornecido, a função busca um experimento
    existente no banco de dados usando esse 'id' como chave de identificação
    e retorna o _id do documento encontrado.
    """

    # Se 'id' foi fornecido, buscar experimento existente
    if id is not None:
        existing_doc = collection.find_one({"id": id})
        if existing_doc:
            print(f"⚠️  Experimento encontrado com id='{id}'. Retornando _id do documento existente.")
            return existing_doc["_id"]
        else:
            print(f"⚠️  Nenhum experimento encontrado com id='{id}'. Criando novo experimento.")

    initial_time = datetime.now()

    doc = {
        "inference_type": inference_type,
        "experiment_name": experiment_name,
        "batch_size": batch_size,
        "save_every": save_every,
        "model_name": model_name,
        "llm_params": llm_params,
        "initial_time": initial_time,
        "final_time": None,                # será preenchido no final
        "total_time": None,                # será preenchido no final
        "experimentIsOver": False          # muda para True ao fim do experimento
    }

    # Se 'id' foi fornecido, incluí-lo no documento
    if id is not None:
        doc["id"] = id

    result = collection.insert_one(doc)
    return result.inserted_id    # retornamos o ID para update posterior



parser = argparse.ArgumentParser(description="Run the script with a specified model and batch size.")
parser.add_argument("--model_name", type=str, required=True, help="Name of the model to load")
parser.add_argument("--batch_size", type=int, required=True, help="Batch size for generation")
parser.add_argument("--output_dir", type=str, required=True, help="Directory to save output results")
parser.add_argument("--save_every", type=int, default=10, help="Number of generations before saving to CSV")


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
def generate_missing_instances(stage_name, k_output_dir, k, input_builder_fn):
    csv_path = os.path.join(k_output_dir, f"free-form-{args.model_name}-STI-{k}-{stage_name}.csv")

    # Ler progresso existente
    done_uids = set()
    if os.path.exists(csv_path):
        existing_df = pd.read_csv(csv_path)
        done_uids = set(existing_df["uid"].astype(str))
        print(f"⚙️  {stage_name}: {len(done_uids)} instâncias já processadas para {k}.")
    else:
        existing_df = pd.DataFrame(columns=["uid", "generation", "generation_time"])

    # Filtrar instâncias pendentes
    pending = [(uid, instance) for uid, instance in data[k]["instance"].items() if str(uid) not in done_uids]
    if not pending:
        print(f"✅ Nenhuma instância restante para {stage_name} de {k}. Pulando...")
        return existing_df["generation"].tolist() if stage_name != "s3" else None

    print(f"🚀 Gerando {len(pending)} instâncias restantes em {stage_name} para {k}...")

    all_new = []
    uids_done = []

    try:
        for i in range(0, len(pending), args.batch_size):
            batch = pending[i:i + args.batch_size]
            uids_batch = [uid for uid, _ in batch]
            inputs = [input_builder_fn(uid, instance) for uid, instance in batch if input_builder_fn(uid, instance) is not None]

            if not inputs:
                continue

            """
            # Nao criar experimento novo
            # Das respostas -> continuar da onde parou

            # Filtrar pelo experiment_id e com experimentIsOver = false 
            # 

            Gravar na collection 'answer_results' o resultado da geração.
            Deve-se gravar um documento para cada instance dentro de cada task gerada.
            Ex: Temos 12 tasks, cada uma com 200 instâncias.
                Logo, teremos 12 * 200 = 2400 documentos na collection 'answer_results'.
                Cada task tem um task_id, e para cada task_id, temos 200 instance_id.

                - id (ObjectId)
                - experiment_id (ObjectId) -- Esse id deve ser igual ao doc gerado no # Passo 1, fazendo um relacionamento entre eles. 
                - task_id (str) -- str(k)
                - instance_id (str) -- data[k]["instance"][]
                    - A estrutura de data é: data['034']['instance'],
                        - Logo, dentro data['034']['instance'] temos: {'034_53021': {...}, '034_52433': {...}, ...}
                        - Nesse caso, o instance_id do primeiro doc seria '034_53021', do segundo '034_52433', etc.
                        - Considere que o instance_id é a chave dentro do dicionário data[k]["instance"]  
                        - E deve-se armazenar de maneira correta qual a instance_id para aquela task_id específica.

                - prompt (juncao da query, instruction, context)
                    - Deve ser o resultado da func: 
                        -     def build_input_s1(uid, instance):
                                return create_prompt_with_tulu_chat_format([{
                                    "role": "user",
                                    "content": (
                                        "### Example:\n\n"
                                        + "### Instruction: " + cot
                                        + "\n\n### Task:\n\n"
                                        + "### Instruction: " + reconstruct_instruction(instance, 1, False)
                                        + "\n\n### Answer:\n\n"
                                    )
                                }])
                - llm_answer:
                    Se o inference_type for igual a "STI": 
                        - (object com chave "s1", "s2", "s3", armazenando as respostas de cada etapa) -- Resposta gerada pelo modelo
                        - OBS: Aqui temos algumas regras importantes, o número de respostas adicionadas por vez
                        depende diretamente do agrs.batch_size, já que cada batch gera várias respostas.
                        Ou seja, precisamos salvar as respostas de acordo com a quantidade de 
                        batch_size. Se batch_size = 4, então adicionamos 4 respostas geradas para aquela instância,
                        e assim sucessivamente até completar todas as instâncias (que são 200 no total) para cada task.
                        
                        Uma coisa importante de se observar é que o código puxa respostas em 3 etapas (s1, s2, s3).

                            - Isso é determinado pelas chamadas das funções:
                                generated_texts_s1 = generate_missing_instances("s1", k_output_dir, k, build_input_s1)
                                generated_texts_s2 = generate_missing_instances("s2", k_output_dir, k, build_input_s2)
                                generate_missing_instances("s3", k_output_dir, k, build_input_s3)
                            - Ou seja, a resposta de uma instância é dividida em 3 etapas, e cada etapa gera uma resposta diferente.

                        Então, para cada task, com 200 instâncias, teremos 200 respostas para s1, 200 para s2 e 200 para s3.
                        Sendo que cada 200 respostas de cada instância é dividida entre as respostas de s1, s2 e s3.
                        Logo, o que precisamos é que por exemplo, após salvar as respostas de s1,
                        o código segue até as respostas de s2, ao chegar nesse estágio, as respostas geradas
                        devem ser acrescentadas, fazendo um append, aos 200 documentos que já existem referentes as respostas de s1.
                        e assim sucessivamente com as respostas de s3, que também devem ser acrescentadas
                        aos mesmos 200 documentos que foram acrescentados na etapa de s2. Considere que para
                        selecionar quais os registros que devem ser atualizados deve-se fazer o match pelo instance_id

                        - Uma coisa importante a se levar em conta também é a dependência entre de qual experimento é esse conjunto de instâncias.
                        - Lembre-se que cada experimento é único, e cada experimento tem um experiment_id.
                        - Logo, para garantir que estamos atualizando o documento correto, devemos também fazer o match pelo experiment_id.
                        - Então se rodamos os experimentos várias vezes, devemos atualizar as instâncias corretas de cada experimento pelo experiment_id e instance_id.
                    Se o inference_type for igual a "MTI":
                        - (object com chave "s1" armazenando uma resposta única) -- Resposta gerada pelo modelo

                - Se o inference_type for igual a "STI"
                        Lista de respostas, uma para cada etapa (s1, s2, s3)         
                - Se o inference_type for igual a "MTI"
                        Lista de respostas, uma única resposta com todas as etapas concatenadas
                - generation_time (float) -- tempo gasto em segundos na geração daquela instância específica
                - consumed_tokens (int) -- número de tokens consumidos na geração daquela instância específica
            """

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

            batch_df = pd.DataFrame({
                "uid": uids_batch,
                "generation": generated_texts,
                "generation_time": generation_time
            })

            all_new.append(batch_df)
            uids_done.extend(uids_batch)

            # Salvar incrementalmente a cada N gerações
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
        print(f"❌ Erro durante {stage_name} para {k}: {e}")
        torch.cuda.empty_cache()
        return None


print("starting evaluation")


experiment_id = create_experiment_record(
    inference_type="STI",
    experiment_name="STI_Experiment_v1",
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
    }
    # id (OPCIONAL)
)

# -------------------------------
# Loop principal
# -------------------------------

for k, v in list(data.items()):
    print(f"\n=== Processando tuid {k} ===")

    k_output_dir = os.path.join(base_output_path, str(k))
    os.makedirs(k_output_dir, exist_ok=True)

    _, s1, s2, s3 = CoT[CoT["tuid"] == int(k)].values[0]
    cot = data[k]["sample"]

    """
       # Passo 1 
       Gravar na collection 'experiment_results' o estado inicial do experimento
       os seguintes campos:

        - inference_type (str) -- "STI" or "MTI"
        - batch_size (int) (args.batch_size)
        - save_every (int) (args.save_every)
        - initial_time (time) -- tempo de início do experimento
        - final_time (time) -- tempo de término do experimento
        - total_time (float) -- tempo total gasto no experimento        
        - type (str) -- "generate_response", "ground_truth", "llm_judge"
        - llm_params (dict) -- parâmetros usados na geração
            - tokenizer (str)
            - model_name (str) (args.model_name)
            - stop_id_sequences=None,
            - add_special_tokens=True,
            - disable_tqdm=False,
            - max_new_tokens=2048,
            - min_new_tokens=32,
            - do_sample=True,
            - temperature=0.7,
            - top_p=1.0
    """

    

    # ---------- Etapa 1 ----------
    def build_input_s1(uid, instance):
        return create_prompt_with_tulu_chat_format([{
            "role": "user",
            "content": (
                "### Example:\n\n"
                + "### Instruction: " + cot
                + "\n\n### Task:\n\n"
                + "### Instruction: " + reconstruct_instruction(instance, 1, False)
                + "\n\n### Answer:\n\n"
            )
        }])
    

    
    generated_texts_s1 = generate_missing_instances("s1", k_output_dir, k, build_input_s1)
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

    generated_texts_s2 = generate_missing_instances("s2", k_output_dir, k, build_input_s2)
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

    generate_missing_instances("s3", k_output_dir, k, build_input_s3)
    torch.cuda.empty_cache()


finalize_experiment(experiment_id)

print("\n✅ Experimento finalizado e registrado no MongoDB!")
print("\n✅ Processo concluído com sucesso!")