import json
import torch
import pandas as pd
from src import reconstruct_instruction
from evaluate.utils import load_hf_lm_and_tokenizer, generate_completions
from evaluate.templates import create_prompt_with_tulu_chat_format 
import argparse
import os

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

# -------------------------------
# Loop principal
# -------------------------------
for k, v in list(data.items()):
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

print("\n✅ Processo concluído com sucesso!")