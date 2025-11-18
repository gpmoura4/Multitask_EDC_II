import json
import random
import pandas as pd

# Caminhos dos arquivos
input_path = "data/Free_Form_Generation.json"
csv_output_path = "amostra_instancias.csv"
json_output_path = "amostra_instancias_estruturado.json"

# Carrega o JSON original
with open(input_path, "r", encoding="utf-8") as file:
    data = json.load(file)

# Lista para armazenar dados planos (para CSV)
resultados_csv = []

# Dicionário final estruturado (para JSON no formato de referência)
resultado_json = {}

# Percorre cada tarefa do conjunto
for task_id, conteudo in data.items():
    instancia_dict = conteudo.get("instance", {})
    
    if not instancia_dict:
        continue  # ignora tarefas vazias
    
    # Seleciona aleatoriamente uma instância
    instance_key = random.choice(list(instancia_dict.keys()))
    instance_data = instancia_dict[instance_key]

    # Extrai os campos relevantes
    query = instance_data.get("query", "")
    instruction = instance_data.get("instruction", {})
    context = instance_data.get("context", {})
    answers = instance_data.get("answers", [])

    # ---------- Estrutura hierárquica para JSON ----------
    resultado_json[task_id] = {
        "instance": {
            instance_key: {
                "query": query,
                "instruction": instruction,
                "context": context,
                "answers": answers
            }
        }
    }

    # ---------- Estrutura plana para CSV ----------
    resultados_csv.append({
        "task_id": task_id,
        "instance_id": instance_key,
        "query": query,
        "instruction": json.dumps(instruction, ensure_ascii=False),
        "context": json.dumps(context, ensure_ascii=False),
        "answers": json.dumps(answers, ensure_ascii=False)
    })

# ---------- Salvar em CSV ----------
df = pd.DataFrame(resultados_csv)
df.to_csv(csv_output_path, index=False, encoding="utf-8-sig")
print(f"Arquivo CSV gerado: {csv_output_path}")

# ---------- Salvar em JSON no formato correto ----------
with open(json_output_path, "w", encoding="utf-8") as outfile:
    json.dump(resultado_json, outfile, indent=4, ensure_ascii=False)
print(f"Arquivo JSON gerado no formato estruturado: {json_output_path}")
