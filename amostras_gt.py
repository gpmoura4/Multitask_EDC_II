import json
import random
import pandas as pd

# Caminhos dos arquivos
input_path = "data/Free_Form_Generation.json"
csv_output_path = "amostra_instancias.csv"
json_output_path = "amostra_instancias.json"

# Carrega o JSON original
with open(input_path, "r", encoding="utf-8") as file:
    data = json.load(file)

# Lista para armazenar os resultados
resultados = []

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

    # Armazena no resultado
    resultados.append({
        "task_id": task_id,
        "instance_id": instance_key,
        "query": query,
        "instruction": instruction,
        "context": context,
        "answers": answers
    })

# ---------- Salvar em CSV ----------
# Convertendo listas/dicionários em strings JSON para evitar erros no CSV
df = pd.DataFrame([
    {
        "task_id": r["task_id"],
        "instance_id": r["instance_id"],
        "query": r["query"],
        "instruction": json.dumps(r["instruction"], ensure_ascii=False),
        "context": json.dumps(r["context"], ensure_ascii=False),
        "answers": json.dumps(r["answers"], ensure_ascii=False),
    }
    for r in resultados
])

df.to_csv(csv_output_path, index=False, encoding="utf-8-sig")
print(f"Arquivo CSV gerado: {csv_output_path}")

# ---------- Salvar em JSON ----------
with open(json_output_path, "w", encoding="utf-8") as outfile:
    json.dump(resultados, outfile, indent=4, ensure_ascii=False)
print(f"Arquivo JSON gerado: {json_output_path}")
