import json

# Caminhos dos arquivos
input_path = "data/Free_Form_Generation.json"
output_path = "data/free_form_first_elem.json"

# Lê o arquivo JSON original
with open(input_path, "r", encoding="utf-8") as file:
    data = json.load(file)

# Obtém o primeiro item do dicionário
primeira_chave = list(data.keys())[0]
primeiro_elemento = {primeira_chave: data[primeira_chave]}

# Salva o novo JSON apenas com esse elemento
with open(output_path, "w", encoding="utf-8") as outfile:
    json.dump(primeiro_elemento, outfile, indent=4, ensure_ascii=False)

print(f"Novo JSON gerado com a primeira chave: {primeira_chave}")
print(f"Caminho do arquivo: {output_path}")
