# Comandos de Execução - Refine Judge Similarity

## 📋 Descrição

Este script calcula a similaridade (Weighted Cohen's Kappa) entre as avaliações de `refine_judge_results` e o ground truth em `final_ground_truth`.

## 🚀 Uso

### Processar experimentos específicos

Para processar apenas experimentos específicos (ex: 2 experimentos):

#### PowerShell
```powershell
python inference/refine_judge_prompt/refine_judge_metric/refine_judge_metric.py `
    --experiments `
        "refine_judge_gpt-4o-mini-2024-07-18_V1" `
        "refine_judge_gpt-4o-mini-2024-07-18_V2"
```

#### Bash
```bash
python inference/refine_judge_prompt/refine_judge_metric/refine_judge_metric.py \
    --experiments \
        "refine_judge_gpt-4o-mini-2024-07-18_V1" \
        "refine_judge_gpt-4o-mini-2024-07-18_V2"
```

### Processar todos os experimentos

Se você não passar o parâmetro `--experiments`, o script processará **todos** os experimentos encontrados em `refine_judge_results`:

```bash
python inference/refine_judge_prompt/refine_judge_metric/refine_judge_metric.py
```

## 📊 Exemplos Completos

### Exemplo 1: Comparar 2 prompts diferentes

```bash
# PowerShell
python inference/refine_judge_prompt/refine_judge_metric/refine_judge_metric.py `
    --experiments `
        "refine_judge_gpt-4o-mini-2024-07-18_V1" `
        "refine_judge_gpt-4o-mini-2024-07-18_V2"
```

### Exemplo 2: Comparar 3 ou mais experimentos

```bash
# Bash
python inference/refine_judge_prompt/refine_judge_metric/refine_judge_metric.py \
    --experiments \
        "refine_judge_gpt-4o-mini_V1" \
        "refine_judge_gpt-4o-mini_V2" \
        "refine_judge_gpt-4o-mini_V3"
```

### Exemplo 3: Processar todos

```bash
python inference/refine_judge_prompt/refine_judge_metric/refine_judge_metric.py
```

## 📈 Output Esperado

O script irá:
1. Buscar os documentos dos experimentos especificados em `refine_judge_results`
2. Parear com documentos correspondentes em `final_ground_truth` (por task_id e instance_id)
3. Calcular Weighted Cohen's Kappa para MTI e STI separadamente
4. Salvar resultados em `refine_judge_similarity_results`
5. Exibir resumo:

```
======================================================================
RESUMO DOS RESULTADOS:
======================================================================

📋 Experimento: refine_judge_gpt-4o-mini-2024-07-18_V1
   MTI Cohen's Kappa: 0.8523
   STI Cohen's Kappa: 0.8234
   Documentos pareados: 240/240

📋 Experimento: refine_judge_gpt-4o-mini-2024-07-18_V2
   MTI Cohen's Kappa: 0.7891
   STI Cohen's Kappa: 0.7654
   Documentos pareados: 240/240

======================================================================
Collection: refine_judge_similarity_results
Total de resultados: 2
======================================================================
```

## 🔧 Parâmetros

| Parâmetro | Obrigatório | Padrão | Descrição |
|-----------|-------------|--------|-----------|
| `--experiments` | ❌ | None (todos) | Lista de nomes de experimentos a processar |

## 📝 Collection de Saída

Os resultados são salvos em `refine_judge_similarity_results` com a estrutura:

```json
{
    "_id": ObjectId,
    "experiment_name": "refine_judge_gpt-4o-mini-2024-07-18_V1",
    "prompt": "...",
    "sti_cohen_kappa": 0.8234,
    "mti_cohen_kappa": 0.8523,
    "calculation_timestamp": ISODate,
    "matched_documents": 240,
    "total_documents": 240
}
```

## 💡 Interpretação dos Resultados

Cohen's Kappa varia de -1 a 1:
- **< 0**: Pior que aleatório
- **0.00 - 0.20**: Concordância leve
- **0.21 - 0.40**: Concordância justa
- **0.41 - 0.60**: Concordância moderada
- **0.61 - 0.80**: Concordância substancial
- **0.81 - 1.00**: Concordância quase perfeita

Quanto **maior** o valor, **mais similar** o experimento está do ground truth.

## ⚠️ Pré-requisitos

1. **Dependências Python**:
   ```bash
   pip install scikit-learn numpy pymongo python-dotenv
   ```

2. **Collections MongoDB**:
   - `refine_judge_results`: Deve conter os experimentos a avaliar
   - `final_ground_truth`: Deve conter o ground truth com medianas calculadas

3. **Variável de ambiente**:
   - `MONGODB_URI` configurada no arquivo `.env`

## 🔍 Verificar Experimentos Disponíveis

Para ver quais experimentos estão disponíveis, você pode consultar diretamente no MongoDB:

```python
from pymongo import MongoClient
import os
from dotenv import load_dotenv

load_dotenv()
client = MongoClient(os.getenv("MONGODB_URI"))
db = client["experiments_db"]

experiments = db["refine_judge_results"].distinct("experiment_gt_name")
print("Experimentos disponíveis:")
for exp in experiments:
    count = db["refine_judge_results"].count_documents({"experiment_gt_name": exp})
    print(f"  - {exp}: {count} documentos")
```
