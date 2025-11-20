# Ground Truth Generation

Este diretório contém o script para geração de Ground Truth (GT) comparando experimentos STI e MTI.

## Descrição

O script `generate_gt.py` realiza a avaliação automática de respostas geradas por modelos em experimentos STI (Single Task Inference) e MTI (Multi Task Inference), utilizando um modelo avaliador (GPT-4o-mini por padrão) para julgar a qualidade das respostas.

## Funcionamento

1. **Seleção de Experimentos**: Busca pares de experimentos STI e MTI com o mesmo `model_name`
2. **Amostragem de Instâncias**: 
   - Modo teste: 2 instâncias por task (24 total)
   - Modo completo: 20 instâncias por task (240 total)
3. **Construção de Prompts**: Combina respostas STI e MTI em um prompt de avaliação
4. **Avaliação por LLM**: Envia para modelo avaliador que julga 5 critérios:
   - Coherence (Coerência)
   - Specificity (Especificidade)
   - Informativeness (Informatividade)
   - Relevance (Relevância)
   - Clarity (Clareza)
5. **Armazenamento**: Salva resultados na collection `ground_truth_results` do MongoDB

## Uso

### Modo Teste (2 instâncias por task)

```bash
uv run -m inference.ground_truth.generate_gt \
    --sti_experiment_id "674a1b2c3d4e5f6g7h8i9j0k" \
    --mti_experiment_id "674a1b2c3d4e5f6g7h8i9j0l" \
    --experiment_gt_name "GT_Test_Llama2_7b" \
    --evaluator_model "gpt-4o-mini-2024-07-18" \
    --is_test \
    --seed 42
```

### Modo Completo (20 instâncias por task)

```bash
uv run -m inference.ground_truth.generate_gt \
    --sti_experiment_id "674a1b2c3d4e5f6g7h8i9j0k" \
    --mti_experiment_id "674a1b2c3d4e5f6g7h8i9j0l" \
    --experiment_gt_name "GT_Complete_Llama2_7b" \
    --evaluator_model "gpt-4o-mini-2024-07-18" \
    --seed 42
```

## Parâmetros

- `--sti_experiment_id`: ID do experimento STI no MongoDB (obrigatório)
- `--mti_experiment_id`: ID do experimento MTI no MongoDB (obrigatório)
- `--experiment_gt_name`: Nome do experimento de ground truth (obrigatório)
- `--evaluator_model`: Modelo usado para avaliação (padrão: "gpt-4o-mini-2024-07-18")
- `--is_test`: Flag para modo teste (2 instâncias/task)
- `--seed`: Seed para reprodutibilidade (padrão: 42)

## Como obter os IDs dos experimentos

Você pode listar os experimentos disponíveis usando MongoDB Compass ou via script Python:

```python
from pymongo import MongoClient
import os

MONGO_URI = os.getenv("MONGODB_URI")
client = MongoClient(MONGO_URI)
db = client["experiments_db"]

# Listar experimentos STI
print("Experimentos STI:")
for exp in db.experiment_results.find({"inference_type": "STI", "experimentIsOver": True}):
    print(f"  ID: {exp['_id']}, Model: {exp['model_name']}")

# Listar experimentos MTI
print("\nExperimentos MTI:")
for exp in db.experiment_results.find({"inference_type": "MTI", "experimentIsOver": True}):
    print(f"  ID: {exp['_id']}, Model: {exp['model_name']}")
```

## Estrutura do Documento GT

Cada documento gerado na collection `ground_truth_results` contém:

```json
{
  "experiment_gt_name": "GT_Test_Llama2_7b",
  "sti_experiment_id": ObjectId,
  "mti_experiment_id": ObjectId,
  "task_id": "034",
  "instance_id": "12345",
  "evaluator_model": "gpt-4o-mini-2024-07-18",
  "evaluation_timestamp": ISODate,
  
  "prompt_MTI": "...",
  "llm_answer_MTI": {
    "s1": "resposta do MTI",
    "position_answer_to_llm": "A"
  },
  
  "prompt_STI": "...",
  "llm_answer_STI": {
    "s1": "resposta step 1",
    "s2": "resposta step 2",
    "s3": "resposta step 3",
    "position_answer_to_llm": "B"
  },
  
  "gt_prompt": "prompt completo enviado ao avaliador",
  "gt_raw_response": "resposta bruta do avaliador",
  
  "gt_MTI_answer": {
    "coherence": {"score": 4, "explanation": "..."},
    "specificity": {"score": 3, "explanation": "..."},
    "informativeness": {"score": 5, "explanation": "..."},
    "relevance": {"score": 4, "explanation": "..."},
    "clarity": {"score": 5, "explanation": "..."}
  },
  
  "gt_STI_answer": {
    "coherence": {"score": 5, "explanation": "..."},
    "specificity": {"score": 4, "explanation": "..."},
    "informativeness": {"score": 4, "explanation": "..."},
    "relevance": {"score": 5, "explanation": "..."},
    "clarity": {"score": 4, "explanation": "..."}
  }
}
```

## Observações

- As respostas STI são concatenadas automaticamente no formato:
  ```
  ### Instruction1:
  [resposta s1]
  
  ### Instruction2:
  [resposta s2]
  
  ### Instruction3:
  [resposta s3]
  ```

- A posição das respostas (A ou B) é randomizada para evitar vieses de posição

- O script é resiliente a erros: continua processando mesmo se uma instância falhar

- Usa MongoDB para armazenamento persistente e recuperação de dados

## Dependências

- pymongo
- torch
- pandas
- tqdm
- bson

As dependências são gerenciadas pelo arquivo `pyproject.toml` do projeto.

---

## Deletar Documentos GT

O script `DELETE_GT_TEST.PY` permite deletar documentos de ground truth de duas formas:

### 1. Por ID do Experimento STI

Deleta todos os documentos GT associados a um experimento STI específico:

```bash
uv run inference/ground_truth/DELETE_GT_TEST.PY --sti_experiment_id "674a1b2c3d4e5f6g7h8i9j0k"
```

### 2. Por Nome do Experimento GT

Deleta todos os documentos GT com um nome específico:

```bash
uv run inference/ground_truth/DELETE_GT_TEST.PY --experiment_gt_name "GT_Test_Llama2_7b"
```

### Saída Esperada

```
📊 Encontrados 24 documentos GT vinculados ao experimento STI 674a1b2c3d4e5f6g7h8i9j0k
✅ 24 documentos GT deletados com sucesso!

📋 Resultado:
{
  'sti_experiment_id': '674a1b2c3d4e5f6g7h8i9j0k',
  'gt_documents_deleted': 24,
  'message': 'Deletados 24 documentos'
}
```
