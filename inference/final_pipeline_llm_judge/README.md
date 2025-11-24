# Final Pipeline LLM Judge

Este diretório contém os scripts para o pipeline de avaliação LLM Judge, onde modelos avaliam suas próprias respostas (auto-avaliação).

## Estrutura

```
final_pipeline_llm_judge/
├── llm_judge.py                    # Script principal de geração de avaliações
├── llm_judge_experiment.py         # Centraliza metadados dos experimentos
├── generate_final_llm_judge.py     # Agrega múltiplos experimentos em collection final
└── README.md                       # Este arquivo
```

## Fluxo de Trabalho

### 1. Geração de Avaliações LLM Judge

O script `llm_judge.py` gera avaliações usando o mesmo modelo que gerou as respostas:

```bash
python inference/final_pipeline_llm_judge/llm_judge.py \
    --sti_experiment_id <STI_EXPERIMENT_ID> \
    --mti_experiment_id <MTI_EXPERIMENT_ID> \
    --experiment_name "LLM_Judge_gpt-4o-mini-2024-07-18_V1" \
    --batch_size 10 \
    --seed 42
```

**Parâmetros:**
- `--sti_experiment_id`: ID do experimento STI (ObjectId do MongoDB)
- `--mti_experiment_id`: ID do experimento MTI (ObjectId do MongoDB)
- `--experiment_name`: Nome identificador do experimento de avaliação
- `--batch_size`: Número de documentos processados antes de salvar (padrão: 10)
- `--seed`: Seed para reprodutibilidade (padrão: 42)

**Saída:**
- Collection `llm_judge_results` com avaliações individuais

### 2. Centralização de Metadados

O script `llm_judge_experiment.py` centraliza os metadados dos experimentos:

```bash
python inference/final_pipeline_llm_judge/llm_judge_experiment.py
```

**Saída:**
- Collection `experiments_llm_judge` com metadados centralizados
- Campo `llm_judge_experiment_id` adicionado aos documentos em `llm_judge_results`

### 3. Agregação Final

O script `generate_final_llm_judge.py` agrega múltiplos experimentos:

```bash
python inference/final_pipeline_llm_judge/generate_final_llm_judge.py \
    --experiments \
        "LLM_Judge_gpt-4o-mini-2024-07-18_V1" \
        "LLM_Judge_llama-3.3-70b-versatile_V1" \
        "LLM_Judge_gpt-4o-2024-08-06_V1"
```

**Parâmetros:**
- `--experiments`: Lista de nomes de experimentos a agregar

**Saída:**
- Collection `final_llm_judge` com avaliações agregadas e medianas calculadas

## Collections MongoDB

### llm_judge_results

Armazena avaliações individuais de cada experimento:

```json
{
    "_id": ObjectId,
    "experiment_name": "LLM_Judge_gpt-4o-mini-2024-07-18_V1",
    "sti_experiment_id": ObjectId,
    "mti_experiment_id": ObjectId,
    "task_id": "030",
    "instance_id": "030_1559",
    "evaluator_model": "gpt-4o-mini-2024-07-18",
    "evaluation_timestamp": ISODate,
    
    "llm_judge_MTI_prompt": "...",
    "llm_judge_STI_prompt": "...",
    "llm_judge_prompt": "teste12345",
    
    "llm_answer_MTI": {
        "s1": "resposta MTI...",
        "position_answer_to_llm": "A"
    },
    
    "llm_answer_STI": {
        "s1": "parte 1...",
        "s2": "parte 2...",
        "s3": "parte 3...",
        "position_answer_to_llm": "B"
    },
    
    "llm_judge_raw_response": "resposta bruta do LLM...",
    
    "llm_judge_MTI_answer": {
        "coherence": {
            "score": 5,
            "explanation": "..."
        },
        "specificity": {...},
        "informativeness": {...},
        "relevance": {...},
        "Understandability": {...}
    },
    
    "llm_judge_STI_answer": {
        "coherence": {...},
        "specificity": {...},
        "informativeness": {...},
        "relevance": {...},
        "Understandability": {...}
    },
    
    "llm_judge_experiment_id": ObjectId  // Adicionado após executar llm_judge_experiment.py
}
```

### experiments_llm_judge

Metadados centralizados dos experimentos:

```json
{
    "_id": ObjectId,
    "experiment_name": "LLM_Judge_gpt-4o-mini-2024-07-18_V1",
    "evaluator_model": "gpt-4o-mini-2024-07-18",
    "evaluation_timestamp": ISODate,
    "sti_experiment_id": ObjectId,
    "mti_experiment_id": ObjectId,
    "created_at": ISODate
}
```

### final_llm_judge

Avaliações agregadas de múltiplos experimentos com medianas:

```json
{
    "_id": ObjectId,
    "experiments": [
        "LLM_Judge_gpt-4o-mini-2024-07-18_V1",
        "LLM_Judge_llama-3.3-70b-versatile_V1",
        "LLM_Judge_gpt-4o-2024-08-06_V1"
    ],
    "models": [
        "gpt-4o-mini-2024-07-18",
        "llama-3.3-70b-versatile",
        "gpt-4o-2024-08-06"
    ],
    "task_id": "030",
    "instance_id": "030_1559",
    
    "llm_judge_MTI_prompt": "...",
    "llm_judge_STI_prompt": "...",
    "llm_judge_prompt": "teste12345",
    
    "llm_judge_MTI_answer": {
        "coherence": {
            "score_gpt-4o-mini-2024-07-18": 5,
            "score_llama-3.3-70b-versatile": 4,
            "score_gpt-4o-2024-08-06": 5,
            "median_score": 5,
            "explanation_gpt-4o-mini-2024-07-18": "...",
            "explanation_llama-3.3-70b-versatile": "...",
            "explanation_gpt-4o-2024-08-06": "..."
        },
        "specificity": {...},
        "informativeness": {...},
        "relevance": {...},
        "Understandability": {...}
    },
    
    "llm_judge_STI_answer": {
        // Mesma estrutura de llm_judge_MTI_answer
    }
}
```

## Diferenças vs Ground Truth

| Aspecto | Ground Truth | LLM Judge |
|---------|--------------|-----------|
| Avaliador | GPT-4o-mini fixo | Mesmo modelo do experimento |
| Propósito | Referência externa | Auto-avaliação |
| Collection | `ground_truth_results` | `llm_judge_results` |
| Prefixo campos | `gt_` | `llm_judge_` |
| Prompt template | `GT_EVALUATION_TEMPLATE` | "teste12345" |

## Notas Importantes

1. **Auto-avaliação**: O LLM Judge usa o mesmo modelo que gerou as respostas, permitindo análise de viés de auto-avaliação

2. **Prompt template**: O prompt está configurado como `"teste12345"` conforme especificação. Deve ser atualizado com o template real antes do uso em produção

3. **Posicionamento aleatório**: As respostas MTI e STI são randomizadas nas posições A/B para evitar viés de posição

4. **Compatibilidade**: Os experimentos STI e MTI devem ter o mesmo `model_name` para avaliação consistente

5. **Batch processing**: Use `--batch_size` para controlar quantos documentos são salvos de uma vez, útil para recuperação em caso de falhas

## Exemplo Completo

```bash
# 1. Gerar avaliações
python inference/final_pipeline_llm_judge/llm_judge.py \
    --sti_experiment_id "691e5cdb47cb353cce0b14b0" \
    --mti_experiment_id "691f07ff79c79aad7e95f16d" \
    --experiment_name "LLM_Judge_gpt-4o-mini_V1"

# 2. Centralizar metadados
python inference/final_pipeline_llm_judge/llm_judge_experiment.py

# 3. Agregar múltiplos experimentos
python inference/final_pipeline_llm_judge/generate_final_llm_judge.py \
    --experiments \
        "LLM_Judge_gpt-4o-mini_V1" \
        "LLM_Judge_llama-70b_V1" \
        "LLM_Judge_gpt-4o_V1"
```
