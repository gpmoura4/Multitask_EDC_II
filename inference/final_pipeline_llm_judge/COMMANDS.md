# Comandos de Execução - LLM Judge

## 📋 Pré-requisitos

Certifique-se de ter:
- IDs dos experimentos STI e MTI (ObjectId do MongoDB)
- Variável de ambiente `MONGODB_URI` configurada
- Dependências instaladas

## 🧪 Modo Teste (10 instâncias)

Use o parâmetro `--is_test` para processar apenas 10 instâncias e validar rapidamente:

### PowerShell
```powershell
python inference/final_pipeline_llm_judge/llm_judge.py `
    --sti_experiment_id "691e5cdb47cb353cce0b14b0" `
    --mti_experiment_id "691f07ff79c79aad7e95f16d" `
    --experiment_name "LLM_Judge_gpt-4o-mini_TEST" `
    --is_test `
    --seed 42
```

### Bash
```bash
python inference/final_pipeline_llm_judge/llm_judge.py \
    --sti_experiment_id "691e5cdb47cb353cce0b14b0" \
    --mti_experiment_id "691f07ff79c79aad7e95f16d" \
    --experiment_name "LLM_Judge_gpt-4o-mini_TEST" \
    --is_test \
    --seed 42
```

## 🚀 Modo Completo (todas as instâncias)

Remova o parâmetro `--is_test` para processar todas as instâncias:

### PowerShell
```powershell
python inference/final_pipeline_llm_judge/llm_judge.py `
    --sti_experiment_id "691e5cdb47cb353cce0b14b0" `
    --mti_experiment_id "691f07ff79c79aad7e95f16d" `
    --experiment_name "LLM_Judge_gpt-4o-mini_V1" `
    --batch_size 10 `
    --seed 42
```

### Bash
```bash
python inference/final_pipeline_llm_judge/llm_judge.py \
    --sti_experiment_id "691e5cdb47cb353cce0b14b0" \
    --mti_experiment_id "691f07ff79c79aad7e95f16d" \
    --experiment_name "LLM_Judge_gpt-4o-mini_V1" \
    --batch_size 10 \
    --seed 42
```

## 📊 Scripts Auxiliares

### 1. Centralizar Metadados dos Experimentos

Execute após gerar as avaliações:

```bash
python inference/final_pipeline_llm_judge/llm_judge_experiment.py
```

### 2. Agregar Múltiplos Experimentos

Combine resultados de diferentes avaliadores:

#### PowerShell
```powershell
python inference/final_pipeline_llm_judge/generate_final_llm_judge.py `
    --experiments `
        "LLM_Judge_gpt-4o-mini_V1" `
        "LLM_Judge_llama-70b_V1" `
        "LLM_Judge_gpt-4o_V1"
```

#### Bash
```bash
python inference/final_pipeline_llm_judge/generate_final_llm_judge.py \
    --experiments \
        "LLM_Judge_gpt-4o-mini_V1" \
        "LLM_Judge_llama-70b_V1" \
        "LLM_Judge_gpt-4o_V1"
```

## 🔧 Parâmetros Disponíveis

| Parâmetro | Obrigatório | Padrão | Descrição |
|-----------|-------------|--------|-----------|
| `--sti_experiment_id` | ✅ | - | ObjectId do experimento STI |
| `--mti_experiment_id` | ✅ | - | ObjectId do experimento MTI |
| `--experiment_name` | ✅ | - | Nome identificador do experimento |
| `--is_test` | ❌ | False | Ativa modo teste (10 instâncias) |
| `--batch_size` | ❌ | 10 | Documentos salvos por lote |
| `--seed` | ❌ | 42 | Seed para reprodutibilidade |

## 📈 Workflow Completo

```bash
# 1. Modo teste primeiro (validação rápida)
python inference/final_pipeline_llm_judge/llm_judge.py \
    --sti_experiment_id "ID_STI" \
    --mti_experiment_id "ID_MTI" \
    --experiment_name "LLM_Judge_Model_TEST" \
    --is_test

# 2. Se tudo OK, execute versão completa
python inference/final_pipeline_llm_judge/llm_judge.py \
    --sti_experiment_id "ID_STI" \
    --mti_experiment_id "ID_MTI" \
    --experiment_name "LLM_Judge_Model_V1"

# 3. Centralize metadados
python inference/final_pipeline_llm_judge/llm_judge_experiment.py

# 4. (Opcional) Agregue múltiplos experimentos
python inference/final_pipeline_llm_judge/generate_final_llm_judge.py \
    --experiments "LLM_Judge_Model_V1" "LLM_Judge_Model2_V1"
```

## 💡 Dicas

1. **Sempre comece com `--is_test`** para validar configurações antes de processar tudo
2. **Use `--batch_size` menor** se tiver problemas de memória
3. **Mantenha `--seed` constante** para reprodutibilidade entre execuções
4. **Nomeie experimentos de teste** com sufixo `_TEST` para identificação fácil

## ⚠️ Notas Importantes

- O modelo avaliador será o mesmo usado no experimento MTI (`model_name`)
- As 10 instâncias do modo teste são as **primeiras** encontradas (não aleatórias)
- Para amostragem aleatória no teste, modifique o código para usar `random.sample()`
- Documentos são salvos em lotes conforme `--batch_size` para recuperação em caso de falha
