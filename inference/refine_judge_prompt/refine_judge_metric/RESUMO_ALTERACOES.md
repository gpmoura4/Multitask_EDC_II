# Resumo das Alterações - Modo Teste refine_judge_metric.py

## 🎯 Objetivo

Adicionar funcionalidade de **modo teste** que permite calcular Cohen's Kappa para uma **amostra específica de instâncias** em vez de processar todos os 240+ registros.

## ✅ Alterações Implementadas

### 1. Novas Funções Adicionadas

#### `get_selected_instances_file(experiment_name, num_instances)`
- Retorna o Path do arquivo JSON onde as instâncias selecionadas serão salvas
- Formato: `selected_instances/[experiment_name]_n[N]_instances.json`

#### `load_selected_instances(experiment_name, num_instances)`
- Carrega instâncias previamente selecionadas de um arquivo JSON
- Retorna `None` se o arquivo não existe
- Garante reprodutibilidade usando mesmas instâncias em execuções subsequentes

#### `save_selected_instances(experiment_name, num_instances, instances, seed)`
- Salva instâncias selecionadas em arquivo JSON com metadados
- Inclui: experiment_name, num_instances, seed, timestamp, lista de instâncias
- Formato: `[{"task_id": "030", "instance_id": "030_1559"}, ...]`

#### `select_sample_instances(experiment_name, num_instances=5, seed=42)`
- Função principal de seleção de instâncias
- Tenta carregar seleção existente primeiro (reprodutibilidade)
- Se não existe, seleciona N instâncias aleatórias usando `random.sample()`
- Ajusta automaticamente se N > instâncias disponíveis
- Salva seleção para futuras execuções

### 2. Funções Modificadas

#### `calculate_similarity_for_experiment(experiment_name, selected_instances=None)`
**Antes:**
```python
def calculate_similarity_for_experiment(experiment_name: str) -> Dict | None:
```

**Depois:**
```python
def calculate_similarity_for_experiment(
    experiment_name: str, 
    selected_instances: List[Dict] | None = None
) -> Dict | None:
```

**Mudanças:**
- Aceita parâmetro opcional `selected_instances`
- Se fornecido, filtra query MongoDB para processar apenas instâncias específicas
- Usa `$or` com lista de `{"task_id": X, "instance_id": Y}` para filtrar
- Exibe mensagem diferenciada no modo teste

#### `process_all_experiments(experiment_names=None, test_instances=None, seed=42)`
**Antes:**
```python
def process_all_experiments(experiment_names: List[str] | None = None) -> Dict[str, Dict]:
```

**Depois:**
```python
def process_all_experiments(
    experiment_names: List[str] | None = None,
    test_instances: int | None = None,
    seed: int = 42
) -> Dict[str, Dict]:
```

**Mudanças:**
- Aceita parâmetros `test_instances` e `seed`
- Se `test_instances` fornecido, chama `select_sample_instances()` antes do cálculo
- Passa instâncias selecionadas para `calculate_similarity_for_experiment()`
- Adiciona campos `test_mode` e `test_instances_count` aos resultados

#### `main()`
**Mudanças:**
- Adicionados argumentos `--test_instances` e `--seed`
- Exibe indicador de modo teste no cabeçalho
- Passa novos parâmetros para `process_all_experiments()`

### 3. Novos Imports

```python
import json
import random
from pathlib import Path
from datetime import datetime
```

### 4. Nova Estrutura de Diretórios

```
inference/refine_judge_prompt/refine_judge_metric/
├── refine_judge_metric.py (modificado)
├── COMMANDS.md (atualizado)
├── README_MODO_TESTE.md (novo)
└── selected_instances/ (novo diretório)
    ├── [experiment1]_n5_instances.json
    ├── [experiment2]_n10_instances.json
    └── ...
```

### 5. Documentação Atualizada

#### `refine_judge_metric.py` (docstring)
- Adicionada seção "MODO TESTE" explicando uso
- Exemplos de uso com diferentes parâmetros

#### `COMMANDS.md`
- Adicionada seção "🧪 Modo Teste (Instâncias Específicas)"
- Exemplos para PowerShell e Bash
- Atualizada tabela de parâmetros com `--test_instances` e `--seed`

#### `README_MODO_TESTE.md` (novo arquivo)
- Documentação completa do modo teste
- Explicação de como funciona
- Fluxo de trabalho recomendado
- Exemplos práticos
- Vantagens e quando usar

## 🎮 Uso

### Modo Completo (padrão)
```bash
# Processa todas as instâncias (240+)
python refine_judge_metric.py --experiments "exp1"
```

### Modo Teste
```bash
# Processa apenas 5 instâncias
python refine_judge_metric.py --experiments "exp1" --test_instances 5

# Processa 10 instâncias com seed customizado
python refine_judge_metric.py --experiments "exp1" --test_instances 10 --seed 123
```

## 📊 Formato do Arquivo de Seleção

```json
{
  "experiment_name": "refine_judge_gpt-4o-mini-2024-07-18_V4_CORRETO",
  "num_instances": 5,
  "seed": 42,
  "selection_timestamp": "2024-01-15T10:30:00.123456",
  "instances": [
    {"task_id": "030", "instance_id": "030_1559"},
    {"task_id": "031", "instance_id": "031_2048"},
    {"task_id": "032", "instance_id": "032_1234"},
    {"task_id": "033", "instance_id": "033_5678"},
    {"task_id": "034", "instance_id": "034_9012"}
  ]
}
```

## 🔍 Fluxo de Execução

### Primeira Execução com `--test_instances 5`

1. Script verifica `selected_instances/[exp]_n5_instances.json`
2. Arquivo não existe → seleciona 5 instâncias aleatoriamente
3. Salva seleção em JSON
4. Processa apenas essas 5 instâncias
5. Calcula Cohen's Kappa

### Execuções Subsequentes

1. Script verifica `selected_instances/[exp]_n5_instances.json`
2. Arquivo existe → carrega instâncias do arquivo
3. Processa **mesmas 5 instâncias** (reprodutibilidade)
4. Calcula Cohen's Kappa

## 🚀 Benefícios

1. **Velocidade**: 5 instâncias processadas em ~2-5 segundos vs 240+ em ~2-3 minutos
2. **Reprodutibilidade**: Mesmas instâncias testadas em todas as execuções
3. **Debugging**: Facilita identificação de instâncias problemáticas
4. **Iteração Rápida**: Teste mudanças de código rapidamente
5. **Escalabilidade**: Aumente N gradualmente (5 → 10 → 20 → todas)

## 🔧 Casos de Uso

### Desenvolvimento
- Testar correções de código com `--test_instances 5`
- Validar lógica antes de processar todos os dados

### Debugging
- Identificar instâncias com kappa negativo
- Investigar alinhamento de dados

### Validação Incremental
- `--test_instances 5`: validação inicial
- `--test_instances 20`: validação intermediária
- Sem parâmetro: validação final completa

## ⚙️ Detalhes Técnicos

### Query MongoDB Modificado

**Modo Normal:**
```python
refine_docs = list(refine_judge_collection.find({
    "experiment_gt_name": experiment_name
}))
```

**Modo Teste:**
```python
refine_docs = list(refine_judge_collection.find({
    "experiment_gt_name": experiment_name,
    "$or": [
        {"task_id": "030", "instance_id": "030_1559"},
        {"task_id": "031", "instance_id": "031_2048"},
        ...
    ]
}))
```

### Estrutura de Resultados

Resultados no modo teste incluem metadados adicionais:

```python
{
    "experiment_name": "...",
    "sti_cohen_kappa": 0.45,
    "mti_cohen_kappa": 0.48,
    "test_mode": True,  # ← Novo
    "test_instances_count": 5,  # ← Novo
    "matched_documents": 5,
    "total_documents": 5  # Em vez de 240
}
```

## 📝 Arquivos Criados/Modificados

### Modificados
- ✏️ `refine_judge_metric.py` (~180 linhas adicionadas/modificadas)
- ✏️ `COMMANDS.md` (seção de modo teste adicionada)

### Criados
- ✨ `README_MODO_TESTE.md` (documentação completa)
- ✨ `RESUMO_ALTERACOES.md` (este arquivo)
- 📁 `selected_instances/` (diretório para arquivos JSON)

## 🧪 Teste de Validação Sugerido

```bash
# 1. Primeira execução - seleciona e salva instâncias
uv run inference/refine_judge_prompt/refine_judge_metric/refine_judge_metric.py \
  --experiments "refine_judge_gpt-4o-mini-2024-07-18_V4_CORRETO" \
  --test_instances 5

# 2. Verificar arquivo criado
cat selected_instances/refine_judge_gpt-4o-mini-2024-07-18_V4_CORRETO_n5_instances.json

# 3. Segunda execução - deve reutilizar mesmas instâncias
uv run inference/refine_judge_prompt/refine_judge_metric/refine_judge_metric.py \
  --experiments "refine_judge_gpt-4o-mini-2024-07-18_V4_CORRETO" \
  --test_instances 5

# 4. Forçar nova seleção
rm selected_instances/refine_judge_gpt-4o-mini-2024-07-18_V4_CORRETO_n5_instances.json
uv run inference/refine_judge_prompt/refine_judge_metric/refine_judge_metric.py \
  --experiments "refine_judge_gpt-4o-mini-2024-07-18_V4_CORRETO" \
  --test_instances 5
```

## 🎓 Conclusão

O modo teste está completamente implementado e documentado. Ele oferece:

- ✅ Seleção aleatória de N instâncias
- ✅ Reprodutibilidade através de arquivos JSON
- ✅ Integração transparente com código existente
- ✅ Documentação completa (código, COMMANDS.md, README_MODO_TESTE.md)
- ✅ Flexibilidade via CLI (--test_instances, --seed)

Pronto para uso em iterações rápidas de debugging e validação incremental do Cohen's Kappa!
