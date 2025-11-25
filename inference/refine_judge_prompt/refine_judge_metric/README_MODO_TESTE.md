# Modo Teste - refine_judge_metric.py

## Visão Geral

O modo teste permite calcular o Cohen's Kappa para um **subconjunto específico de instâncias**, em vez de processar todos os 240+ registros. Isso é útil para:

- ✅ Iteração rápida durante debugging
- ✅ Validação de correções em amostras pequenas
- ✅ Identificação de instâncias problemáticas
- ✅ Reprodutibilidade de testes

## Uso

### Sintaxe Básica

```bash
python refine_judge_metric.py --test_instances N [--seed SEED]
```

### Parâmetros

- `--test_instances N`: Número de instâncias a selecionar aleatoriamente (padrão: processa todas)
- `--seed SEED`: Seed para seleção aleatória (padrão: 42, garante reprodutibilidade)
- `--experiments`: Lista de experimentos a processar (opcional, padrão: todos)

### Exemplos

#### Teste com 5 instâncias (configuração padrão)

```bash
uv run inference/refine_judge_prompt/refine_judge_metric/refine_judge_metric.py --experiments "refine_judge_gpt-4o-mini-2024-07-18_V4_CORRETO" --test_instances 5
```

#### Teste com 10 instâncias e seed customizado

```bash
uv run inference/refine_judge_prompt/refine_judge_metric/refine_judge_metric.py --experiments "refine_judge_gpt-4o-mini-2024-07-18_V4_CORRETO" --test_instances 10 --seed 123
```

#### Teste múltiplos experimentos

```bash
uv run inference/refine_judge_prompt/refine_judge_metric/refine_judge_metric.py --experiments "exp1" "exp2" --test_instances 5
```

## Como Funciona

### 1. Seleção de Instâncias

Na primeira execução com `--test_instances N`:

1. Script busca todas as instâncias do experimento
2. Seleciona N instâncias aleatoriamente usando `random.sample()`
3. Salva a seleção em `selected_instances/[experiment_name]_n[N]_instances.json`

### 2. Reprodutibilidade

Em execuções subsequentes:

1. Script verifica se existe arquivo de seleção
2. Se existe, **reutiliza as mesmas instâncias**
3. Se não existe, cria nova seleção

Isso garante que você sempre teste as mesmas instâncias, mesmo após múltiplas execuções.

### 3. Arquivo de Seleção

Formato: `selected_instances/[nome_experimento]_n[N]_instances.json`

Exemplo: `selected_instances/refine_judge_gpt-4o-mini-2024-07-18_V4_CORRETO_n5_instances.json`

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

## Fluxo de Trabalho Recomendado

### Fase 1: Desenvolvimento/Debugging (N=5)

```bash
# Teste rápido com 5 instâncias
uv run inference/refine_judge_prompt/refine_judge_metric/refine_judge_metric.py \
  --experiments "seu_experimento" \
  --test_instances 5
```

### Fase 2: Validação Intermediária (N=20)

```bash
# Teste com amostra maior
uv run inference/refine_judge_prompt/refine_judge_metric/refine_judge_metric.py \
  --experiments "seu_experimento" \
  --test_instances 20
```

### Fase 3: Validação Final (Todas as instâncias)

```bash
# Executar sem --test_instances para processar tudo
uv run inference/refine_judge_prompt/refine_judge_metric/refine_judge_metric.py \
  --experiments "seu_experimento"
```

## Quando Usar

### Use Modo Teste Quando:

- 🔬 Testando correções de código
- 🐛 Debugando problemas de Cohen's Kappa
- ⚡ Precisar de feedback rápido (segundos vs minutos)
- 🔍 Investigando instâncias específicas
- 📊 Validando mudanças incrementais

### Use Modo Completo Quando:

- 📈 Gerar resultados finais para análise
- 📄 Preparar dados para paper
- ✅ Validação final após todas as correções
- 📊 Comparar experimentos completos

## Vantagens

1. **Velocidade**: Processa 5 instâncias em segundos vs 240+ em minutos
2. **Reprodutibilidade**: Mesmas instâncias em todas as execuções
3. **Rastreabilidade**: Arquivo JSON com metadados completos
4. **Flexibilidade**: Escolha N conforme necessidade
5. **Debugging**: Fácil inspeção manual de instâncias específicas

## Notas Técnicas

- O modo teste filtra documentos no MongoDB usando `$or` com lista de (task_id, instance_id)
- Se N > número de instâncias disponíveis, usa todas disponíveis
- Cada combinação de (experimento, N) gera arquivo único
- Diferentes valores de `--seed` geram seleções diferentes

## Limpeza

Para forçar nova seleção de instâncias:

```bash
# Deletar arquivo de seleção específico
rm selected_instances/[experiment_name]_n5_instances.json

# Ou deletar todos os arquivos de seleção
rm -r selected_instances/
```

Na próxima execução, novas instâncias serão selecionadas.
