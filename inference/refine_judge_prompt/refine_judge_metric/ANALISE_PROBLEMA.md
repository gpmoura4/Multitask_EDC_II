# Análise do Problema: Cohen's Kappa Baixo ou Negativo

## 🔴 Problema Identificado

Os valores de Cohen's Kappa estão anormalmente baixos, mesmo quando usando o **mesmo prompt** que gerou o ground truth:

```
MTI Cohen's Kappa por métrica:
   - coherence: 0.3967
   - specificity: 0.3942
   - informativeness: 0.3840
   - relevance: 0.2445
   - Understandability: 0.4123
   -> média MTI: 0.3663

STI Cohen's Kappa por métrica:
   - coherence: 0.4906
   - specificity: 0.4994
   - informativeness: 0.4890
   - relevance: -0.0067  ⚠️ NEGATIVO!
   - Understandability: 0.4800
   -> média STI: 0.3905
```

**Expectativa:** Com o mesmo prompt, os valores deveriam estar próximos de 1.0 (concordância quase perfeita).

---

## 🔍 Possíveis Causas Identificadas

### 1. **PROBLEMA DE EXTRAÇÃO DE DADOS** ⚠️ MAIS PROVÁVEL

O código original usava funções auxiliares que podem estar extraindo dados incorretamente:

```python
# CÓDIGO ANTIGO - PROBLEMÁTICO
mti_refine = extract_scores_from_answer(
    refine_doc.get("gt_MTI_answer", {}), 
    metrics
)
mti_gt = extract_median_scores_from_gt(
    gt_doc.get("gt_MTI_answer", {}), 
    metrics
)
```

**Problemas potenciais:**
- A função `extract_scores_from_answer` busca `score` diretamente
- A função `extract_median_scores_from_gt` busca `median_score`
- Se a estrutura dos documentos for ligeiramente diferente, os scores podem não corresponder
- Erros silenciosos (try/except vazios) podem mascarar problemas

### 2. **PROBLEMA DE ALINHAMENTO DE LISTAS**

Quando usamos índices (`mti_gt[i]`, `mti_refine[i]`), assumimos que:
- Ambas as listas têm o mesmo tamanho
- Os elementos na posição `i` correspondem à mesma métrica
- Não há valores None ou faltantes

**Se houver qualquer desalinhamento:**
- Score de `coherence` pode ser comparado com median de `specificity`
- Isso resulta em Cohen's Kappa incorreto e possivelmente negativo

### 3. **PROBLEMA NA ESTRUTURA DOS DOCUMENTOS**

#### Ground Truth (`final_ground_truth`):
```json
{
  "gt_MTI_answer": {
    "coherence": {
      "score_gpt-4o-mini-2024-07-18": 5,
      "score_llama-3.3-70b-versatile": 4,
      "score_gpt-4o-2024-08-06": 5,
      "median_score": 5  ← ESTE É O VALOR USADO
    }
  }
}
```

#### Refine Judge (`refine_judge_results`):
```json
{
  "gt_MTI_answer": {
    "coherence": {
      "score": 5,  ← ESTE É O VALOR USADO
      "explanation": "..."
    }
  }
}
```

**Risco:** Se a estrutura for diferente do esperado, a extração falha silenciosamente.

### 4. **VALORES AUSENTES OU NONE**

Se algumas métricas estão ausentes em certos documentos:
- O código antigo com try/except vazio ignora o erro
- Mas não garante que as listas mantêm correspondência 1:1
- Resultado: desalinhamento que causa kappa baixo ou negativo

### 5. **PROBLEMA NO CÁLCULO DA MEDIANA** ❗ CRÍTICO

**DESCOBERTA IMPORTANTE:**

Olhando o código de `generate_final_gt.py`, a mediana é calculada assim:

```python
def aggregate_metric_scores(evaluations: List[Dict], metric_name: str, models_map: Dict[str, str]) -> Dict:
    result = {}
    scores = []
    
    for eval_doc in evaluations:
        exp_name = eval_doc["experiment_gt_name"]
        model_name = models_map.get(exp_name, exp_name)
        
        metric_data = eval_doc.get(metric_name, {})
        
        if "score" in metric_data:
            score = metric_data["score"]
            result[f"score_{model_name}"] = score
            scores.append(score)
    
    if scores:
        result["median_score"] = statistics.median(scores)
    
    return result
```

**Problema:** Esse código está buscando `metric_name` diretamente de `eval_doc`, mas deveria buscar de dentro de `gt_MTI_answer` ou `gt_STI_answer`!

O código correto deveria ser:
```python
for eval_doc in evaluations:
    answer_data = eval_doc.get("gt_MTI_answer", {})  # ← FALTAVA ISSO
    metric_data = answer_data.get(metric_name, {})
    # ...
```

Isso significa que **o ground truth pode ter sido calculado incorretamente desde o início**! 🚨

---

## ✅ Solução Implementada

### Mudanças no Código:

1. **Extração Direta por Métrica:**
   ```python
   # NOVO - Mais seguro e explícito
   refine_mti_metric = refine_doc.get("gt_MTI_answer", {}).get(metric, {})
   refine_mti_score = refine_mti_metric.get("score", None)
   
   gt_mti_metric = gt_doc.get("gt_MTI_answer", {}).get(metric, {})
   gt_mti_median = gt_mti_metric.get("median_score", None)
   ```

2. **Validação de Dados:**
   ```python
   # Só adiciona se AMBOS existirem
   if refine_mti_score is not None and gt_mti_median is not None:
       mti_ref_by_metric[metric].append(refine_mti_score)
       mti_gt_by_metric[metric].append(gt_mti_median)
   ```

3. **Debug Detalhado:**
   ```python
   print("\n🔍 DEBUG - Tamanhos das listas por métrica:")
   for metric in metrics:
       mti_ref_size = len(mti_ref_by_metric[metric])
       mti_gt_size = len(mti_gt_by_metric[metric])
       if mti_ref_size != mti_gt_size:
           print(f"      ⚠️  ALERTA MTI: Tamanhos diferentes!")
   ```

4. **Informação sobre Cálculo:**
   ```python
   print(f"   MTI {metric}: kappa={k:.4f} (n={len(gt_list)} pares)")
   ```

---

## 🔬 Como Diagnosticar o Problema Real

Execute o código atualizado e observe:

### 1. **Verificar Tamanhos das Listas**
```
🔍 DEBUG - Tamanhos das listas por métrica:
   coherence:
      MTI: refine=240, gt=240  ✅
      STI: refine=240, gt=240  ✅
```

- Se os tamanhos não batem → problema de extração de dados
- Se algum for 0 → métrica ausente em alguma collection

### 2. **Verificar Número de Pares**
```
   MTI coherence: kappa=0.3967 (n=240 pares)
   STI relevance: kappa=-0.0067 (n=240 pares)
```

- Se `n` for diferente entre métricas → dados inconsistentes
- Se `n` for menor que esperado → instâncias sem correspondência

### 3. **Verificar Correspondência de Instâncias**
```
⚠️  5 instâncias sem correspondência no GT
```

- Se houver muitas instâncias não encontradas → problema de matching
- Pode ser que `refine_judge_results` tenha instâncias diferentes de `final_ground_truth`

### 4. **Inspecionar Valores Manualmente**

Adicione este código temporário para ver os valores reais:

```python
# Depois de coletar os dados, antes de calcular kappa
if metric == "relevance":  # Métrica problemática
    print(f"\n🔍 DEBUG DETALHADO - Relevance MTI:")
    print(f"   Ground Truth (primeiros 10): {mti_gt_by_metric['relevance'][:10]}")
    print(f"   Refine Judge (primeiros 10): {mti_ref_by_metric['relevance'][:10]}")
```

---

## 🎯 Próximos Passos

### 1. **Execute o código atualizado**
```bash
python inference/refine_judge_prompt/refine_judge_metric/refine_judge_metric.py \
    --experiments "refine_judge_gpt-4o-mini-2024-07-18_GT_CORRETO"
```

### 2. **Analise os outputs de debug**
- Tamanhos das listas batem?
- Todas as métricas têm dados?
- Quantas instâncias não foram encontradas?

### 3. **Se o problema persistir:**

#### Opção A: Verificar Ground Truth
```python
# Script de verificação
from pymongo import MongoClient
client = MongoClient(MONGO_URI)
db = client["experiments_db"]

# Pegar um documento de exemplo
gt_doc = db["final_ground_truth"].find_one({"task_id": "030"})
print("Estrutura do GT:")
print(json.dumps(gt_doc.get("gt_MTI_answer", {}), indent=2))

refine_doc = db["refine_judge_results"].find_one({
    "task_id": "030",
    "experiment_gt_name": "refine_judge_gpt-4o-mini-2024-07-18_GT_CORRETO"
})
print("\nEstrutura do Refine:")
print(json.dumps(refine_doc.get("gt_MTI_answer", {}), indent=2))
```

#### Opção B: Verificar se o prompt é REALMENTE o mesmo
```python
# Comparar prompts
gt_experiments = db["experiments_ground_truth"].find({})
for exp in gt_experiments:
    print(f"\nGT Experiment: {exp['experiment_gt_name']}")
    # Buscar um doc de resultado
    result = db["ground_truth_results"].find_one({
        "experiment_gt_name": exp['experiment_gt_name']
    })
    print(f"Prompt (primeiros 100 chars): {result.get('gt_prompt', '')[:100]}")
```

#### Opção C: Recalcular o Ground Truth
Se o problema estiver no `generate_final_gt.py`, será necessário:
1. Corrigir a função `aggregate_metric_scores`
2. Reprocessar a collection `final_ground_truth`
3. Executar o cálculo de similaridade novamente

---

## 📊 Interpretação dos Resultados

### Cohen's Kappa Esperado:
- **≥ 0.81:** Concordância quase perfeita (esperado com mesmo prompt)
- **0.61 - 0.80:** Concordância substancial
- **0.41 - 0.60:** Concordância moderada
- **0.21 - 0.40:** Concordância justa (valores atuais)
- **0.00 - 0.20:** Concordância leve
- **< 0:** Pior que aleatório (problema sério!)

### Valores Atuais (0.24 - 0.49):
Indicam **concordância justa a moderada**, o que é **muito baixo** para o mesmo prompt.

### Relevance Negativo (-0.0067):
Indica que o modelo está **discordando sistematicamente** do ground truth nessa métrica específica, o que é um **forte indicador de problema de dados**.

---

## 🚨 Conclusão

O problema mais provável é um **erro na extração ou alinhamento de dados**. O código atualizado deve ajudar a identificar exatamente onde está o problema através dos outputs de debug detalhados.

**Ação imediata:** Execute o código atualizado e compartilhe os outputs de debug completos.
