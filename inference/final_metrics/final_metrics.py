"""
"Context" - 
Estou escrevendo um artigo científico que tem como objetivo avaliar a eficiencia das técnicas 
de Single-Task Inference (STI) e Multi-Task Inference (MTI) com relação à qualidade de respostas
de acordo com métricas de preferência humana.

- O Single-Task Inference refere-se a habilidade do 
LLM de seguir uma instrução por chamada de inferência, ou seja, aborda sub-tarefas sequencialmente 

- Já o Multi-Task Inference refere-se a habilidade de seguir instruções complexas que lidam com múltiplas 
sub-tasks em uma única chamada de inferência, ou seja, lidar com tarefas compostas por várias instruções em 
uma única chamada de inferência. 

- As respostas geradas pelo Single-Task Inference e pelo Multi-Task Inference 
são avalidas com as características: coerência, especificidade, compreensibilidade, informatividade e relevância. 
- Isso será feito para diferentes modelos de linguagem, ou seja, eu vou pedir para que diferentes modelos 
de linguagem gerem a reposta tanto utilizando a técnica de Single-Task Inference e Multi-Task Inference. 
E utilizarei um outro modelo de linguagem como juiz para avaliar essas respostas de acordo com as métricas 
de preferência humana definidas anteriormente. As notas (score) de cada métrica variam de 1 a 5, de acordo com a escala Likert.

[doc da collection refine_judge_similarity_k_alpha]: 
{ "_id": { "$oid": "69271df3b0bff0313a699a84" }, 
"sti_k_alpha": 0.145, "mti_k_alpha": 0.285, 
"sti_k_alpha_by_metric": { "coherence": 0.143, "specificity": 0.127, "informativeness": 0.162, "relevance": 0.188, "Understandability": 0.107 },
"mti_k_alpha_by_metric": { "coherence": 0.32, "specificity": 0.245, "informativeness": 0.227, "relevance": 0.354, "Understandability": 0.277 },
"sti_exact_match_percentage": 68.333, "mti_exact_match_percentage": 68.5, "sti_exact_match_percentage_by_metric":
{ "coherence": 62.917, "specificity": 58.333, "informativeness": 60.833, "relevance": 96.25, "Understandability": 63.333 },
"mti_exact_match_percentage_by_metric": { "coherence": 65.833, "specificity": 56.667, "informativeness": 58.333, "relevance": 94.583,
"Understandability": 67.083 }, "calculation_timestamp": { "$date": "2025-11-26T12:34:11.251Z" }, "matched_documents": 240, "total_documents": 240 } 

- Perceba que nesse json , temos duas métricas que são feitas referentes a um conjunto de respostas. 
As métricas são o krippendorff alpha para cada métrica e uma média delas e uma métrica que define quanto
% das instâncias foram dadas notas iguais em cada métrica e uma média delas. Esse json é de uma collection intermediária apenas
feita para refinar um estágio do desenvolvimento 
   
- Na minha collection 'llm_judge_results', para cada modelo, temos as notas de cada instância de acordo com as 5 métricas fornecidas anteriormente 

- O que eu pretendo comparar, por meio de métricas como essa, é o desempenho, para cada modelo de linguagem,
aa abordagem MTI e STI e qual foi a diferença das notas entre cada uma das abordagens. 
Por exemplo, digamos que o MTI teve mais notas 5 para a métrica 'relevance', considerando
todas as instâncias, do que o STI, isso deve ser metrificado. - Pretendo avaliar também o desvio padrão de cada modelo em relação
as notas de cada métrica para todas as instâncias - Eu pretendo apresentar o resultado do meu trabalho em um artigo científico,
e essas métricas devem ficar bem claras, para cada métrica.

[doc com notas finais na collection llm_judge_results]:
{
  "_id": {
    "$oid": "69274d8f5af958558d8c1382"
  },
  "llm_judge_experiment_id": {
    "$oid": "69274d215af958558d8c1381"
  },
  "experiment_name": "LLM_Judge_gpt-4o-mini",
  "sti_experiment_id": {
    "$oid": "691e5cdb47cb353cce0b14b0"
  },
  "mti_experiment_id": {
    "$oid": "691f07ff79c79aad7e95f16d"
  },
  "task_id": "030",
  "instance_id": "030_1002",
  "evaluator_model": "gpt-4o-mini-2024-07-18",
  "evaluation_timestamp": {
    "$date": "2025-11-26T15:56:48.253Z"
  },

  "llm_judge_MTI_answer": {
    "coherence": {
      "score": 4,
      "explanation": "The response is well-structured and follows a logical flow, making it easy to follow."
    },
    "specificity": {
      "score": 4,
      "explanation": "It provides relevant details about feral children and their upbringing, but could include more context."
    },
    "informativeness": {
      "score": 4,
      "explanation": "The answer covers the essential points from the text, though it could delve deeper into the implications."
    },
    "relevance": {
      "score": 5,
      "explanation": "The response directly addresses the question regarding what does not apply to feral children."
    },
    "Understandability": {
      "score": 4,
      "explanation": "The language is clear and mostly straightforward, with minor phrasing issues that do not hinder comprehension."
    }
  },
  "llm_judge_STI_answer": {
    "coherence": {
      "score": 4,
      "explanation": "The response is logically organized and easy to follow, maintaining a clear structure."
    },
    "specificity": {
      "score": 4,
      "explanation": "It includes specific details about feral children but could enhance context for clarity."
    },
    "informativeness": {
      "score": 4,
      "explanation": "The answer effectively summarizes the key points from the text but lacks deeper analysis."
    },
    "relevance": {
      "score": 5,
      "explanation": "The response is directly relevant to the question about feral children's upbringing."
    },
    "Understandability": {
      "score": 4,
      "explanation": "The text is generally clear and understandable, with minor issues that do not significantly affect clarity."
    }
  }
}

- "Task" - Produza um código em Python que leia os documentos da collection 'llm_judge_results' 
e calcule a métrica de comparação entre as abordagens STI e MTI para cada modelo de linguagem.
Para cada experimento, identificado pelo experiment_name, calcule as seguintes métricas:
    1. A média das notas para cada métrica (coerência, especificidade, compreensibilidade, informatividade e relevância) 
    para ambas as abordagens (STI e MTI). Indicando assim sua pontuação média. 
    2. O desvio padrão das notas para cada métrica para ambas as abordagens (STI e MTI).
    3. A contagem de quantas vezes cada abordagem (STI e MTI) recebeu a nota máxima (5) para cada métrica.
    4. A diferença entre as médias das notas de cada métrica entre as abordagens STI e MTI.

Deve-se elencar, para cada experimento, os resultados comparativos entre MTI e STI, indicando qual abordagem resultou 
em melhores desempenhos para cada métrica avaliada e para a média geral das métricas.

Com base na collection 'llm_judge_results', produza uma nova collection chamada 'llm_judge_comparative_metrics'
 

"""

import os
import numpy as np
from pymongo import MongoClient
from bson import ObjectId
from datetime import datetime
from typing import Dict, List
from collections import defaultdict
from dotenv import load_dotenv

load_dotenv()

# Conexão MongoDB
MONGO_URI = os.getenv("MONGODB_URI")
mongo_client = MongoClient(MONGO_URI)
db = mongo_client["experiments_db"]
llm_judge_results_collection = db["llm_judge_results"]
comparative_metrics_collection = db["llm_judge_comparative_metrics"]

# Métricas a serem avaliadas
METRICS = ["coherence", "specificity", "informativeness", "relevance", "Understandability"]


def extract_scores_by_experiment() -> Dict[str, Dict[str, Dict[str, List[int]]]]:
    """
    Extrai as notas de todos os documentos da collection llm_judge_results
    agrupadas por experiment_name, approach (STI/MTI) e métrica.
    
    Returns:
        Dict com estrutura: {experiment_name: {approach: {metric: [scores]}}}
    """
    print("📊 Extraindo notas da collection llm_judge_results...")
    
    # Estrutura: experiment_name -> approach -> metric -> list of scores
    scores_by_experiment = defaultdict(lambda: {
        "STI": defaultdict(list),
        "MTI": defaultdict(list)
    })
    
    # Buscar todos os documentos
    documents = llm_judge_results_collection.find()
    doc_count = 0
    
    for doc in documents:
        doc_count += 1
        experiment_name = doc.get("experiment_name")
        
        if not experiment_name:
            continue
            
        # Extrair notas STI
        sti_answer = doc.get("llm_judge_STI_answer", {})
        for metric in METRICS:
            if metric in sti_answer and "score" in sti_answer[metric]:
                score = sti_answer[metric]["score"]
                scores_by_experiment[experiment_name]["STI"][metric].append(score)
        
        # Extrair notas MTI
        mti_answer = doc.get("llm_judge_MTI_answer", {})
        for metric in METRICS:
            if metric in mti_answer and "score" in mti_answer[metric]:
                score = mti_answer[metric]["score"]
                scores_by_experiment[experiment_name]["MTI"][metric].append(score)
    
    print(f"✅ {doc_count} documentos processados")
    print(f"✅ {len(scores_by_experiment)} experimentos encontrados")
    
    return scores_by_experiment


def calculate_statistics(scores: List[int]) -> Dict:
    """
    Calcula estatísticas para uma lista de notas.
    
    Args:
        scores: Lista de notas (1-5)
    
    Returns:
        Dict com mean, std, e count_max_score
    """
    if not scores:
        return {
            "mean": 0.0,
            "std": 0.0,
            "count_max_score": 0,
            "total_samples": 0
        }
    
    scores_array = np.array(scores)
    
    return {
        "mean": float(np.mean(scores_array)),
        "std": float(np.std(scores_array, ddof=1) if len(scores) > 1 else 0.0),
        "count_max_score": int(np.sum(scores_array == 5)),
        "total_samples": len(scores)
    }


def calculate_comparative_metrics(scores_by_experiment: Dict) -> List[Dict]:
    """
    Calcula as métricas comparativas entre STI e MTI para cada experimento.
    
    Args:
        scores_by_experiment: Dict com as notas agrupadas
    
    Returns:
        Lista de documentos prontos para inserir na collection
    """
    print("\n📈 Calculando métricas comparativas...")
    
    results = []
    
    for experiment_name, approaches in scores_by_experiment.items():
        print(f"\n  Processando experimento: {experiment_name}")
        
        sti_scores = approaches["STI"]
        mti_scores = approaches["MTI"]
        
        # Calcular estatísticas por métrica
        sti_metrics_stats = {}
        mti_metrics_stats = {}
        differences = {}
        
        overall_sti_scores = []
        overall_mti_scores = []
        
        for metric in METRICS:
            sti_metric_scores = sti_scores.get(metric, [])
            mti_metric_scores = mti_scores.get(metric, [])
            
            sti_stats = calculate_statistics(sti_metric_scores)
            mti_stats = calculate_statistics(mti_metric_scores)
            
            sti_metrics_stats[metric] = sti_stats
            mti_metrics_stats[metric] = mti_stats
            
            # Calcular diferença (MTI - STI)
            differences[metric] = {
                "mean_difference": mti_stats["mean"] - sti_stats["mean"],
                "better_approach": "MTI" if mti_stats["mean"] > sti_stats["mean"] else "STI" if sti_stats["mean"] > mti_stats["mean"] else "TIE"
            }
            
            # Acumular para média geral
            overall_sti_scores.extend(sti_metric_scores)
            overall_mti_scores.extend(mti_metric_scores)
        
        # Calcular estatísticas gerais (média de todas as métricas)
        overall_sti_stats = calculate_statistics(overall_sti_scores)
        overall_mti_stats = calculate_statistics(overall_mti_scores)
        
        # Determinar abordagem vencedora geral
        wins_mti = sum(1 for d in differences.values() if d["better_approach"] == "MTI")
        wins_sti = sum(1 for d in differences.values() if d["better_approach"] == "STI")
        ties = sum(1 for d in differences.values() if d["better_approach"] == "TIE")
        
        # Construir documento de resultado
        result_doc = {
            "experiment_name": experiment_name,
            "calculation_timestamp": datetime.utcnow(),
            
            # Estatísticas STI por métrica
            "STI_metrics": sti_metrics_stats,
            
            # Estatísticas MTI por métrica
            "MTI_metrics": mti_metrics_stats,
            
            # Diferenças entre MTI e STI
            "differences": differences,
            
            # Estatísticas gerais (todas as métricas combinadas)
            "overall_statistics": {
                "STI": overall_sti_stats,
                "MTI": overall_mti_stats,
                "mean_difference": overall_mti_stats["mean"] - overall_sti_stats["mean"],
                "better_approach_overall": "MTI" if overall_mti_stats["mean"] > overall_sti_stats["mean"] else "STI" if overall_sti_stats["mean"] > overall_mti_stats["mean"] else "TIE"
            },
            
            # Resumo comparativo
            "comparative_summary": {
                "metrics_won_by_MTI": wins_mti,
                "metrics_won_by_STI": wins_sti,
                "metrics_tied": ties,
                "dominant_approach": "MTI" if wins_mti > wins_sti else "STI" if wins_sti > wins_mti else "BALANCED"
            }
        }
        
        results.append(result_doc)
        
        # Imprimir resumo
        print(f"    ✓ STI geral: μ={overall_sti_stats['mean']:.3f}, σ={overall_sti_stats['std']:.3f}")
        print(f"    ✓ MTI geral: μ={overall_mti_stats['mean']:.3f}, σ={overall_mti_stats['std']:.3f}")
        print(f"    ✓ Melhor abordagem: {result_doc['comparative_summary']['dominant_approach']}")
    
    return results


def save_results_to_collection(results: List[Dict]):
    """
    Salva os resultados na collection llm_judge_comparative_metrics.
    Remove documentos antigos do mesmo experimento antes de inserir novos.
    """
    print("\n💾 Salvando resultados na collection llm_judge_comparative_metrics...")
    
    for result in results:
        experiment_name = result["experiment_name"]
        
        # Remover documentos antigos do mesmo experimento
        delete_result = comparative_metrics_collection.delete_many({
            "experiment_name": experiment_name
        })
        
        if delete_result.deleted_count > 0:
            print(f"  🗑️  Removidos {delete_result.deleted_count} documento(s) antigo(s) de {experiment_name}")
        
        # Inserir novo documento
        insert_result = comparative_metrics_collection.insert_one(result)
        print(f"  ✅ Documento inserido para {experiment_name} (ID: {insert_result.inserted_id})")
    
    print(f"\n✨ Total de {len(results)} experimento(s) processado(s) e salvos!")


def print_detailed_report(results: List[Dict]):
    """
    Imprime um relatório detalhado das métricas comparativas.
    """
    print("\n" + "="*80)
    print("📊 RELATÓRIO DETALHADO DE MÉTRICAS COMPARATIVAS STI vs MTI")
    print("="*80)
    
    for result in results:
        print(f"\n{'='*80}")
        print(f"EXPERIMENTO: {result['experiment_name']}")
        print(f"{'='*80}")
        
        print("\n📋 MÉTRICAS INDIVIDUAIS:")
        print("-" * 80)
        print(f"{'Métrica':<20} {'STI μ':<10} {'STI σ':<10} {'STI #5':<8} {'MTI μ':<10} {'MTI σ':<10} {'MTI #5':<8} {'Δ(MTI-STI)':<12} {'Melhor':<8}")
        print("-" * 80)
        
        for metric in METRICS:
            sti_stats = result["STI_metrics"][metric]
            mti_stats = result["MTI_metrics"][metric]
            diff = result["differences"][metric]
            
            print(f"{metric:<20} "
                  f"{sti_stats['mean']:<10.3f} "
                  f"{sti_stats['std']:<10.3f} "
                  f"{sti_stats['count_max_score']:<8} "
                  f"{mti_stats['mean']:<10.3f} "
                  f"{mti_stats['std']:<10.3f} "
                  f"{mti_stats['count_max_score']:<8} "
                  f"{diff['mean_difference']:<12.3f} "
                  f"{diff['better_approach']:<8}")
        
        print("\n📊 ESTATÍSTICAS GERAIS:")
        print("-" * 80)
        overall = result["overall_statistics"]
        print(f"STI - Média Geral: {overall['STI']['mean']:.3f} | Desvio Padrão: {overall['STI']['std']:.3f} | Notas 5: {overall['STI']['count_max_score']}")
        print(f"MTI - Média Geral: {overall['MTI']['mean']:.3f} | Desvio Padrão: {overall['MTI']['std']:.3f} | Notas 5: {overall['MTI']['count_max_score']}")
        print(f"Diferença (MTI - STI): {overall['mean_difference']:.3f}")
        print(f"Melhor Abordagem Geral: {overall['better_approach_overall']}")
        
        print("\n🏆 RESUMO COMPARATIVO:")
        print("-" * 80)
        summary = result["comparative_summary"]
        print(f"Métricas vencidas por MTI: {summary['metrics_won_by_MTI']}")
        print(f"Métricas vencidas por STI: {summary['metrics_won_by_STI']}")
        print(f"Métricas empatadas: {summary['metrics_tied']}")
        print(f"Abordagem Dominante: {summary['dominant_approach']}")


def main():
    """
    Função principal que executa todo o pipeline de cálculo de métricas comparativas.
    """
    print("🚀 Iniciando cálculo de métricas comparativas STI vs MTI\n")
    
    try:
        # 1. Extrair notas da collection
        scores_by_experiment = extract_scores_by_experiment()
        
        if not scores_by_experiment:
            print("⚠️ Nenhum experimento encontrado na collection llm_judge_results")
            return
        
        # 2. Calcular métricas comparativas
        results = calculate_comparative_metrics(scores_by_experiment)
        
        # 3. Salvar na collection
        save_results_to_collection(results)
        
        # 4. Imprimir relatório detalhado
        print_detailed_report(results)
        
        print("\n" + "="*80)
        print("✅ PROCESSAMENTO CONCLUÍDO COM SUCESSO!")
        print("="*80)
        
    except Exception as e:
        print(f"\n❌ ERRO durante o processamento: {str(e)}")
        raise
    finally:
        mongo_client.close()


if __name__ == "__main__":
    main()