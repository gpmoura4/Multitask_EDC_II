"""
Script para calcular a similaridade entre avaliações refine_judge_results e final_ground_truth
usando Weighted Cohen's Kappa.

Este script:
1. Busca documentos de refine_judge_results por experimento
2. Encontra os correspondentes em final_ground_truth (por task_id e instance_id)
3. Calcula Weighted Cohen's Kappa para STI e MTI separadamente
4. Salva resultados na collection refine_judge_similarity_results

Dependências:
    - scikit-learn: para cálculo de Cohen's Kappa
    - numpy: para operações numéricas
    
Instalação:
    pip install scikit-learn numpy
"""

import os
import numpy as np
from pymongo import MongoClient
from bson import ObjectId
from datetime import datetime
from typing import List, Dict, Tuple
from dotenv import load_dotenv

try:
    from sklearn.metrics import cohen_kappa_score
except ImportError:
    print("⚠️ AVISO: scikit-learn não está instalado. Execute: pip install scikit-learn")
    raise

load_dotenv()

# Conexão MongoDB
MONGO_URI = os.getenv("MONGODB_URI")
mongo_client = MongoClient(MONGO_URI)
db = mongo_client["experiments_db"]
refine_judge_collection = db["refine_judge_results"]
final_gt_collection = db["final_ground_truth"]
similarity_results_collection = db["refine_judge_similarity_results"]


"""
    "Context" - Estou escrevendo um artigo científico que tem como objetivo avaliar a eficiencia das técnicas de Single-Task Inference (STI) e Multi-Task Inference (MTI) com relação à qualidade de respostas de acordo com métricas de preferência humana.

    O Single-Task Inference refere-se a habilidade do LLM de seguir uma instrução por chamada de inferência, ou seja, aborda sub-tarefas sequencialmente . Já o Multi-Task Inference refere-se a habilidade de seguir instruções complexas que lidam com múltiplas sub-tasks em uma única chamada de inferência, ou seja, lidar com tarefas compostas por várias instruções em uma única chamada de inferência.

    As respostas geradas pelo Single-Task Inference e pelo Multi-Task Inference são avalidas com as características: coerência, especificidade, compreensibilidade, informatividade e relevância. Isso será feito para diferentes modelos de linguagem, ou seja, eu vou pedir para que diferentes modelos de linguagem gerem a reposta tanto utilizando a técnica de Single-Task Inference e Multi-Task Inference. E utilizarei um outro modelo de linguagem como juiz para avaliar essas respostas de acordo com as métricas de preferência humana definidas anteriormente. As notas (score) de cada métrica variam de 1 a 5, de acordo com a escala Likert.


    Eu possuo uma base de dados com meu ground truth, isto é, a avaliação esperada para uma resposta de acordo com cada critério que eu defini. A ground truth está em uma coleção do mongodb chamada final_ground_truth, existem 240 documentos nessa collection, sendo 12 tasks, identificadas pelo task_id, onde cada task possui 12 instâncias, identificadas pelo instance_id. Os documentos do ground truth tem o seguinte formato:
    <gt_documents>
    {
    "_id": {
        "$oid": "69221caac45c7312531adb5f"
    },
    "experiments": [
        "GT_gpt-4o-mini-2024-07-18_V3",
        "GT_llama-3.3-70b-versatile_V3",
        "GT_gpt-4o-2024-08-06_V3"
    ],
    "models": [
        "gpt-4o-2024-08-06",
        "gpt-4o-mini-2024-07-18",
        "llama-3.3-70b-versatile"
    ],
    "task_id": "036",
    "instance_id": "036_28565",
    "prompt_MTI": "<|user|> ### Example:\n\n### Instruction: Read the following passage, and follow the given steps.",
    "gt_MTI_answer": {
        "coherence": {
        "score_llama-3.3-70b-versatile": 5,
        "explanation_llama-3.3-70b-versatile": "The response is well-structured, following the instructions clearly and logically.",
        "score_gpt-4o-mini-2024-07-18": 4,
        "explanation_gpt-4o-mini-2024-07-18": "The response is mostly clear, though slightly less organized than the first.",
        "score_gpt-4o-2024-08-06": 4,
        "explanation_gpt-4o-2024-08-06": "The response is coherent and follows the task steps logically.",
        "median_score": 4
        },
        "specificity": {
        "score_llama-3.3-70b-versatile": 5,
        "explanation_llama-3.3-70b-versatile": "It provides a detailed and accurate translation of the original text.",
        "score_gpt-4o-mini-2024-07-18": 4,
        "explanation_gpt-4o-mini-2024-07-18": "It provides relevant details but could be more concise.",
        "score_gpt-4o-2024-08-06": 4,
        "explanation_gpt-4o-2024-08-06": "It includes specific details from the text in both translation and summary.",
        "median_score": 4
        },
        "informativeness": {
        "score_llama-3.3-70b-versatile": 5,
        "explanation_llama-3.3-70b-versatile": "The answer includes all necessary information for completing the task.",
        "score_gpt-4o-mini-2024-07-18": 4,
        "explanation_gpt-4o-mini-2024-07-18": "Contains all necessary components, but some details are repetitive.",
        "score_gpt-4o-2024-08-06": 4,
        "explanation_gpt-4o-2024-08-06": "The response provides a complete translation and a concise summary.",
        "median_score": 4
        },
        "relevance": {
        "score_llama-3.3-70b-versatile": 5,
        "explanation_llama-3.3-70b-versatile": "The content is fully aligned with the given instructions and task.",
        "score_gpt-4o-mini-2024-07-18": 5,
        "explanation_gpt-4o-mini-2024-07-18": "Fully aligned with the task requirements.",
        "score_gpt-4o-2024-08-06": 5,
        "explanation_gpt-4o-2024-08-06": "The content is entirely relevant to the given instructions.",
        "median_score": 5
        },
        "Understandability": {
        "score_llama-3.3-70b-versatile": 5,
        "explanation_llama-3.3-70b-versatile": "The language used is clear, precise, and easy to understand.",
        "score_gpt-4o-mini-2024-07-18": 4,
        "explanation_gpt-4o-mini-2024-07-18": "Generally clear, but some sentences could be more straightforward.",
        "score_gpt-4o-2024-08-06": 4,
        "explanation_gpt-4o-2024-08-06": "The language is clear and easy to understand, with minor phrasing issues.",
        "median_score": 4
        }
    },
    "gt_STI_answer": {
        "coherence": {
        "score_llama-3.3-70b-versatile": 4,
        "explanation_llama-3.3-70b-versatile": "The response generally follows the instructions but with minor inconsistencies.",
        "score_gpt-4o-mini-2024-07-18": 5,
        "explanation_gpt-4o-mini-2024-07-18": "The response is well-structured and follows the required steps clearly.",
        "score_gpt-4o-2024-08-06": 4,
        "explanation_gpt-4o-2024-08-06": "The response is well-structured, following the steps logically with minor awkward phrasing.",
        "median_score": 4
        },
        "specificity": {
        "score_llama-3.3-70b-versatile": 4,
        "explanation_llama-3.3-70b-versatile": "It provides a mostly accurate translation, though some details might be slightly off or missing.",
        "score_gpt-4o-mini-2024-07-18": 5,
        "explanation_gpt-4o-mini-2024-07-18": "It includes detailed instructions relevant to the task.",
        "score_gpt-4o-2024-08-06": 4,
        "explanation_gpt-4o-2024-08-06": "The translation and summary include relevant details from the text.",
        "median_score": 4
        },
        "informativeness": {
        "score_llama-3.3-70b-versatile": 4,
        "explanation_llama-3.3-70b-versatile": "The answer covers the main points but might lack some depth or clarity.",
        "score_gpt-4o-mini-2024-07-18": 5,
        "explanation_gpt-4o-mini-2024-07-18": "All necessary information for translating and summarizing is provided.",
        "score_gpt-4o-2024-08-06": 4,
        "explanation_gpt-4o-2024-08-06": "The response covers both tasks, though the summary could be more concise.",
        "median_score": 4
        },
        "relevance": {
        "score_llama-3.3-70b-versatile": 4,
        "explanation_llama-3.3-70b-versatile": "The content is largely relevant to the task, with only minor deviations.",
        "score_gpt-4o-mini-2024-07-18": 5,
        "explanation_gpt-4o-mini-2024-07-18": "The content is entirely focused on the task at hand.",
        "score_gpt-4o-2024-08-06": 5,
        "explanation_gpt-4o-2024-08-06": "The answer fully aligns with the task requirements.",
        "median_score": 5
        },
        "Understandability": {
        "score_llama-3.3-70b-versatile": 4,
        "explanation_llama-3.3-70b-versatile": "The language is mostly clear, though there might be some minor issues with phrasing or grammar.",
        "score_gpt-4o-mini-2024-07-18": 5,
        "explanation_gpt-4o-mini-2024-07-18": "The language is clear and easy to follow.",
        "score_gpt-4o-2024-08-06": 4,
        "explanation_gpt-4o-2024-08-06": "The language is mostly clear, with minor phrasing issues that do not hinder comprehension.",
        "median_score": 4
        }
    },
    "sti_experiment_id": {
        "$oid": "691e5cdb47cb353cce0b14b0"
    },
    "mti_experiment_id": {
        "$oid": "691f07ff79c79aad7e95f16d"
    },
    "gt_experiment_ids": [
        {
        "$oid": "6922047501710f166e7e9d00"
        },
        {
        "$oid": "6922047501710f166e7e9cff"
        },
        {
        "$oid": "6922047501710f166e7e9cfe"
        }
    ]
    }
    </gt_documents> 

    O documento foi gerado a partir de três avaliações diferentes e então foi realizada uma mediana de cada avaliação para cada um dos 5 critérios, esse valor está armazenado nos campos "median_score" dentro do objeto de cada critério.

    Com isso, eu quero definir qual prompt final utilizarei para gerar as avaliações no passo final que é quando o LLM Judge age. Foram selecionadas as mesmas respostas MTI e STI na avaliação ground truth e enviei cada um LLM julgar utilizando os mesmos critérios e características definidas anteriormente, o que foi alterado foi apenas o prompt enviado para esse LLM. O Objetivo avaliar a diferença de cada nota gerada de acordo com cada prompt, verificando assim qual prompt gerou respostas mais semelhantes com o ground_truth.

    Esse resultado está armazenado na collection do mongodb refine_judge_results. Cada elemento dessa collection deve ser identificado pelo _id e pelo seu experiment_gt_name, já que tem mais de um experiment_gt_name, pois testei para mais de um prompt. Os documentos dessa coleção tem o seguinte formato:
    <refine_collection>
    {
    "_id": {
        "$oid": "69224b43d6562bd8c9d346f9"
    },
    "experiment_gt_name": "refine_judge_gpt-4o-mini-2024-07-18_V2",
    "sti_experiment_id": {
        "$oid": "691e5cdb47cb353cce0b14b0"
    },
    "mti_experiment_id": {
        "$oid": "691f07ff79c79aad7e95f16d"
    },
    "task_id": "031",
    "instance_id": "031_2076",
    "evaluator_model": "gpt-4o-mini-2024-07-18",
    "evaluation_timestamp": {
        "$date": "2025-11-22T20:36:04.995Z"
    },
    "prompt_MTI": "<|user|> ### Example:\n\n### Instruction: Read the following passage, and follow the given steps.",
    "prompt_STI": "<|user|> ### Example:\n\n### (1) Instruction: Transate the given text to English,
    "gt_prompt": "\n[System]\nAct as a neutral and unbiased evaluator. You will evaluate two AI-generated answers to the same user query. For each answer, follow this ",
    "gt_MTI_answer": {
        "coherence": {
        "score": 5,
        "explanation": "The response is logically organized and maintains a clear progression of ideas."
        },
        "specificity": {
        "score": 5,
        "explanation": "It includes precise information about the countries involved with the H5N1 virus."
        },
        "informativeness": {
        "score": 5,
        "explanation": "The answer provides comprehensive details and directly addresses the user's question."
        },
        "relevance": {
        "score": 5,
        "explanation": "Every part of the response is pertinent to the task and the question asked."
        },
        "Understandability": {
        "score": 5,
        "explanation": "The writing is clear, making it easy for the reader to comprehend."
        }
    },
    "gt_STI_answer": {
        "coherence": {
        "score": 5,
        "explanation": "The answer is well-structured and flows logically from the translation to the conclusion."
        },
        "specificity": {
        "score": 5,
        "explanation": "It provides specific details about the countries affected by the H5N1 virus."
        },
        "informativeness": {
        "score": 5,
        "explanation": "The response thoroughly explains the context and directly answers the question."
        },
        "relevance": {
        "score": 5,
        "explanation": "All content is directly related to the user's question and the provided text."
        },
        "Understandability": {
        "score": 5,
        "explanation": "The language is clear and easy to understand."
        }
    }
    }
    </refine_collection>


    Diante disso, deve-se gerar o resultado de quão similar as notas da etapa da refine_judge_results foram em relação as  medianas de cada métrica dos documentos de final_ground_truth, para assim determinar quais dos experimentos foram mais similares com as notas de ground_truth e quais foram menos similares. A diferença de um experimento na refine_judge_results para outro pode ser identificado pelo campo 'refine_judge_results'. Atualmente, a collection conta com 480 documentos, sendo 240 seguindo a abordagem com o gt_prompt 'A' e 240 seguindo a abordagem com o gt_prompt 'B', cada registro de cada experimento são referentes as mesmas instâncias de cada task da collection 'final_ground_truth'. Os registros do prompt 'A' possuem experiment_gt_name igual a "refine_judge_gpt-4o-mini-2024-07-18_V1" e o do prompt 'B' "refine_judge_gpt-4o-mini-2024-07-18_V2". Considere que podem existir diversos experimentos para além desses dois descritos, a ideia descrita foi só um exemplo de como estão distribuídos os elementos. 

    Considere que para calcular essa métrica utilizaremos
    a abordagem Weighted Cohen's Kappa para assim verificar o nível de concordância (similaridade) entre as respostas. Ao final, deve-se indicar qual a pontuação de similaridade tanto das notas da abordagem STI quanto MTI referente a um dado experimento específico da collection refine_judge_results.

    Deve-se criar uma nova collection chamada 'refine_judge_similarity_results'
    que armazenará os resultados dessa similaridade, contendo os seguintes campos:
    {
        "_id": ObjectId,
        "refine_judge_experiment_id": ObjectId, (id do elemento refine judge refine_judge_results)
        "experiment_name": str, (deve ser o valor de experiment_gt_name de  refine_judge_results)
        "prompt": str, # Deve guardar o gt_prompt utilizado no experimento (refine_judge_results)
        "sti_cohen_kappa": float,
        "mti_cohen_kappa": float,
        "calculation_timestamp": datetime
    }
"""


def extract_scores_from_answer(answer_data: Dict, metrics: List[str]) -> List[int]:
    """
    Extrai scores de um documento de answer (MTI ou STI).
    
    Args:
        answer_data: Dict contendo dados de gt_MTI_answer ou gt_STI_answer
        metrics: Lista de nomes de métricas a extrair
    
    Returns:
        Lista de scores na ordem das métricas
    """
    scores = []
    for metric in metrics:
        metric_data = answer_data.get(metric, {})
        score = metric_data.get("score", 0)
        scores.append(score)
    
    return scores


def extract_median_scores_from_gt(gt_answer_data: Dict, metrics: List[str]) -> List[int]:
    """
    Extrai median_scores de um documento ground truth.
    
    Args:
        gt_answer_data: Dict contendo dados de gt_MTI_answer ou gt_STI_answer do ground truth
        metrics: Lista de nomes de métricas a extrair
    
    Returns:
        Lista de median_scores na ordem das métricas
    """
    scores = []
    for metric in metrics:
        metric_data = gt_answer_data.get(metric, {})
        median_score = metric_data.get("median_score", 0)
        scores.append(median_score)
    
    return scores


def calculate_weighted_cohen_kappa(y_true: List[int], y_pred: List[int]) -> float:
    """
    Calcula Weighted Cohen's Kappa entre duas listas de scores.
    
    Args:
        y_true: Lista de scores ground truth
        y_pred: Lista de scores preditos
    
    Returns:
        Valor de Cohen's Kappa (weighted)
    """
    if not y_true or not y_pred or len(y_true) != len(y_pred):
        return 0.0
    
    try:
        # Usar sklearn para calcular Cohen's Kappa com pesos quadráticos
        kappa = cohen_kappa_score(y_true, y_pred, weights='quadratic')
        return float(kappa)
    except Exception as e:
        print(f"⚠️ Erro ao calcular Cohen's Kappa: {e}")
        return 0.0


def get_ground_truth_document(task_id: str, instance_id: str) -> Dict | None:
    """
    Busca documento correspondente na collection final_ground_truth.
    
    Args:
        task_id: ID da task
        instance_id: ID da instância
    
    Returns:
        Documento de ground truth ou None se não encontrado
    """
    gt_doc = final_gt_collection.find_one({
        "task_id": task_id,
        "instance_id": instance_id
    })
    
    return gt_doc


def calculate_similarity_for_experiment(experiment_name: str) -> Dict | None:
    """
    Calcula Cohen's Kappa para todos os documentos de um experimento.
    
    Args:
        experiment_name: Nome do experimento (experiment_gt_name)
    
    Returns:
        Dict com resultados agregados ou None se houver erro
    """
    print(f"\n🔄 Processando experimento: {experiment_name}")
    
    # Buscar todos os documentos deste experimento
    refine_docs = list(refine_judge_collection.find({
        "experiment_gt_name": experiment_name
    }))
    
    if not refine_docs:
        print(f"⚠️ Nenhum documento encontrado para experimento '{experiment_name}'")
        return None
    
    print(f"📊 Encontrados {len(refine_docs)} documentos")
    
    # Métricas a serem comparadas
    metrics = ["coherence", "specificity", "informativeness", "relevance", "Understandability"]
    
    # Listas para armazenar todos os scores
    mti_gt_scores = []
    mti_refine_scores = []
    sti_gt_scores = []
    sti_refine_scores = []
    
    matched_count = 0
    
    for refine_doc in refine_docs:
        task_id = refine_doc.get("task_id")
        instance_id = refine_doc.get("instance_id")
        
        # Buscar documento correspondente no ground truth
        gt_doc = get_ground_truth_document(task_id, instance_id)
        
        if not gt_doc:
            print(f"  ⚠️ Ground truth não encontrado para {instance_id}")
            continue
        
        matched_count += 1
        
        # Extrair scores MTI
        mti_refine = extract_scores_from_answer(
            refine_doc.get("gt_MTI_answer", {}), 
            metrics
        )
        mti_gt = extract_median_scores_from_gt(
            gt_doc.get("gt_MTI_answer", {}), 
            metrics
        )
        
        # Extrair scores STI
        sti_refine = extract_scores_from_answer(
            refine_doc.get("gt_STI_answer", {}), 
            metrics
        )
        sti_gt = extract_median_scores_from_gt(
            gt_doc.get("gt_STI_answer", {}), 
            metrics
        )
        
        # Adicionar às listas
        mti_gt_scores.extend(mti_gt)
        mti_refine_scores.extend(mti_refine)
        sti_gt_scores.extend(sti_gt)
        sti_refine_scores.extend(sti_refine)
    
    print(f"✅ {matched_count} documentos pareados com ground truth")
    
    if matched_count == 0:
        print("❌ Nenhum documento pareado, não é possível calcular similaridade")
        return None
    
    # Calcular Cohen's Kappa para MTI e STI
    mti_kappa = calculate_weighted_cohen_kappa(mti_gt_scores, mti_refine_scores)
    sti_kappa = calculate_weighted_cohen_kappa(sti_gt_scores, sti_refine_scores)
    
    print(f"📈 MTI Cohen's Kappa: {mti_kappa:.4f}")
    print(f"📈 STI Cohen's Kappa: {sti_kappa:.4f}")
    
    # Pegar o gt_prompt de um documento representativo
    sample_doc = refine_docs[0]
    gt_prompt = sample_doc.get("gt_prompt", "")
    
    return {
        "experiment_name": experiment_name,
        "prompt": gt_prompt,
        "sti_cohen_kappa": sti_kappa,
        "mti_cohen_kappa": mti_kappa,
        "matched_documents": matched_count,
        "total_documents": len(refine_docs)
    }


def get_unique_experiment_names() -> List[str]:
    """
    Obtém lista de experiment_gt_name únicos da collection refine_judge_results.
    
    Returns:
        Lista de nomes únicos de experimentos
    """
    experiment_names = refine_judge_collection.distinct("experiment_gt_name")
    return [name for name in experiment_names if name]


def save_similarity_results(experiment_name: str, results: Dict) -> ObjectId | None:
    """
    Salva resultados de similaridade na collection refine_judge_similarity_results.
    
    Args:
        experiment_name: Nome do experimento
        results: Dict com resultados calculados
    
    Returns:
        ObjectId do documento inserido ou None se houver erro
    """
    # Verificar se já existe resultado para este experimento
    existing = similarity_results_collection.find_one({
        "experiment_name": experiment_name
    })
    
    if existing:
        print(f"⚠️ Resultado já existe para '{experiment_name}', atualizando...")
        
        similarity_results_collection.update_one(
            {"experiment_name": experiment_name},
            {"$set": {
                "prompt": results["prompt"],
                "sti_cohen_kappa": results["sti_cohen_kappa"],
                "mti_cohen_kappa": results["mti_cohen_kappa"],
                "calculation_timestamp": datetime.now(),
                "matched_documents": results["matched_documents"],
                "total_documents": results["total_documents"]
            }}
        )
        
        return existing["_id"]
    
    # Criar novo documento
    similarity_doc = {
        "experiment_name": experiment_name,
        "prompt": results["prompt"],
        "sti_cohen_kappa": results["sti_cohen_kappa"],
        "mti_cohen_kappa": results["mti_cohen_kappa"],
        "calculation_timestamp": datetime.now(),
        "matched_documents": results["matched_documents"],
        "total_documents": results["total_documents"]
    }
    
    result = similarity_results_collection.insert_one(similarity_doc)
    print(f"✅ Resultado salvo com ID: {result.inserted_id}")
    
    return result.inserted_id


def process_all_experiments() -> Dict[str, Dict]:
    """
    Processa todos os experimentos únicos e calcula similaridade.
    
    Returns:
        Dict mapeando experiment_name para resultados
    """
    experiment_names = get_unique_experiment_names()
    
    if not experiment_names:
        print("⚠️ Nenhum experimento encontrado na collection refine_judge_results")
        return {}
    
    print(f"\n📊 Encontrados {len(experiment_names)} experimentos únicos:")
    for name in experiment_names:
        print(f"  - {name}")
    print()
    
    results_map = {}
    
    for experiment_name in experiment_names:
        try:
            # Calcular similaridade
            results = calculate_similarity_for_experiment(experiment_name)
            
            if results:
                # Salvar resultados
                save_similarity_results(experiment_name, results)
                results_map[experiment_name] = results
            
        except Exception as e:
            print(f"❌ Erro ao processar experimento '{experiment_name}': {e}")
    
    return results_map


def main():
    """
    Função principal para execução do script.
    """
    print("\n" + "="*70)
    print("CÁLCULO DE SIMILARIDADE - REFINE JUDGE vs GROUND TRUTH")
    print("="*70)
    
    # Processar todos os experimentos
    results_map = process_all_experiments()
    
    if results_map:
        print("\n" + "="*70)
        print("RESUMO DOS RESULTADOS:")
        print("="*70)
        
        for exp_name, results in results_map.items():
            print(f"\n📋 Experimento: {exp_name}")
            print(f"   MTI Cohen's Kappa: {results['mti_cohen_kappa']:.4f}")
            print(f"   STI Cohen's Kappa: {results['sti_cohen_kappa']:.4f}")
            print(f"   Documentos pareados: {results['matched_documents']}/{results['total_documents']}")
        
        print("\n" + "="*70)
        print(f"Collection: refine_judge_similarity_results")
        print(f"Total de resultados: {similarity_results_collection.count_documents({})}")
        print("="*70 + "\n")
    else:
        print("\n⚠️ Nenhum resultado foi gerado")


if __name__ == "__main__":
    main()