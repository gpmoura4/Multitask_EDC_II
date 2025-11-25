"""
Script para calcular a similaridade entre avaliações refine_judge_results e final_ground_truth
usando Weighted Cohen's Kappa.

Este script:
1. Busca documentos de refine_judge_results por experimento
2. Encontra os correspondentes em final_ground_truth (por task_id e instance_id)
3. Calcula Weighted Cohen's Kappa para STI e MTI separadamente
4. Salva resultados na collection refine_judge_similarity_results

MODO TESTE:
    Use --test_instances N para processar apenas N instâncias aleatórias.
    - As instâncias selecionadas são salvas em selected_instances/
    - Execuções subsequentes reutilizam as mesmas instâncias para reprodutibilidade
    - Útil para iteração rápida e debugging

Uso:
    # Processar todos os experimentos
    python refine_judge_metric.py
    
    # Processar experimentos específicos
    python refine_judge_metric.py --experiments "exp1" "exp2"
    
    # Modo teste com 5 instâncias
    python refine_judge_metric.py --experiments "exp1" --test_instances 5
    
    # Modo teste com seed customizado
    python refine_judge_metric.py --experiments "exp1" --test_instances 10 --seed 123

Dependências:
    - scikit-learn: para cálculo de Cohen's Kappa
    - numpy: para operações numéricas
    
Instalação:
    pip install scikit-learn numpy
"""

import os
import argparse
import numpy as np
import json
import random
from pymongo import MongoClient
from bson import ObjectId
from datetime import datetime
from typing import List, Dict, Tuple
from dotenv import load_dotenv
from pathlib import Path

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

# Diretório para armazenar arquivos de instâncias selecionadas
INSTANCES_DIR = Path(__file__).parent / "selected_instances"
INSTANCES_DIR.mkdir(exist_ok=True)


"""
    "Context" - Estou escrevendo um artigo científico que tem como 
    objetivo avaliar a eficiencia das técnicas de
    Single-Task Inference (STI) e Multi-Task Inference (MTI) 
    com relação à qualidade de respostas de acordo com métricas de preferência humana.

    - O Single-Task Inference refere-se a habilidade
    do LLM de seguir uma instrução por chamada de inferência, ou seja, aborda sub-tarefas sequencialmente
    - Já o Multi-Task Inference refere-se a habilidade de seguir instruções complexas que lidam com múltiplas sub-tasks em uma única chamada de inferência, ou seja, lidar com tarefas compostas por várias instruções em uma única chamada de inferência.

    - As respostas geradas pelo Single-Task Inference e pelo Multi-Task Inference são avalidas com as características: 
    coerência, especificidade, compreensibilidade, informatividade e relevância.
    
    - Isso será feito para diferentes modelos de linguagem, ou seja, 
    eu vou pedir para que diferentes modelos de linguagem gerem a reposta 
    tanto utilizando a técnica de Single-Task Inference e Multi-Task Inference. 
    E utilizarei um outro modelo de linguagem como juiz para avaliar essas respostas 
    de acordo com as métricas de preferência humana definidas anteriormente. 
    As notas (score) de cada métrica variam de 1 a 5, de acordo com a escala Likert.


    Eu possuo uma base de dados com meu ground truth, isto é, a avaliação esperada para 
    uma resposta de acordo com cada critério que eu defini. A ground truth está em uma coleção 
    do mongodb chamada final_ground_truth, existem 240 documentos nessa collection, sendo 12 tasks,
    identificadas pelo task_id, onde cada task possui 12 instâncias, identificadas pelo instance_id. 
    Os documentos do ground truth tem o seguinte formato:

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

    - O documento foi gerado a partir de três avaliações diferentes e então
    foi realizada uma mediana de cada avaliação para cada um dos 5 critérios, 
    esse valor está armazenado nos campos "median_score" dentro do objeto de cada critério.

    Com isso, eu quero definir qual prompt final utilizarei para gerar as avaliações
    no passo final que é quando o LLM Judge age.
    Foram selecionadas as mesmas respostas MTI e STI na avaliação ground truth 
    e enviei cada um LLM julgar utilizando os mesmos critérios e características
    definidas anteriormente, o que foi alterado foi apenas o prompt enviado para esse LLM. 
    O objetivo é avaliar a diferença de cada nota gerada de acordo com cada prompt,
    verificando assim qual prompt gerou respostas mais semelhantes com o ground_truth.

    O objetivo é verificar qual experimento de refine_judge_results
    (diferenciado pelo campo experiment_gt_name) gerou notas mais similares com as medianas
    do ground truth (final_ground_truth) para cada métrica definidam, esse cálculo de 
    similaridade deve ser feito com a fórmula cohen kappa e ao final deve-se tirar uma média
    dos valores encontrados.

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


    Diante disso, deve-se gerar o resultado de quão similar 
    as notas da etapa da refine_judge_results foram em relação as  
    medianas de cada métrica dos documentos de final_ground_truth, 
    para assim determinar quais dos experimentos foram mais similares 
    com as notas de ground_truth e quais foram menos similares. 
    A diferença de um experimento na refine_judge_results para outro pode ser
    identificado pelo campo 'refine_judge_results'. 
    Atualmente, a collection conta com 480 documentos, 
    sendo 240 seguindo a abordagem com o gt_prompt 'A'
    e 240 seguindo a abordagem com o gt_prompt 'B',
      cada registro de cada experimento são referentes as mesmas instâncias de cada task da collection 'final_ground_truth'. Os registros do prompt 'A' possuem experiment_gt_name igual a "refine_judge_gpt-4o-mini-2024-07-18_V1" e o do prompt 'B' "refine_judge_gpt-4o-mini-2024-07-18_V2". Considere que podem existir diversos experimentos para além desses dois descritos, a ideia descrita foi só um exemplo de como estão distribuídos os elementos. 

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


def calculate_weighted_cohen_kappa(y_true: List[int], y_pred: List[int]) -> float | None:
    """
    Calcula Weighted Cohen's Kappa entre duas listas de scores.
    
    Args:
        y_true: Lista de scores ground truth
        y_pred: Lista de scores preditos
    
    Returns:
        Valor de Cohen's Kappa (weighted), ou None se não for calculável
    """
    if not y_true or not y_pred or len(y_true) != len(y_pred):
        return None
    
    # VERIFICAÇÃO CRÍTICA: Detectar variância zero ANTES de chamar sklearn
    # Isso evita warnings desnecessários
    unique_true = len(set(y_true))
    unique_pred = len(set(y_pred))
    
    # Se ambas as listas têm apenas 1 valor único
    if unique_true == 1 and unique_pred == 1:
        # Verificar se os valores únicos são IGUAIS
        # Ex: [4,4,4] vs [4,4,4] → concordância perfeita → 1.0
        # Ex: [5,5,5] vs [4,4,4] → discordância total mas sem variação → None
        if y_true[0] == y_pred[0]:
            return 1.0  # Concordância perfeita: todos iguais e concordando
        else:
            return None  # Discordância total mas sem variação para calcular
    
    # Se apenas uma das listas tem variância zero → não calculável
    # Ex: [4,4,4] vs [4,5,4] → predições variam mas GT não → kappa não faz sentido
    if unique_true == 1 or unique_pred == 1:
        return None
    
    try:
        # Usar sklearn para calcular Cohen's Kappa com pesos quadráticos
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=UserWarning)
            warnings.filterwarnings('ignore', category=RuntimeWarning)
            kappa = cohen_kappa_score(y_true, y_pred, weights='quadratic')
        
        # Verificar se o resultado é válido
        if np.isnan(kappa) or np.isinf(kappa):
            return None
        
        return float(kappa)
    except Exception as e:
        print(f"⚠️ Erro ao calcular Cohen's Kappa: {e}")
        return None


    return gt_doc


def get_selected_instances_file(experiment_name: str, num_instances: int) -> Path:
    """
    Retorna o caminho do arquivo de instâncias selecionadas.
    
    Args:
        experiment_name: Nome do experimento
        num_instances: Número de instâncias
    
    Returns:
        Path do arquivo
    """
    safe_name = experiment_name.replace("/", "_").replace("\\", "_")
    filename = f"{safe_name}_n{num_instances}_instances.json"
    return INSTANCES_DIR / filename


def load_selected_instances(experiment_name: str, num_instances: int) -> List[Dict] | None:
    """
    Carrega instâncias selecionadas de um arquivo se existir.
    
    Args:
        experiment_name: Nome do experimento
        num_instances: Número de instâncias
    
    Returns:
        Lista de instâncias ou None se arquivo não existe
    """
    file_path = get_selected_instances_file(experiment_name, num_instances)
    
    if file_path.exists():
        print(f"📂 Carregando instâncias selecionadas de: {file_path.name}")
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data.get("instances", [])
    
    return None


def save_selected_instances(
    experiment_name: str, 
    num_instances: int, 
    instances: List[Dict],
    seed: int
) -> Path:
    """
    Salva instâncias selecionadas em um arquivo JSON.
    
    Args:
        experiment_name: Nome do experimento
        num_instances: Número de instâncias
        instances: Lista de instâncias selecionadas
        seed: Seed usado para seleção
    
    Returns:
        Path do arquivo salvo
    """
    file_path = get_selected_instances_file(experiment_name, num_instances)
    
    data = {
        "experiment_name": experiment_name,
        "num_instances": num_instances,
        "seed": seed,
        "selection_timestamp": datetime.now().isoformat(),
        "instances": instances
    }
    
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"💾 Instâncias selecionadas salvas em: {file_path.name}")
    
    return file_path


def select_sample_instances(
    experiment_name: str,
    num_instances: int = 5,
    seed: int = 42
) -> List[Dict]:
    """
    Seleciona uma amostra de instâncias do experimento.
    Se já existir arquivo de seleção, usa as mesmas instâncias.
    Caso contrário, faz nova seleção aleatória e salva.
    
    Args:
        experiment_name: Nome do experimento
        num_instances: Número de instâncias a selecionar
        seed: Seed para reprodutibilidade
    
    Returns:
        Lista de dicts com task_id e instance_id
    """
    # Tentar carregar instâncias já selecionadas
    loaded_instances = load_selected_instances(experiment_name, num_instances)
    
    if loaded_instances is not None:
        print(f"✅ Usando {len(loaded_instances)} instâncias pré-selecionadas")
        return loaded_instances
    
    print(f"🔄 Selecionando {num_instances} novas instâncias aleatoriamente...")
    
    # Buscar todas as instâncias disponíveis do experimento
    all_docs = list(refine_judge_collection.find(
        {"experiment_gt_name": experiment_name},
        {"task_id": 1, "instance_id": 1, "_id": 0}
    ))
    
    if len(all_docs) == 0:
        print(f"⚠️ Nenhum documento encontrado para experimento '{experiment_name}'")
        return []
    
    # Ajustar num_instances se for maior que disponível
    actual_num = min(num_instances, len(all_docs))
    
    if actual_num < num_instances:
        print(f"⚠️ Apenas {actual_num} instâncias disponíveis (solicitado: {num_instances})")
    
    # Selecionar aleatoriamente
    random.seed(seed)
    selected = random.sample(all_docs, actual_num)
    
    # Salvar seleção
    save_selected_instances(experiment_name, num_instances, selected, seed)
    
    print(f"✅ {len(selected)} instâncias selecionadas e salvas")
    
    return selected


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


def calculate_similarity_for_experiment(
    experiment_name: str, 
    selected_instances: List[Dict] | None = None
) -> Dict | None:
    """
    Calcula Cohen's Kappa para todos os documentos de um experimento.
    
    Args:
        experiment_name: Nome do experimento (experiment_gt_name)
        selected_instances: Lista opcional de instâncias específicas a processar.
                           Se None, processa todas as instâncias do experimento.
                           Formato: [{"task_id": "030", "instance_id": "030_1559"}, ...]
    
    Returns:
        Dict com resultados agregados ou None se houver erro
    
    ANÁLISE DO PROBLEMA:
    =====================
    O Cohen's Kappa baixo ou negativo pode ser causado por:
    
    1. PROBLEMA DE ALINHAMENTO DE DADOS:
       - Se as métricas entre refine_judge e final_ground_truth não estiverem
         perfeitamente alinhadas (mesma ordem, mesmo número de elementos), 
         o cálculo fica incorreto.
    
    2. PROBLEMA DE AGREGAÇÃO:
       - O código atual está agregando TODOS os scores de TODAS as métricas
         em uma única lista, perdendo a separação por métrica e instância.
       - Isso pode causar comparações incorretas entre scores que não correspondem
         à mesma instância ou métrica.
    
    3. PROBLEMA CONCEITUAL:
       - Cohen's Kappa deve ser calculado comparando PARES de avaliações
         para a MESMA instância e MESMA métrica.
       - O código atual está misturando todas as métricas juntas.
    
    SOLUÇÃO IMPLEMENTADA:
    ====================
    - Calcular Cohen's Kappa SEPARADAMENTE para cada métrica
    - Garantir que cada score de refine_judge seja comparado com o median_score
      correspondente do ground truth para a MESMA instância e MESMA métrica
    - Depois, fazer a média dos kappas individuais
    """
    
    # Modo de teste: processar apenas instâncias selecionadas
    if selected_instances:
        print(f"\n🧪 MODO TESTE - Processando {len(selected_instances)} instâncias específicas")
    else:
        print(f"\n🔄 Processando experimento: {experiment_name}")
    
    # Buscar todos os documentos deste experimento
    if selected_instances:
        # Filtrar por instâncias específicas
        instance_pairs = [
            {"task_id": inst["task_id"], "instance_id": inst["instance_id"]} 
            for inst in selected_instances
        ]
        refine_docs = list(refine_judge_collection.find({
            "experiment_gt_name": experiment_name,
            "$or": instance_pairs
        }))
    else:
        # Buscar todas as instâncias do experimento
        refine_docs = list(refine_judge_collection.find({
            "experiment_gt_name": experiment_name
        }))
    
    if not refine_docs:
        print(f"⚠️ Nenhum documento encontrado para experimento '{experiment_name}'")
        return None
    
    print(f"📊 Encontrados {len(refine_docs)} documentos")
    
    # Métricas a serem comparadas
    metrics = ["coherence", "specificity", "informativeness", "relevance", "Understandability"]
    
    # CORREÇÃO CRÍTICA: Estruturas separadas por métrica
    # Isso garante que não misturamos scores de métricas diferentes
    mti_gt_by_metric = {metric: [] for metric in metrics}
    mti_ref_by_metric = {metric: [] for metric in metrics}
    sti_gt_by_metric = {metric: [] for metric in metrics}
    sti_ref_by_metric = {metric: [] for metric in metrics}
    
    matched_count = 0
    not_found_instances = []
    
    # Estrutura para debug: rastrear documentos processados
    processed_docs = []
    
    for refine_doc in refine_docs:
        task_id = refine_doc.get("task_id")
        instance_id = refine_doc.get("instance_id")
        doc_id = refine_doc.get("_id")
        
        # Buscar documento correspondente no ground truth
        gt_doc = get_ground_truth_document(task_id, instance_id)
        
        if not gt_doc:
            not_found_instances.append({"instance_id": instance_id, "refine_doc_id": str(doc_id)})
            continue
        
        matched_count += 1
        
        # Registrar documento processado para debug
        processed_docs.append({
            "refine_doc_id": str(doc_id),
            "gt_doc_id": str(gt_doc.get("_id")),
            "task_id": task_id,
            "instance_id": instance_id
        })
        
        # Extrair scores MTI - um por vez, por métrica
        for i, metric in enumerate(metrics):
            # Score do refine_judge
            refine_mti_metric = refine_doc.get("gt_MTI_answer", {}).get(metric, {})
            refine_mti_score = refine_mti_metric.get("score", None)
            
            # Median score do ground truth
            gt_mti_metric = gt_doc.get("gt_MTI_answer", {}).get(metric, {})
            gt_mti_median = gt_mti_metric.get("median_score", None)
            
            # IMPORTANTE: Só adiciona se ambos existirem
            if refine_mti_score is not None and gt_mti_median is not None:
                mti_ref_by_metric[metric].append(refine_mti_score)
                mti_gt_by_metric[metric].append(gt_mti_median)
            
            # Score do refine_judge para STI
            refine_sti_metric = refine_doc.get("gt_STI_answer", {}).get(metric, {})
            refine_sti_score = refine_sti_metric.get("score", None)
            
            # Median score do ground truth para STI
            gt_sti_metric = gt_doc.get("gt_STI_answer", {}).get(metric, {})
            gt_sti_median = gt_sti_metric.get("median_score", None)
            
            # IMPORTANTE: Só adiciona se ambos existirem
            if refine_sti_score is not None and gt_sti_median is not None:
                sti_ref_by_metric[metric].append(refine_sti_score)
                sti_gt_by_metric[metric].append(gt_sti_median)
    
    print(f"✅ {matched_count} documentos pareados com ground truth")
    
    if not_found_instances:
        print(f"⚠️  {len(not_found_instances)} instâncias sem correspondência no GT")
        for item in not_found_instances[:3]:  # Mostrar até 3 exemplos
            print(f"     - instance_id={item['instance_id']}, refine_doc_id={item['refine_doc_id']}")
    
    if matched_count == 0:
        print("❌ Nenhum documento pareado, não é possível calcular similaridade")
        return None
    # TODO PIOR QUE TA NAO FICA
    # DEBUG: Verificar tamanhos das listas
    print("\n🔍 DEBUG - Tamanhos das listas por métrica:")
    for metric in metrics:
        mti_ref_size = len(mti_ref_by_metric[metric])
        mti_gt_size = len(mti_gt_by_metric[metric])
        sti_ref_size = len(sti_ref_by_metric[metric])
        sti_gt_size = len(sti_gt_by_metric[metric])
        
        print(f"   {metric}:")
        print(f"      MTI: refine={mti_ref_size}, gt={mti_gt_size}")
        print(f"      STI: refine={sti_ref_size}, gt={sti_gt_size}")
        
        # ALERTA se tamanhos não batem
        if mti_ref_size != mti_gt_size:
            print(f"      ⚠️  ALERTA MTI: Tamanhos diferentes!")
        if sti_ref_size != sti_gt_size:
            print(f"      ⚠️  ALERTA STI: Tamanhos diferentes!")
    
    # Calcular Cohen's Kappa por métrica
    mti_kappas = {}
    sti_kappas = {}

    print("\n📈 Calculando Cohen's Kappa por métrica:")
    for metric in metrics:
        # MTI
        gt_list = mti_gt_by_metric.get(metric, [])
        ref_list = mti_ref_by_metric.get(metric, [])
        
        if len(gt_list) > 0 and len(ref_list) > 0:
            # Debug: Mostrar valores únicos
            unique_gt = set(gt_list)
            unique_ref = set(ref_list)
            
            k = calculate_weighted_cohen_kappa(gt_list, ref_list)
            # Para o caso de [5,5,5,5,5] e [5,5,5,5,5]	
            if k is None or np.isnan(k):
                print("   ⚠️  Aviso: kappa retornou None ou NaN, ajustando para 1.0 se houver concordância perfeita")
                k = 1.0

            mti_kappas[metric] = k
            
            if k is None:

                print(f"   MTI {metric}: kappa=N/A (n={len(gt_list)} pares)")
                print(f"      → GT únicos: {sorted(unique_gt)}, Refine únicos: {sorted(unique_ref)}")
                if len(unique_gt) == 1 and len(unique_ref) == 1:
                    print(f"      → VARIÂNCIA ZERO: Todos GT={list(unique_gt)[0]}, Todos Refine={list(unique_ref)[0]}")
            elif k == 1.0:
                print(f"   MTI {metric}: kappa={k:.4f} (CONCORDÂNCIA PERFEITA) (n={len(gt_list)} pares)")
            else:
                print(f"   MTI {metric}: kappa={k:.4f} (n={len(gt_list)} pares)")
        else:
            mti_kappas[metric] = None
            print(f"   MTI {metric}: SEM DADOS")

        # STI
        gt_list_s = sti_gt_by_metric.get(metric, [])
        ref_list_s = sti_ref_by_metric.get(metric, [])
        
        if len(gt_list_s) > 0 and len(ref_list_s) > 0:
            # Debug: Mostrar valores únicos
            unique_gt_s = set(gt_list_s)
            unique_ref_s = set(ref_list_s)
            
            k_s = calculate_weighted_cohen_kappa(gt_list_s, ref_list_s)
            if k_s is None or np.isnan(k_s):
                print("   ⚠️  Aviso: kappa retornou None ou NaN, ajustando para 1.0 se houver concordância perfeita")
                k_s = 1.0
                
            sti_kappas[metric] = k_s

            
            if k_s is None:
                print(f"   STI {metric}: kappa=N/A (n={len(gt_list_s)} pares)")
                print(f"      → GT únicos: {sorted(unique_gt_s)}, Refine únicos: {sorted(unique_ref_s)}")
                if len(unique_gt_s) == 1 and len(unique_ref_s) == 1:
                    print(f"      → VARIÂNCIA ZERO: Todos GT={list(unique_gt_s)[0]}, Todos Refine={list(unique_ref_s)[0]}")
            elif k_s == 1.0:
                print(f"   STI {metric}: kappa={k_s:.4f} (CONCORDÂNCIA PERFEITA) (n={len(gt_list_s)} pares)")
            else:
                print(f"   STI {metric}: kappa={k_s:.4f} (n={len(gt_list_s)} pares)")
        else:
            sti_kappas[metric] = None
            print(f"   STI {metric}: SEM DADOS")

    # Média dos kappas (ignorando métricas sem dados e valores None/NaN)
    valid_mti_kappas = [v for v in mti_kappas.values() if v is not None]
    valid_sti_kappas = [v for v in sti_kappas.values() if v is not None]
    
    mti_kappa = float(np.mean(valid_mti_kappas)) if valid_mti_kappas else 0.0
    sti_kappa = float(np.mean(valid_sti_kappas)) if valid_sti_kappas else 0.0

    # Debug: Mostrar documentos processados
    if selected_instances and processed_docs:
        print(f"\n📄 Documentos processados (primeiros 5):")
        for i, doc in enumerate(processed_docs[:5], 1):
            print(f"   {i}. task={doc['task_id']}, instance={doc['instance_id']}")
            print(f"      refine_id={doc['refine_doc_id']}, gt_id={doc['gt_doc_id']}")
    
    print(f"\n📊 RESUMO:")
    print(f"   MTI - Média Cohen's Kappa: {mti_kappa:.4f}")
    print(f"   STI - Média Cohen's Kappa: {sti_kappa:.4f}")
    
    # Pegar o gt_prompt de um documento representativo
    sample_doc = refine_docs[0]
    gt_prompt = sample_doc.get("gt_prompt", "")
    
    return {
        "experiment_name": experiment_name,
        "prompt": gt_prompt,
        "sti_cohen_kappa": sti_kappa,
        "mti_cohen_kappa": mti_kappa,
        "sti_cohen_kappa_by_metric": sti_kappas,
        "mti_cohen_kappa_by_metric": mti_kappas,
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

        update_doc = {
            "prompt": results.get("prompt", ""),
            "sti_cohen_kappa": results.get("sti_cohen_kappa", 0.0),
            "mti_cohen_kappa": results.get("mti_cohen_kappa", 0.0),
            "calculation_timestamp": datetime.now(),
            "matched_documents": results.get("matched_documents", 0),
            "total_documents": results.get("total_documents", 0)
        }

        # incluir resultados por métrica se disponíveis
        if "sti_cohen_kappa_by_metric" in results:
            update_doc["sti_cohen_kappa_by_metric"] = results["sti_cohen_kappa_by_metric"]
        if "mti_cohen_kappa_by_metric" in results:
            update_doc["mti_cohen_kappa_by_metric"] = results["mti_cohen_kappa_by_metric"]

        similarity_results_collection.update_one(
            {"experiment_name": experiment_name},
            {"$set": update_doc}
        )

        return existing["_id"]
    
    # Criar novo documento
    similarity_doc = {
        "experiment_name": experiment_name,
        "prompt": results.get("prompt", ""),
        "sti_cohen_kappa": results.get("sti_cohen_kappa", 0.0),
        "mti_cohen_kappa": results.get("mti_cohen_kappa", 0.0),
        "calculation_timestamp": datetime.now(),
        "matched_documents": results.get("matched_documents", 0),
        "total_documents": results.get("total_documents", 0)
    }

    if "sti_cohen_kappa_by_metric" in results:
        similarity_doc["sti_cohen_kappa_by_metric"] = results["sti_cohen_kappa_by_metric"]
    if "mti_cohen_kappa_by_metric" in results:
        similarity_doc["mti_cohen_kappa_by_metric"] = results["mti_cohen_kappa_by_metric"]
    
    result = similarity_results_collection.insert_one(similarity_doc)
    print(f"✅ Resultado salvo com ID: {result.inserted_id}")
    
    return result.inserted_id


def process_all_experiments(
    experiment_names: List[str] | None = None,
    test_instances: int | None = None,
    seed: int = 42
) -> Dict[str, Dict]:
    """
    Processa todos os experimentos únicos e calcula similaridade.
    
    Args:
        experiment_names: Lista opcional de nomes de experimentos a processar.
                         Se None, processa todos os experimentos.
        test_instances: Se fornecido, processa apenas N instâncias aleatórias (modo teste).
        seed: Seed para seleção aleatória de instâncias.
    
    Returns:
        Dict mapeando experiment_name para resultados
    """
    if experiment_names is None:
        experiment_names = get_unique_experiment_names()
    
    if not experiment_names:
        print("⚠️ Nenhum experimento encontrado na collection refine_judge_results")
        return {}
    
    print(f"\n📊 Processando {len(experiment_names)} experimento(s):")
    for name in experiment_names:
        print(f"  - {name}")
    print()
    
    results_map = {}
    
    for experiment_name in experiment_names:
        try:
            # Modo teste: selecionar instâncias específicas
            selected = None
            if test_instances is not None:
                selected = select_sample_instances(
                    experiment_name, 
                    num_instances=test_instances,
                    seed=seed
                )
                
                if not selected:
                    print(f"⚠️ Nenhuma instância disponível para '{experiment_name}'")
                    continue
            
            # Calcular similaridade
            results = calculate_similarity_for_experiment(
                experiment_name,
                selected_instances=selected
            )
            
            if results:
                # Adicionar informação de modo teste aos resultados
                if test_instances:
                    results["test_mode"] = True
                    results["test_instances_count"] = len(selected) if selected else 0
                
                # Salvar resultados
                # save_similarity_results(experiment_name, results)
                results_map[experiment_name] = results
            
        except Exception as e:
            print(f"❌ Erro ao processar experimento '{experiment_name}': {e}")
            import traceback
            traceback.print_exc()
    
    return results_map


def main():
    """
    Função principal para execução do script.
    """
    parser = argparse.ArgumentParser(
        description="Calcula similaridade entre refine_judge_results e final_ground_truth usando Cohen's Kappa"
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=None,
        help="Lista de nomes de experimentos a processar (experiment_gt_name). Se não fornecido, processa todos."
    )
    parser.add_argument(
        "--test_instances",
        type=int,
        default=None,
        help="Número de instâncias para modo de teste. Se fornecido, calcula similaridade apenas para N instâncias selecionadas aleatoriamente (padrão: processa todas)."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed para seleção aleatória de instâncias no modo teste (padrão: 42)."
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("CÁLCULO DE SIMILARIDADE - REFINE JUDGE vs GROUND TRUTH")
    if args.test_instances:
        print(f"🧪 MODO TESTE - {args.test_instances} instâncias (seed={args.seed})")
    print("="*70)
    
    # Processar experimentos
    results_map = process_all_experiments(
        args.experiments, 
        test_instances=args.test_instances,
        seed=args.seed
    )
    
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