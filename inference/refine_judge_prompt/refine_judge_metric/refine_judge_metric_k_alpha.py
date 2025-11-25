"""
Script para calcular a similaridade entre avaliações refine_judge_results e final_ground_truth
usando Krippendorff's Alpha e porcentagem de concordância exata.

Este script:
1. Busca documentos de refine_judge_results por experimento
2. Encontra os correspondentes em final_ground_truth (por task_id e instance_id)
3. Calcula Krippendorff's Alpha para STI e MTI separadamente
4. Calcula porcentagem de concordância exata para cada métrica
5. Salva resultados na collection refine_judge_similarity_k_alpha

MODO TESTE:
    Use --test_instances N para processar apenas N instâncias aleatórias.
    - As instâncias selecionadas são salvas em selected_instances/
    - Execuções subsequentes reutilizam as mesmas instâncias para reprodutibilidade
    - Útil para iteração rápida e debugging

Uso:
    # Processar todos os experimentos
    python refine_judge_metric_k_alpha.py
    
    # Processar experimentos específicos
    python refine_judge_metric_k_alpha.py --experiments "exp1" "exp2"
    
    # Modo teste com 5 instâncias
    python refine_judge_metric_k_alpha.py --experiments "exp1" --test_instances 5
    
    # Modo teste com seed customizado
    python refine_judge_metric_k_alpha.py --experiments "exp1" --test_instances 10 --seed 123

Dependências:
    - krippendorff: para cálculo de Krippendorff's Alpha
    - numpy: para operações numéricas
    
Instalação:
    pip install krippendorff numpy
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
    import krippendorff
except ImportError:
    print("⚠️ AVISO: krippendorff não está instalado. Execute: pip install krippendorff")
    raise

load_dotenv()

# Conexão MongoDB
MONGO_URI = os.getenv("MONGODB_URI")
mongo_client = MongoClient(MONGO_URI)
db = mongo_client["experiments_db"]
refine_judge_collection = db["refine_judge_results"]
final_gt_collection = db["final_ground_truth"]
similarity_results_collection = db["refine_judge_similarity_k_alpha"]

# Diretório para armazenar arquivos de instâncias selecionadas
INSTANCES_DIR = Path(__file__).parent / "selected_instances"
INSTANCES_DIR.mkdir(exist_ok=True)


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


def calculate_krippendorff_alpha(y_true: List[int], y_pred: List[int]) -> float | None:
    """
    Calcula Krippendorff's Alpha entre duas listas de scores.
    
    Args:
        y_true: Lista de scores ground truth
        y_pred: Lista de scores preditos
    
    Returns:
        Valor de Krippendorff's Alpha, ou None se não for calculável
    """
    if not y_true or not y_pred or len(y_true) != len(y_pred):
        return None
    
    # VERIFICAÇÃO CRÍTICA: Detectar variância zero ANTES de chamar krippendorff
    unique_true = len(set(y_true))
    unique_pred = len(set(y_pred))
    
    # Se ambas as listas têm apenas 1 valor único
    if unique_true == 1 and unique_pred == 1:
        # Verificar se os valores únicos são IGUAIS
        if y_true[0] == y_pred[0]:
            return 1.0  # Concordância perfeita: todos iguais e concordando
        else:
            return None  # Discordância total mas sem variação para calcular
    
    # Se apenas uma das listas tem variância zero → não calculável
    if unique_true == 1 or unique_pred == 1:
        return None
    
    try:
        # Krippendorff's Alpha espera matriz (n_raters x n_items)
        # Onde cada linha é um avaliador (rater) e cada coluna é um item
        reliability_data = np.array([y_true, y_pred])
        
        # Calcular Krippendorff's Alpha com métrica ordinal
        alpha = krippendorff.alpha(reliability_data, level_of_measurement='ordinal')
        
        # Verificar se o resultado é válido
        if np.isnan(alpha) or np.isinf(alpha):
            return None
        
        return float(alpha)
    except Exception as e:
        print(f"⚠️ Erro ao calcular Krippendorff's Alpha: {e}")
        return None


def calculate_exact_match_percentage(y_true: List[int], y_pred: List[int]) -> float | None:
    """
    Calcula a porcentagem de concordância exata entre duas listas de scores.
    
    Args:
        y_true: Lista de scores ground truth
        y_pred: Lista de scores preditos
    
    Returns:
        Porcentagem de concordância (0-100), ou None se não for calculável
    """
    if not y_true or not y_pred or len(y_true) != len(y_pred):
        return None
    
    matches = sum(1 for a, b in zip(y_true, y_pred) if a == b)
    percentage = (matches / len(y_true)) * 100
    
    return float(percentage)


def calculate_similarity_for_experiment(
    experiment_name: str, 
    selected_instances: List[Dict] | None = None
) -> Dict | None:
    """
    Calcula Krippendorff's Alpha e porcentagem de concordância exata 
    para todos os documentos de um experimento.
    
    Args:
        experiment_name: Nome do experimento (experiment_gt_name)
        selected_instances: Lista opcional de instâncias específicas a processar.
                           Se None, processa todas as instâncias do experimento.
                           Formato: [{"task_id": "030", "instance_id": "030_1559"}, ...]
    
    Returns:
        Dict com resultados agregados ou None se houver erro
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
    
    # Estruturas separadas por métrica
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
        for metric in metrics:
            # Score do refine_judge
            refine_mti_metric = refine_doc.get("gt_MTI_answer", {}).get(metric, {})
            refine_mti_score = refine_mti_metric.get("score", None)
            
            # Median score do ground truth
            gt_mti_metric = gt_doc.get("gt_MTI_answer", {}).get(metric, {})
            gt_mti_median = gt_mti_metric.get("median_score", None)
            
            # Só adiciona se ambos existirem
            if refine_mti_score is not None and gt_mti_median is not None:
                mti_ref_by_metric[metric].append(refine_mti_score)
                mti_gt_by_metric[metric].append(gt_mti_median)
            
            # Score do refine_judge para STI
            refine_sti_metric = refine_doc.get("gt_STI_answer", {}).get(metric, {})
            refine_sti_score = refine_sti_metric.get("score", None)
            
            # Median score do ground truth para STI
            gt_sti_metric = gt_doc.get("gt_STI_answer", {}).get(metric, {})
            gt_sti_median = gt_sti_metric.get("median_score", None)
            
            # Só adiciona se ambos existirem
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
    
    # Calcular Krippendorff's Alpha e porcentagem de concordância por métrica
    mti_k_alphas = {}
    sti_k_alphas = {}
    mti_exact_match_percentages = {}
    sti_exact_match_percentages = {}

    print("\n📈 Calculando Krippendorff's Alpha e Concordância Exata por métrica:")
    
    for metric in metrics:
        # ========== MTI ==========
        gt_list = mti_gt_by_metric.get(metric, [])
        ref_list = mti_ref_by_metric.get(metric, [])
        
        if len(gt_list) > 0 and len(ref_list) > 0:
            # Calcular Krippendorff's Alpha
            alpha = calculate_krippendorff_alpha(gt_list, ref_list)
            
            # Ajuste para concordância perfeita
            if alpha is None or np.isnan(alpha):
                unique_gt = set(gt_list)
                unique_ref = set(ref_list)
                if len(unique_gt) == 1 and len(unique_ref) == 1 and list(unique_gt)[0] == list(unique_ref)[0]:
                    alpha = 1.0
            
            mti_k_alphas[metric] = alpha
            
            # Calcular porcentagem de concordância exata
            exact_match = calculate_exact_match_percentage(gt_list, ref_list)
            mti_exact_match_percentages[metric] = exact_match
            
            # Exibir resultados
            alpha_str = f"{alpha:.4f}" if alpha is not None else "N/A"
            exact_str = f"{exact_match:.2f}%" if exact_match is not None else "N/A"
            print(f"   MTI {metric}:")
            print(f"      α={alpha_str}, Concordância Exata={exact_str} (n={len(gt_list)} pares)")
        else:
            mti_k_alphas[metric] = None
            mti_exact_match_percentages[metric] = None
            print(f"   MTI {metric}: SEM DADOS")

        # ========== STI ==========
        gt_list_s = sti_gt_by_metric.get(metric, [])
        ref_list_s = sti_ref_by_metric.get(metric, [])
        
        if len(gt_list_s) > 0 and len(ref_list_s) > 0:
            # Calcular Krippendorff's Alpha
            alpha_s = calculate_krippendorff_alpha(gt_list_s, ref_list_s)
            
            # Ajuste para concordância perfeita
            if alpha_s is None or np.isnan(alpha_s):
                unique_gt_s = set(gt_list_s)
                unique_ref_s = set(ref_list_s)
                if len(unique_gt_s) == 1 and len(unique_ref_s) == 1 and list(unique_gt_s)[0] == list(unique_ref_s)[0]:
                    alpha_s = 1.0
            
            sti_k_alphas[metric] = alpha_s
            
            # Calcular porcentagem de concordância exata
            exact_match_s = calculate_exact_match_percentage(gt_list_s, ref_list_s)
            sti_exact_match_percentages[metric] = exact_match_s
            
            # Exibir resultados
            alpha_str_s = f"{alpha_s:.4f}" if alpha_s is not None else "N/A"
            exact_str_s = f"{exact_match_s:.2f}%" if exact_match_s is not None else "N/A"
            print(f"   STI {metric}:")
            print(f"      α={alpha_str_s}, Concordância Exata={exact_str_s} (n={len(gt_list_s)} pares)")
        else:
            sti_k_alphas[metric] = None
            sti_exact_match_percentages[metric] = None
            print(f"   STI {metric}: SEM DADOS")

    # Calcular médias (ignorando métricas sem dados e valores None/NaN)
    valid_mti_alphas = [v for v in mti_k_alphas.values() if v is not None]
    valid_sti_alphas = [v for v in sti_k_alphas.values() if v is not None]
    valid_mti_exact = [v for v in mti_exact_match_percentages.values() if v is not None]
    valid_sti_exact = [v for v in sti_exact_match_percentages.values() if v is not None]
    
    mti_k_alpha_mean = float(np.mean(valid_mti_alphas)) if valid_mti_alphas else 0.0
    sti_k_alpha_mean = float(np.mean(valid_sti_alphas)) if valid_sti_alphas else 0.0
    mti_exact_match_mean = float(np.mean(valid_mti_exact)) if valid_mti_exact else 0.0
    sti_exact_match_mean = float(np.mean(valid_sti_exact)) if valid_sti_exact else 0.0

    # Debug: Mostrar documentos processados
    if selected_instances and processed_docs:
        print(f"\n📄 Documentos processados (primeiros 5):")
        for i, doc in enumerate(processed_docs[:5], 1):
            print(f"   {i}. task={doc['task_id']}, instance={doc['instance_id']}")
            print(f"      refine_id={doc['refine_doc_id']}, gt_id={doc['gt_doc_id']}")
    
    print(f"\n📊 RESUMO:")
    print(f"   MTI - Média Krippendorff's Alpha: {mti_k_alpha_mean:.4f}")
    print(f"   STI - Média Krippendorff's Alpha: {sti_k_alpha_mean:.4f}")
    print(f"   MTI - Média Concordância Exata: {mti_exact_match_mean:.2f}%")
    print(f"   STI - Média Concordância Exata: {sti_exact_match_mean:.2f}%")
    
    # Pegar o gt_prompt de um documento representativo
    sample_doc = refine_docs[0]
    gt_prompt = sample_doc.get("gt_prompt", "")
    
    return {
        "experiment_name": experiment_name,
        "prompt": gt_prompt,
        "sti_k_alpha": sti_k_alpha_mean,
        "mti_k_alpha": mti_k_alpha_mean,
        "sti_k_alpha_by_metric": sti_k_alphas,
        "mti_k_alpha_by_metric": mti_k_alphas,
        "sti_exact_match_percentage": sti_exact_match_mean,
        "mti_exact_match_percentage": mti_exact_match_mean,
        "sti_exact_match_percentage_by_metric": sti_exact_match_percentages,
        "mti_exact_match_percentage_by_metric": mti_exact_match_percentages,
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
    Salva resultados de similaridade na collection refine_judge_similarity_k_alpha.
    
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
            "sti_k_alpha": results.get("sti_k_alpha", 0.0),
            "mti_k_alpha": results.get("mti_k_alpha", 0.0),
            "sti_k_alpha_by_metric": results.get("sti_k_alpha_by_metric", {}),
            "mti_k_alpha_by_metric": results.get("mti_k_alpha_by_metric", {}),
            "sti_exact_match_percentage": results.get("sti_exact_match_percentage", 0.0),
            "mti_exact_match_percentage": results.get("mti_exact_match_percentage", 0.0),
            "sti_exact_match_percentage_by_metric": results.get("sti_exact_match_percentage_by_metric", {}),
            "mti_exact_match_percentage_by_metric": results.get("mti_exact_match_percentage_by_metric", {}),
            "calculation_timestamp": datetime.now(),
            "matched_documents": results.get("matched_documents", 0),
            "total_documents": results.get("total_documents", 0)
        }

        similarity_results_collection.update_one(
            {"experiment_name": experiment_name},
            {"$set": update_doc}
        )

        return existing["_id"]
    
    # Criar novo documento
    similarity_doc = {
        "experiment_name": experiment_name,
        "prompt": results.get("prompt", ""),
        "sti_k_alpha": results.get("sti_k_alpha", 0.0),
        "mti_k_alpha": results.get("mti_k_alpha", 0.0),
        "sti_k_alpha_by_metric": results.get("sti_k_alpha_by_metric", {}),
        "mti_k_alpha_by_metric": results.get("mti_k_alpha_by_metric", {}),
        "sti_exact_match_percentage": results.get("sti_exact_match_percentage", 0.0),
        "mti_exact_match_percentage": results.get("mti_exact_match_percentage", 0.0),
        "sti_exact_match_percentage_by_metric": results.get("sti_exact_match_percentage_by_metric", {}),
        "mti_exact_match_percentage_by_metric": results.get("mti_exact_match_percentage_by_metric", {}),
        "calculation_timestamp": datetime.now(),
        "matched_documents": results.get("matched_documents", 0),
        "total_documents": results.get("total_documents", 0)
    }
    
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
                save_similarity_results(experiment_name, results)
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
        description="Calcula similaridade entre refine_judge_results e final_ground_truth usando Krippendorff's Alpha e concordância exata"
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
    print("CÁLCULO DE SIMILARIDADE - KRIPPENDORFF'S ALPHA")
    print("REFINE JUDGE vs GROUND TRUTH")
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
            print(f"   MTI Krippendorff's Alpha: {results['mti_k_alpha']:.4f}")
            print(f"   STI Krippendorff's Alpha: {results['sti_k_alpha']:.4f}")
            print(f"   MTI Concordância Exata: {results['mti_exact_match_percentage']:.2f}%")
            print(f"   STI Concordância Exata: {results['sti_exact_match_percentage']:.2f}%")
            print(f"   Documentos pareados: {results['matched_documents']}/{results['total_documents']}")
        
        print("\n" + "="*70)
        print(f"Collection: refine_judge_similarity_k_alpha")
        print(f"Total de resultados: {similarity_results_collection.count_documents({})}")
        print("="*70 + "\n")
    else:
        print("\n⚠️ Nenhum resultado foi gerado")


if __name__ == "__main__":
    main()
