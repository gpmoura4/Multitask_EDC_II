

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