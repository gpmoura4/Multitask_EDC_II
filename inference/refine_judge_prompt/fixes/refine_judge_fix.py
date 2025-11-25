"""
Script para corrigir valores zerados nos campos gt_MTI_answer e gt_STI_answer 
da collection refine_judge_results.

Este script:
1. Seleciona registros com experiment_gt_name específico
2. Extrai scores e explanations corretos de gt_raw_response
3. Determina o mapeamento correto usando position_answer_to_llm
4. Atualiza gt_MTI_answer e gt_STI_answer com valores corretos
"""

import os
import re
from pymongo import MongoClient
from bson import ObjectId
from typing import Dict, Tuple
from dotenv import load_dotenv

load_dotenv()

# Conexão MongoDB
MONGO_URI = os.getenv("MONGODB_URI")
mongo_client = MongoClient(MONGO_URI)
db = mongo_client["experiments_db"]
refine_judge_collection = db["refine_judge_results"]


def parse_gt_raw_response(raw_response: str) -> Tuple[Dict, Dict]:
    """
    Extrai scores e explanations de gt_raw_response.
    
    Args:
        raw_response: String contendo a resposta bruta do avaliador
    
    Returns:
        Tupla (scores_dict, explanations_dict) onde cada dict tem chaves 'a' e 'b'
    """
    scores = {"a": {}, "b": {}}
    explanations = {"a": {}, "b": {}}
    
    # Tentar extrair scores_a
    score_a_match = re.search(r'scores_a\s*=\s*(\{[^}]+\})', raw_response)
    if score_a_match:
        try:
            scores["a"] = eval(score_a_match.group(1))
        except Exception as e:
            print(f"⚠️ Erro ao parsear scores_a: {e}")
    
    # Tentar extrair scores_b (pode estar como "b_scores" também)
    score_b_match = re.search(r'(scores_b|b_scores)\s*=\s*(\{[^}]+\})', raw_response)
    if score_b_match:
        try:
            scores["b"] = eval(score_b_match.group(2))
        except Exception as e:
            print(f"⚠️ Erro ao parsear scores_b: {e}")
    
    # Tentar extrair explanations_a
    expl_a_match = re.search(r'explanations_a\s*=\s*\{([^}]+)\}', raw_response, re.DOTALL)
    if expl_a_match:
        try:
            explanations["a"] = eval("{" + expl_a_match.group(1) + "}")
        except Exception as e:
            print(f"⚠️ Erro ao parsear explanations_a: {e}")
    
    # Tentar extrair explanations_b (pode estar como "b_explanations" também)
    expl_b_match = re.search(r'(explanations_b|b_explanations)\s*=\s*\{([^}]+)\}', raw_response, re.DOTALL)
    if expl_b_match:
        try:
            explanations["b"] = eval("{" + expl_b_match.group(2) + "}")
        except Exception as e:
            print(f"⚠️ Erro ao parsear explanations_b: {e}")
    
    return scores, explanations


def normalize_metric_name(name: str) -> str:
    """
    Normaliza nomes de métricas para lowercase (exceto Understandability).
    
    Args:
        name: Nome da métrica
    
    Returns:
        Nome normalizado
    """
    # Mapeamento de possíveis variações
    metric_map = {
        "Coherence": "coherence",
        "coherence": "coherence",
        "Specificity": "specificity",
        "specificity": "specificity",
        "Specific": "specificity",  # Variação encontrada no exemplo
        "Informativeness": "informativeness",
        "informativeness": "informativeness",
        "Relevance": "relevance",
        "relevance": "relevance",
        "Understandability": "Understandability",  # Mantém uppercase U
        "understandability": "Understandability"
    }
    
    return metric_map.get(name, name.lower())


def build_answer_structure(scores: Dict, explanations: Dict) -> Dict:
    """
    Constrói a estrutura de answer (gt_MTI_answer ou gt_STI_answer).
    
    Args:
        scores: Dict com scores das métricas
        explanations: Dict com explanations das métricas
    
    Returns:
        Dict com estrutura completa de answer
    """
    answer_structure = {}
    
    # Processar todas as métricas dos scores
    for metric_name, score_value in scores.items():
        normalized_name = normalize_metric_name(metric_name)
        
        answer_structure[normalized_name] = {
            "score": score_value,
            "explanation": explanations.get(metric_name, "")
        }
    
    return answer_structure


def fix_document(doc: Dict) -> Dict | None:
    """
    Corrige um documento extraindo valores corretos de gt_raw_response.
    
    Args:
        doc: Documento da collection refine_judge_results
    
    Returns:
        Dict com campos atualizados ou None se não houver correção necessária
    """
    # Verificar se há gt_raw_response
    if "gt_raw_response" not in doc or not doc["gt_raw_response"]:
        return None
    
    # Parsear gt_raw_response
    scores, explanations = parse_gt_raw_response(doc["gt_raw_response"])
    
    if not scores["a"] and not scores["b"]:
        print(f"⚠️ Não foi possível extrair scores do documento {doc.get('_id')}")
        return None
    
    # Determinar posições MTI e STI
    position_mti = doc.get("llm_answer_MTI", {}).get("position_answer_to_llm", "A")
    position_sti = doc.get("llm_answer_STI", {}).get("position_answer_to_llm", "B")
    
    # Mapear scores e explanations para MTI e STI
    position_map = {
        "A": "a",
        "B": "b"
    }
    
    mti_key = position_map.get(position_mti, "a")
    sti_key = position_map.get(position_sti, "b")
    
    # Construir estruturas corretas
    gt_mti_answer = build_answer_structure(scores[mti_key], explanations[mti_key])
    gt_sti_answer = build_answer_structure(scores[sti_key], explanations[sti_key])
    
    return {
        "gt_MTI_answer": gt_mti_answer,
        "gt_STI_answer": gt_sti_answer
    }


def fix_refine_judge_results(experiment_name: str = "refine_judge_gpt-4o-mini-2024-07-18_V1") -> int:
    """
    Corrige todos os documentos de um experimento específico.
    
    Args:
        experiment_name: Nome do experimento a corrigir
    
    Returns:
        Número de documentos corrigidos
    """
    print(f"\n🚀 Iniciando correção de documentos do experimento: {experiment_name}\n")
    
    # Buscar documentos do experimento
    documents = list(refine_judge_collection.find({
        "experiment_gt_name": experiment_name
    }))
    
    if not documents:
        print(f"⚠️ Nenhum documento encontrado para o experimento '{experiment_name}'")
        return 0
    
    print(f"📊 Encontrados {len(documents)} documentos para processar\n")
    
    fixed_count = 0
    error_count = 0
    
    for doc in documents:
        instance_id = doc.get("instance_id", "unknown")
        task_id = doc.get("task_id", "unknown")
        
        try:
            # Tentar corrigir documento
            updates = fix_document(doc)
            
            if updates:
                # Atualizar no MongoDB
                result = refine_judge_collection.update_one(
                    {"_id": doc["_id"]},
                    {"$set": updates}
                )
                
                if result.modified_count > 0:
                    fixed_count += 1
                    print(f"  ✅ {instance_id} (task {task_id})")
                else:
                    print(f"  ℹ️ {instance_id} (task {task_id}) - Sem alterações necessárias")
            else:
                print(f"  ⚠️ {instance_id} (task {task_id}) - Não foi possível corrigir")
                error_count += 1
                
        except Exception as e:
            print(f"  ❌ {instance_id} (task {task_id}) - Erro: {e}")
            error_count += 1
    
    return fixed_count


def main():
    """
    Função principal para execução do script.
    """
    print("\n" + "="*70)
    print("CORREÇÃO DE REFINE JUDGE RESULTS")
    print("="*70)
    
    # Experimento a ser corrigido
    experiment_name = "refine_judge_gpt-4o-mini-2024-07-18_V1"
    
    # Executar correção
    fixed_count = fix_refine_judge_results(experiment_name)
    
    print("\n" + "="*70)
    print("RESUMO:")
    print(f"  - Documentos corrigidos: {fixed_count}")
    print(f"  - Experimento: {experiment_name}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()