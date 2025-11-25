#!/usr/bin/env python
"""
Script de exemplo para demonstrar o uso do modo teste do refine_judge_metric.py

Este script mostra como:
1. Executar o modo teste com 5 instâncias
2. Verificar o arquivo de instâncias selecionadas criado
3. Executar novamente para demonstrar reprodutibilidade
"""

import subprocess
import json
from pathlib import Path

# Configuração
EXPERIMENT_NAME = "refine_judge_gpt-4o-mini-2024-07-18_V4_CORRETO"
NUM_INSTANCES = 5
SEED = 42

# Paths
SCRIPT_PATH = Path(__file__).parent / "refine_judge_metric.py"
INSTANCES_DIR = Path(__file__).parent / "selected_instances"
INSTANCE_FILE = INSTANCES_DIR / f"{EXPERIMENT_NAME}_n{NUM_INSTANCES}_instances.json"

print("=" * 80)
print("EXEMPLO DE USO DO MODO TESTE - refine_judge_metric.py")
print("=" * 80)

# Passo 1: Limpar arquivo anterior se existir
if INSTANCE_FILE.exists():
    print(f"\n🗑️ Removendo arquivo de instâncias anterior...")
    INSTANCE_FILE.unlink()
    print(f"   Arquivo removido: {INSTANCE_FILE.name}")

# Passo 2: Primeira execução - cria seleção
print(f"\n📝 PASSO 1: Primeira execução (cria seleção de instâncias)")
print(f"   Comando: python {SCRIPT_PATH} --experiments \"{EXPERIMENT_NAME}\" --test_instances {NUM_INSTANCES} --seed {SEED}")
print()

cmd = [
    "python",
    str(SCRIPT_PATH),
    "--experiments", EXPERIMENT_NAME,
    "--test_instances", str(NUM_INSTANCES),
    "--seed", str(SEED)
]

result1 = subprocess.run(cmd, capture_output=True, text=True)
print(result1.stdout)

if result1.returncode != 0:
    print("❌ Erro na execução:")
    print(result1.stderr)
    exit(1)

# Passo 3: Verificar arquivo criado
print("\n" + "=" * 80)
print("📂 PASSO 2: Verificando arquivo de instâncias criado")
print("=" * 80)

if INSTANCE_FILE.exists():
    print(f"✅ Arquivo criado: {INSTANCE_FILE.name}")
    
    with open(INSTANCE_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"\n📊 Conteúdo do arquivo:")
    print(f"   Experimento: {data['experiment_name']}")
    print(f"   Número de instâncias: {data['num_instances']}")
    print(f"   Seed: {data['seed']}")
    print(f"   Timestamp: {data['selection_timestamp']}")
    print(f"\n   Instâncias selecionadas:")
    for i, inst in enumerate(data['instances'], 1):
        print(f"      {i}. task_id={inst['task_id']}, instance_id={inst['instance_id']}")
else:
    print(f"⚠️ Arquivo não foi criado: {INSTANCE_FILE}")

# Passo 4: Segunda execução - reutiliza seleção
print("\n" + "=" * 80)
print("🔁 PASSO 3: Segunda execução (reutiliza instâncias selecionadas)")
print("=" * 80)
print(f"   Comando: python {SCRIPT_PATH} --experiments \"{EXPERIMENT_NAME}\" --test_instances {NUM_INSTANCES}")
print()

result2 = subprocess.run(cmd, capture_output=True, text=True)
print(result2.stdout)

# Verificar se usou mesmas instâncias
if "Usando" in result2.stdout and "instâncias pré-selecionadas" in result2.stdout:
    print("\n✅ SUCESSO: Segunda execução reutilizou as mesmas instâncias (reprodutibilidade confirmada)")
else:
    print("\n⚠️ AVISO: Não foi possível confirmar reprodutibilidade")

print("\n" + "=" * 80)
print("🎓 CONCLUSÃO")
print("=" * 80)
print("""
O modo teste permite:
1. ✅ Selecionar N instâncias aleatoriamente (primeira execução)
2. ✅ Salvar seleção em arquivo JSON
3. ✅ Reutilizar mesmas instâncias em execuções subsequentes
4. ✅ Garantir reprodutibilidade dos testes

Para forçar nova seleção, delete o arquivo:
    rm {}

Para testar com quantidade diferente:
    python {} --experiments "{}" --test_instances 10
""".format(INSTANCE_FILE, SCRIPT_PATH, EXPERIMENT_NAME))

print("=" * 80)
