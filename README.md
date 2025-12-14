# Avaliação de Critérios de Preferência Humana para a Multi-Task Inference

## Resumo

Este repositório contém o código, dados e resultados usados em um experimento científico cujo objetivo foi avaliar a eficiência de duas técnicas de inferência para modelos de linguagem (LLMs): Single-Task Inference (STI) e Multi-Task Inference (MTI). Comparamos respostas geradas por ambas as abordagens segundo critérios de preferência humana: coerência, especificidade, compreensibilidade, informatividade e relevância, usando uma escala Likert de 1 a 5.

## Contexto e motivação

- Single-Task Inference (STI): o LLM é acionado para cada sub-tarefa individualmente (abordagem sequencial).
- Multi-Task Inference (MTI): o LLM recebe instruções compostas que contêm múltiplas sub-tarefas em uma única chamada.

O objetivo foi investigar se e como as estratégias STI e MTI influenciam a qualidade das respostas em diferentes modelos de linguagem.

## Relação com trabalho anterior

Este trabalho é uma extensão do repositório [MTI-Bench](https://github.com/guijinSON/MTI-Bench) que é a implemetação utilizada no trabalho [Multi-Task Inference: Can Large Language Models Follow Multiple Instructions at Once?](https://aclanthology.org/2024.acl-long.304/). Adaptamos o dataset "Free-Form Generation" para construir um ground truth (GT) e calibrar o prompt do juiz (LLM Judge) usado na avaliação final.

<!-- ## Contribuições principais

1. Pipelines para geração de respostas usando STI e MTI com múltiplos LLMs.
2. Criação de um ground truth (GT) gerado por um LLM avaliador para calibragem do juiz final.
3. Implementação de um LLM Judge que atribui notas (Likert 1–5) para as métricas: coerência, especificidade, compreensibilidade, informatividade e relevância.
4. Armazenamento dos resultados em MongoDB para facilitar análise e reprodutibilidade. -->

## Estrutura do repositório

- `inference/` — scripts para gerar inferências (STI, MTI, geração do ground truth, refinamento do prompt de avaliação).
- `evaluate/` — utilitários de avaliação.
- `dataset/` — datasets base do [MTI-Bench](https://github.com/guijinSON/MTI-Bench) e o ground truth proposto por nós do Free-Form Generation que se encontra no arquivo .
- `ground_truth/` — scripts para geração e validação do ground truth com LLM `gt_human_preferences_metrics_dataset.json`.
- `Prompts/` — templates de prompt para criação do ground truth, sequencia de prompts testados para avaliação final(`refine_judge_prompt_templates.py`) e prompt avaliação escolhido após experimentos(`llm_judge_templates.py`).
- `final_metrics/` — scripts e notebooks para sumarização e exportação dos resultados.
- `main.py`, `src.py` — scripts utilitários / runners.
- `pyproject.toml` — dependências do projeto.


## Dados e ground truth

- Não havia um ground truth humano disponível; geramos um GT automatizado usando um LLM avaliador que anotou respostas segundo as mesmas cinco métricas (Likert 1–5). O GT foi utilizado para calibrar o prompt do LLM Judge na avaliação final.
- Arquivos derivados e brutos estão em `dataset/`. Veja também `ground_truth/` para os scripts de geração e validação.

## Avaliação (LLM Judge)

- O LLM Judge compara respostas (por exemplo, STI vs MTI) e atribui notas Likert (1–5) para cada uma das métricas:
  - Coerência
  - Especificidade
  - Compreensibilidade
  - Informatividade
  - Relevância
- O prompt do juiz foi calibrado usando o ground truth gerado previamente.

## Banco de dados

- Para armazenar instâncias, respostas e anotações, este trabalho utiliza MongoDB. Configure `MONGO_URI` no ambiente antes de rodar os scripts que gravam/consultam o banco.


## Como executar

Antes de rodar qualquer script, faça uma cópia do arquivo de exemplo de variáveis de ambiente e preencha os valores corretos (por exemplo: chaves de API, URI do MongoDB). Você pode usar os comandos abaixo para copiar o arquivo `.env`:

```bash
# copie o arquivo de exemplo para .env
cp .env.example .env
```

Substitua `<...>` pelos valores adequados (ex.: `MODEL_NAME`, `MONGO_URI`, `PATH_TO_INSTANCES`). Os exemplos assumem o uso de um shell bash.

1) Preparação do ambiente

```bash
# Usamos o gerenciador 'uv' (https://docs.astral.sh/uv/).
# O 'uv' resolve e instala automaticamente as dependências declaradas em
# 'pyproject.toml' ao executar um script com 'uv run'.
# Portanto, para instalar as dependências basta executar qualquer arquivo Python
# via 'uv run python <script>' — NÃO é necessário executar 'uv install'.

# Exemplo: ao rodar o script de inferência abaixo, o 'uv' instalará as deps
# declaradas em 'pyproject.toml' automaticamente:
uv run python inference/STI_Inference_S3T.py 
```

2) Inferência Single-Task Inference (STI)

```bash
# Exemplo — ajuste flags conforme os scripts reais
uv run python inference/STI_Inference_S3T.py --model <MODEL_NAME> --instances <PATH_TO_INSTANCES> --out_dir data/results/sti --mongo_uri "<MONGO_URI>"
```

3) Inferência Multi-Task Inference (MTI)

```bash
uv run python inference/MTI_Inference_S3T.py --model <MODEL_NAME> --instances <PATH_TO_INSTANCES> --out_dir data/results/mti --mongo_uri "<MONGO_URI>"
```

4) Gerar ground truth (GT) com um LLM

```bash
uv run python ground_truth/generate_gt.py --model <LLM_MODEL_FOR_GT> --instances <PATH_TO_INSTANCES> --out_file data/ground_truth/gt_llm.json --mongo_uri "<MONGO_URI>"
```

5) Avaliar respostas com o LLM Judge (pipeline final)

```bash
uv run python inference/final_pipeline_llm_judge/llm_judge.py --sti_dir data/results/sti --mti_dir data/results/mti --gt data/ground_truth/gt_llm.json --model <LLM_JUDGE_MODEL> --out_dir data/results/final --mongo_uri "<MONGO_URI>"
```

6) Gerar métricas finais e relatórios

```bash
uv run python final_metrics/final_metrics.py --input_dir data/results/final --out_dir final_metrics/exports
# ou abra os notebooks em final_metrics/ para análises interativas
```