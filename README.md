# Human Preference Criteria Evaluation for Multi-Task Inference

## Summary

This repository contains the code, data, and results used in a scientific experiment that aimed to evaluate the effectiveness of two inference techniques for language models (LLMs): Single-Task Inference (STI) and Multi-Task Inference (MTI). We compared responses produced by both approaches according to human-preference criteria: coherence, specificity, comprehensibility, informativeness, and relevance, using a 1–5 Likert scale.

## Context and motivation

- Single-Task Inference (STI): the LLM is invoked for individual sub-tasks (a sequential approach).
- Multi-Task Inference (MTI): the LLM receives composite instructions that contain multiple sub-tasks in a single call.

The goal was to investigate whether and how STI and MTI strategies influence response quality across different language models.

## Relation to prior work

This project extends the "MTI-Bench" repository (https://github.com/guijinSON/MTI-Bench), which was used in the paper "Multi-Task Inference: Can Large Language Models Follow Multiple Instructions at Once?" (https://aclanthology.org/2024.acl-long.304/). We adapted the "Free-Form Generation" dataset to build a ground truth (GT) and to calibrate the LLM Judge prompt used in the final evaluation.

## Repository structure

- `inference/` — scripts to produce inferences (STI, MTI, ground-truth generation, prompt refinement).
- `evaluate/` — evaluation utilities.
- `dataset/` — base datasets from [MTI-Bench](https://github.com/guijinSON/MTI-Bench) and the adapted Free-Form Generation GT that we proposed(`gt_human_preferences_metrics_dataset.json`).
- `ground_truth/` — scripts for GT generation and validation (`gt_human_preferences_metrics_dataset.json`).
- `Prompts/` — prompt templates used to create the GT and the judge prompt experiments (e.g., `refine_judge_prompt_templates.py`, `llm_judge_templates.py`).
- `final_metrics/` — scripts and notebooks for summarizing and exporting results.
- `main.py`, `src.py` — utility scripts / runners.
- `pyproject.toml` — project dependencies.


## Data and ground truth

- No human-labeled ground truth existed initially; we generated an automated GT using an LLM evaluator that annotated responses according to the five metrics (Likert 1–5). The GT was used to calibrate the LLM Judge prompt for the final evaluation.
- Derived and raw files are stored under `dataset/`. See `ground_truth/` for GT generation and validation scripts.

## Evaluation (LLM Judge)

- The LLM Judge compares model outputs (for example, STI vs MTI) and assigns Likert scores (1–5) for each metric:
  - Coherence
  - Specificity
  - Comprehensibility
  - Informativeness
  - Relevance
- The judge prompt was calibrated using the generated ground truth.

## Database

- This project uses MongoDB to store instances, model outputs, and annotations. Configure `MONGO_URI` in the environment before running scripts that read/write the database.

## How to run

Before running any script, copy the example environment file and fill in the correct values (e.g., API keys, MongoDB URI):

```bash
# copy the example env file to .env
cp .env.example .env
```

Replace `<...>` with appropriate values (e.g., `MODEL_NAME`, `MONGO_URI`, `PATH_TO_INSTANCES`). Examples below assume a bash shell.

1) Environment preparation

```bash
# We use the 'uv' package manager (https://docs.astral.sh/uv/).
# 'uv' will resolve and install dependencies declared in 'pyproject.toml'
# automatically when you run a script with 'uv run'.
# Therefore, to install dependencies simply execute any Python file using
# 'uv run python <script>' — there is no need to run 'uv install'.

# Example: running the STI inference help will cause 'uv' to ensure deps are available
uv run inference/STI_Inference_S3T.py
```

2) Single-Task Inference (STI)

```bash
# Example — adjust flags according to the actual scripts
uv run inference/STI_Inference_S3T.py --model <MODEL_NAME> --instances <PATH_TO_INSTANCES> --out_dir data/results/sti --mongo_uri "<MONGO_URI>"
```

3) Multi-Task Inference (MTI)

```bash
uv run inference/MTI_Inference_S3T.py --model <MODEL_NAME> --instances <PATH_TO_INSTANCES> --out_dir data/results/mti --mongo_uri "<MONGO_URI>"
```

4) Generate ground truth (GT) with an LLM

```bash
uv run ground_truth/generate_gt.py --model <LLM_MODEL_FOR_GT> --instances <PATH_TO_INSTANCES> --out_file data/ground_truth/gt_llm.json --mongo_uri "<MONGO_URI>"
```

5) Evaluate responses with the LLM Judge (final pipeline)

```bash
uv run inference/final_pipeline_llm_judge/llm_judge.py --sti_dir data/results/sti --mti_dir data/results/mti --gt data/ground_truth/gt_llm.json --model <LLM_JUDGE_MODEL> --out_dir data/results/final --mongo_uri "<MONGO_URI>"
```

6) Produce final metrics and reports

```bash
uv run final_metrics/final_metrics.py --input_dir data/results/final --out_dir final_metrics/exports
# or open the notebooks in final_metrics/ for interactive analysis
```
---