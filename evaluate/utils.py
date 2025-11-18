import os
import time
from typing import List, Tuple, Optional

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from groq import Groq
from openai import OpenAI  # nova lib oficial OpenAI 1.x


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


class RemoteChatModel:
    """
    Wrapper simples para modelos remotos de chat (OpenAI ou Groq).
    Mantém compatibilidade com a assinatura (model, tokenizer=None).
    """
    def __init__(self, client, model_name: str, provider: str):
        self.client = client          # OpenAI ou Groq
        self.model_name = model_name  # nome do modelo (ex: "gpt-4o", "llama-3.1-70b-versatile")
        self.provider = provider      # "openai" ou "groq"


def generate_completions(
    model,
    tokenizer,
    prompts: List[str],
    batch_size: int = 1,
    stop_id_sequences=None,   # mantido por compatibilidade, ainda não usado
    add_special_tokens: bool = True,  # idem
    disable_tqdm: bool = False,
    max_new_tokens: int = 512,
    min_new_tokens: int = 1,
    do_sample: bool = False,
    temperature: float = 0.7,
    top_p: float = 1.0
) -> Tuple[List[float], List[str]]:
    """
    Gera completions com:
      - Modelos remotos (OpenAI ou Groq) via chat.completions
      - Modelos locais Hugging Face (causal LM)

    Retorna:
        generation_times: lista com tempos de geração por prompt
        generations:      lista com textos gerados
    """

    generations: List[str] = []
    generation_times: List[float] = []

    # 🔹 Caso 1: modelo REMOTO (OpenAI ou Groq) → tokenizer == None
    if tokenizer is None:
        if not isinstance(model, RemoteChatModel):
            raise TypeError(
                "Esperado RemoteChatModel quando tokenizer é None. "
                "Verifique o retorno de load_hf_lm_and_tokenizer."
            )

        client = model.client
        remote_model_name = model.model_name

        for prompt in tqdm(prompts, disable=disable_tqdm):
            start = time.time()

            # mapeia do_sample para temperature (se quiser determinístico, basta usar temperature=0)
            temp = temperature if do_sample else temperature

            response = client.chat.completions.create(
                model=remote_model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=temp,
                max_tokens=max_new_tokens,
                top_p=top_p,
                # se quiser, pode adicionar 'stop' aqui futuramente
            )

            end = time.time()
            generations.append(response.choices[0].message.content.strip())
            generation_times.append(end - start)

        return generation_times, generations

    # 🔹 Caso 2: modelo Hugging Face local
    model.eval()

    for i in tqdm(range(0, len(prompts), batch_size), disable=disable_tqdm):
        batch_prompts = prompts[i:i + batch_size]
        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            add_special_tokens=add_special_tokens,
        ).to(model.device)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.time()

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                min_new_tokens=min_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=tokenizer.eos_token_id,
            )

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end = time.time()

        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        # remove o prompt original do começo, se estiver incluído na saída
        cleaned = [
            out.replace(prompt, "").strip()
            for out, prompt in zip(decoded, batch_prompts)
        ]

        generations.extend(cleaned)
        generation_times.extend([end - start] * len(batch_prompts))

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return generation_times, generations


def load_hf_lm_and_tokenizer(
    model_name: str,
    torch_dtype: torch.dtype = torch.float16
):
    """
    Carrega um "modelo" de linguagem e tokenizer, com roteamento:

      - Modelos GPT (gpt-3.*, gpt-4.*, o1, o3, etc.)  → OpenAI (API)
      - Modelos com 'llama' ou 'groq' no nome         → Groq (API)
      - Demais modelos                                → Hugging Face local

    Retorna:
        (model, tokenizer)

    Onde:
      - Para OpenAI/Groq: model é RemoteChatModel e tokenizer == None
      - Para HF: model é AutoModelForCausalLM e tokenizer é AutoTokenizer
    """

    # Normaliza para facilitar checagens
    name_lower = model_name.lower()

    # 🔹 Caso 1 – Modelos GPT → OpenAI
    if any(prefix in name_lower for prefix in ["gpt-3", "gpt-4", "o1", "o3"]):
        if not OPENAI_API_KEY:
            raise ValueError("❌ Variável de ambiente OPENAI_API_KEY não encontrada.")

        client = OpenAI(api_key=OPENAI_API_KEY)
        print(f"✅ Usando modelo OpenAI: {model_name}")

        remote = RemoteChatModel(client=client, model_name=model_name, provider="openai")
        return remote, None

    # 🔹 Caso 2 – Modelos LLaMA ou Groq → Groq
    if ("llama" in name_lower) or ("groq" in name_lower) or ("openai/" in name_lower):
        if not GROQ_API_KEY:
            raise ValueError("❌ Variável de ambiente GROQ_API_KEY não encontrada.")

        client = Groq(api_key=GROQ_API_KEY)
        print(f"✅ Usando modelo Groq: {model_name}")

        remote = RemoteChatModel(client=client, model_name=model_name, provider="groq")
        return remote, None

    # 🔹 Caso 3 – Demais modelos → Hugging Face local
    if torch.cuda.is_available():
        device_map = "auto"
    elif torch.backends.mps.is_available():
        device_map = {"": "mps"}
    else:
        device_map = {"": "cpu"}

    print(f"🔧 Carregando modelo HF '{model_name}' com dtype={torch_dtype} no dispositivo {device_map}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=True,
    )
    model.eval()
    print("✅ Modelo Hugging Face carregado com sucesso.")
    return model, tokenizer
