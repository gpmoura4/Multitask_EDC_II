import os
import time
from typing import List, Tuple, Optional

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from groq import Groq
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


class RemoteChatModel:
    """
    Wrapper para modelos remotos (OpenAI, Groq, Ollama).
    """

    def __init__(self, client, model_name: str, provider: str):
        self.client = client
        self.model_name = model_name
        self.provider = provider


# ============================================================
#   CLIENTE OLLAMA (Completo e Funcional)
# ============================================================
class OllamaClient:
    """
    Cliente mínimo para usar o endpoint oficial do Ollama:
    POST http://localhost:11434/api/chat
    """

    def __init__(self, model_name, base_url="http://localhost:11434/api/chat"):
        self.model_name = model_name
        self.base_url = base_url

    def chat(self, prompt):
        import requests

        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "stream": False
        }

        resp = requests.post(self.base_url, json=payload)
        resp.raise_for_status()
        data = resp.json()

        # Resposta correta: data["message"]["content"]
        return data["message"]["content"]


# ============================================================
#      generate_completions (AGORA COM SUPORTE A OLLAMA)
# ============================================================
def generate_completions(
        model,
        tokenizer,
        prompts: List[str],
        batch_size: int = 1,
        stop_id_sequences=None,
        add_special_tokens: bool = True,
        disable_tqdm: bool = False,
        max_new_tokens: int = 512,
        min_new_tokens: int = 1,
        do_sample: bool = False,
        temperature: float = 0.7,
        top_p: float = 1.0
) -> Tuple[List[float], List[str]]:
    generations: List[str] = []
    generation_times: List[float] = []

    # -------------------------------------------------------------
    # CASO MODELOS REMOTOS (OpenAI, Groq, Ollama)
    # -------------------------------------------------------------
    if tokenizer is None:
        if not isinstance(model, RemoteChatModel):
            raise TypeError("Esperado RemoteChatModel para modelos remotos.")

        provider = model.provider

        # ---------------------------------------------
        #   🔵 PROVIDER = OLLAMA
        # ---------------------------------------------
        if provider == "ollama":
            for prompt in tqdm(prompts, disable=disable_tqdm):
                start = time.time()
                reply = model.client.chat(prompt)
                end = time.time()

                generations.append(reply.strip())
                generation_times.append(end - start)

            return generation_times, generations

        # ---------------------------------------------
        #   🟩 OpenAI / 🟥 Groq (permanece igual)
        # ---------------------------------------------
        client = model.client
        remote_model_name = model.model_name

        for prompt in tqdm(prompts, disable=disable_tqdm):
            start = time.time()

            response = client.chat.completions.create(
                model=remote_model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )

            end = time.time()
            generations.append(response.choices[0].message.content.strip())
            generation_times.append(end - start)

        return generation_times, generations

    # -------------------------------------------------------------
    # CASO MODELOS LOCAIS HF (continua como estava)
    # -------------------------------------------------------------
    model.eval()

    for i in tqdm(range(0, len(prompts), batch_size), disable=disable_tqdm):
        batch_prompts = prompts[i:i + batch_size]
        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
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
                eos_token_id=tokenizer.eos_token_id,
            )

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end = time.time()

        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        cleaned = [
            out.replace(prompt, "").strip()
            for out, prompt in zip(decoded, batch_prompts)
        ]

        generations.extend(cleaned)
        generation_times.extend([end - start] * len(batch_prompts))

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return generation_times, generations


# ============================================================
#   load_hf_lm_and_tokenizer (AGORA COM SUPORTE A OLLAMA)
# ============================================================
def load_hf_lm_and_tokenizer(
        model_name: str,
        torch_dtype: torch.dtype = torch.float16
):
    name_lower = model_name.lower()

    # -------------------------------------------------------------
    # 🔵 Caso especial: Llama-2-7b → rodar via OLLAMA LOCAL
    # -------------------------------------------------------------
    if model_name in ["Llama-2-7b", "meta-llama/Llama-2-7b-hf", "llama2:7b"]:
        print("🟦 Usando modelo local via Ollama: http://localhost:11434/api/chat")

        client = OllamaClient(model_name=model_name)
        remote = RemoteChatModel(client=client, model_name=model_name, provider="ollama")
        return remote, None

    # -------------------------------------------------------------
    # 🟩 OpenAI
    # -------------------------------------------------------------
    if any(prefix in name_lower for prefix in ["gpt-3", "gpt-4", "o1", "o3"]):
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY não encontrada.")
        client = OpenAI(api_key=OPENAI_API_KEY)
        print(f"🟩 Usando modelo OpenAI: {model_name}")
        remote = RemoteChatModel(client=client, model_name=model_name, provider="openai")
        return remote, None

    # -------------------------------------------------------------
    # 🟥 Groq
    # -------------------------------------------------------------
    if ("llama-3.3-70b-versatile" in name_lower) or ("groq" in name_lower) or ("openai/" in name_lower):
        if not GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY não encontrada.")
        client = Groq(api_key=GROQ_API_KEY)
        print(f"🟥 Usando modelo Groq: {model_name}")
        remote = RemoteChatModel(client=client, model_name=model_name, provider="groq")
        return remote, None

    # -------------------------------------------------------------
    # Hugging Face local
    # -------------------------------------------------------------
    if torch.cuda.is_available():
        device_map = "auto"
    else:
        device_map = {"": "cpu"}

    print(f"🔧 Carregando modelo HF '{model_name}' no dispositivo {device_map}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=True,
    )

    model.eval()
    print("✅ Modelo Hugging Face carregado com sucesso.")
    return model, tokenizer