import time
from tqdm import tqdm
import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def generate_completions(
    model,
    tokenizer,
    prompts,
    batch_size=1,
    stop_id_sequences=None,
    add_special_tokens=True,
    disable_tqdm=False,
    max_new_tokens=512,
    min_new_tokens=1,
    do_sample=False,
    temperature=0.7,
    top_p=1.0
):
    """
    Gera completions tanto com modelos Hugging Face quanto OpenAI.
    Retorna:
        generation_times: lista com tempos
        generations: lista com textos gerados
    """

    generations = []
    generation_times = []

    # 🔹 Caso 1: modelo OpenAI
    if tokenizer is None:
        client = model  # aqui model == cliente OpenAI

        for prompt in tqdm(prompts, disable=disable_tqdm):
            start = time.time()
            response = client.chat.completions.create(
                model="gpt-3.5-turbo-0125",
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_new_tokens,
                top_p=top_p,
            )
            end = time.time()
            generations.append(response.choices[0].message.content.strip())
            generation_times.append(end - start)

        return generation_times, generations

    # 🔹 Caso 2: modelo Hugging Face
    model.eval()
    for i in tqdm(range(0, len(prompts), batch_size), disable=disable_tqdm):
        batch_prompts = prompts[i:i + batch_size]
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)

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

        torch.cuda.synchronize()
        end = time.time()

        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        cleaned = [out.replace(prompt, "").strip() for out, prompt in zip(decoded, batch_prompts)]
        generations.extend(cleaned)
        generation_times.extend([end - start] * len(batch_prompts))
        torch.cuda.empty_cache()

    return generation_times, generations


def load_hf_lm_and_tokenizer(model_name, torch_dtype=torch.float16):
    """
    Carrega um modelo de linguagem e tokenizer, suportando:
      - Modelos Hugging Face (ex: meta-llama/Llama-2-7b-chat-hf)
      - Modelos OpenAI (ex: gpt-3.5-turbo, gpt-4-turbo, gpt-4o, etc.)
    """
    
    # 🔹 Caso 1 — Modelos da OpenAI
    if any(prefix in model_name.lower() for prefix in ["gpt-3", "gpt-4", "o1", "o3"]):
        # Inicializa cliente da OpenAI
        # api_key = os.getenv("OPENAI_API_KEY")
        api_key = OPENAI_API_KEY
        if not api_key:
            raise ValueError("❌ Variável de ambiente OPENAI_API_KEY não encontrada. Defina-a antes de executar.")
        
        client = OpenAI(api_key=api_key)
        print(f"✅ Usando modelo OpenAI: {model_name}")
        # Não há 'tokenizer' real aqui — retornamos um placeholder
        return client, None

    # 🔹 Caso 2 — Modelos Hugging Face locais ou do hub
    else:
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
            trust_remote_code=True
        )
        model.eval()
        print("✅ Modelo Hugging Face carregado com sucesso.")
        return model, tokenizer
