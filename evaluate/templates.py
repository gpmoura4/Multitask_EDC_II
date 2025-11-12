def create_prompt_with_tulu_chat_format(messages):
    """
    Constrói um prompt de chat compatível com o formato Tulu.

    Args:
        messages (list[dict]): Lista de mensagens, cada uma no formato:
            {"role": "system"|"user"|"assistant", "content": str}

    Returns:
        str: prompt formatado como string pronta para o modelo
    """

    # Prefixos típicos do formato Tulu / LLaMA
    role_map = {
        "system": "<|system|>",
        "user": "<|user|>",
        "assistant": "<|assistant|>"
    }

    # Concatena todas as mensagens em sequência
    formatted = ""
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "").strip()
        formatted += f"{role_map.get(role, '<|user|>')} {content}\n"

    # O modelo espera terminar com a tag do assistente
    formatted += "<|assistant|>"

    return formatted
