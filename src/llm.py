"""Ligação ao LLM da plataforma IAedu (FCT), usada pelo chatbot do agente.

No IAedu (https://iaedu.pt) cria-se um agente e, na conversa modelo do agente, em engrenagem →
«Informação da API», copiam-se três valores para um ficheiro `.env` na raiz do repositório
(o `.env` está no .gitignore e nunca vai para o git; ver `.env.example`):

    IAEDU_API_KEY=...
    IAEDU_AGENT_ID=...
    IAEDU_CHANNEL_ID=...

A chave é de cada agente. A API recebe a mensagem num POST (multipart/form-data) e responde em
streaming, com um objeto JSON por linha: {"type": "token", "content": "..."} com cada pedaço do texto,
{"type": "message", ...} com o texto completo, {"type": "done"} no fim e {"type": "error"} se falhar.
O formato não tem documentação pública: foi visto na prática e pode mudar.
"""

import json
import os
import uuid

import requests
from dotenv import load_dotenv

from .config import ROOT

BASE_URL = "https://api.iaedu.pt/agent-chat/api/v1"
VARIABLES = ["IAEDU_API_KEY", "IAEDU_AGENT_ID", "IAEDU_CHANNEL_ID"]


def iaedu_configured():
    """True se o `.env` (ou o ambiente) tem as três variáveis do IAedu."""
    load_dotenv(ROOT / ".env")
    return all(os.environ.get(v) for v in VARIABLES)


class IAedu:
    """Conversa com um agente do IAedu. As mensagens da mesma instância ficam na mesma conversa (thread)."""

    def __init__(self, timeout=120):
        load_dotenv(ROOT / ".env")
        missing = [v for v in VARIABLES if not os.environ.get(v)]
        if missing:
            raise RuntimeError(f"Faltam variáveis no .env: {', '.join(missing)} (ver src/llm.py e .env.example)")
        self._key = os.environ["IAEDU_API_KEY"]
        self.agent_id = os.environ["IAEDU_AGENT_ID"]
        self.channel_id = os.environ["IAEDU_CHANNEL_ID"]
        self.timeout = timeout
        self.new_conversation()

    def __repr__(self):
        return "IAedu(api_key=<escondida>)"   # a chave nunca aparece em prints nem em erros

    def new_conversation(self):
        self.thread_id = f"thr-{uuid.uuid4()}"

    def ask(self, message):
        """Envia uma mensagem e devolve o texto da resposta."""
        response = requests.post(
            f"{BASE_URL}/agent/{self.agent_id}/stream",
            headers={"x-api-key": self._key},
            files={"channel_id": (None, self.channel_id), "thread_id": (None, self.thread_id),
                   "user_info": (None, "{}"), "message": (None, message)},   # files= força multipart/form-data
            stream=True, timeout=self.timeout,
            allow_redirects=False,   # não reenviar a chave para outro endereço
        )
        with response:
            lines = [line for line in response.iter_lines(decode_unicode=True) if line and line.strip()]
        if response.status_code == 429:
            raise RuntimeError("IAedu: limite de pedidos atingido; esperar um pouco antes de tentar outra vez.")
        if response.status_code != 200:
            raise RuntimeError(f"IAedu respondeu {response.status_code}: {' '.join(lines)[:300]}")
        return parse_stream(lines)


def parse_stream(lines):
    """Junta os pedaços de texto da resposta em streaming (um objeto JSON por linha)."""
    tokens, full_message = [], None
    for line in lines:
        line = line[len("data:"):].strip() if line.startswith("data:") else line   # aceita também SSE
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if not isinstance(event, dict):
            continue
        content = event.get("content")
        if event.get("type") == "token" and isinstance(content, str):
            tokens.append(content)
        elif event.get("type") == "message":
            full_message = content.get("content") if isinstance(content, dict) else content
        elif event.get("type") == "error":
            raise RuntimeError(f"IAedu devolveu um erro: {content}")
    return "".join(tokens) if tokens else (full_message or "")
