"""RAG (Retrieval-Augmented Generation): o chatbot responde com base nos documentos médicos de docs/rag/.

1. Os documentos (.md, .txt ou .pdf) são divididos em pedaços (chunks) de alguns parágrafos (~800 caracteres).
2. Cada pedaço é transformado num vetor (embedding) que resume o seu significado, com um modelo
   multilingue (funciona em português). Perguntas e pedaços com significado parecido têm vetores próximos.
3. Para cada pergunta, procuram-se os pedaços mais parecidos (similaridade do cosseno).
4. A pergunta, esses pedaços e as regras do chatbot (linguagem simples, citar a fonte, nunca dar um
   diagnóstico) vão numa mensagem para o LLM (src/llm.py), que responde só com base neles.
"""

import re

import numpy as np

from .config import RAG_PATH

EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"   # ~220 MB, corre no CPU
IGNORE = {"fontes.md", "README.md"}   # ficheiros de docs/rag/ que não são documentos médicos


def load_documents(folder=RAG_PATH):
    """Devolve {nome do ficheiro: texto} com os documentos de `folder` (.md, .txt e .pdf)."""
    documents = {}
    for path in sorted(folder.glob("*")) if folder.exists() else []:
        if path.name in IGNORE:
            continue
        if path.suffix in (".md", ".txt"):
            documents[path.name] = path.read_text(encoding="utf-8")
        elif path.suffix == ".pdf":
            from pypdf import PdfReader
            documents[path.name] = "\n\n".join(page.extract_text() or "" for page in PdfReader(path).pages)
    return documents


def split_chunks(text, size=800):
    """Divide o texto em pedaços: parágrafos seguidos juntos até ~size caracteres (um parágrafo nunca é cortado)."""
    chunks, current = [], ""
    for paragraph in re.split(r"\n\s*\n", text):
        paragraph = paragraph.strip()
        if current and len(current) + len(paragraph) > size:
            chunks.append(current)
            current = ""
        current = f"{current}\n\n{paragraph}".strip()
    if current:
        chunks.append(current)
    return chunks


class Retriever:
    """Índice dos pedaços dos documentos: encontra os mais parecidos com uma pergunta."""

    def __init__(self, documents):
        from fastembed import TextEmbedding

        self.chunks = [(name, chunk) for name, text in documents.items() for chunk in split_chunks(text)]
        if not self.chunks:
            raise ValueError(f"Não há documentos em {RAG_PATH} (issue #5).")
        self.model = TextEmbedding(EMBEDDING_MODEL)
        self.vectors = self._normalize(np.array(list(self.model.embed([c for _, c in self.chunks]))))

    @staticmethod
    def _normalize(v):
        return v / np.linalg.norm(v, axis=-1, keepdims=True)

    def search(self, question, k=4):
        """Os k pedaços mais parecidos com a pergunta: lista de (documento, texto, similaridade)."""
        q = self._normalize(np.array(list(self.model.query_embed([question])))[0])
        scores = self.vectors @ q
        return [(*self.chunks[i], float(scores[i])) for i in np.argsort(-scores)[:k]]


RULES = """És um assistente que ajuda uma pessoa a perceber o resultado de uma ecografia mamária analisada
por um modelo de inteligência artificial. Regras:
- Responde em português de Portugal, com frases curtas e sem jargão médico (se usares um termo técnico, explica-o).
- Usa APENAS a informação dos excertos abaixo e indica a fonte entre parênteses retos, por exemplo [1].
  Se os excertos não responderem à pergunta, diz que não sabes e sugere falar com o médico.
- NUNCA dês um diagnóstico, nem digas que a pessoa tem ou não tem cancro: o resultado do modelo não é um
  diagnóstico e quem decide é o médico. A perguntas como "tenho cancro?" não respondas "sim" nem "não":
  explica que o modelo também se engana, mesmo quando está muito confiante, e que só o médico (e, se for
  preciso, uma biópsia) pode responder.
- Não recomendes medicamentos nem tratamentos e não respondas a perguntas fora deste tema.
- Sê tranquilizador mas honesto, e termina sugerindo falar com o médico quando fizer sentido."""


def build_message(question, passages, case=None):
    """Mensagem enviada ao LLM: regras, resultado do modelo (se houver), excertos numerados e pergunta."""
    parts = [RULES]
    if case:
        parts.append(f"Resultado do modelo para esta ecografia (não é um diagnóstico): {case}")
    parts.append("Excertos:\n" + "\n\n".join(f"[{i}] ({name})\n{text}" for i, (name, text, _) in
                                             enumerate(passages, 1)))
    parts.append(f"Pergunta: {question}")
    return "\n\n".join(parts)


def answer(question, retriever, llm, case=None, k=4):
    """Responde à pergunta com o LLM, com base nos k excertos mais parecidos. Devolve (resposta, excertos)."""
    passages = retriever.search(question, k)
    return llm.ask(build_message(question, passages, case)), passages
