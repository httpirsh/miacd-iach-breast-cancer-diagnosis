"""RAG (Retrieval-Augmented Generation): o chatbot responde com base nos documentos médicos de docs/rag/.

1. Os documentos (.md, .txt ou .pdf) são divididos em pedaços (chunks) de ~800 caracteres.
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


def split_chunks(text, size=800, overlap=150):
    """Divide o texto em pedaços de ~size caracteres, sem cortar parágrafos sempre que possível.

    Cada pedaço começa com o fim (overlap caracteres) do anterior, para não perder o contexto
    de uma frase que fique na fronteira.
    """
    pieces = []
    for paragraph in (p.strip() for p in re.split(r"\n\s*\n", text)):
        while len(paragraph) > size:   # parágrafo maior do que um pedaço: cortar no fim de uma frase
            cut = paragraph.rfind(". ", 0, size) + 1 or size
            pieces.append(paragraph[:cut].strip())
            paragraph = paragraph[cut:].strip()
        if paragraph:
            pieces.append(paragraph)

    chunks, current = [], ""
    for piece in pieces:
        if current and len(current) + len(piece) + 2 > size:
            chunks.append(current)
            current = current[-overlap:].split(" ", 1)[-1]   # o próximo começa com o fim deste
        current = f"{current}\n\n{piece}" if current else piece
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


RULES = {
    "paciente": """És um assistente que ajuda uma pessoa a perceber o resultado de uma ecografia mamária analisada
por um modelo de inteligência artificial. Regras:
- Responde em português de Portugal, com frases curtas e sem jargão médico (se usares um termo técnico, explica-o).
- Usa APENAS a informação dos excertos abaixo e indica a fonte entre parênteses retos, por exemplo [1].
  Se os excertos não responderem à pergunta, diz que não sabes e sugere falar com o médico.
- NUNCA dês um diagnóstico, nem digas que a pessoa tem ou não tem cancro: o resultado do modelo não é um
  diagnóstico e quem decide é o médico.
- Não recomendes medicamentos nem tratamentos e não respondas a perguntas fora deste tema.
- Sê tranquilizador mas honesto, e termina sugerindo falar com o médico quando fizer sentido.""",
    "medico": """És um assistente para um médico que revê o resultado de um modelo de classificação de ecografias
mamárias (normal / benigno / maligno). Responde em português, em linguagem técnica e concisa, usando APENAS os
excertos abaixo e citando a fonte entre parênteses retos, por exemplo [1]. Se os excertos não chegarem, di-lo.
Lembra as limitações do modelo quando forem relevantes para a pergunta.""",
}


def build_message(question, passages, case=None, mode="paciente"):
    """Mensagem enviada ao LLM: regras, resultado do modelo (se houver), excertos numerados e pergunta."""
    parts = [RULES[mode]]
    if case:
        parts.append(f"Resultado do modelo para esta ecografia (não é um diagnóstico): {case}")
    parts.append("Excertos:\n" + "\n\n".join(f"[{i}] ({name})\n{text}" for i, (name, text, _) in
                                             enumerate(passages, 1)))
    parts.append(f"Pergunta: {question}")
    return "\n\n".join(parts)


def answer(question, retriever, llm, case=None, mode="paciente", k=4):
    """Responde à pergunta com o LLM, com base nos k excertos mais parecidos. Devolve (resposta, excertos)."""
    passages = retriever.search(question, k)
    return llm.ask(build_message(question, passages, case, mode)), passages
