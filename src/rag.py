"""RAG (Retrieval-Augmented Generation): o chatbot responde com base nos documentos médicos de docs/rag/.

1. Os documentos (.md) são divididos em pedaços (chunks) de alguns parágrafos (~800 caracteres).
2. Cada pedaço é transformado num vetor TF-IDF: conta os grupos de 3 a 5 letras seguidas que tem, dando mais
   peso aos que aparecem em poucos pedaços. Grupos de letras (e não palavras inteiras) apanham variações
   como "biópsia"/"biópsias" ou "maligno"/"malignos".
3. Para cada pergunta, procuram-se os pedaços com mais grupos de letras em comum (similaridade do cosseno).
4. A pergunta, esses pedaços e as regras do chatbot (linguagem simples, citar a fonte, nunca dar um
   diagnóstico) vão numa mensagem para o LLM (src/llm.py), que responde só com base neles.
"""

import re

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from .config import RAG_PATH


def load_documents(folder=RAG_PATH):
    """Devolve {nome do ficheiro: texto} com os documentos .md de `folder`."""
    return {path.name: path.read_text(encoding="utf-8")
            for path in sorted(folder.glob("*.md")) if path.name != "fontes.md"}   # fontes.md é a lista das fontes


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
        self.chunks = [(name, chunk) for name, text in documents.items() for chunk in split_chunks(text)]
        # char_wb: grupos de letras dentro de cada palavra; strip_accents: "biopsia" = "biópsia"
        self.tfidf = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), strip_accents="unicode")
        self.vectors = self.tfidf.fit_transform([chunk for _, chunk in self.chunks])   # já normalizados

    def search(self, question, k=4):
        """Os k pedaços mais parecidos com a pergunta: lista de (documento, texto, similaridade)."""
        scores = (self.vectors @ self.tfidf.transform([question]).T).toarray().ravel()
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
