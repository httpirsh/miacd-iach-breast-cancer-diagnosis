"""RAG (Retrieval-Augmented Generation): o chatbot responde com base nos documentos médicos de docs/rag/.

1. Os documentos (.md) são divididos em pedaços (chunks) de alguns parágrafos (~650 caracteres).
2. Cada pedaço é indexado de forma híbrida (palavras inteiras + grupos de 3 a 5 letras) juntamente com o
   título derivado do nome do ficheiro.
3. Antes da pesquisa, a pergunta passa por uma expansão clínica de sinónimos (ex.: "pancada" -> "embates
   violentos", "hereditário" -> "história familiar genética") para ligar linguagem comum a termos médicos.
4. Procuram-se os pedaços com maior similaridade do cosseno, garantindo diversidade entre documentos.
"""

import re

import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer

from .config import RAG_PATH

PT_STOPWORDS = [
    "a", "as", "o", "os", "um", "uma", "uns", "umas", "de", "da", "das", "do", "dos",
    "em", "na", "nas", "no", "nos", "por", "para", "com", "sem", "sob", "sobre",
    "e", "ou", "mas", "que", "se", "como", "quando", "onde", "qual", "quais", "quem",
    "quer", "dizer", "significa", "pode", "podem", "posso", "ser", "estar", "ter", "ha",
    "sao", "foi", "era", "isso", "isto", "aquilo", "este", "esta", "esse", "essa",
    "meu", "minha", "meus", "minhas", "seu", "sua", "seus", "suas", "ao", "aos", "resultado",
]

# Expansão de linguagem comum do paciente para termos usados nos documentos clínicos (SNS24, LPCC, DGS)
CLINICAL_SYNONYMS = {
    "benigno": "tumor benigno não é cancro metastiza",
    "pancada": "embates violentos trauma fatores de risco",
    "hereditario": "história familiar alterações genéticas BRCA fatores de risco",
    "hereditário": "história familiar alterações genéticas BRCA fatores de risco",
    "proximos passos": "diagnóstico exames adicionais consulta de avaliação biópsia",
    "próximos passos": "diagnóstico exames adicionais consulta de avaliação biópsia",
}


def expand_query(question):
    """Acrescenta sinónimos clínicos se a pergunta contiver termos comuns do paciente."""
    q_lower = question.lower()
    extras = [syn for term, syn in CLINICAL_SYNONYMS.items() if term in q_lower]
    return f"{question} {' '.join(extras)}".strip() if extras else question


def load_documents(folder=RAG_PATH):
    """Devolve {nome do ficheiro: texto} com os documentos .md de `folder`."""
    return {path.name: path.read_text(encoding="utf-8")
            for path in sorted(folder.glob("*.md")) if path.name != "fontes.md"}


def split_chunks(text, size=650):
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


def _clean_doc_title(filename):
    """Converte 'lpcc_fatores_de_risco.md' em 'lpcc fatores de risco' para reforçar o contexto."""
    return filename.replace(".md", "").replace("_", " ").replace("-", " ")


class Retriever:
    """Índice híbrido dos pedaços dos documentos: encontra os mais parecidos com uma pergunta."""

    def __init__(self, documents):
        self.chunks = [(name, chunk) for name, text in documents.items() for chunk in split_chunks(text)]
        indexed_texts = [f"{_clean_doc_title(name)}: {chunk}" for name, chunk in self.chunks]

        self.word_tfidf = TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            strip_accents="unicode",
            stop_words=PT_STOPWORDS,
            sublinear_tf=True,
        )
        self.char_tfidf = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            strip_accents="unicode",
            sublinear_tf=True,
        )

        w_mat = self.word_tfidf.fit_transform(indexed_texts)
        c_mat = self.char_tfidf.fit_transform(indexed_texts)
        self.vectors = hstack([w_mat * 1.5, c_mat]).tocsr()

    def search(self, question, k=4):
        """Os k pedaços mais parecidos com a pergunta (priorizando documentos distintos):
        lista de (documento, texto, similaridade).
        """
        expanded_q = expand_query(question)
        q_w = self.word_tfidf.transform([expanded_q]) * 1.5
        q_c = self.char_tfidf.transform([expanded_q])
        q_vec = hstack([q_w, q_c])

        q_norm = np.sqrt(q_vec.multiply(q_vec).sum())
        v_norms = np.sqrt(self.vectors.multiply(self.vectors).sum(axis=1)).A1
        raw_scores = (self.vectors @ q_vec.T).toarray().ravel()
        scores = np.divide(raw_scores, v_norms * q_norm, out=np.zeros_like(raw_scores), where=(v_norms * q_norm) > 0)

        ranked = np.argsort(-scores)
        selected = []
        seen_docs = set()

        for i in ranked:
            doc_name = self.chunks[i][0]
            if doc_name not in seen_docs:
                selected.append((*self.chunks[i], float(scores[i])))
                seen_docs.add(doc_name)
                if len(selected) == k:
                    return selected

        for i in ranked:
            if len(selected) == k:
                break
            item = (*self.chunks[i], float(scores[i]))
            if item not in selected:
                selected.append(item)

        return selected


RULES = """És um assistente que ajuda uma pessoa a perceber o resultado de uma ecografia mamária analisada
por um modelo de inteligência artificial. Regras:
- Responde em português de Portugal, com frases curtas e sem jargão médico (se usares um termo técnico, explica-o).
  Trata a pessoa sempre por "você" (nunca por "tu").
- Começa por responder diretamente à pergunta. Só fales do resultado do modelo se a pergunta for sobre o
  resultado ou o exame da pessoa; numa pergunta geral (por exemplo, "o cancro da mama tem cura?"), não o refiras.
- Usa APENAS a informação dos excertos abaixo e indica a fonte entre parênteses retos, por exemplo [1].
  Se os excertos não responderem à pergunta, diz que não sabes e sugere falar com o médico.
- NUNCA dês um diagnóstico, nem digas que a pessoa tem ou não tem cancro: o resultado do modelo não é um
  diagnóstico e quem decide é o médico. A perguntas como "tenho cancro?" não respondas "sim" nem "não", nem
  comeces a resposta por "Sim" ou "Não" (lido sozinho, um "Não" soa a "não tem cancro"): explica que o modelo também se engana, mesmo quando está muito confiante, e que só o médico (e, se for
  preciso, uma biópsia) pode responder.
- Nunca digas que o resultado do modelo é tranquilizador, bom sinal ou motivo para não se preocupar.
- Não apliques à pessoa o que os excertos dizem sobre a probabilidade de algo ser cancro. Se a pessoa
  descrever um sintoma (por exemplo, um nódulo), diz-lhe que deve ser visto pelo médico, seja qual for o
  resultado do modelo.
- Não recomendes medicamentos n em tratamentos e não respondas a perguntas fora deste tema.
- Sê calmo e honesto, e termina sugerindo falar com o médico quando fizer sentido."""


def build_message(question, passages, case=None):
    """Mensagem enviada ao LLM: regras, resultado do modelo (se houver), pergunta, excertos numerados e a pergunta
    outra vez.
    """
    parts = [RULES]
    if case:
        parts.append(f"Resultado do modelo para esta ecografia (não é um diagnóstico): {case}")
    parts.append(f"Pergunta da pessoa: {question}")
    parts.append("Excertos:\n" + "\n\n".join(f"[{i}] ({name})\n{text}" for i, (name, text, _) in
                                             enumerate(passages, 1)) + "\n\n(fim dos excertos)")
    parts.append(f"Responde agora à pergunta da pessoa: {question}")
    return "\n\n".join(parts)


def answer(question, retriever, llm, case=None, k=4):
    """Responde à pergunta com o LLM, com base nos k excertos mais parecidos. Devolve (resposta, excertos)."""
    passages = retriever.search(question, k)
    return llm.ask(build_message(question, passages, case)), passages