import pandas as pd
from src.config import RAG_PATH
from src.rag import Retriever, load_documents

perguntas = pd.read_csv(RAG_PATH / "perguntas_teste.csv")
retriever = Retriever(load_documents(RAG_PATH))

normais = perguntas[perguntas["deve_recusar"] == "nao"]
hits = 0
for _, r in normais.iterrows():
    esperados = {f.strip() + ".md" for f in r["documento_fonte"].split(";")}
    encontrados = [nome for nome, _, _ in retriever.search(r["pergunta"], k=3)]
    ok = bool(esperados & set(encontrados))
    hits += ok
    print("OK   " if ok else "FALHA", r["id"], r["pergunta"], "" if ok else f"-> {encontrados}")
print(f"{hits} de {len(normais)} com o documento esperado nos 3 primeiros excertos")