import os
import warnings

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings("ignore")

from src.agent import TEXTS, case_for_chat, decide, choose_threshold
from src.data import load_dataset
from src.explain import load_model
from src.rag import answer, build_retriever  # ajusta o import ao nome do teu ficheiro RAG

print("A carregar modelo e documentos médicos...")
_, x_val, x_test, _, y_val, y_test, names_test = load_dataset(
    "ds/img", size=128, test_names=True
)
model = load_model("models/mobilenet.keras")
threshold = choose_threshold(model.predict(x_val, verbose=0), y_val)
retriever = build_retriever("docs/rag")

# Simula um caso de teste (ex: índice 0) para dar contexto ao chatbot
probs = model.predict(x_test[0:1], verbose=0)[0]
decision = decide(probs, threshold)
case_summary = case_for_chat(decision)

print("\n" + "=" * 60)
print(TEXTS["boas_vindas"])
print(TEXTS["aviso"])
print(f"[Contexto interno do exame: {case_summary}]")
print("=" * 60)

while True:
  q = input("\nPaciente (escreve 'sair' para terminar): ").strip()
  if q.lower() in ("sair", "exit", ""):
    break
  resposta, fontes = answer(q, retriever, llm=None, case=case_summary)
  print("\nAssistente:", resposta)
  print("Fontes:", sorted({name for name, _, _ in fontes}))