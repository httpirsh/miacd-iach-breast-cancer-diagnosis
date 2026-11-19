"""Agente: o que fazer com cada ecografia depois de o modelo a classificar.

- Human-in-the-Loop: se o modelo prevê maligno, ou se a confiança dele é menor do que um limiar, o caso vai
  SEMPRE para o médico e nada é comunicado como resultado final.
- Human-on-the-Loop: os restantes casos (normal ou benigno com confiança alta) são decididos pelo sistema,
  mas ficam registados num log que o médico pode rever e corrigir.

O limiar é escolhido no conjunto de validação (choose_threshold), nunca no teste.
"""

import csv
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np

from .data import CLASSES
from .evaluation import MALIGNANT

DOCTOR, AUTOMATIC = "médico", "automático"
CLASSES_PT = {"normal": "normal", "benign": "benigno", "malignant": "maligno"}
PATIENT_CLASSES_PT = {
    "normal": "«normal»",
    "benign": "«benigna» (com aspeto não canceroso)",
}
LOG_COLUMNS = [
    "data",
    "imagem",
    "classe_prevista",
    "confianca",
    "encaminhamento",
    "revisto",
    "classe_medico",
]

# Textos para o paciente.
TEXTS = {
    "boas_vindas": (
        "Olá. Este protótipo usa inteligência artificial para analisar uma"
        " imagem de ecografia da mama. Mostra uma análise automática ou"
        " assinala a imagem para revisão por um médico. É uma demonstração:"
        " não envia o exame a um médico. Quando a conversa está disponível,"
        " também pode fazer perguntas gerais sobre a ecografia."
    ),
    "aviso": (
        "Isto não é um diagnóstico. Fale com o seu médico para interpretar o"
        " exame e esclarecer as suas dúvidas."
    ),
    DOCTOR: (
        "O protótipo assinalou esta imagem para revisão por um médico. "
        "Por isso, não mostra a análise automática. "
        "Este aviso, por si só, não diz se há ou não um problema."
    ),
    AUTOMATIC: (
        "O programa sugeriu a categoria {classe} para esta imagem. "
        "Esta análise automática não foi confirmada por um médico."
    ),
    "confianca": (
        "Confiança do modelo: {confianca:.0%}. A percentagem indica quão seguro"
        " o programa está da sua escolha. Não indica o seu risco de ter cancro."
        " Mesmo com uma confiança alta, o programa pode enganar-se."
    ),
    "heatmap": (
        "No mapa de cores, as zonas a vermelho são as que mais influenciaram a"
        " escolha do programa. As cores, por si só, não dizem se existe um"
        " problema nem onde ele está."
    ),
}


def to_doctor(probs, threshold):
  """Para cada imagem: vai para o médico?

  (maligno previsto ou confiança < threshold)
  """
  probs = np.atleast_2d(probs)
  return (probs.argmax(axis=1) == MALIGNANT) | (probs.max(axis=1) < threshold)


def choose_threshold(probs_val, y_val, target=0.95, max_threshold=0.95):
  """Menor limiar com que pelo menos `target` dos malignos de VALIDAÇÃO chegam ao médico,

  limitado a `max_threshold` para evitar que o sistema envie tudo para o
  médico.
  """
  malignant = np.asarray(y_val) == MALIGNANT

  # Passo de 0.01 até 0.90, e passo fino de 0.005 entre 0.90 e max_threshold
  if max_threshold > 0.90:
    grid = np.concatenate([
        np.arange(0.50, 0.90, 0.01),
        np.arange(0.90, max_threshold + 0.001, 0.005),
    ])
  else:
    grid = np.arange(0.50, max_threshold + 0.01, 0.01)

  for threshold in np.round(grid, 3):
    if to_doctor(probs_val, threshold)[malignant].mean() >= target:
      return float(threshold)

  # Fallback: se nenhum limiar atingir o target, devolve o max_threshold em vez de 1.0
  # para manter o agente funcional e não bloquear a triagem
  return float(max_threshold)


def referral_summary(probs, y_true, threshold):
  """Quantas imagens vão para o médico, quantos malignos lá chegam e quantos erros ficam no automático."""
  y_true = np.asarray(y_true)
  doctor = to_doctor(probs, threshold)
  wrong = probs.argmax(axis=1) != y_true
  return {
      "enviadas ao médico": f"{doctor.sum()} de {len(y_true)} ({doctor.mean():.0%})",
      "malignos que chegam ao médico": (
          f"{doctor[y_true == MALIGNANT].sum()} de {(y_true == MALIGNANT).sum()}"
      ),
      "erros decididos sem médico": f"{(wrong & ~doctor).sum()} de {(~doctor).sum()}",
  }


def decide(probs, threshold):
  """Decisão para uma imagem: classe prevista, confiança e para onde vai (médico ou automático)."""
  pred, confidence = int(np.argmax(probs)), float(np.max(probs))
  if not to_doctor(probs, threshold)[0]:  # a mesma regra que se avalia em choose_threshold
    route, reason = AUTOMATIC, f"{CLASSES[pred]} com confiança alta ({confidence:.0%})"
  elif pred == MALIGNANT:
    route, reason = DOCTOR, "o modelo prevê maligno"
  else:
    route, reason = (
        DOCTOR,
        f"confiança ({confidence:.0%}) abaixo do limiar ({threshold:.0%})",
    )
  return {
      "classe": CLASSES[pred],
      "confianca": confidence,
      "encaminhamento": route,
      "motivo": reason,
  }


def patient_message(decision):
  """O que o paciente vê: sem classe nem confiança nos casos para revisão médica."""
  route = decision["encaminhamento"]
  text = TEXTS[route]
  if route == AUTOMATIC:
    text = text.format(classe=PATIENT_CLASSES_PT[decision["classe"]])
    confidence = TEXTS["confianca"].format(confianca=decision["confianca"])
    text = f"{text}\n\n{confidence}"
  return f"{text}\n\n{TEXTS['aviso']}"


def case_for_chat(decision):
  """Resumo do caso para o chatbot.

  Se o caso vai para o médico, a classe não é dada ao LLM, para não chegar ao
  paciente antes de o médico ver o exame.
  """
  if decision["encaminhamento"] == DOCTOR:
    return (
        "o exame foi encaminhado para o médico e ainda não há resultado para"
        " comunicar ao paciente"
    )
  return (
      f"{CLASSES_PT[decision['classe']]}, com confiança de"
      f" {decision['confianca']:.0%} (decidido sem médico, mas o médico pode"
      " rever)"
  )


def show_case(image, cam, decision, title="", *, patient_view=False):
  """Mostra o mapa; na vista do paciente, só o mostra para casos automáticos."""
  if patient_view and decision["encaminhamento"] != AUTOMATIC:
    return
  fig, axs = plt.subplots(1, 2, figsize=(7, 3.6))
  axs[0].imshow(image, cmap="gray")
  axs[1].imshow(image, cmap="gray")
  axs[1].imshow(cam, cmap="jet", alpha=0.4, vmin=0, vmax=1)
  if patient_view:
    axs[0].set_title("Ecografia")
    axs[1].set_title("Mapa de cores")
  else:
    axs[0].set_title(title or "ecografia")
    axs[1].set_title(
        f"{decision['classe']} ({decision['confianca']:.0%}) →"
        f" {decision['encaminhamento']}",
        color="firebrick" if decision["encaminhamento"] == DOCTOR else "black",
    )
  for ax in axs:
    ax.axis("off")
  plt.tight_layout()
  plt.show()


# --- Log para o Human-on-the-Loop ---------------------------------------------------------------


def log_case(path, image_name, decision):
  """Acrescenta um caso ao log (cria o ficheiro se não existir)."""
  new = not path.exists()
  with open(path, "a", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=LOG_COLUMNS)
    if new:
      writer.writeheader()
    writer.writerow({
        "data": datetime.now().isoformat(timespec="seconds"),
        "imagem": image_name,
        "classe_prevista": decision["classe"],
        "confianca": round(decision["confianca"], 3),
        "encaminhamento": decision["encaminhamento"],
        "revisto": False,
        "classe_medico": "",
    })


def read_log(path):
  import pandas as pd

  return pd.read_csv(
      path,
      encoding="utf-8",
      dtype={"classe_medico": str},
      keep_default_na=False,
)


def review_case(path, image_name, doctor_class):
  """O médico revê um caso do log e indica a classe correta (igual à prevista se concordar)."""
  if doctor_class not in CLASSES:
    raise ValueError(
        f"Classe inválida: {doctor_class!r} (usar uma de {CLASSES})"
    )
  log = read_log(path)
  rows = log["imagem"] == image_name
  if not rows.any():
    raise ValueError(f"{image_name!r} não está no log")
  log.loc[rows, "revisto"] = True
  log.loc[rows, "classe_medico"] = doctor_class
  log.to_csv(path, index=False, encoding="utf-8")
  return log[rows]