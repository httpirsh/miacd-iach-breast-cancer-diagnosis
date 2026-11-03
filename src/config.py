"""Caminhos do projeto, iguais no Google Colab e localmente (relativos à raiz do repositório)."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data/breast-ultrasound-images-DatasetNinja/ds/img"
RESULTS_PATH = ROOT / "results"
RESULTS_PATH.mkdir(exist_ok=True)
