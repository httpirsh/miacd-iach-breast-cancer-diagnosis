"""Caminhos do projeto (relativos à raiz do repositório)."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data/breast-ultrasound-images-DatasetNinja/ds/img"
RESULTS_PATH = ROOT / "results"
RESULTS_PATH.mkdir(exist_ok=True)
MODELS_PATH = ROOT / "models"   # modelos treinados (não vão para o git: são gerados pelo notebook 01)
MODELS_PATH.mkdir(exist_ok=True)
RAG_PATH = ROOT / "docs/rag"    # documentos médicos para o chatbot (issue #5)
