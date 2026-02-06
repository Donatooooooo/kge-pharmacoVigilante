# ----------------------
# Build graph
# ----------------------
DMAX = 1   # numero di medicinali con cui creare il grafo

# ----------------------
# Paths
# ----------------------
from pathlib import Path

PROJ_ROOT = Path(__file__).resolve().parents[1]

# Data directories
DATA_DIR = PROJ_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

# Raw data files
RAW_MEDICINE_CSV = RAW_DIR / "medicine_dataset.csv"
RAW_COMPOSITION_CSV = RAW_DIR / "medicine_composition_dataset.csv"

# Processed data files
PROCESSED_MEDICINE_CSV = DATA_DIR / "processed_medicine.csv"
TRIPLES_TSV = DATA_DIR / "triples.tsv"

# Models directories
MODELS_DIR = PROJ_ROOT / "models"
HPO_MODELS_DIR = MODELS_DIR / "hpo_models"
HPO_TRANSE_DIR = HPO_MODELS_DIR / "transe"
HPO_COMPLEX_DIR = HPO_MODELS_DIR / "complex"
HPO_CONVE_DIR = HPO_MODELS_DIR / "conve"
TRAINED_MODEL_PKL = MODELS_DIR / "trained_model.pkl"

# Results
HPO_RESULTS_JSON = PROJ_ROOT / "hpo_results_summary.json"
