from dotenv import load_dotenv
load_dotenv()

# ----------------------
# Graph settings
# ----------------------
"""
Gestisce sia la creazione del grafo che la selezione del file delle triple 
per il training. Se DMAX = 0 allora il grafo viene creato con tutto il dataset
"""
DMAX = 5000


# ----------------------
# Training settings
# ----------------------
N_TRIALS = 50
MLFLOW = True
SEED = 42


# ----------------------
# Paths
# ----------------------
from pathlib import Path
PROJ_ROOT = Path(__file__).resolve().parents[1]

# Data directories
DATA_DIR = PROJ_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
GRAPH_DIR = DATA_DIR / "graph"

REPORTS_DIR = PROJ_ROOT / "reports"

# Raw data files
RAW_MEDICINE_CSV = RAW_DIR / "medicine_dataset.csv"
RAW_COMPOSITION_CSV = RAW_DIR / "medicine_composition_dataset.csv"

# Processed data files
PROCESSED_MEDICINE_CSV = PROCESSED_DIR / "processed_medicine.csv"
SIDE_EFFETCS_MEDICINE_CSV = PROCESSED_DIR / "side_effects.csv"

if DMAX == 0:
    TRIPLES_TSV = GRAPH_DIR / f"triples_full.tsv"
else:
    TRIPLES_TSV = GRAPH_DIR / f"triples_{int(DMAX / 1000)}k.tsv"

# Models directories
MODELS_DIR = PROJ_ROOT / "models"
HPO_MODELS_DIR = MODELS_DIR / "hpo_models"
HPO_TRANSE_DIR = HPO_MODELS_DIR / "transe"
HPO_COMPLEX_DIR = HPO_MODELS_DIR / "complex"
HPO_ROTATE_DIR = HPO_MODELS_DIR / "rotate"
TRAINED_MODEL_PKL = MODELS_DIR / "trained_model.pkl"

# Results
HPO_RESULTS_JSON = REPORTS_DIR / "hpo_results_summary.json"
