from dotenv import load_dotenv

load_dotenv()


# ----------------------
# Data and graph settings
# ----------------------
DMAX = 50000
INCLUDE_SIDE_EFFECTS = False

MIN_SE_FREQ = 200
MIN_USE_FREQ = 10
MIN_CC_FREQ = 20


# ----------------------
# Labels cleaning settings
# ----------------------
UMLS_SCORE_THRESHOLD = 0.85
JACCARD_THRESHOLD = 0.5


# ----------------------
# Training settings
# ----------------------
TARGET_RELATION = "has_side_effect"
N_TRIALS = 50
MLFLOW = True
SEED = 42

PARAMS = {
    False: {
        "embedding_dim": 256,
        "margin": 9.110810283295493,
        "adversarial_temperature": 1.2254690573112397,
        "lr": 0.006559683288625212,
        "gamma": 0.9540410886033941,
        "num_negs_per_pos": 39,
        "batch_size": 512,
        "stopper_frequency": 20,
        "stopper_patience": 10,
        "save_dir": "learned_kge",
    },
    True: {
        "embedding_dim": 512,
        "margin": 6.150084600489551,
        "adversarial_temperature": 1.1219555067661364,
        "lr": 0.004072230731461761,
        "gamma": 0.9855083571393716,
        "num_negs_per_pos": 96,
        "batch_size": 1792,
        "stopper_frequency": 10,
        "stopper_patience": 5,
        "save_dir": "learned_kge_se",
    },
}


# ----------------------
# HPO settings
# ----------------------
N_TRIALS = 50


def get_rotate_hpo_config(training, testing, validation, device):
    if INCLUDE_SIDE_EFFECTS:
        margin_range = dict(type=float, low=6.0, high=30.0)
        adv_temp_range = dict(type=float, low=0.3, high=2.0)
        gamma_range = dict(type=float, low=0.95, high=0.995)
        emb_range = dict(type=int, low=128, high=512, q=64)
        lr_range = dict(type=float, low=5e-5, high=5e-3, log=True)
        batch_range = dict(type=int, low=512, high=2048, q=256)
        negs_range = dict(type=int, low=10, high=100)
    else:
        margin_range = dict(type=float, low=9.0, high=24.0)
        adv_temp_range = dict(type=float, low=0.5, high=1.5)
        gamma_range = dict(type=float, low=0.93, high=0.99)
        emb_range = dict(type=int, low=64, high=256, q=64)
        lr_range = dict(type=float, low=1e-4, high=1e-2, log=True)
        batch_range = dict(type=int, low=256, high=1024, q=256)
        negs_range = dict(type=int, low=5, high=50)

    return {
        "model": "RotatE",
        "training": training,
        "testing": testing,
        "validation": validation,
        "negative_sampler": "bernoulli",
        "n_trials": N_TRIALS,
        "save_model_directory": str(HPO_ROTATE_DIR),
        "loss": "NSSALoss",
        "loss_kwargs_ranges": dict(
            margin=margin_range,
            adversarial_temperature=adv_temp_range,
        ),
        "lr_scheduler": "ExponentialLR",
        "lr_scheduler_kwargs_ranges": dict(gamma=gamma_range),
        "model_kwargs": dict(
            entity_initializer="xavier_uniform_",
            relation_initializer="init_phases",
        ),
        "model_kwargs_ranges": dict(embedding_dim=emb_range),
        "optimizer": "Adam",
        "optimizer_kwargs_ranges": dict(lr=lr_range),
        "training_kwargs": dict(num_epochs=500, use_tqdm=True),
        "training_kwargs_ranges": dict(batch_size=batch_range),
        "negative_sampler_kwargs_ranges": dict(num_negs_per_pos=negs_range),
        "stopper": "early",
        "stopper_kwargs": dict(
            frequency=10, patience=10, relative_delta=0.0005, metric="hits_at_10"
        ),
        "device": device,
    }


# ----------------------
# Paths
# ----------------------
from pathlib import Path

PROJ_ROOT = Path(__file__).resolve().parents[1]

# Data directories
DATA_DIR = PROJ_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
REPORTS_DIR = PROJ_ROOT / "reports"

# Raw data files
RAW_MEDICINE_CSV = RAW_DIR / "medicine_dataset.csv"
RAW_COMPOSITION_CSV = RAW_DIR / "medicine_composition.csv"

# Processed data files
PROCESSED_MEDICINE_CSV = PROCESSED_DIR / "processed_medicine.csv"
SIDE_EFFETCS_MEDICINE_CSV = PROCESSED_DIR / "side_effects.csv"

# Graph files
if DMAX == 0:
    _graph_base = "graph_full"
else:
    _graph_base = f"graph_{int(DMAX / 1000)}k"

GRAPH_DIR = DATA_DIR / "graph"
TRIPLES_TSV = GRAPH_DIR / f"{_graph_base}{'_se' if INCLUDE_SIDE_EFFECTS else ''}.tsv"
TRIPLES_TSV_SE = GRAPH_DIR / f"{_graph_base}_se.tsv"
EMB_DATASET = GRAPH_DIR / "embeddings.pkl"

# Models directories
MODELS_DIR = PROJ_ROOT / "models"
HPO_ROTATE_DIR = MODELS_DIR / "rotate"

MODEL = MODELS_DIR / "learned_kge" / "trained_model.pkl"
MODEL_SE = MODELS_DIR / "learned_kge_se" / "trained_model.pkl"
XGB = MODELS_DIR / "xgboost"

# UMLS Cache
UMLS_CACHE_PATH = MODELS_DIR / "umls" / "umls_cache.json"
