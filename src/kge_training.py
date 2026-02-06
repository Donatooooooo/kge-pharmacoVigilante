import torch, pandas as pd
from pykeen.triples import TriplesFactory
from pykeen.pipeline import pipeline
from pykeen.hpo import hpo_pipeline
from .config import TRIPLES_TSV, MODELS_DIR, HPO_MODELS_DIR, TRAINED_MODEL_PKL

import logging, warnings
warnings.filterwarnings("ignore")
logging.getLogger('pykeen').setLevel(logging.ERROR)

class LinkPredictor:
    def __init__(self, tsv_path: str):
        self.tsv_path = tsv_path
        self.dataset = None
        self.model = None
        self.results = None
        
        self.triples_factory = self.load_triples()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    
    def load_triples(self, delimiter: str = '\t'):
        data = pd.read_csv(self.tsv_path, delimiter=delimiter, 
                            names=['subject', 'relation', 'object'], encoding='utf-16')
        data = data.fillna("").astype(str)
        print(data['relation'].unique())
        print(f"- Loaded {len(data)} triples")
        print(f"   ├─ Entities: {len(set(data['subject'].tolist() + data['object'].tolist()))}")
        print(f"   └─ Relations: {len(data['relation'].unique())}")

        self.triples_factory = TriplesFactory.from_labeled_triples(
            data.values,
            create_inverse_triples=True
        )
        return self.triples_factory
        
    
    def create_dataset(self, train_ratio: float = 0.8, test_ratio: float = 0.1):
        self.training, self.testing, self.validation = self.triples_factory.split([train_ratio, test_ratio])    
        return 
    
    
    def train_model(self, model_name):    
        hpo_result = hpo_pipeline(
            
            model = model_name,
            
            training=self.training,
            testing=self.testing,
            validation=self.validation,
            negative_sampler="basic",
            
            n_trials=1,
            
            save_model_directory = str(HPO_MODELS_DIR),
            loss='MarginRankingLoss',
            loss_kwargs_ranges=dict(
                margin=dict(type=float, low=1.0, high=2.0),
            ),
            
            lr_scheduler='ExponentialLR',
            lr_scheduler_kwargs_ranges=dict(
                gamma=dict(type=float, low=0.8, high=0.99),
            ),
            
            model_kwargs_ranges=dict(
                embedding_dim=dict(type=int, low=100, high=400, step=25),
                entity_initializer=dict(type='categorical', choices=[
                    'normal',
                    'zeros',
                    'uniform',
                ]),
            ),
            
            optimizer="Adam",
            optimizer_kwargs_ranges=dict(
                lr=dict(type=float, low=1e-5, high=1e-2, scale="log"),
            ),
            
            training_kwargs_ranges=dict(
                batch_size=dict(type=int, low=256, high=1024, step=256),
                num_epochs=dict(type=int, low=100, high=500, step=25),
            ),
            
            stopper='early',
            stopper_kwargs=dict(frequency=20, patience=10, relative_delta=0.002),
            
            device=self.device
        )
        
        best_params = hpo_result.study.best_params
        print(f"\n\n- Best Hyperparameters: {best_params}\n")
        print("- TRAINING ON BEST HYPERPARAMETERS\n")
        
        self.results = pipeline(
            model=model_name,
            training=self.training,
            validation=self.validation,
            testing=self.testing,
            model_kwargs={
                "embedding_dim": best_params.get("model.embedding_dim"),
                "scoring_fct_norm": best_params.get("model.scoring_fct_norm"),
                "entity_initializer": best_params.get("model.entity_initializer"),
            },
            loss=best_params.get("loss"),
            loss_kwargs={
                "margin": best_params.get("loss.margin"),
            },
            optimizer=best_params.get("optimizer"),
            optimizer_kwargs={
                "lr": best_params.get("optimizer.lr"),
            },
            training_kwargs={
                "num_epochs": best_params.get("training.num_epochs"),
                "batch_size": best_params.get("training.batch_size"),
                "use_tqdm": True,
            },
            lr_scheduler=best_params.get("lr_scheduler"),
            lr_scheduler_kwargs={
                "gamma": best_params.get("lr_scheduler.gamma"),
            },
            stopper='early',
            stopper_kwargs=dict(frequency=20, patience=10, relative_delta=0.002),
            device=self.device,
        )
        
        print("- Model Trained")
        print("  └─ Evaluation on test set:")
        print(f"      Mean Rank: {self.results.get_metric('mean_rank'):.4f}")
        print(f"      Mean Reciprocal Rank: {self.results.get_metric('mean_reciprocal_rank'):.4f}")
        print(f"      Hits@1: {self.results.get_metric('hits_at_1'):.4f}")
        print(f"      Hits@3: {self.results.get_metric('hits_at_3'):.4f}")
        print(f"      Hits@10: {self.results.get_metric('hits_at_10'):.4f}")
        
        return self.results

    def save_model(self):
        self.results.save_to_directory(str(MODELS_DIR))
        print(f"- Model saved")
    
    
    def load_model(self):
        self.model = torch.load(str(TRAINED_MODEL_PKL), weights_only=False)
        print(f"- Loaded model")


def run():
    tsv_file = str(TRIPLES_TSV)

    predictor = LinkPredictor(tsv_file)
    predictor.create_dataset(train_ratio=0.8, test_ratio=0.1)
    
    predictor.train_model(
        model_name='TransE'
    )

    predictor.save_model()

run()