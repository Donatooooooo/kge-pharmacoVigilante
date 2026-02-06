import torch, pandas as pd
from pykeen.triples import TriplesFactory
from pykeen.hpo import hpo_pipeline
from .config import (
    TRIPLES_SHORT_TSV, HPO_TRANSE_DIR, HPO_COMPLEX_DIR,
    HPO_CONVE_DIR, HPO_RESULTS_JSON,
)

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

  def get_transe_hpo_config(self):
      return {
          'model': 'TransE',
          'training': self.training,
          'testing': self.testing,
          'validation': self.validation,
          'negative_sampler': "basic",

          'n_trials': 1,

          'save_model_directory': str(HPO_TRANSE_DIR),

          'loss': 'MarginRankingLoss',
          'loss_kwargs_ranges': dict(
              margin=dict(type=float, low=0.5, high=3.0),
          ),

          'lr_scheduler': 'ExponentialLR',
          'lr_scheduler_kwargs_ranges': dict(
              gamma=dict(type=float, low=0.85, high=0.99),
          ),

          'model_kwargs_ranges': dict(
              embedding_dim=dict(type=int, low=16, high=256, q=16),
              scoring_fct_norm=dict(type=int, low=1, high=2),
              entity_initializer=dict(type='categorical', choices=[
                  'xavier_uniform_',
                  'xavier_uniform_norm_',
                  'xavier_normal_',
                  'normal_',
              ]),
              relation_initializer=dict(type='categorical', choices=[
                  'xavier_uniform_',
                  'xavier_uniform_norm_',
                  'xavier_normal_',
                  'normal_',
              ]),
          ),

          'optimizer': "Adam",
          'optimizer_kwargs_ranges': dict(
              lr=dict(type=float, low=1e-5, high=1e-1, log=True),
              weight_decay=dict(type=float, low=1e-7, high=1e-3, log=True),
          ),

          'training_kwargs': dict(
              batch_size=256,
          ),
          'training_kwargs_ranges': dict(
              num_epochs=dict(type=int, low=100, high=1000, q=100),
          ),

          'negative_sampler_kwargs_ranges': dict(
              num_negs_per_pos=dict(type=int, low=1, high=100, log=True),
          ),

          'stopper': 'early',
          'stopper_kwargs': dict(frequency=25, patience=15, relative_delta=0.001),
          'device': self.device
      }

  def get_complex_hpo_config(self):
      return {
          'model': 'ComplEx',
          'training': self.training,
          'testing': self.testing,
          'validation': self.validation,
          'negative_sampler': "basic",

          'n_trials': 1,

          'save_model_directory': str(HPO_COMPLEX_DIR),


          'loss': 'SoftplusLoss',
          'lr_scheduler': 'ExponentialLR',
          'lr_scheduler_kwargs_ranges': dict(
              gamma=dict(type=float, low=0.9, high=0.99),
          ),

          'model_kwargs_ranges': dict(
              embedding_dim=dict(type=int, low=16, high=256, q=16),
              entity_initializer=dict(type='categorical', choices=[
                  'normal_',
                  'xavier_uniform_',
                  'xavier_normal_',
              ]),
              relation_initializer=dict(type='categorical', choices=[
                  'normal_',
                  'xavier_uniform_',
                  'xavier_normal_',
              ]),
          ),

          'regularizer': 'LpRegularizer',
          'regularizer_kwargs_ranges': dict(
              weight=dict(type=float, low=1e-6, high=1e-2, log=True),
              p=dict(type='categorical', choices=[2.0, 3.0]),
          ),

          'optimizer': "Adam",
          'optimizer_kwargs_ranges': dict(
              lr=dict(type=float, low=1e-5, high=5e-2, log=True),
              weight_decay=dict(type=float, low=1e-7, high=1e-2, log=True),
          ),

          'training_kwargs': dict(
              batch_size=256,
          ),
          'training_kwargs_ranges': dict(
              num_epochs=dict(type=int, low=100, high=1000, q=100),
          ),

          'negative_sampler_kwargs_ranges': dict(
              num_negs_per_pos=dict(type=int, low=1, high=100, log=True),
          ),

          'stopper': 'early',
          'stopper_kwargs': dict(frequency=30, patience=20, relative_delta=0.0005),

          'device': self.device
      }

  def get_conve_hpo_config(self):
      return {
          'model': 'ConvE',
          'training': self.training,
          'testing': self.testing,
          'validation': self.validation,
          'negative_sampler': "basic",

          'n_trials': 1,

          'save_model_directory': str(HPO_CONVE_DIR),

          'loss': 'BCEAfterSigmoidLoss',

          'lr_scheduler': 'StepLR',
          'lr_scheduler_kwargs_ranges': dict(
              step_size=dict(type=int, low=100, high=400),
              gamma=dict(type=float, low=0.1, high=0.9),
          ),

          'model_kwargs_ranges': dict(
              embedding_dim=dict(type=int, low=50, high=200, q=50),
              output_channels=dict(type=int, low=4, high=6, scale='power_two'),

              input_dropout=dict(type=float, low=0.0, high=0.5, q=0.1),
              feature_map_dropout=dict(type=float, low=0.0, high=0.5, q=0.1),
              output_dropout=dict(type=float, low=0.0, high=0.5, q=0.1),

              apply_batch_normalization=dict(type='categorical', choices=[True, False]),

              entity_initializer=dict(type='categorical', choices=[
                  'xavier_uniform_',
                  'xavier_normal_',
                  'normal_',
              ]),
              relation_initializer=dict(type='categorical', choices=[
                  'xavier_uniform_',
                  'xavier_normal_',
                  'normal_',
              ]),
          ),

          'optimizer': "Adam",
          'optimizer_kwargs_ranges': dict(
              lr=dict(type=float, low=1e-5, high=1e-2, log=True),
              weight_decay=dict(type=float, low=1e-7, high=1e-3, log=True),
          ),

          'training_kwargs': dict(
              batch_size=128,
          ),
          'training_kwargs_ranges': dict(
              num_epochs=dict(type=int, low=100, high=500, q=100),
          ),

          'negative_sampler_kwargs_ranges': dict(
              num_negs_per_pos=dict(type=int, low=1, high=100, log=True),
          ),

          'stopper': 'early',
          'stopper_kwargs': dict(frequency=15, patience=12, relative_delta=0.001),

          'device': self.device
      }

  def run_hpo(self):

    self.create_dataset()
    results = {}

    print("Iniziando HPO per TransE...")
    results['TransE'] = hpo_pipeline(**self.get_transe_hpo_config())
    print("TransE completato!")

    print("Iniziando HPO per ComplEx...")
    results['ComplEx'] = hpo_pipeline(**self.get_complex_hpo_config())
    print("ComplEx completato!")

    print("Iniziando HPO per ConvE...")
    results['ConvE'] = hpo_pipeline(**self.get_conve_hpo_config())
    print("ConvE completato!")

    print("\nRISULTATI FINALI:")
    import json
    serializable_results = {}
    valid_results = {}
    for model_name, result in results.items():
        completed = [t for t in result.study.trials if t.state.name == 'COMPLETE']
        if completed:
            best_score = result.study.best_trial.value
            print(f"{model_name}: Best Score = {best_score:.4f}")
            valid_results[model_name] = result
            serializable_results[model_name] = {
                'best_score': result.study.best_trial.value,
                'best_params': result.study.best_trial.params,
                'n_trials': len(result.study.trials),
            }
        else:
            print(f"{model_name}: Nessun trial completato ({len(result.study.trials)} trials falliti)")
            serializable_results[model_name] = {
                'best_score': None,
                'best_params': None,
                'n_trials': len(result.study.trials),
            }

    with open(HPO_RESULTS_JSON, "w") as f:
        json.dump(serializable_results, f, indent=4)

    if valid_results:
        best_model = max(valid_results.keys(), key=lambda x: valid_results[x].study.best_trial.value)
        print(f"\nMiglior modello: {best_model}")
    else:
        print("\nNessun modello ha completato i trial con successo.")
    return results

results = LinkPredictor(str(TRIPLES_SHORT_TSV)).run_hpo()
print(results)