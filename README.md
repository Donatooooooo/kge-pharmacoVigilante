# ML-pykeen-template
This name will be changed

## Info
### Codice
- `hpo.py` (Hyperparams Pipeline Optimization): contiene i tre modelli e la definizione degli iperparamteri (tipo gridSearch) da ottimizzare. Va cambiato il valore `n_trials = 1`, messo solo per vedere se funzionasse il tutto. Valori ottimali per `n_trials` sarebbero [50, 80].

- `kge_training.py`: addestra un modello TransE (il più semplice) trovando i miglior params (anche in questo caso `n_trials = 1`).

### Dati
I dataset sono due e sono complementari. Difatti il dataset finale è il merge di entrambi per avere più info per creare relazioni significative. Da questo vengono estratte le triple. Per ora non teniamo in considerazione i side effect ma come puoi vedere ci sono.

Attualmente le triple sono 23 (solo per taestare che tutto funzionasse). Elaborando un solo medicinale (il numero da elaborare per creare il grafo è indicato in `config.py` con la costante `DMAX`) vengono create quelle relazioni che vedi. In generale possono essere definite altre relazioni, c'è da capire un po'.  

## Idea formalizzata da Claude
---
# Side Effect Prediction via Knowledge Graph Embeddings

## Obiettivo

Predire gli effetti collaterali dei farmaci utilizzando embeddings appresi da un Knowledge Graph farmaceutico. Gli effetti collaterali sono **esclusi dal grafo**: il modello deve inferirli esclusivamente dalla struttura relazionale (composizione, classe terapeutica, sostituibilità).

---

## Formulazione del Problema

**Input:** embedding $\mathbf{e}_d \in \mathbb{R}^k$ di un farmaco $d$, ottenuto da un modello KGE allenato sul grafo.

**Output:** vettore binario $\mathbf{y}_d \in \{0, 1\}^C$, dove $C$ è il numero di effetti collaterali distinti e $y_{d,c} = 1$ se il farmaco $d$ causa l'effetto collaterale $c$.

**Tipo di task:** Multi-Label Classification.

---

## Pipeline

### Fase 1 — Costruzione del Knowledge Graph

Generazione delle triple dal dataset `processed_medicine.csv`, utilizzando le seguenti relazioni:

| Relazione                    | Soggetto          | Oggetto                          |
|------------------------------|-------------------|----------------------------------|
| `has_form`                   | farmaco           | forma farmaceutica               |
| `has_route`                  | farmaco           | via di somministrazione          |
| `composed_by`                | farmaco           | principio attivo                 |
| `composed_by_at_dose`        | farmaco           | principio attivo + dosaggio      |
| `has_therapeutic_class`      | farmaco           | classe terapeutica               |
| `has_substitute`             | farmaco           | farmaco sostitutivo              |

> **Nota:** gli effetti collaterali NON sono inseriti nel grafo. Sono usati solo come label per il task downstream.

### Fase 2 — HPO e Training dei Modelli KGE

Quattro modelli candidati allenati sullo stesso grafo:

| Modello  | Spazio         | Scoring Function                                  |
|----------|----------------|---------------------------------------------------|
| TransE   | Reale          | $-\|\mathbf{h} + \mathbf{r} - \mathbf{t}\|$      |
| RotatE   | Complesso      | $-\|\mathbf{h} \circ \mathbf{r} - \mathbf{t}\|$  |
| ComplEx  | Complesso      | $\text{Re}(\langle \mathbf{h}, \mathbf{r}, \bar{\mathbf{t}} \rangle)$ |
| ConvE    | Reale (conv.)  | $f(\text{vec}(\mathbf{h}, \mathbf{r}) * \omega) \cdot \mathbf{t}$ |

**Libreria:** PyKEEN  
**HPO:** Optuna (con pruning)  
**Iperparametri ottimizzati:** embedding dimension, learning rate, batch size, num epochs, loss function, regularizer, negative sampler.

**Metriche di selezione KGE:**

| Metrica     | Descrizione                                             |
|-------------|---------------------------------------------------------|
| MRR         | Mean Reciprocal Rank (metrica primaria)                 |
| Hits@1      | % di triple con entità corretta al primo posto          |
| Hits@3      | % di triple con entità corretta nei primi 3             |
| Hits@10     | % di triple con entità corretta nei primi 10            |

**Tracking:** MLflow / DagsHub.

### Fase 3 — Estrazione degli Embeddings

Dal modello KGE migliore (per MRR):

1. Estrarre la matrice degli entity embeddings $\mathbf{E} \in \mathbb{R}^{N \times k}$
2. Filtrare solo le entità che corrispondono a farmaci (escludendo dosaggi, forme, classi)
3. Associare a ciascun farmaco $d$ il suo embedding $\mathbf{e}_d$ e il vettore di label $\mathbf{y}_d$

**Risultato:** dataset tabellare $\mathcal{D} = \{(\mathbf{e}_d, \mathbf{y}_d)\}_{d=1}^{D}$, dove $D$ è il numero di farmaci.

### Fase 4 — Multi-Label Classification

**Split:** stratificato per farmaco (non per tripla), 80/20.

**Classificatori candidati:**

| Modello              | Strategia multi-label     |
|----------------------|---------------------------|
| Logistic Regression  | OneVsRest / ClassifierChain |
| Random Forest        | OneVsRest                 |
| MLP                  | Binary Cross-Entropy      |
| XGBoost              | OneVsRest                 |

**Metriche di valutazione:**

| Metrica             | Descrizione                                                |
|---------------------|------------------------------------------------------------|
| Hamming Loss        | Frazione media di label predette erroneamente              |
| F1 Micro            | F1 globale su tutte le coppie (farmaco, effetto)           |
| F1 Macro            | F1 medio per classe (bilancia classi rare)                 |
| F1 per label        | Performance su ciascun effetto collaterale                 |
| AUC-ROC (micro)     | Capacità discriminativa globale                            |
| Precision@K         | Precisione sui top-K effetti predetti                      |

---

## Struttura del Progetto

```
drugs-KGE/
├── data/
│   ├── processed_medicine.csv
│   ├── triples.tsv
│   └── side_effects.csv           # label per la classificazione
├── src/
│   ├── triple_generator.py        # Fase 1
│   ├── hpo_pipeline.py            # Fase 2
│   ├── embedding_extractor.py     # Fase 3
│   └── classifier.py              # Fase 4
├── notebooks/
│   ├── eda.ipynb
│   └── results_analysis.ipynb
├── models/                         # checkpoint dei modelli
├── mlruns/                         # tracking MLflow
└── visualize_pyvis.py
```

---

## Ipotesi da Validare

1. **Gli embeddings KGE catturano informazione sufficiente per predire effetti collaterali**, nonostante questi non siano mai stati osservati dal grafo.
2. **Farmaci con composizioni e classi terapeutiche simili hanno embeddings vicini**, e condividono effetti collaterali.
3. **La relazione `has_substitute` codifica una similarità farmacologica** che implica profili di effetti collaterali sovrapposti.

---

## Analisi dello Spazio di Ricerca HPO

### Dati: statistiche del Knowledge Graph di ricerca

| Metrica | Valore |
|---|---|
| Triple totali | 32,767 |
| Soggetti unici (farmaci) | 3,081 |
| Oggetti unici | 6,851 |
| Relazioni uniche | 6 |

**Distribuzione delle relazioni:**

| Relazione | Conteggio | % |
|---|---|---|
| `has_substitute` | 12,833 | 39% |
| `composed_by_at_dose` | 5,650 | 17% |
| `composed_by` | 4,761 | 15% |
| `has_form` | 3,218 | 10% |
| `has_route` | 3,190 | 10% |
| `has_therapeutic_class` | 3,115 | 9% |

### Spazio di ricerca attuale

#### TransE

| Parametro | Range | Motivazione |
|---|---|---|
| `embedding_dim` | {32, 64, 96, 128} | Controlla l'espressivita delle rappresentazioni. Con ~9K entita e 6 relazioni, 32-128 bilancia capacita e rischio overfitting |
| `scoring_fct_norm` | {1, 2} | Specifico di TransE: scelta tra norma L1 e L2 nella scoring function $\|\|h + r - t\|\|_p$ |
| `margin` (MarginRankingLoss) | [0.5, 1.5] | Definisce la separazione minima tra score positivi e negativi. Critico per la margin-based loss |
| `lr` | [1e-4, 1e-2] log | Learning rate dell'ottimizzatore Adam |
| `gamma` (ExponentialLR) | [0.90, 0.99] | Decay del learning rate per epoca |
| `num_negs_per_pos` | [1, 30] log | Numero di triple negative generate per ogni positiva |

#### ComplEx

| Parametro | Range | Motivazione |
|---|---|---|
| `embedding_dim` | {32, 64, 96, 128} | Come TransE |
| `regularizer_weight` (L2) | [1e-5, 1e-3] log | Specifico di ComplEx: opera in spazio complesso ed e molto incline all'overfitting. Il paper originale enfatizza la necessita di regolarizzazione |
| `lr` | [1e-4, 1e-2] log | Learning rate |
| `gamma` (ExponentialLR) | [0.93, 0.99] | Decay del learning rate |
| `num_negs_per_pos` | [1, 30] log | Negative sampling |
| Loss: `SoftplusLoss` | fissa | Versione smooth della logistic loss, standard per ComplEx |

#### RotatE

| Parametro | Range | Motivazione |
|---|---|---|
| `embedding_dim` | {32, 64, 96, 128} | Come TransE |
| `margin` (NSSALoss) | [6.0, 24.0] | Specifico di RotatE: il paper propone la self-adversarial negative sampling loss. Il margine e piu alto rispetto a TransE per la diversa scala dello scoring |
| `adversarial_temperature` | [0.5, 2.0] | Specifico di RotatE/NSSA: controlla il peso dato ai negativi piu difficili. Temperatura alta = focus su hard negatives |
| `lr` | [1e-4, 1e-2] log | Learning rate |
| `gamma` (ExponentialLR) | [0.93, 0.99] | Decay del learning rate |
| `num_negs_per_pos` | [1, 30] log | Negative sampling |

### Possibili estensioni dello spazio di ricerca

Le seguenti estensioni sono motivate dalle caratteristiche specifiche del grafo (forte asimmetria relazionale, ~9K entita, 6 relazioni).

| # | Modifica | Modelli | Motivazione |
|---|---|---|---|
| 1 | **Bernoulli negative sampler** (al posto di `basic`) | Tutti | Le relazioni hanno arita molto diverse: `has_substitute` e many-to-many, `has_therapeutic_class` e many-to-few. Il sampler bernoulli corrompe il lato corretto della tripla in base alla distribuzione head/tail di ciascuna relazione |
| 2 | **N3 regularization** (`p=3` invece di `p=2`) | ComplEx | La regolarizzazione con norma nucleare (Lacroix et al., 2018) funziona meglio di L2 per i modelli tensoriali come ComplEx |
| 3 | **`num_negs_per_pos`**: estendere a [1, 100] | Tutti | Con ~9K entita lo spazio dei possibili negativi e ampio. L'attuale range (1-30) e conservativo, specialmente per `has_substitute` (39% delle triple) |
| 4 | **`batch_size`** come iperparametro: {128, 256, 512} | Tutti | Con 32K triple, la dimensione del batch influenza la qualita dei gradienti e la generalizzazione |
| 5 | **`embedding_dim`**: estendere a {32, 64, ..., 256} | Tutti | ~9K entita e un grafo non banale. Dimensioni piu alte possono aiutare a distinguere farmaci simili (es. sostituti) |
| 6 | **`entity_initializer`** per RotatE | RotatE | L'inizializzazione uniforme delle fasi e preferibile per le rotazioni in spazio complesso (coerenza con il paper originale) |

### Stima dell'impatto sui tempi di ricerca

Ipotesi base: **6 ore** con lo spazio attuale.

| Fattore | Impatto |
|---|---|
| `embedding_dim` fino a 256 | Media della dimensione campionata passa da ~80 a ~144. Costo computazionale per trial: **~1.4x** |
| `num_negs_per_pos` fino a 100 | Media geometrica passa da ~5.5 a ~10. Costo per trial: **~1.3x** |
| `batch_size` variabile | Effetto neutro (batch piu piccoli = piu step ma piu leggeri) |
| Bernoulli / N3 / initializer | Overhead trascurabile |

**Costo medio per trial con tutte le estensioni: ~1.5x**

| Scenario | N_TRIALS | Tempo stimato | Note |
|---|---|---|---|
| Attuale | N | ~6 ore | Baseline |
| Esteso, stessi trial | N | **~9 ore** | +50% dal costo per trial |
| Esteso, trial aumentati per coprire lo spazio | ~1.5-1.8N | **~12-15 ore** | Raccomandato per esplorare adeguatamente lo spazio ampliato |

> Lo spazio di ricerca cresce di ~1.5-1.8x in volume (nuove dimensioni + range estesi). Mantenere lo stesso numero di trial significa accettare una copertura meno densa. Per un'esplorazione equivalente servirebbero circa il 50-80% di trial in piu.

---

## Baseline

Per contestualizzare i risultati, confrontare con:

- **Random baseline:** predizione casuale rispettando la distribuzione delle label.
- **Majority baseline:** predire gli effetti collaterali più frequenti per tutti i farmaci.
- **Feature-based:** usare feature categoriali (classe terapeutica, principi attivi) direttamente come one-hot encoding, senza embeddings.
