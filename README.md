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

## Baseline

Per contestualizzare i risultati, confrontare con:

- **Random baseline:** predizione casuale rispettando la distribuzione delle label.
- **Majority baseline:** predire gli effetti collaterali più frequenti per tutti i farmaci.
- **Feature-based:** usare feature categoriali (classe terapeutica, principi attivi) direttamente come one-hot encoding, senza embeddings.
