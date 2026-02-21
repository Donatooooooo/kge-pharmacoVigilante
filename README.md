# Side Effect Ranking via Knowledge Graph Embeddings

## Pitch

Migliaia di farmaci nel nostro Knowledge Graph hanno profili di effetti collaterali incompleti: nuovi side effects emergono anni dopo l'immissione in commercio attraverso segnalazioni spontanee (FAERS, EudraVigilance, sistema AIFA). Il nostro sistema produce un **ranking ordinato dei possibili effetti collaterali** per ciascun farmaco, permettendo un monitoraggio mirato nella farmacovigilanza.

**Tre approcci confrontati:**

1. **Baseline — Marginal Frequency:** assegna a ogni farmaco lo stesso ranking, basato sulla frequenza marginale di ciascun effetto collaterale nel training set. Non utilizza alcuna informazione specifica del farmaco: serve come lower bound per quantificare quanto valore aggiungano gli embeddings.

2. **Approccio A — KGE con side effects nel grafo (link prediction diretta):** gli effetti collaterali sono inclusi nel Knowledge Graph come relazione `has_side_effect`. Il modello KGE apprende direttamente la plausibilità di triple `(farmaco, has_side_effect, effetto)`. Lo split è strategico: il training contiene tutte le relazioni, mentre validation e test contengono **solo** triple `has_side_effect`. Questo valuta la capacità del modello di predire nuovi link side effect sfruttando la struttura complessiva del grafo.

3. **Approccio B — KGE senza side effects + XGBoost per ranking:** gli effetti collaterali sono **esclusi** dal grafo. Il modello KGE apprende embeddings dalla sola struttura relazionale (composizione, classe terapeutica, usi, sostituti, etc.). Gli embeddings dei farmaci vengono poi estratti e usati come feature per un classificatore XGBoost OneVsRest, le cui probabilità producono il ranking finale. Questo valuta se la struttura relazionale *da sola* cattura informazione sufficiente per inferire effetti collaterali mai osservati nel grafo.

**Valore applicativo:**
- Le segnalazioni post-marketing sono lente e incomplete
- Un sistema di pre-screening automatico riduce i tempi di rilevamento
- Qualsiasi filtro che prioritizza cosa monitorare ha valore economico in farmacovigilanza

**Limite:** il modello suggerisce candidati da indagare, non certifica effetti collaterali. E' un sistema di ranking/alert, non un sostituto della validazione clinica.

---

## EDA

#### Dati raw

| Metrica | Valore |
|---|---|
| Righe totali (merge) | 256,726 |
| Nomi farmaci unici | 221,415 |
| Duplicati per `name` | 35,311 (dovuti a `pack_size_label` diversi, stessi SE) |
| Side effects unici | 1,050 |
| Media SE per farmaco | 6.5 |
| Mediana SE per farmaco | 6 |
| Max SE per farmaco | 42 |
| SE con < 10 farmaci | 269 (25.6%) |
| SE in 1 solo farmaco | 81 |
| Occorrenze "No common side effects seen" | 2,809 |
| Classi terapeutiche | 14 |
| Farmaci con `Therapeutic Class` NaN | 79 |
| Farmaci con 2 principi attivi | 43.7% |

**Distribuzione percentili SE per farmaco:** P10=3, P25=4, P50=6, P75=8, P90=11, P95=14, P99=25

**Soglie di frequenza SE:**

| Soglia min farmaci | SE rimanenti | % del totale |
|---|---|---|
| >= 50 | 561 | 53% |
| >= 100 | 456 | 43% |
| >= 200 | 356 | 34% |
| >= 500 | 257 | 24% |
| >= 1000 | 180 | 17% |

---

## Formulazione del Problema

### Definizione generale

Dato un farmaco $d$ e un insieme di $C$ effetti collaterali noti nel dominio, il task consiste nel produrre un **ranking** $\pi_d$ degli effetti collaterali ordinati per plausibilità decrescente.

**Input:**
- **Approccio A:** triple del Knowledge Graph contenenti `has_side_effect` $\rightarrow$ scoring diretto $f(d, \texttt{has\_side\_effect}, s_c)$ per ogni side effect $s_c$
- **Approccio B:** embedding $\mathbf{e}_d \in \mathbb{R}^k$ del farmaco $d$, ottenuto da un KGE addestrato su un grafo *senza* side effects $\rightarrow$ probabilità $P(y_{d,c} = 1 \mid \mathbf{e}_d)$ da un classificatore downstream

**Output:** ranking $\pi_d = (\pi_d(1), \pi_d(2), \ldots, \pi_d(C))$ dove $\pi_d(j)$ è la posizione del $j$-esimo effetto collaterale.

**Tipo di task:** Learning to Rank (per-query ranking dove ogni farmaco è una query e i side effects sono i documenti da ordinare).

### Metriche di valutazione

| Metrica | Formula | Descrizione |
|---|---|---|
| **MR** (Mean Rank) | $\frac{1}{\|Q\|} \sum_{q} \text{rank}(q)$ | Rank medio dei veri positivi. Più basso = meglio |
| **MRR** (Mean Reciprocal Rank) | $\frac{1}{\|Q\|} \sum_{q} \frac{1}{\text{rank}^*(q)}$ | Media del reciproco del rank del miglior TP per query. Più alto = meglio (metrica primaria) |
| **Hits@K** | $\frac{1}{\|Q\|} \sum_{q} \mathbb{1}[\text{rank}^*(q) \leq K]$ | Frazione di query con almeno un TP nei top-K |

Dove $\text{rank}^*(q)$ indica il rank del vero positivo con rank migliore per la query $q$.

---

## Pipeline

### Fase 0 — Preprocessing e Label Cleaning

Le label degli effetti collaterali nei dati raw contengono ridondanze e sinonimi (es. *hives* vs *urticaria*, *allergy* vs *allergic reaction*). La normalizzazione avviene in due passaggi:

1. **Risoluzione UMLS** (Bodenreider, 2004): ogni label viene mappata al Concept Unique Identifier (CUI) più vicino tramite scispaCy (Neumann et al., 2019) con entity linking su UMLS. Soglia di accettazione: score $\geq 0.85$. Le label che risolvono allo stesso CUI vengono fuse sotto il nome canonico UMLS.

2. **Merge Jaccard:** le label non risolte da UMLS vengono confrontate per similarità lessicale (tokenizzazione con lemmatizzazione, rimozione stopwords, sostituzione sinonimi). Coppie con Jaccard $\geq 0.5$ vengono fuse, con precedenza alla label con supporto maggiore. Il merge rispetta vincoli semantici: coppie contenenti antonimi (es. *hyper-/hypo-*) o appartenenti a gruppi bloccati non vengono mai fuse.

**Filtraggio frequenza:** solo side effects con $\geq$ `MIN_SE_FREQ` occorrenze vengono mantenuti (default: 200).

### Fase 1 — Costruzione del Knowledge Graph

| Relazione | Soggetto | Oggetto |
|---|---|---|
| `has_form` | farmaco | forma farmaceutica |
| `has_route` | farmaco | via di somministrazione |
| `composed_by` | farmaco | principio attivo |
| `has_dose_of` | farmaco | principio attivo + dosaggio |
| `has_therapeutic_class` | farmaco | classe terapeutica |
| `has_use` | farmaco | indicazione terapeutica |
| `has_substitute` | farmaco | farmaco sostitutivo |
| `has_chemical_class` | farmaco | classe chimica |
| `has_side_effect` | farmaco | effetto collaterale *(solo Approccio A)* |

**Inverse triples:** abilitate nel training del KGE (`create_inverse_triples=True`) per permettere al modello di apprendere pattern bidirezionali.

### Fase 2 — HPO di RotatE KGE

Tre modelli candidati:

| Modello | Spazio | Scoring Function | Riferimento |
|---|---|---|---|
| RotatE | $\mathbb{C}^k$ | $-\|\mathbf{h} \circ \mathbf{r} - \mathbf{t}\|$ | Sun et al., 2019 |

**Libreria:** PyKEEN (Ali et al., 2021)
**HPO:** Optuna con early stopping

**Split dei dati:**
- **Senza side effects (Approccio B):** split random 60/20/20 su tutte le triple
- **Con side effects (Approccio A):** split strategico — tutte le relazioni non-SE vanno nel training, le triple `has_side_effect` vengono splittate 60/20/20 tra train/validation/test. Validation e test contengono *solo* triple SE.

### Fase 3A — Link Prediction Diretta (Approccio A)

Quando `INCLUDE_SIDE_EFFECTS=True`, il modello KGE predice direttamente la plausibilità di `(farmaco, has_side_effect, effetto)`. Il ranking per un farmaco $d$ è dato dagli score:

$$\pi_d(c) = \text{rank}(f(d, \texttt{has\_side\_effect}, s_c))$$

Non è necessario un classificatore downstream: il KGE produce il ranking end-to-end.

### Fase 3B — Embeddings + XGBoost (Approccio B)

Quando `INCLUDE_SIDE_EFFECTS=False`:

1. **Estrazione embeddings:** dal modello KGE migliore (per MRR), si estrae la matrice $\mathbf{E} \in \mathbb{R}^{N \times k}$ (o $\mathbb{R}^{N \times 2k}$ per modelli complessi, concatenando parte reale e immaginaria). Si filtrano solo le entità-farmaco.

2. **Dataset tabellare:** $\mathcal{D} = \{(\mathbf{e}_d, \mathbf{y}_d)\}_{d=1}^{D}$, dove $\mathbf{y}_d \in \{0,1\}^C$ è il vettore multi-label dei side effects.

3. **XGBoost OVR** (Chen & Guestrin, 2016): un classificatore binario per ogni side effect. Le probabilità $P(y_{d,c} = 1 \mid \mathbf{e}_d)$ definiscono il ranking.

4. **Validazione:** K-Fold CV (default: 10 fold). Le probabilità out-of-fold vengono raccolte per l'intera matrice, garantendo che la valutazione avvenga su dati mai visti in training.

### Fase 4 — Valutazione

**Metriche globali:** MR, MRR, Hits@1, Hits@3, Hits@5, Hits@10 calcolate sull'intera matrice drug $\times$ side effect.

**Valutazione stratificata per frequenza:**

| Strato | Range occorrenze | Scopo |
|---|---|---|
| Rare | 1–10 | Effetti collaterali con pochissimi esempi nel dataset |
| Uncommon | 11–50 | Frequenza bassa ma non trascurabile |
| Moderate | 51–200 | Frequenza media |
| Common | > 200 | Effetti collaterali frequenti |

La stratificazione è cruciale per misurare se gli embeddings aggiungono valore *oltre* la semplice frequenza: se XGBoost supera la baseline soprattutto sugli strati rari, significa che il KGE ha catturato pattern strutturali non riducibili a statistica.

---

## Ipotesi da Validare

1. **Gli embeddings KGE catturano informazione sufficiente per rankare effetti collaterali**, nonostante questi non siano mai stati osservati dal grafo (Approccio B).
2. **La link prediction diretta con SE nel grafo produce ranking migliori** rispetto all'approccio indiretto via embeddings + XGBoost (Approccio A vs B).
3. **Farmaci con composizioni e classi terapeutiche simili hanno embeddings vicini** e condividono profili di effetti collaterali.
4. **La relazione `has_substitute` codifica una similarità farmacologica** che implica profili di effetti collaterali sovrapposti.
5. **Il vantaggio degli embeddings è più marcato sugli effetti collaterali rari**, dove la baseline frequentista fallisce.

---

## Spazio di Ricerca HPO

### Statistiche del Knowledge Graph (graph_50k.tsv, senza side effects)
| Relation | Count | % |
|----------|------:|------:|
| has_dose_of | 52912 | 14.09% |
| has_substitute | 50614 | 13.48% |
| has_form | 46353 | 12.34% |
| has_route | 45868 | 12.22% |
| composed_by | 45012 | 11.99% |
| has_use | 44945 | 11.97% |
| has_chemical_class | 44921 | 11.96% |
| has_therapeutic_class | 44866 | 11.95% |

### Statistiche del knowledge graph (graph_50k_se.tsv, include side effects)
| Relation | Count | % |
|----------|------:|------:|
| has_side_effect | 260785 | 40.99% |
| has_dose_of | 52912 | 8.32% |
| has_substitute | 50614 | 7.95% |
| has_form | 46353 | 7.29% |
| has_route | 45868 | 7.21% |
| composed_by | 45012 | 7.07% |
| has_use | 44945 | 7.06% |
| has_chemical_class | 44921 | 7.06% |
| has_therapeutic_class | 44866 | 7.05% |

### Parametri RotatE (Sun et al., 2019)

- **Negative sampler:** Bernoulli
- **Entity initializer:** `xavier_uniform_`
- **Optimizer:** Adam
- **Num epochs:** 500 (fisso, con early stopping)
- **Inverse triples:** abilitate (`create_inverse_triples=True`)

Lo spazio di ricerca varia in base alla modalità (`INCLUDE_SIDE_EFFECTS`). Con side effects nel grafo, il grafo è significativamente più grande e lo spazio è ampliato (dimensioni più alte, batch più grandi, range più ampi) per gestire la complessità aggiuntiva.

**Scoring:** $f(h, r, t) = -\|\mathbf{h} \circ \mathbf{r} - \mathbf{t}\|$
**Loss:** NSSALoss (self-adversarial negative sampling)
**Relation initializer:** `init_phases`

#### Senza side effects

Early stopping: `frequency=15`, `patience=10`, `relative_delta=0.001`

| Parametro | Range | Tipo |
|---|---|---|
| `embedding_dim` | {64, 128, 192, 256} | categorico (step 64) |
| `margin` (NSSALoss) | [9.0, 24.0] | continuo |
| `adversarial_temperature` | [0.5, 1.5] | continuo |
| `lr` | [1e-4, 1e-2] | log-uniform |
| `gamma` (ExponentialLR) | [0.93, 0.99] | continuo |
| `batch_size` | {256, 512, 768, 1024} | categorico (step 256) |
| `num_negs_per_pos` | [5, 50] | intero |

Best params trovati:
| Categoria | Parametro | Valore |
|-----------|-----------|--------|
| **Modello** | model | RotatE |
| | embedding_dim | 256 |
| | entity_initializer | xavier_uniform_ |
| | relation_initializer | init_phases |
| **Loss** | loss | NSSALoss |
| | margin | 9.1108 |
| | adversarial_temperature | 1.2255 |
| **Negative Sampler** | negative_sampler | bernoulli |
| | num_negs_per_pos | 39 |
| **Optimizer** | optimizer | Adam |
| | lr | 0.00656 |
| **LR Scheduler** | lr_scheduler | ExponentialLR |
| | gamma | 0.9540 |
| **Training** | num_epochs | 500 |
| | batch_size | 512 |
| | use_tqdm | True |
| **Early Stopping** | stopper | early |
| | frequency | 20 |
| | patience | 10 |
| | relative_delta | 0.0005 |
| **Dataset Split** | train_ratio | 0.7 |
| | val_ratio | 0.1 |
| | test_ratio | 0.2 |
| **Triples** | create_inverse_triples | True |

final model trained on 350+k triples. Trained for 220 epochs.
#### Con side effects

Early stopping: `frequency=15`, `patience=10`, `relative_delta=0.0005`

| Parametro | Range | Tipo |
|---|---|---|
| `embedding_dim` | {128, 192, 256, ..., 512} | categorico (step 64) |
| `margin` (NSSALoss) | [6.0, 30.0] | continuo |
| `adversarial_temperature` | [0.3, 2.0] | continuo |
| `lr` | [5e-5, 5e-3] | log-uniform |
| `gamma` (ExponentialLR) | [0.95, 0.995] | continuo |
| `batch_size` | {512, 768, 1024, ..., 2048} | categorico (step 256) |
| `num_negs_per_pos` | [10, 100] | intero |

Best Params Trovati:
| Categoria | Parametro | Valore |
|-----------|-----------|--------|
| **Modello** | model | RotatE |
| | embedding_dim | 512 |
| | entity_initializer | xavier_uniform_ |
| | relation_initializer | init_phases |
| **Loss** | loss | NSSALoss |
| | margin | 6.1501 |
| | adversarial_temperature | 1.1220 |
| **Negative Sampler** | negative_sampler | bernoulli |
| | num_negs_per_pos | 96 |
| **Optimizer** | optimizer | Adam |
| | lr | 0.00407 |
| **LR Scheduler** | lr_scheduler | ExponentialLR |
| | gamma | 0.9855 |
| **Training** | num_epochs | 500 |
| | batch_size | 1792 |
| | use_tqdm | True |
| **Early Stopping** | stopper | early |
| | frequency | 20 |
| | patience | 10 |
| | relative_delta | 0.0005 |
| **Strategic Split** | train | tutte le relazioni + 70% SE |
| | val | 10% solo has_side_effect |
| | test | 20% solo has_side_effect |
| **Triples** | create_inverse_triples | True |
| **Output** | save_directory | models/learned_kge_se |
---
## Test stratificato
## Strategia di Valutazione

### Definizione del Task

Il task consiste nel produrre, per ciascun farmaco, un ranking degli effetti collaterali ordinati per plausibilità decrescente. Un sistema efficace posiziona gli effetti collaterali realmente associati al farmaco nelle posizioni più alte del ranking.

### Partizionamento dei Dati

Sia $\mathcal{D} = \{(d_i, s_j) : \text{il farmaco } d_i \text{ causa l'effetto collaterale } s_j\}$ l'insieme delle associazioni farmaco-effetto collaterale note. Partizioniamo $\mathcal{D}$ in tre sottoinsiemi disgiunti mediante campionamento casuale con seed fisso:

- $\mathcal{D}_{train}$ (70%): associazioni utilizzate per l'addestramento
- $\mathcal{D}_{val}$ (10%): associazioni utilizzate per early stopping e selezione degli iperparametri
- $\mathcal{D}_{test}$ (20%): associazioni riservate esclusivamente alla valutazione finale

È importante sottolineare che il partizionamento avviene a livello di singole associazioni, non a livello di farmaci. Questo significa che uno stesso farmaco può avere alcune delle sue associazioni nel training set e altre nel test set. Tutti i modelli osservano tutti i farmaci durante l'addestramento, ma per ciascun farmaco alcune associazioni con effetti collaterali vengono nascoste e riservate alla valutazione.

### Addestramento dei Modelli

**Baseline (Frequenza Marginale):** Per ciascun effetto collaterale $s_j$, calcoliamo la sua frequenza nel training set, ovvero la proporzione di farmaci associati a $s_j$ in $\mathcal{D}_{train}$. Il ranking per ogni farmaco è identico e corrisponde all'ordinamento degli effetti collaterali per frequenza decrescente. Questa baseline non utilizza alcuna informazione specifica del farmaco.

**Knowledge Graph Embedding con Link Prediction (KGE_SE):** Il grafo di conoscenza include le relazioni strutturali (composizione, classe terapeutica, indicazioni, sostituibilità) per tutti i farmaci, più le associazioni farmaco-effetto collaterale presenti in $\mathcal{D}_{train}$. Il modello apprende embeddings per tutte le entità ottimizzando la plausibilità delle triple osservate. Per produrre il ranking, calcoliamo lo score $f(d_i, \texttt{has\_side\_effect}, s_j)$ per ogni coppia farmaco-effetto collaterale e ordiniamo per score decrescente.

**Embeddings + Classificatore (XGBoost):** Gli embeddings dei farmaci vengono estratti da un modello KGE addestrato sul grafo strutturale senza le relazioni di effetto collaterale. Questi embeddings costituiscono le feature di input per un classificatore XGBoost multi-label. La matrice delle label utilizzata per l'addestramento contiene esclusivamente le associazioni presenti in $\mathcal{D}_{train}$. L'addestramento avviene mediante K-fold cross-validation, dove in ciascun fold un sottoinsieme di farmaci viene escluso dal training e utilizzato per generare predizioni out-of-fold. Al termine della procedura, ogni farmaco dispone di predizioni generate da un modello che non lo ha osservato durante l'addestramento di quel fold. Il ranking è dato dall'ordinamento delle probabilità predette per ciascun effetto collaterale.

### Procedura di Valutazione

Per ciascun farmaco $d_i$, ogni modello produce uno score di plausibilità per tutti gli effetti collaterali nel dominio, generando un ranking completo. Sia $\pi_{d_i}(s_j)$ la posizione dell'effetto collaterale $s_j$ nel ranking prodotto per il farmaco $d_i$, dove posizione 1 indica il più plausibile.

La valutazione considera esclusivamente le associazioni in $\mathcal{D}_{test}$. Per ciascuna associazione $(d_i, s_j) \in \mathcal{D}_{test}$, estraiamo la posizione $\pi_{d_i}(s_j)$ dal ranking del rispettivo modello. Le metriche aggregate sono:

**Mean Rank (MR):** Media aritmetica delle posizioni delle associazioni positive nel test set. Valori più bassi indicano performance migliori.

**Mean Reciprocal Rank (MRR):** Media dei reciproci delle posizioni. Più sensibile alle posizioni alte del ranking. Valori più alti indicano performance migliori.

**Hits@K:** Proporzione di associazioni positive nel test set che compaiono entro le prime K posizioni del ranking.

### Valutazione Stratificata

Per analizzare le performance al variare della disponibilità di esempi durante l'addestramento, stratifichiamo gli effetti collaterali in base alla loro frequenza in $\mathcal{D}_{train}$:

- Rari: 1-10 associazioni nel training
- Non comuni: 11-50 associazioni
- Moderati: 51-200 associazioni
- Comuni: oltre 200 associazioni

Le metriche vengono calcolate separatamente per ciascuno strato, permettendo di identificare se i modelli basati su embeddings offrono vantaggi rispetto alla baseline soprattutto per gli effetti collaterali con pochi esempi di addestramento.

### Garanzia di Equità del Confronto

Questa strategia garantisce un confronto equo poiché tutti i modelli:
- Hanno accesso alle stesse informazioni durante l'addestramento: la struttura relazionale completa del grafo e le associazioni in $\mathcal{D}_{train}$
- Non osservano mai le associazioni in $\mathcal{D}_{test}$ durante l'addestramento
- Vengono valutati sulle stesse coppie held-out con le stesse metriche


## Riferimenti

1. Bordes, A., Usunier, N., Garcia-Duran, A., Weston, J., & Yakhnenko, O. (2013). *Translating Embeddings for Modeling Multi-relational Data*. NeurIPS 2013.
2. Trouillon, T., Welbl, J., Riedel, S., Gaussier, E., & Bouchard, G. (2016). *Complex Embeddings for Simple Link Prediction*. ICML 2016.
3. Sun, Z., Deng, Z.-H., Nie, J.-Y., & Tang, J. (2019). *RotatE: Knowledge Graph Embedding by Relational Rotation in Complex Space*. ICLR 2019.
4. Ali, M., Berrendorf, M., Hoyt, C. T., et al. (2021). *PyKEEN 1.0: A Python Library for Training and Evaluating Knowledge Graph Embeddings*. JMLR, 22(82), 1-6.
5. Chen, T. & Guestrin, C. (2016). *XGBoost: A Scalable Tree Boosting System*. KDD 2016.
6. Bodenreider, O. (2004). *The Unified Medical Language System (UMLS): Integrating Biomedical Terminology*. Nucleic Acids Research, 32(suppl_1), D116-D122.
7. Neumann, M., King, D., Beltagy, I., & Ammar, W. (2019). *ScispaCy: Fast and Robust Models for Biomedical Natural Language Processing*. BioNLP Workshop, ACL 2019.

# Releted Work
1. Mohamed et al. (2020): "Discovering protein drug targets using knowledge graph embeddings" Oxford Academic — Bioinformatics, Volume 36, Issue 2
2. Celebi et al. (2019): "Evaluation of knowledge graph embedding approaches for drug-drug interaction prediction in reali

Da valutare:
1. RANEDDI (2021) — "Relation-aware network embedding for drug-drug interaction prediction" ScienceDirect

Information Sciences, 2021
Usa esplicitamente RotatE per catturare le informazioni multi-relazionali tra farmaci e apprendere l'embedding delle entità ScienceDirect, poi le combina con network structure embedding per la predizione downstream

2. Zhang et al. (2021) — "Drug repurposing for COVID-19 via knowledge graph completion"

Journal of Biomedical Informatics
Confronta cinque algoritmi: TransE, RotatE, DistMult, ComplEx, e STELP per predire candidati drug repurposing ACS Publications

3. Understanding the performance of KGE in drug discovery (2022)

Artificial Intelligence in the Life Sciences
Confronto sistematico di TransE, ComplEx, DistMult, SimplE e RotatE su Hetionet per drug-target interactions ScienceDirect

Pattern embeddings + classifier (non RotatE ma stesso approccio):
4. DREAMwalk (2023) — Nature Communications

"Usa random walks per generare embeddings di farmaci e malattie, che vengono poi usati come input per un classificatore XGBoost" Oxford Academic