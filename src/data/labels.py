import warnings

warnings.filterwarnings("ignore")

from collections import defaultdict
import json

from patterns import SYNONYMS, antonyms, blocked_groups
from scispacy.linking import EntityLinker  # noqa: F401
import spacy
from spacy.tokens import Span
import tqdm

from src.config import (
    JACCARD_THRESHOLD,
    UMLS_CACHE_PATH,
    UMLS_SCORE_THRESHOLD,
)


def _load_cache():
    if UMLS_CACHE_PATH.exists():
        with open(UMLS_CACHE_PATH, encoding="utf-8") as f:
            return json.load(f)
    return None


def _save_cache(mapping, cui_to_canonical, umls_unmapped, umls_scores):
    UMLS_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)

    cache_data = {
        "mapping": mapping,
        "cui_to_canonical": cui_to_canonical,
        "umls_unmapped": umls_unmapped,
        "umls_scores": umls_scores,
    }

    with open(UMLS_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache_data, f, indent=2)


def _load_linker():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        nlp = spacy.load("en_core_sci_sm")
        nlp.add_pipe(
            "scispacy_linker",
            config={"resolve_abbreviations": True, "linker_name": "umls"},
        )
    return nlp, nlp.get_pipe("scispacy_linker")


def _resolve_label(nlp, linker, label):
    doc = nlp.make_doc(label)
    doc = nlp.get_pipe("ner")(doc)

    span = Span(doc, 0, len(doc), label="ENTITY")
    doc.ents = [span]
    nlp.get_pipe("scispacy_linker")(doc)

    ent = list(doc.ents)[0]
    if not ent._.kb_ents:
        return None, None, 0.0

    cui, score = ent._.kb_ents[0]
    if score < UMLS_SCORE_THRESHOLD:
        return None, None, score

    concept = linker.kb.cui_to_entity[cui]
    return cui, concept.canonical_name.lower(), score


def _tokenize(nlp, text):
    doc = nlp(text)
    tokens = set()
    for token in doc:
        if token.is_stop or token.is_punct:
            continue
        lemma = token.lemma_.lower()
        lemma = SYNONYMS.get(lemma, lemma)
        tokens.add(lemma)
    return tokens


def _jaccard(set_a, set_b):
    if not set_a or not set_b:
        return 0.0

    for word1, word2 in antonyms:
        if (word1 in set_a and word2 in set_b) or (word2 in set_a and word1 in set_b):
            return 0.0

    a_str = " ".join(set_a)
    b_str = " ".join(set_b)
    if ("hyper" in a_str and "hypo" in b_str) or ("hypo" in a_str and "hyper" in b_str):
        return 0.0

    for group in blocked_groups:
        a_matches = set_a & group
        b_matches = set_b & group
        if a_matches and b_matches and a_matches != b_matches:
            return 0.0

    return len(set_a & set_b) / len(set_a | set_b)


def _jaccard_merge(unmapped, all_labels, umls_mapping, raw_supports, nlp):
    tokenized = {label: _tokenize(nlp, label) for label in all_labels}

    jaccard_mapping = {}
    unmapped_sorted = sorted(unmapped, key=lambda x: -raw_supports.get(x, 0))
    processed = set()

    for label in unmapped_sorted:
        if label in processed:
            continue

        label_tokens = tokenized[label]
        if not label_tokens:
            continue

        best_match = None
        best_score = 0

        for candidate in all_labels:
            if candidate == label or candidate in jaccard_mapping:
                continue

            score = _jaccard(label_tokens, tokenized[candidate])
            if score > best_score and score >= JACCARD_THRESHOLD:
                best_score = score
                best_match = candidate

        if best_match:
            canonical = umls_mapping.get(best_match, best_match)

            if canonical in unmapped:
                if raw_supports.get(label, 0) > raw_supports.get(canonical, 0):
                    jaccard_mapping[canonical] = label
                    canonical = label

            jaccard_mapping[label] = canonical
            processed.add(label)

    return jaccard_mapping


def _run_umls_resolution(labels):
    nlp, linker = _load_linker()

    mapping = {}
    cui_to_canonical = {}
    umls_unmapped = []
    umls_scores = {}

    for label in tqdm.tqdm(labels, desc="UMLS resolution"):
        cui, pref, score = _resolve_label(nlp, linker, label)
        umls_scores[label] = score

        if cui is not None:
            cui_to_canonical.setdefault(cui, pref)
            mapping[label] = cui_to_canonical[cui]
        else:
            umls_unmapped.append(label)

    return mapping, cui_to_canonical, umls_unmapped, umls_scores


def build_mapping_and_supports(labels, raw_supports):
    cache = _load_cache()

    if cache:
        mapping = cache["mapping"]
        umls_unmapped = cache["umls_unmapped"]
        umls_scores = cache["umls_scores"]
    else:
        mapping, cui_to_canonical, umls_unmapped, umls_scores = _run_umls_resolution(labels)
        _save_cache(mapping, cui_to_canonical, umls_unmapped, umls_scores)

    nlp = spacy.load("en_core_sci_sm")
    jaccard_matches = _jaccard_merge(umls_unmapped, labels, mapping, raw_supports, nlp)

    final_unmapped = []
    for label in tqdm.tqdm(umls_unmapped, "Jaccard resolution"):
        if label in jaccard_matches:
            mapping[label] = jaccard_matches[label]
        else:
            mapping[label] = label
            final_unmapped.append((label, umls_scores[label]))

    groups = defaultdict(list)
    for orig, canonical in mapping.items():
        groups[canonical].append(orig)

    supports = {}
    for canonical, originals in groups.items():
        supports[canonical] = sum(raw_supports.get(orig, 0) for orig in originals)

    return mapping


def normalize_side_effects(df):
    se_cols = [c for c in df.columns if c.startswith("sideEffect")]

    stacked = df[se_cols].stack().dropna()

    labels = sorted(stacked.unique().tolist())
    raw_supports = stacked.value_counts().to_dict()

    mapping = build_mapping_and_supports(labels, raw_supports)

    return mapping
