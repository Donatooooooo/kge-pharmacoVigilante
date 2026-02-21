from collections import defaultdict

import pandas as pd
from tqdm.auto import tqdm

from src.config import (
    DMAX,
    INCLUDE_SIDE_EFFECTS,
    PROCESSED_MEDICINE_CSV,
    SEED,
    SIDE_EFFETCS_MEDICINE_CSV,
    TRIPLES_TSV,
)
from src.data.patterns import *


def get_route(form):
    if not form:
        return None
    for route, forms in CATEGORIES.items():
        if form in forms:
            return route
    return None


def process_drug(drug):
    drug = drug.lower()
    form = get_drug_form(drug)
    quantities = get_drug_quantities(drug)

    if form:
        drug = drug.replace(form, "").strip()
        form = form.replace(" ", "").strip()

    if quantities:
        for qty in quantities:
            drug = drug.replace(str(qty), "").strip()
        drug = CLEANUP_PATTERN.sub("", drug).strip()

    return drug, form


def process_composition(cmp):
    if not pd.notna(cmp) or not cmp:
        return None, []

    cmp = str(cmp).strip()
    match = COMPOSITION_PATTERN.match(cmp)
    if not match:
        return cmp, []

    name, dose_str = match.groups()
    dose_str = DOSE_CLEANUP_PATTERN.sub("", dose_str).replace("/", " ")

    doses = [f"{num}{unit or 'mg'}" for num, unit in DOSE_EXTRACT_PATTERN.findall(dose_str)]
    return name.strip(), doses


def generate_triples(row, substitute_cols, use_cols):
    triples = set()
    ingredients = set()

    drug, form = process_drug(row["name"])

    # Form triple
    if form:
        triples.add(f"{drug}\thas_form\t{form}")

    # Route triple
    route = get_route(form)
    if route:
        triples.add(f"{drug}\thas_route\t{route}")

    # Composition triples
    for col in ["short_composition1", "short_composition2"]:
        composition, doses = process_composition(row.get(col))
        if composition:
            triples.add(f"{drug}\tcomposed_by\t{composition}")
            ingredients.add(composition)
            for dose in doses:
                composite = f"{composition}_{dose}"
                triples.add(f"{drug}\thas_dose_of\t{composite}")

    # Therapeutic class
    therapeutic_class = row.get("Therapeutic Class")
    if pd.notna(therapeutic_class):
        tc_normalized = therapeutic_class.lower().replace(" ", "_").strip()
        triples.add(f"{drug}\thas_therapeutic_class\t{tc_normalized}")

    # Uses
    for col in use_cols:
        use_val = row.get(col)
        if pd.notna(use_val) and use_val:
            for _, condition in normalize_use(use_val):
                condition_normalized = condition.replace(" ", "_").strip()
                triples.add(f"{drug}\thas_use\t{condition_normalized}")

    # Substitutes
    for col in substitute_cols:
        substitute = row.get(col)
        if pd.notna(substitute) and substitute:
            sub_drug, _ = process_drug(substitute)
            triples.add(f"{drug}\thas_substitute\t{sub_drug}")

    # Chemical class
    chemical_class = row.get("Chemical Class")
    if pd.notna(chemical_class):
        triples.add(f"{drug}\thas_chemical_class\t{chemical_class}")

    return triples, drug, ingredients


def generate_side_effect_triples(drug, side_effects_row, se_cols):
    triples = set()
    for col in se_cols:
        se = side_effects_row.get(col)
        if pd.notna(se) and se:
            se_normalized = str(se).lower().replace(" ", "_").strip()
            triples.add(f"{drug}\thas_side_effect\t{se_normalized}")
    return triples


def build_graph():
    dataset = pd.read_csv(PROCESSED_MEDICINE_CSV)
    dataset = dataset.sample(frac=1, random_state=SEED).reset_index(drop=True)

    if INCLUDE_SIDE_EFFECTS:
        side_effects = pd.read_csv(SIDE_EFFETCS_MEDICINE_CSV)
        se_cols = [col for col in side_effects.columns if col.startswith("sideEffect")]
        dataset = dataset.merge(side_effects, on="name", how="left")

    substitute_cols = [col for col in dataset.columns if col.startswith("substitute0")]
    use_cols = [col for col in dataset.columns if col.startswith("use0")]
    name_to_idx = dataset.groupby("name").groups

    all_triples = set()
    ingredient_to_drugs = defaultdict(set)

    for idx, row in tqdm(dataset.iterrows(), total=DMAX, desc="Building triples"):
        try:
            # Process principal drugs
            triples, drug, ingredients = generate_triples(row, substitute_cols, use_cols)
            all_triples.update(triples)
            for ing in ingredients:
                ingredient_to_drugs[ing].add(drug)

            if INCLUDE_SIDE_EFFECTS:
                all_triples.update(generate_side_effect_triples(drug, row, se_cols))

            # Process substitute drugs
            for col in substitute_cols:
                substitute = row.get(col)
                if pd.notna(substitute) and substitute in name_to_idx:
                    sub_idx = name_to_idx[substitute][0]
                    sub_row = dataset.loc[sub_idx]
                    sub_triples, sub_drug, sub_ingredients = generate_triples(
                        sub_row, substitute_cols, use_cols
                    )
                    all_triples.update(sub_triples)
                    for ing in sub_ingredients:
                        ingredient_to_drugs[ing].add(sub_drug)

                    if INCLUDE_SIDE_EFFECTS:
                        all_triples.update(
                            generate_side_effect_triples(sub_drug, sub_row, se_cols)
                        )

            if DMAX != 0 and idx == DMAX:
                break

        except Exception as e:
            print(f"Error at row {idx}: {e}")

    triples = sorted(all_triples)
    relation_counts = defaultdict(int)
    for t in triples:
        relation = t.split("\t")[1]
        relation_counts[relation] += 1

    total = len(triples)
    print(f"\n{'Relation':<30} {'Count':>8} {'%':>8}")
    for rel, count in sorted(relation_counts.items(), key=lambda x: -x[1]):
        print(f"{rel:<30} {count:>8} {count / total * 100:>7.2f}%")

    with open(TRIPLES_TSV, "w", encoding="utf-16") as f:
        for triple in sorted(triples):
            f.write(triple.replace("  ", "").replace(" ", "").replace("-", "") + "\n")


if __name__ == "__main__":
    build_graph()
