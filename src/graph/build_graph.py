from collections import defaultdict
import re

import pandas as pd
import tqdm

from src.config import DMAX, PROCESSED_MEDICINE_CSV, TRIPLES_TSV
from src.data.patterns import CATEGORIES, get_drug_form, get_drug_quantities


COMPOSITION_PATTERN = re.compile(r"^(.*?)\s*\((.*?)\)$")
DOSE_CLEANUP_PATTERN = re.compile(r"/[a-zA-Z%]+")
DOSE_EXTRACT_PATTERN = re.compile(r"(\d+(?:\.\d+)?)([a-zA-Z%]*)")
CLEANUP_PATTERN = re.compile(r"[\/\\\s\d%]+")


def get_route(form):
    """Get administration route from drug form using CATEGORIES."""
    if not form:
        return None
    for route, forms in CATEGORIES.items():
        if form in forms:
            return route
    return None


def process_drug(drug):
    """Extract drug name and form, removing quantities."""
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
    """Parse composition string into name and doses."""
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


def generate_triples(row, substitute_cols):
    """Generate all triples for a single row.
    Returns (triples, drug_name, ingredients_set)."""
    triples = set()
    ingredients = set()

    drug, form = process_drug(row["name"])

    # Form triple
    if form:
        triples.add(f"{drug}\thas_form\t{form}")

    # Route triple (from CATEGORIES mapping)
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
                triples.add(f"{drug}\tcomposed_by_at_dose\t{composite}")

    # Therapeutic class
    therapeutic_class = row.get("Therapeutic Class")
    if pd.notna(therapeutic_class):
        tc_normalized = therapeutic_class.lower().replace(" ", "_").strip()
        triples.add(f"{drug}\thas_therapeutic_class\t{tc_normalized}")

    # Substitutes
    for col in substitute_cols:
        substitute = row.get(col)
        if pd.notna(substitute) and substitute:
            sub_drug, _ = process_drug(substitute)
            triples.add(f"{drug}\thas_substitute\t{sub_drug}")

    return triples, drug, ingredients


def main():
    dataset = pd.read_csv(PROCESSED_MEDICINE_CSV)

    # Pre-identify substitute columns once
    substitute_cols = [col for col in dataset.columns if col.startswith("substitute")]

    # Build index for fast substitute lookups
    name_to_idx = dataset.groupby("name").groups

    all_triples = set()
    # ingredient -> set of drugs (for shares_active_ingredient)
    ingredient_to_drugs = defaultdict(set)

    for idx, row in tqdm.tqdm(dataset.iterrows()):
        try:
            # Process principal medicine
            triples, drug, ingredients = generate_triples(row, substitute_cols)
            all_triples.update(triples)
            for ing in ingredients:
                ingredient_to_drugs[ing].add(drug)

            # Process substitute medicines
            for col in substitute_cols:
                substitute = row.get(col)
                if pd.notna(substitute) and substitute in name_to_idx:
                    sub_idx = name_to_idx[substitute][0]
                    sub_row = dataset.loc[sub_idx]
                    sub_triples, sub_drug, sub_ingredients = generate_triples(
                        sub_row, substitute_cols
                    )
                    all_triples.update(sub_triples)
                    for ing in sub_ingredients:
                        ingredient_to_drugs[ing].add(sub_drug)

            if DMAX != 0 and idx == DMAX:
                break

        except Exception as e:
            print(f"Error at row {idx}: {e}")

    unique_triples = all_triples

    print(f"- Generated {len(unique_triples)} unique triples")

    # Count relations
    relations = {}
    for triple in unique_triples:
        parts = triple.split("\t")
        if len(parts) >= 2:
            rel = parts[1]
            relations[rel] = relations.get(rel, 0) + 1

    print("\nRelations summary:")
    for rel, count in sorted(relations.items(), key=lambda x: -x[1]):
        print(f"    {rel}: {count}")

    with open(TRIPLES_TSV, "w", encoding="utf-16") as f:
        for triple in unique_triples:
            f.write(triple.replace("  ", "").replace(" ", "") + "\n")


if __name__ == "__main__":
    main()
