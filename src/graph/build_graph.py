from .patterns import get_drug_form, get_drug_quantities
import pandas as pd
import re
import tqdm
from ..config import PROCESSED_MEDICINE_CSV, TRIPLES_TSV, DMAX


COMPOSITION_PATTERN = re.compile(r'^(.*?)\s*\((.*?)\)$')
DOSE_CLEANUP_PATTERN = re.compile(r'/[a-zA-Z%]+')
DOSE_EXTRACT_PATTERN = re.compile(r'(\d+(?:\.\d+)?)([a-zA-Z%]*)')
QUANTITY_PATTERN = re.compile(r"\b\d+(?:\.\d+)?(?:\s*(?:mg|mcg|iu|ml|g|gm|kg|%))?\b", re.IGNORECASE)
UNIT_PATTERN = re.compile(r'(mg|mcg|iu|ml|g|gm|kg|%)', re.IGNORECASE)
CLEANUP_PATTERN = re.compile(r"[\/\\\s\d%]+")

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
    dose_str = DOSE_CLEANUP_PATTERN.sub('', dose_str).replace('/', ' ')

    doses = [
        f"{num}{unit or 'mg'}" 
        for num, unit in DOSE_EXTRACT_PATTERN.findall(dose_str)
    ]
    return name.strip(), doses


def extract_quantity(text):
    """Extract quantity with unit from pack size label."""
    match = QUANTITY_PATTERN.search(text)
    if not match:
        return None
    
    result = match.group(0).strip()
    if UNIT_PATTERN.search(result):
        return result.replace(" ", "").lower()
    return f"{result}pz"


def generate_triples(row, substitute_cols):
    """Generate all triples for a single row."""
    triples = set()
    
    drug, form = process_drug(row["name"])
    quantity = extract_quantity(str(row.get("pack_size_label", "")))
    
    # Form and quantity triples
    if form:
        triples.add(f"{drug}\thas_form\t{form}")
        
    if quantity:
        triples.add(f"{drug}\tcontains\t{quantity}")

    # Composition triples
    for col in ["short_composition1", "short_composition2"]:
        composition, doses = process_composition(row.get(col))
        if composition:
            triples.add(f"{drug}\tcomposed_by\t{composition}")
            for dose in doses:
                triples.add(f"{composition}\thas_quantity\t{dose}")
                triples.add(f"{drug}\tcontains_active_quantity\t{dose}")
    
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

    return triples


def main():
    dataset = pd.read_csv(PROCESSED_MEDICINE_CSV)
    
    # Pre-identify substitute columns once
    substitute_cols = [col for col in dataset.columns if col.startswith('substitute')]
    
    # Build index for fast substitute lookups
    name_to_idx = dataset.groupby('name').groups
    
    all_triples = set()
    
    for idx, row in tqdm.tqdm(dataset.iterrows()):
        try:
            # Process principal medicine
            all_triples.update(generate_triples(row, substitute_cols))
            
            # Process substitute medicines
            for col in substitute_cols:
                substitute = row.get(col)
                if pd.notna(substitute) and substitute in name_to_idx:
                    sub_idx = name_to_idx[substitute][0]
                    sub_row = dataset.loc[sub_idx]
                    all_triples.update(generate_triples(sub_row, substitute_cols))
        
            if idx == DMAX:
                break
        
        except Exception as e:
            print(f"Error at row {idx}: {e}")

    # Sort and write output
    unique_triples = all_triples
    
    print(f"- Generated {len(unique_triples)} unique triples")
    
    # Count relations
    relations = {}
    for triple in unique_triples:
        parts = triple.split('\t')
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