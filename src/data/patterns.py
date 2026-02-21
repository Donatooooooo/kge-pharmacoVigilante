import re

import numpy as np
import pandas as pd

DRUG_FORM = [
    "syrup",
    "tablet",
    "injection",
    "capsule",
    "suspension",
    "gel",
    "drops",
    "cream",
    "inhaler",
    "liquid",
    "spray",
    "penfill",
    "infusion",
    "emulsion",
    "expectorant",
    "lotion",
    "oral solution",
    "solution",
    "foam wash",
    "vaccine",
    "shampoo",
    "ointment",
    "kit",
    "powder",
    "mouth wash",
    "dusting powder",
    "transdermal patch",
    "drop",
    "scrub",
    "sachet",
    "soap",
    "strip",
    "toothpaste",
    "flexpen",
    "rheocap",
    "octacap",
    "captab",
    "shots",
    "jelly",
    "gums",
    "tabcap",
    "pellets",
    "rotacap",
    "serum",
    "suppository",
    "patch",
    "oral paste",
    "pessaries",
    "caps",
    "respicaps",
    "transcaps",
    "gargle",
    "respules",
    "divicap",
    "instacap",
    "testocap",
    "foam",
    "divitabs",
    "paste",
    "pen",
    "nail lacquer",
    "cartridge",
    "granules",
    "mouth paint",
    "combipack",
    "fleshtab",
    "multihaler",
    "autohaler",
    "evohaler",
    "turbuhaler",
    "transhaler",
    "transpules",
    "shot",
    "pill",
    "pills",
    "smartules",
    "surgical handwash",
    "syringe",
    "nexcap",
    "capsule",
    "liquidfilm",
    "film",
    "face wash",
    "respicap",
    "caplet",
    "gum",
    "vaginal wash",
    "particles",
    "lozenges",
    "elixir",
    "nexhaler",
    "plaster",
    "enema",
    "eye droppaint",
    "puch",
    "accuhaler",
    "ear drop",
    "synchrobreathe",
    "flextouch",
    "axacap",
    "breezhaler",
    "bar",
    "roll-on applicator",
    "pastilles",
    "vaginal pessary",
    "dry syup",
    "combo pack",
    "restore formula",
    "flashTab",
]

CATEGORIES = {
    "solid_oral": [
        "tablet",
        "capsule",
        "pill",
        "pills",
        "caplet",
        "divitabs",
        "fleshtab",
        "flashTab",
        "captab",
        "tabcap",
        "caps",
        "lozenges",
        "pastilles",
        "gum",
        "gums",
        "jelly",
        "granules",
        "powder",
        "pellets",
        "strip",
    ],
    "liquid_oral": [
        "syrup",
        "oral solution",
        "solution",
        "liquid",
        "suspension",
        "emulsion",
        "expectorant",
        "elixir",
        "dry syup",
    ],
    "topical": [
        "cream",
        "gel",
        "lotion",
        "ointment",
        "paste",
        "oral paste",
        "foam",
        "patch",
        "transdermal patch",
        "plaster",
        "nail lacquer",
        "paint",
        "mouth paint",
        "film",
        "liquidfilm",
        "bar",
    ],
    "inhalation": [
        "inhaler",
        "spray",
        "multihaler",
        "autohaler",
        "evohaler",
        "turbuhaler",
        "transhaler",
        "nexhaler",
        "accuhaler",
        "synchrobreathe",
        "breezhaler",
        "respicaps",
        "transcaps",
        "respules",
        "transpules",
        "smartules",
        "rotacap",
        "rheocap",
        "octacap",
        "divicap",
        "instacap",
        "testocap",
        "nexcap",
        "respicap",
        "axacap",
    ],
    "injectable": [
        "injection",
        "infusion",
        "shots",
        "shot",
        "penfill",
        "flexpen",
        "pen",
        "flextouch",
        "syringe",
        "cartridge",
        "vaccine",
        "serum",
    ],
    "liquid_drops": ["drops", "drop", "eye drop", "ear drop"],
    "suppository_pessary": ["suppository", "pessaries", "vaginal pessary"],
    "hygiene_cleansing": [
        "shampoo",
        "soap",
        "mouth wash",
        "foam wash",
        "scrub",
        "face wash",
        "vaginal wash",
        "surgical handwash",
        "toothpaste",
        "gargle",
        "enema",
    ],
    "specialized_delivery": [
        "kit",
        "combipack",
        "combo pack",
        "sachet",
        "puch",
        "roll-on applicator",
        "dusting powder",
        "particles",
        "restore formula",
    ],
}

SYNONYMS = {
    "syndrome": "symptom",
    "symptoms": "symptom",
    "symptom": "symptom",
}

COMPOSITION_PATTERN = re.compile(r"^(.*?)\s*\((.*?)\)$")
DOSE_CLEANUP_PATTERN = re.compile(r"/[a-zA-Z%]+")
DOSE_EXTRACT_PATTERN = re.compile(r"(\d+(?:\.\d+)?)([a-zA-Z%]*)")
CLEANUP_PATTERN = re.compile(r"[\/\\\s\d%]+")


def drug_form_pattern(keyword):
    kw = re.sub(r"\s+", r"\\s*", keyword)
    if not kw.endswith("s"):
        kw = kw + "s?"
    else:
        kw = kw + "?"
    return re.compile(rf"\b{kw}\b", re.IGNORECASE)


patterns = {kw: drug_form_pattern(kw) for kw in DRUG_FORM}


def get_drug_quantities(text):
    pattern = re.compile(r"\b\d+(?:\.\d+)?(?:\s*(?:mg|mcg|iu|ml|g|gm|kg|%))?\b", re.IGNORECASE)
    return pattern.findall(text)


def get_drug_form(text):
    for kw, pattern in patterns.items():
        if pattern.search(text):
            return kw
    return None


USE_ALIAS = {
    "liver  disease": "liver disease",
    "resistance tuberculosis (tb)": "resistant tuberculosis",
    "softening earwax": "softening of earwax",
    "treatment of softening of earwax": "softening of earwax",
    "treatment of prevention of heart attack and stroke": "prevention of heart attack and stroke",
    "treatment of infective diarrhea": "infectious diarrhea",
}

PAREN_PATTERN = re.compile(r"\s*\([^)]*\)")

USE_PREFIX_PATTERN = re.compile(
    r"^((?:(?:syndromic|acute|chronic)\s+)?"
    r"(?:treatment|management|prevention|relief|control|reduction"
    r"|suppression|correction|softening|prophylaxis|alleviation)"
    r"(?:\s+and\s+(?:treatment|management|prevention|relief|control"
    r"|reduction|suppression|prophylaxis))*"
    r")\s+of\s+"
)

_CC_TYPOS = {
    "specttrum": "spectrum",
    "sesquiterpine": "sesquiterpene",
    "denzamide": "benzamide",
    "drivative": "derivative",
    "derivetive": "derivative",
    "derivatieve": "derivative",
    "derivatve": "derivative",
    "derivate": "derivative",
    "aanalogue": "analogue",
    "asnalogue": "analogue",
    "analogu": "analogue",
    "aacid": "acid",
    "aacids": "acids",
    "aalcohol": "alcohol",
}

_CC_TYPO_PATTERN = re.compile(r"\b(" + "|".join(re.escape(k) for k in _CC_TYPOS) + r")\b")


def normalize_chemical_class(text):
    if pd.isna(text) or not str(text).strip():
        return np.nan

    s = str(text).lower().strip().rstrip(".")
    s = re.sub(r"\s+", " ", s)

    s = s.replace("{", "(").replace("}", ")")
    open_n, close_n = s.count("("), s.count(")")
    if open_n > close_n:
        s += ")" * (open_n - close_n)
    s = re.sub(r"\([^)]*\)", "", s)

    s = re.sub(r"['\"]", "", s)

    s = re.sub(r"[\d-]+", " ", s)

    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"^[^a-z]+", "", s)
    s = re.sub(r"[^a-z]+$", "", s)

    s = re.sub(r",(?=\S)", ", ", s)

    s = _CC_TYPO_PATTERN.sub(lambda m: _CC_TYPOS[m.group()], s)

    s = re.sub(r"\bderivatives\b", "derivative", s)
    s = re.sub(r"\banalogues\b", "analogue", s)
    s = re.sub(r"\banalogs?\b", "analogue", s)

    s = re.sub(r"(\w)s\s+(derivative|analogue)\b", r"\1 \2", s)

    s = re.sub(
        r"\b(chloride|fluoride|sulfate|hydroxide|oxide|nitrate|phosphate)s\b",
        r"\1",
        s,
    )

    s = s.strip()
    if not s:
        return np.nan
    return s


def _clean_condition(text):
    text = PAREN_PATTERN.sub("", text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_use(use):
    if pd.isna(use) or not use:
        return []

    use = str(use).lower().strip()
    use = re.sub(r"\s+", " ", use)
    use = USE_ALIAS.get(use, use)

    match = USE_PREFIX_PATTERN.match(use)
    if match:
        condition = _clean_condition(use[match.end() :])
    else:
        condition = _clean_condition(use)

    if not condition:
        return []
    prefix = match.group(1) if match else ""
    return [(prefix, condition)]


antonyms = [
    ("increased", "decreased"),
    ("increase", "decrease"),
]

blocked_groups = [
    {
        "magnesium",
        "sodium",
        "potassium",
        "glucose",
        "uric",
        "prolactin",
        "lipid",
        "calcium",
        "phosphate",
    },
    {"genital", "oropharynx", "nasal", "oral", "ocular", "vaginal", "rectal", "pharyngeal"},
    {"white", "red"},
]
