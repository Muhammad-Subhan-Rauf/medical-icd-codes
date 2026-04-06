"""
NTA Comorbidity Detail Lookup and Scoring for PDPM.

Maps ICD-10-CM codes to their specific NTA comorbidity categories,
MDS fields, and computes aggregate NTA scores per CMS PDPM rules.
"""

import os
import pandas as pd

# CMS PDPM NTA Comorbidity Category → MDS Field mapping
# Source: CMS PDPM ICD-10-CM Mappings, MDS 3.0 Section I
NTA_MDS_FIELD_MAP = {
    "HIV/AIDS": "I0100",
    "Lung Transplant Status": "I0100",
    "Major Organ Transplant Status": "I0100",
    "Opportunistic Infections": "I2500",
    "Chronic Myeloid Leukemia": "I0100",
    "Myelodysplastic Syndromes and Myelofibrosis": "I0100",
    "Endocarditis": "I0100",
    "Cardio-Respiratory Failure and Shock": "I8000",
    "Respiratory Arrest": "I8000",
    "Pulmonary Fibrosis and Other Chronic Lung Disorders": "I6200",
    "Cystic Fibrosis": "I6200",
    "End-Stage Liver Disease": "I0100",
    "Cirrhosis of Liver": "I0100",
    "Chronic Pancreatitis": "I0100",
    "Morbid Obesity": "I0100",
    "Bone/Joint/Muscle Infections/Necrosis - Except : RxCC80: Aseptic Necrosis of Bone": "I2500",
    "Aseptic Necrosis of Bone": "I8000",
    "Intractable Epilepsy": "I5200",
    "Narcolepsy and Cataplexy": "I5200",
    "Severe Skin Burn or Condition": "I2500",
    "Proliferative Diabetic Retinopathy and Vitreous Hemorrhage": "I2900",
    "Diabetic Retinopathy - Except : CC122: Proliferative Diabetic Retinopathy and Vitreous Hemorrhage": "I2900",
    "Complications of Specified Implanted Device or Graft": "I8000",
    "Disorders of Immunity - Except : RxCC97: Immune Disorders": "I0100",
    "Immune Disorders": "I0100",
    "Specified Hereditary Metabolic/Immune Disorders": "I0100",
    "Psoriatic Arthropathy and Systemic Sclerosis": "I0100",
    "Systemic Lupus Erythematosus": "I0100",
}

_nta_detail_lookup = None


def build_nta_detail_lookup(nta_csv_path=None):
    """Build a lookup dict: ICD-10 code (no dots) -> {comorbidity_name, mds_field}."""
    global _nta_detail_lookup
    if _nta_detail_lookup is not None:
        return _nta_detail_lookup

    if nta_csv_path is None:
        nta_csv_path = os.path.join(
            os.path.dirname(__file__), "mappings",
            "PDPM-ICD10-Mappings-FY2026-NTA-Comorbidity.csv"
        )

    nta_df = pd.read_csv(
        nta_csv_path, skiprows=6,
        names=["Sort_Order", "Comorbidity_Desc", "RxCC_CC", "ICD10_Code", "Description"],
        dtype=str
    )
    nta_df["ICD10_Code"] = nta_df["ICD10_Code"].str.strip().str.replace(".", "", regex=False)
    nta_df["Comorbidity_Desc"] = nta_df["Comorbidity_Desc"].str.strip()

    lookup = {}
    for _, row in nta_df.iterrows():
        code = row["ICD10_Code"]
        comorb = row["Comorbidity_Desc"]
        mds_field = NTA_MDS_FIELD_MAP.get(comorb, "I0100")
        lookup[code] = {
            "comorbidity_name": comorb,
            "mds_field": mds_field,
            "rxcc_cc": str(row.get("RxCC_CC", "N/A")).strip(),
        }

    _nta_detail_lookup = lookup
    return _nta_detail_lookup


def score_nta_comorbidities(enriched_codes):
    """
    Given a list of enriched code dicts (each having 'exact_code'),
    compute the NTA comorbidity detail and aggregate score.

    Returns:
        {
            "itemized": [
                {"comorbidity_name": str, "mds_field": str, "icd10_codes": [str, ...]}
            ],
            "total_categories": int,
            "total_nta_points": int
        }
    """
    lookup = build_nta_detail_lookup()

    # Group matched codes by comorbidity category
    categories = {}  # comorbidity_name -> {mds_field, codes: []}
    for code_dict in enriched_codes:
        code = code_dict.get("exact_code", "").replace(".", "")
        if code in lookup:
            info = lookup[code]
            name = info["comorbidity_name"]
            if name not in categories:
                categories[name] = {
                    "comorbidity_name": name,
                    "mds_field": info["mds_field"],
                    "icd10_codes": [],
                }
            if code not in categories[name]["icd10_codes"]:
                categories[name]["icd10_codes"].append(code)

    itemized = list(categories.values())
    total_categories = len(itemized)

    # Under PDPM, each unique NTA comorbidity category = 1 count.
    # The aggregate NTA score equals the count (capped at 8 per CMS rules).
    total_nta_points = min(total_categories, 8)

    return {
        "itemized": itemized,
        "total_categories": total_categories,
        "total_nta_points": total_nta_points,
    }


def get_nta_detail_for_code(icd10_code):
    """Get NTA comorbidity info for a single ICD-10 code, or None."""
    lookup = build_nta_detail_lookup()
    return lookup.get(icd10_code)
