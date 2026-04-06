import os
import logging
import numpy as np
import pandas as pd
from pydantic import BaseModel
from google import genai
from indexer import load_data, build_or_load_index, build_or_load_code_index
from nta_scoring import score_nta_comorbidities, get_nta_detail_for_code
from pdf_extractor import find_source_for_quote
from typing import Literal
import faiss

logger = logging.getLogger("medic.pipeline")

# Model constants
MODEL_FAST = "gemini-2.5-flash"
MODEL_QUALITY = "gemini-2.5-flash"

# Global caches
_index = None
_group_list = None
_df = None
_embedder = None
_genai_client = None
_crosswalk_df = None
_code_index = None
_code_list = None


def _get_client():
    global _genai_client
    if _genai_client is None:
        api_key = os.environ.get("GEMINI_API_KEY")
        _genai_client = genai.Client(api_key=api_key)
    return _genai_client


def _load_crosswalk():
    global _crosswalk_df
    if _crosswalk_df is not None:
        return _crosswalk_df

    mappings_dir = os.path.join(os.path.dirname(__file__), "mappings")

    cat_df = pd.read_csv(
        os.path.join(mappings_dir, "PDPM-ICD10-Mappings-FY2026-Clinical-Categories-by-Dx.csv"),
        skiprows=5, names=["Sort_Order", "ICD10_Code", "Description", "PDPM_Category", "Major_Procedure"],
        dtype=str
    )
    cat_df["ICD10_Code"] = cat_df["ICD10_Code"].str.strip().str.replace(".", "", regex=False)
    cat_df["PDPM_Category"] = cat_df["PDPM_Category"].str.strip()
    cat_df["Major_Procedure"] = cat_df["Major_Procedure"].str.strip()

    nta_df = pd.read_csv(
        os.path.join(mappings_dir, "PDPM-ICD10-Mappings-FY2026-NTA-Comorbidity.csv"),
        skiprows=6, names=["Sort_Order", "Comorbidity_Desc", "RxCC_CC", "ICD10_Code", "Description"],
        dtype=str
    )
    nta_df["ICD10_Code"] = nta_df["ICD10_Code"].str.strip().str.replace(".", "", regex=False)
    nta_lookup = nta_df.groupby("ICD10_Code").agg(
        NTA_Comorbidity=("Comorbidity_Desc", lambda x: "; ".join(x.str.strip().unique()))
    ).reset_index()
    nta_lookup["NTA_Points"] = 1

    slp_df = pd.read_csv(
        os.path.join(mappings_dir, "PDPM-ICD10-Mappings-FY2026-SLP-Comorbidity.csv"),
        skiprows=5, names=["Sort_Order", "Comorbidity_Desc", "ICD10_Code", "Description"],
        dtype=str
    )
    slp_df["ICD10_Code"] = slp_df["ICD10_Code"].str.strip().str.replace(".", "", regex=False)
    slp_lookup = slp_df[["ICD10_Code"]].drop_duplicates()
    slp_lookup["SLP_Comorbidity"] = True

    crosswalk = cat_df[["ICD10_Code", "PDPM_Category", "Major_Procedure"]].drop_duplicates(subset="ICD10_Code")
    crosswalk = crosswalk.merge(nta_lookup[["ICD10_Code", "NTA_Points", "NTA_Comorbidity"]], on="ICD10_Code", how="left")
    crosswalk = crosswalk.merge(slp_lookup, on="ICD10_Code", how="left")
    crosswalk["NTA_Points"] = crosswalk["NTA_Points"].fillna(0).astype(int)
    crosswalk["SLP_Comorbidity"] = crosswalk["SLP_Comorbidity"].fillna(False)

    logger.info("Loaded CMS crosswalk: %d codes, %d NTA, %d SLP",
                len(crosswalk), len(nta_lookup), len(slp_lookup))
    _crosswalk_df = crosswalk
    return _crosswalk_df


# -------------------------------------------------------------
# PYDANTIC MODELS
# -------------------------------------------------------------

class RefinedClinicalSection(BaseModel):
    section_name: str
    content: str

class InferredCondition(BaseModel):
    condition: str
    evidence_type: str
    evidence: str
    confidence: str

class RefinedClinicalNote(BaseModel):
    sections: list[RefinedClinicalSection]
    inferred_conditions: list[InferredCondition]
    abbreviations_expanded: list[str]
    warnings: list[str]
    merged_note: str

class ExtractedCondition(BaseModel):
    condition: str
    category: Literal["acute", "chronic", "historical", "incidental"]
    body_site: str | None = None
    laterality: str | None = None
    severity: str | None = None
    episode: str | None = None
    source_section: str | None = None

class ExtractedConditions(BaseModel):
    conditions: list[ExtractedCondition]

class RankedCode(BaseModel):
    exact_code: str
    description: str
    designation: Literal["Primary", "Secondary"]
    status: Literal["Active", "Resolved"]
    acuity: Literal["Acute", "Chronic"]
    clinical_evidence_quote: str
    document_reference: str
    probability_score: int
    reasoning: str

class FinalCodingReport(BaseModel):
    ranked_codes: list[RankedCode]


# -------------------------------------------------------------
# INITIALIZATION
# -------------------------------------------------------------

def initialize_system():
    global _index, _group_list, _df, _embedder, _code_index, _code_list
    if _index is None:
        _index, _group_list, _df = build_or_load_index()
    if _code_index is None:
        _code_index, _code_list, _ = build_or_load_code_index()
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        _embedder = SentenceTransformer('FremyCompany/BioLORD-2023')


# -------------------------------------------------------------
# ICD-10-CM VOCABULARY ANCHOR
# -------------------------------------------------------------

ICD10_CHAPTERS = """A00-B99: Certain infectious and parasitic diseases
C00-D49: Neoplasms
D50-D89: Diseases of the blood and blood-forming organs
E00-E89: Endocrine, nutritional and metabolic diseases
F01-F99: Mental, behavioral and neurodevelopmental disorders
G00-G99: Diseases of the nervous system
H00-H59: Diseases of the eye and adnexa
H60-H95: Diseases of the ear and mastoid process
I00-I99: Diseases of the circulatory system
J00-J99: Diseases of the respiratory system
K00-K95: Diseases of the digestive system
L00-L99: Diseases of the skin and subcutaneous tissue
M00-M99: Diseases of the musculoskeletal system and connective tissue
N00-N99: Diseases of the genitourinary system
O00-O9A: Pregnancy, childbirth and the puerperium
P00-P96: Certain conditions originating in the perinatal period
Q00-Q99: Congenital malformations, deformations and chromosomal abnormalities
R00-R99: Symptoms, signs and abnormal clinical and laboratory findings
S00-T88: Injury, poisoning and certain other consequences of external causes
V00-Y99: External causes of morbidity
Z00-Z99: Factors influencing health status and contact with health services"""


# -------------------------------------------------------------
# STEP 0: REFINE CLINICAL NOTE (single LLM call)
# -------------------------------------------------------------

def refine_clinical_note(note: str) -> RefinedClinicalNote:
    """Preprocess raw clinical note: expand abbreviations, structure into sections,
    normalize terminology, and infer conditions from medications/labs."""
    client = _get_client()

    prompt = f"""You are a clinical documentation specialist. Your job is to REFINE a raw clinical note into a structured, expanded format suitable for ICD-10-CM medical coding.

STRICT RULES — VIOLATION OF ANY RULE IS UNACCEPTABLE:
1. NEVER add medical information not present in or directly implied by the original note.
2. NEVER invent diagnoses, symptoms, findings, or history not supported by the text.
3. NEVER change the clinical meaning of any statement.
4. If an abbreviation is ambiguous, choose the MOST LIKELY medical interpretation given the surrounding context, and log it in the warnings field.
5. Every inferred condition MUST have explicit evidence in the original note (a medication name or a lab value). Document this in inferred_conditions.
6. PRESERVE ALL ORIGINAL INFORMATION. Every fact from the original note must appear in the output. Do not drop or omit anything.

TASK 1 — EXPAND ABBREVIATIONS:
Convert medical shorthand to full terminology. Common examples:
- "HTN" → "hypertension", "DM2" → "type 2 diabetes mellitus", "DM1" → "type 1 diabetes mellitus"
- "CAD" → "coronary artery disease", "CHF" → "congestive heart failure", "COPD" → "chronic obstructive pulmonary disease"
- "CKD" → "chronic kidney disease", "ESRD" → "end-stage renal disease", "AKI" → "acute kidney injury"
- "AFib" → "atrial fibrillation", "DVT" → "deep vein thrombosis", "PE" → "pulmonary embolism"
- "UTI" → "urinary tract infection", "AMS" → "altered mental status", "SOB" → "shortness of breath"
- "s/p" → "status post", "p/w" → "presents with", "hx" → "history of", "w/" → "with", "w/o" → "without"
- "yo" → "year-old", "M" → "male", "F" → "female", "pt" → "patient"
- "CABG" → "coronary artery bypass graft", "TKA" → "total knee arthroplasty", "THA" → "total hip arthroplasty"
- "BID" → "twice daily", "TID" → "three times daily", "QHS" → "every night at bedtime", "PRN" → "as needed"
- "Cr" → "creatinine", "WBC" → "white blood cell count", "HgA1c"/"HbA1c" → "hemoglobin A1c"
- "BMP" → "basic metabolic panel", "CBC" → "complete blood count"
Log each expansion in abbreviations_expanded as "ABBR -> Full Term".

TASK 2 — STRUCTURE INTO SECTIONS:
Organize the content into standard clinical sections. Only create sections that have content in the original note:
- History of Present Illness (HPI)
- Past Medical History (PMH)
- Medications
- Laboratory Values
- Assessment and Plan
- Physical Examination
- Social History
- Review of Systems
- Procedures/Surgical History
- Discharge Diagnoses
If the original note does not clearly separate sections, infer section placement from content type.

TASK 3 — NORMALIZE TERMINOLOGY:
Use standard medical terminology compatible with ICD-10-CM coding conventions.
Examples: "sugar disease" → "diabetes mellitus", "water pill" → "diuretic", "blood thinner" → "anticoagulant"

TASK 4 — INFER IMPLIED CONDITIONS:
Infer conditions from medications and lab values ONLY when the implication is clinically standard:
HIGH confidence (always infer):
- metformin → type 2 diabetes mellitus
- insulin (with type 2 context) → type 2 diabetes mellitus
- lisinopril/losartan/amlodipine → hypertension
- atorvastatin/rosuvastatin/simvastatin → hyperlipidemia
- levothyroxine → hypothyroidism
- warfarin/apixaban/rivaroxaban → condition requiring anticoagulation
- furosemide → fluid overload or heart failure
- HgA1c/HbA1c > 6.5% → diabetes mellitus
MODERATE confidence (infer but note uncertainty):
- albuterol/ipratropium → asthma or COPD (flag ambiguity in warnings)
- elevated creatinine → acute or chronic kidney disease
- WBC > 11,000 → possible infection or leukocytosis
- low hemoglobin → anemia
LOW confidence (infer with strong caveat):
- elevated BNP → possible heart failure
Include inferred conditions in the relevant section text AND list each in inferred_conditions.

TASK 5 — MERGED NOTE:
In the merged_note field, combine all sections into a single coherent clinical note with section headers formatted as "## Section Name" followed by the section content.

ORIGINAL CLINICAL NOTE:
{note}"""

    response = client.models.generate_content(
        model=MODEL_FAST,
        contents=prompt,
        config={"response_mime_type": "application/json", "response_schema": RefinedClinicalNote, "temperature": 0.0}
    )
    return RefinedClinicalNote.model_validate_json(response.text)


# -------------------------------------------------------------
# SECTION PARSER (deterministic)
# -------------------------------------------------------------

def parse_sections(text: str) -> list[dict]:
    """Split refined note into named sections using ## headers."""
    import re
    sections = []
    parts = re.split(r'^## (.+)$', text, flags=re.MULTILINE)
    # parts = ['preamble', 'Section Name 1', 'content 1', 'Section Name 2', 'content 2', ...]
    if len(parts) > 1:
        for i in range(1, len(parts), 2):
            section_name = parts[i].strip()
            content = parts[i + 1].strip() if i + 1 < len(parts) else ""
            if content:
                sections.append({"section_name": section_name, "content": content})
    if not sections:
        sections.append({"section_name": "Full Note", "content": text})
    return sections


# -------------------------------------------------------------
# EMBEDDING QUERY BUILDER (deterministic)
# -------------------------------------------------------------

def build_embedding_query(c: ExtractedCondition) -> str:
    """Deterministically build a normalized embedding query from structured fields."""
    parts = [c.condition]
    if c.body_site:
        parts.append(c.body_site)
    if c.laterality:
        parts.append(c.laterality)
    if c.severity:
        parts.append(c.severity)
    if c.episode:
        parts.append(c.episode)
    return ", ".join(parts)


# -------------------------------------------------------------
# EMBED-BACK VALIDATION (deterministic)
# -------------------------------------------------------------

def validate_extractions(conditions: list[ExtractedCondition],
                         code_index, embedder,
                         low_threshold=0.35) -> list[ExtractedCondition]:
    """Check each extracted condition against the FAISS index. Log warnings for low matches."""
    queries = [build_embedding_query(c) for c in conditions]
    vectors = embedder.encode(queries, convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(vectors)
    sims, idxs = code_index.search(vectors, 1)

    validated = []
    for i, condition in enumerate(conditions):
        sim = float(sims[i][0])
        if sim < low_threshold:
            logger.warning("Low embed match for '%s': %.3f — may not align with ICD-10 terminology",
                          condition.condition, sim)
        validated.append(condition)
    return validated


# -------------------------------------------------------------
# STEP 1: EXTRACT ALL CONDITIONS (single LLM call)
# -------------------------------------------------------------

def extract_all_conditions(note: str, sections: list[dict] = None) -> ExtractedConditions:
    client = _get_client()

    # Format sections for the prompt
    if sections:
        formatted_sections = "\n\n".join(
            f"### {s['section_name']}\n{s['content']}" for s in sections
        )
    else:
        formatted_sections = note

    prompt = f"""You are an expert medical coder. Extract EVERY medical condition, diagnosis, comorbidity, and clinical finding from this clinical note.

ICD-10-CM TERMINOLOGY REFERENCE — align your extracted condition names with these standard categories:
{ICD10_CHAPTERS}

INSTRUCTIONS:
- Extract from EACH section of the note. Tag each condition with its source_section (the section name where you found it).
- Extract ALL conditions: acute admission diagnoses, chronic comorbidities, historical/resolved conditions, and incidental findings.
- Use standard medical terminology matching ICD-10-CM conventions. Align condition names with the chapter categories above.
- For EACH condition, fill in structured fields where documented:
  * body_site: anatomical site (e.g., "kidney", "left lung", "lumbar spine"). Use null if not specified.
  * laterality: "left", "right", or "bilateral". Use null if not specified.
  * severity: severity or stage (e.g., "stage 3", "moderate", "severe", "mild"). Use null if not specified.
  * episode: episode of care (e.g., "initial encounter", "subsequent encounter", "sequela"). Use null if not specified.
- Categorize each as: "acute" (new/active), "chronic" (ongoing), "historical" (resolved/past), or "incidental" (found but not treated).
- Be thorough — capture every condition mentioned, even if only in the past medical history or problem list.
- Include conditions like deconditioning, obesity, hypertension, etc. that may seem minor but are billable.

MEDICATION-IMPLIED CONDITIONS — extract these even if not explicitly stated as diagnoses:
- Patient on metformin/insulin → type 2 diabetes mellitus
- Patient on lisinopril/losartan/amlodipine → hypertension
- Patient on atorvastatin/simvastatin → hyperlipidemia
- Patient on levothyroxine → hypothyroidism
- Patient on albuterol → asthma or COPD
- Patient on warfarin/apixaban → condition requiring anticoagulation (e.g., atrial fibrillation, DVT history)
- Patient on furosemide → fluid overload or heart failure

LAB-IMPLIED CONDITIONS — extract these when lab values clearly indicate a condition:
- HgA1c/HbA1c > 6.5% → diabetes mellitus
- Elevated creatinine → acute kidney injury or chronic kidney disease
- WBC > 11,000 → leukocytosis or active infection
- Low hemoglobin → anemia

ABBREVIATION HANDLING — if any abbreviations remain unexpanded, expand them:
- HTN → hypertension, DM2 → type 2 diabetes mellitus, CAD → coronary artery disease
- CHF → congestive heart failure, COPD → chronic obstructive pulmonary disease
- CKD → chronic kidney disease, ESRD → end-stage renal disease, AFib → atrial fibrillation
- DVT → deep vein thrombosis, PE → pulmonary embolism, AKI → acute kidney injury

NEGATION HANDLING — DO NOT extract negated conditions:
- "Denies chest pain" → do NOT extract chest pain
- "No evidence of malignancy" → do NOT extract malignancy
- "Negative for DVT" → do NOT extract DVT
- Only extract conditions that are PRESENT, ACTIVE, HISTORICAL, or INCIDENTAL — never negated ones.

EXAMPLES:
- "Septic shock secondary to left lower extremity cellulitis" → condition: "Septic shock secondary to cellulitis", category: "acute", body_site: "left lower extremity", laterality: "left"
- "Stage 3 chronic kidney disease" → condition: "Chronic kidney disease", category: "chronic", severity: "stage 3", body_site: "kidney"
- "Stage II acute kidney injury, resolved" → condition: "Acute kidney injury", category: "historical", severity: "stage II"
- "Initial encounter for right hip fracture" → condition: "Hip fracture", category: "acute", body_site: "hip", laterality: "right", episode: "initial encounter"

CLINICAL NOTE (by section):
{formatted_sections}"""

    response = client.models.generate_content(
        model=MODEL_FAST,
        contents=prompt,
        config={"response_mime_type": "application/json", "response_schema": ExtractedConditions, "temperature": 0.0}
    )
    return ExtractedConditions.model_validate_json(response.text)


# -------------------------------------------------------------
# STEP 2: PROGRAMMATIC CODE MATCHING (zero LLM calls)
# -------------------------------------------------------------

def match_codes_programmatic(conditions: list[ExtractedCondition],
                              top_k_per_condition=5,
                              similarity_threshold=0.45) -> list[dict]:
    """
    For each extracted condition, embed it and search the 71K code-level
    FAISS index. Return deduplicated candidate codes above the threshold.
    """
    initialize_system()

    # Batch-encode all condition strings using structured fields
    condition_texts = [build_embedding_query(c) for c in conditions]
    query_vectors = _embedder.encode(condition_texts, convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(query_vectors)  # Normalize for cosine similarity

    # Search all conditions in one batch
    similarities, indices = _code_index.search(query_vectors, top_k_per_condition)

    # Collect and deduplicate results
    best_matches = {}  # code -> best match dict
    for cond_idx, condition in enumerate(conditions):
        for rank in range(top_k_per_condition):
            code_idx = indices[cond_idx][rank]
            sim_score = float(similarities[cond_idx][rank])

            if sim_score < similarity_threshold:
                continue

            code_info = _code_list[code_idx]
            code = code_info["Specific_Code"]

            # Keep the best similarity score for each code
            if code not in best_matches or sim_score > best_matches[code]["similarity_score"]:
                best_matches[code] = {
                    "exact_code": code,
                    "description": code_info["Specific_Description"],
                    "group_code": code_info["Group_Code"],
                    "group_description": code_info["Group_Description"],
                    "similarity_score": sim_score,
                    "matched_condition": condition.condition,
                    "condition_category": condition.category,
                    "reasoning": f"Matched to '{condition.condition}' (cosine similarity: {sim_score:.3f})",
                }

    # Sort by similarity score descending
    results = sorted(best_matches.values(), key=lambda x: x["similarity_score"], reverse=True)
    return results


# -------------------------------------------------------------
# STEP 3: FINAL RANKER (single LLM call — unchanged)
# -------------------------------------------------------------

def step_3_final_ranker(note: str, validated_codes: list[dict]) -> FinalCodingReport:
    client = _get_client()

    candidates_text = ""
    for c in validated_codes:
        candidates_text += (
            f"- Code: {c['exact_code']} | Description: {c['description']} | "
            f"Matched Condition: {c['matched_condition']} | Similarity: {c['similarity_score']:.3f}\n"
        )

    prompt = f"""You are a Master Clinical Documentation Improvement (CDI) Specialist and certified ICD-10-CM coder reviewing codes for Medicare SNF/PDPM submission.

TASK: Review the candidate ICD-10-CM codes matched by vector similarity. Include each code that is supported by the clinical note. REMOVE any code that does NOT have clear clinical evidence or that violates ICD-10-CM coding conventions below.

=== ICD-10-CM CODING CONVENTIONS (MUST FOLLOW) ===

SYMPTOM EXCLUSION RULE:
Do NOT include symptom codes (R00-R99 chapter, or pain/weakness codes) when the underlying diagnosis that causes the symptom is already coded. Examples:
- Do NOT code "shortness of breath" (R06.02) if respiratory failure (J96.x) or COPD (J44.x) is coded
- Do NOT code "leg pain" (M79.6x) if cellulitis (L03.x) of the same site is coded
- Do NOT code "hypotension" (I95.x) if septic shock (R65.21) is coded
- Do NOT code "elevated WBC" (D72.8x) or other lab findings as standalone diagnoses when the infection causing them is coded
- Do NOT code "fever" (R50.x) if the underlying infection is coded
Exception: Include symptom codes ONLY if no underlying diagnosis in the final code list explains them.

SEPSIS SEQUENCING RULE:
When coding severe sepsis (R65.20) or septic shock (R65.21), you MUST also include:
- The underlying systemic infection code (e.g., A41.9 Sepsis, unspecified organism)
- The code for the localized infection that caused the sepsis (e.g., L03.116 Cellulitis of left lower limb)
Sepsis codes are never coded alone — they require the full causal chain.

WOUND CODING RULE:
When clinical notes document wounds, ulcers, or skin lesions, code them with maximum specificity:
- Include the specific anatomical site (foot, toe, ankle, lower leg, etc.)
- Include laterality (left/right)
- Include depth/severity if documented (e.g., limited to breakdown of skin, with fat layer exposed)
- Infected wounds need BOTH the wound code AND the infection code
- Non-pressure ulcers of the lower extremity use L97.x codes with full site specificity
- Foot infections are coded separately from leg cellulitis when documented

COMPLETENESS CHECK:
After reviewing the candidate codes, scan the ENTIRE clinical note for common chronic conditions documented anywhere (PMH, problem list, medication list, HPI) that may NOT be in the candidate list. If clearly documented, ADD them even though they are not in the candidate codes. Common conditions to check for:
- Hypertension (I10)
- Type 2 Diabetes Mellitus (E11.9 or more specific)
- Hyperlipidemia (E78.5 or more specific)
- Obesity (E66.x)
- CKD stages (N18.x)
- Hypothyroidism (E03.9)
- GERD (K21.0)
You may add codes not in the candidate list if they have CLEAR documentation in the note.

CONFIDENCE THRESHOLD:
Only include codes with DEFINITE or PROBABLE documentation. Do NOT code conditions described as "possible", "suspected", "concern for", "rule out", or "cannot exclude" UNLESS:
- The condition was actively treated (e.g., antibiotics given for "possible diverticulitis" = code it)
- ICD-10 inpatient coding guidelines explicitly apply (uncertain diagnoses coded as if confirmed for inpatient stays)
If documentation is speculative AND no treatment was directed at it, do NOT include it.

=== CLASSIFICATION RULES ===

1. DESIGNATION (STRICT — READ CAREFULLY):
   - "Primary": The SINGLE principal diagnosis that MOST DIRECTLY drove or occasioned the SNF admission. This is the ONE condition that is the primary reason the patient needs skilled nursing care. Almost always exactly ONE code is Primary.
   - A second Primary is allowed ONLY if two conditions were truly co-equal, inseparable reasons for the SNF admission (extremely rare). A third Primary is virtually never appropriate.
   - "Secondary": EVERYTHING else — all other active diagnoses, complications during the hospital stay, chronic comorbidities, historical conditions, and incidental findings.
   - Complications that occurred DURING the hospital stay (e.g., AKI, pulmonary edema, respiratory failure) are ALWAYS Secondary — they are complications, not the reason for admission.
   - Sequencing codes required by convention (e.g., A41.9 required alongside R65.21) are Secondary — they exist for coding compliance, not because they independently drove the admission.
   - When in doubt, designate as Secondary. Primary is the exception, not the rule.

2. STATUS:
   - "Active": Condition is CURRENTLY being treated, monitored, or evaluated AT THE TIME OF SNF ADMISSION. The patient still has this condition.
   - "Resolved": Condition was documented during the hospital stay but resolved BEFORE discharge/SNF transfer. The patient no longer has this condition.
   - Key test: Was the condition present and requiring management when the patient arrived at the SNF? If yes = Active. If it resolved during the hospital stay = Resolved.
   - Chronic conditions from PMH that are ongoing = Active.
   - Acute complications that resolved before discharge (e.g., AKI that resolved, septic shock that cleared) = Resolved.

3. ACUITY:
   - "Acute": New onset, acute exacerbation, or active infection.
   - "Chronic": Stable, ongoing condition managed long-term.

4. EVIDENCE: Extract a VERBATIM quote from the clinical note that supports each code. Must be an exact substring — do not paraphrase.

5. REFERENCE: Identify the section. If the note contains source markers like ---[filename pN]---, include them (e.g., "Brunk_DC12.26.25.pdf p2, Assessment/Plan").

6. PROBABILITY SCORE: Rate 0-100 based on documentation support.

PATIENT NOTE:
{note}

CANDIDATE CODES (from vector similarity search):
{candidates_text}
"""

    response = client.models.generate_content(
        model=MODEL_QUALITY,
        contents=prompt,
        config={"response_mime_type": "application/json", "response_schema": FinalCodingReport, "temperature": 0.0}
    )
    return FinalCodingReport.model_validate_json(response.text)


# -------------------------------------------------------------
# ORCHESTRATION PIPELINE
# -------------------------------------------------------------

def process_patient_note(note, top_k_per_condition=5, similarity_threshold=0.45,
                          on_progress=None, page_chunks=None):
    """
    Main pipeline orchestrator.

    New architecture (2 LLM calls, rest programmatic):
    1. LLM: Extract ALL conditions from note
    2. Programmatic: FAISS code-level search per condition
    3. LLM: Final ranking and classification
    """
    initialize_system()
    logs = []

    # Step 0: Refine clinical note
    logs.append("Step 0: Refining clinical note (expand abbreviations, structure, normalize)...")
    refined = None
    try:
        refined = refine_clinical_note(note)
        refined_text = refined.merged_note
        logs.append(f"  Refinement complete: {len(refined.sections)} sections")
        if refined.abbreviations_expanded:
            logs.append(f"  {len(refined.abbreviations_expanded)} abbreviations expanded:")
            for abbr in refined.abbreviations_expanded:
                logs.append(f"    {abbr}")
        if refined.inferred_conditions:
            logs.append(f"  {len(refined.inferred_conditions)} conditions inferred from meds/labs:")
            for inf in refined.inferred_conditions:
                logs.append(f"    {inf.condition} (from {inf.evidence_type}: {inf.evidence}, confidence: {inf.confidence})")
        for warn in refined.warnings:
            logs.append(f"  WARNING: {warn}")
    except Exception as e:
        logger.error("Clinical note refinement failed: %s", e, exc_info=True)
        logs.append(f"WARNING: Note refinement failed ({e}), proceeding with raw note")
        refined_text = note
    if on_progress: on_progress(0.1)

    # Step 1a: Parse sections deterministically
    sections = parse_sections(refined_text)
    logs.append(f"\nStep 1a: Parsed {len(sections)} sections: {[s['section_name'] for s in sections]}")

    # Step 1b: Extract ALL conditions (single LLM call, section-aware + structured)
    logs.append("Step 1b: Extracting conditions (structured, section-aware)...")
    try:
        extracted = extract_all_conditions(refined_text, sections)
    except Exception as e:
        logger.error("Condition extraction failed: %s", e, exc_info=True)
        if on_progress: on_progress(1.0)
        return {"status": "failed", "logs": logs, "error": f"Condition extraction failed: {e}"}

    logs.append(f"  Extracted {len(extracted.conditions)} conditions:")
    for c in extracted.conditions:
        fields = f"[{c.category}] {c.condition}"
        if c.body_site:
            fields += f" | site: {c.body_site}"
        if c.laterality:
            fields += f" | lat: {c.laterality}"
        if c.severity:
            fields += f" | sev: {c.severity}"
        if c.source_section:
            fields += f" | from: {c.source_section}"
        logs.append(f"  {fields}")
    if on_progress: on_progress(0.25)

    # Step 1c: Embed-back validation
    logs.append("\nStep 1c: Validating extraction quality via embedding check...")
    validated_conditions = validate_extractions(extracted.conditions, _code_index, _embedder)
    if on_progress: on_progress(0.35)

    # Step 2: Programmatic code matching (zero LLM calls)
    logs.append(f"\nStep 2: Programmatic FAISS code search (top_k={top_k_per_condition}, threshold={similarity_threshold})...")
    candidates = match_codes_programmatic(
        validated_conditions,
        top_k_per_condition=top_k_per_condition,
        similarity_threshold=similarity_threshold
    )
    logs.append(f"Found {len(candidates)} candidate codes above similarity threshold")
    for c in candidates[:20]:  # Log top 20
        logs.append(f"  {c['exact_code']} ({c['similarity_score']:.3f}) — {c['description'][:60]}")
    if on_progress: on_progress(0.55)

    if not candidates:
        logs.append("ERROR: No codes matched above similarity threshold.")
        if on_progress: on_progress(1.0)
        return {"status": "failed", "logs": logs, "error": "No codes matched above similarity threshold"}

    # Step 3: Final LLM ranking (single LLM call)
    logs.append(f"\nStep 3: LLM final ranking of {len(candidates)} candidates...")
    try:
        final_report = step_3_final_ranker(refined_text, candidates)
    except Exception as e:
        logger.error("Step 3 final ranker failed: %s", e, exc_info=True)
        logs.append(f"ERROR: Final ranking failed: {e}")
        if on_progress: on_progress(1.0)
        return {"status": "failed", "logs": logs, "error": str(e)}

    logs.append(f"Final report: {len(final_report.ranked_codes)} codes")
    if on_progress: on_progress(0.8)

    # Post-Process: Enrich with Medicare metadata
    enriched_results = []
    crosswalk_df = _load_crosswalk()
    for code_obj in final_report.ranked_codes:
        code_dict = code_obj.model_dump()
        # Normalize: strip dots for crosswalk lookup (crosswalk stores codes without dots)
        lookup_code = code_obj.exact_code.replace(".", "")
        match = crosswalk_df[crosswalk_df['ICD10_Code'] == lookup_code]
        if not match.empty:
            row = match.iloc[0]
            code_dict['PDPM_Category'] = row['PDPM_Category']
            code_dict['SLP_Comorbidity'] = bool(row.get('SLP_Comorbidity', False))
            code_dict['Major_Procedure'] = row.get('Major_Procedure', 'N/A')
        else:
            code_dict['PDPM_Category'] = "Unmapped"
            code_dict['SLP_Comorbidity'] = False
            code_dict['Major_Procedure'] = "N/A"

        nta_info = get_nta_detail_for_code(lookup_code)
        if nta_info:
            code_dict['NTA_Comorbidity_Name'] = nta_info['comorbidity_name']
            code_dict['NTA_MDS_Field'] = nta_info['mds_field']
            code_dict['is_nta'] = True
        else:
            code_dict['NTA_Comorbidity_Name'] = None
            code_dict['NTA_MDS_Field'] = None
            code_dict['is_nta'] = False

        enriched_results.append(code_dict)

    # NTA summary
    nta_summary = score_nta_comorbidities(enriched_results)
    logs.append(f"NTA: {nta_summary['total_categories']} categories, {nta_summary['total_nta_points']} points")

    # QM Defense
    primary_codes = [c for c in enriched_results if c['designation'] == 'Primary']
    qm_warnings = []
    if primary_codes and primary_codes[0].get('PDPM_Category') == 'Unmapped':
        qm_warnings.append("Primary diagnosis is not mapped to a PDPM clinical category.")

    # Source-reference fallback
    if page_chunks:
        for code_dict in enriched_results:
            doc_ref = code_dict.get('document_reference', '')
            quote = code_dict.get('clinical_evidence_quote', '')
            if quote and '---[' not in doc_ref:
                source = find_source_for_quote(quote, page_chunks)
                if source:
                    code_dict['document_reference'] = f"{source}, {doc_ref}" if doc_ref else source

    if on_progress: on_progress(1.0)

    return {
        "status": "success",
        "report": enriched_results,
        "logs": logs,
        "nta_summary": nta_summary,
        "qm_warnings": qm_warnings,
        "total_nta_points": nta_summary['total_nta_points'],
        "extracted_conditions": [c.model_dump() for c in extracted.conditions],
        "refinement": refined.model_dump() if refined else None,
        "raw_note": note,
    }
