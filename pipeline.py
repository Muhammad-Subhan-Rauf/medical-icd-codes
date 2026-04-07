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
Convert ALL medical shorthand, acronyms, and abbreviations to their full standard medical terminology.
Use your clinical knowledge to interpret abbreviations in context. Log each expansion in abbreviations_expanded as "ABBR -> Full Term".

TASK 2 — STRUCTURE INTO SECTIONS:
Organize the content into standard clinical sections (HPI, PMH, Medications, Labs, Assessment/Plan, Physical Exam, Social History, ROS, Procedures/Surgical History, Discharge Diagnoses, etc.).
Only create sections that have content in the original note. If the original note does not clearly separate sections, infer section placement from content type.

TASK 3 — NORMALIZE TERMINOLOGY:
Convert lay terms, colloquialisms, and non-standard medical language to standard ICD-10-CM compatible medical terminology.

TASK 4 — INFER IMPLIED CONDITIONS:
Infer conditions from medications and lab values ONLY when the clinical implication is standard and well-established.
- For each medication, consider what condition it is most commonly prescribed for.
- For abnormal lab values, consider what clinical condition they indicate.
- Rate each inference as HIGH, MODERATE, or LOW confidence based on how directly the evidence implies the condition.
- Do NOT infer conditions from lab values that are noted as artifactual, hemolyzed, or unreliable.
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

def extract_all_conditions(note: str, sections: list[dict] = None,
                           already_extracted: list[str] = None) -> ExtractedConditions:
    client = _get_client()

    # Format sections for the prompt
    if sections:
        formatted_sections = "\n\n".join(
            f"### {s['section_name']}\n{s['content']}" for s in sections
        )
    else:
        formatted_sections = note

    # Build context of already-extracted conditions (for batched processing)
    already_extracted_text = ""
    if already_extracted:
        already_extracted_text = f"""
ALREADY EXTRACTED (from previous pages) — DO NOT re-extract these conditions.
Only extract NEW conditions not already in this list:
{chr(10).join(f'- {c}' for c in already_extracted)}

"""

    prompt = f"""You are an expert medical coder. Extract EVERY medical condition, diagnosis, comorbidity, and clinical finding from this clinical note.
{already_extracted_text}
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
- Be STRICT about "historical" — use this category for:
  * Conditions prefixed with "history of", "h/o", "status post", "s/p", "prior", "previous", "old"
  * Conditions in the Past Medical History that are no longer active or treated
  * Resolved conditions (e.g., "AKI resolved", "fracture healed", "infection cleared")
  * Conditions from years ago with no current treatment (e.g., "pancreatitis 2007", "seizure-like activity")
  * Healed surgical procedures (e.g., "prior hip replacement", "appendectomy 2010")
- Only categorize as "chronic" if the condition is CURRENTLY being managed with medications or ongoing monitoring.
- Be thorough — capture every condition mentioned, even if only in the past medical history or problem list.
- Include conditions that may seem minor but are billable (functional limitations, nutritional disorders, chronic pain, etc.).

MEDICATION-IMPLIED CONDITIONS:
- If a patient is on a medication that treats a specific condition, and no other reason for that medication is documented, extract the implied condition.
- Use your clinical knowledge to infer the most likely condition from each medication class.
- Mark medication-implied conditions with source_section: "Medications".

LAB-IMPLIED CONDITIONS:
- Extract conditions when lab values are clearly outside normal range AND clinically significant.
- Do NOT extract conditions from lab values that are noted as artifactual, hemolyzed, contaminated, or otherwise unreliable.
- Do NOT extract conditions from borderline values that normalized without treatment.

ABBREVIATION HANDLING:
- Expand all medical abbreviations to their full terminology.
- Use standard ICD-10-CM compatible terminology.

NEGATION HANDLING — DO NOT extract negated conditions:
- Phrases like "denies", "no evidence of", "negative for", "ruled out", "absent" indicate the condition is NOT present.
- Only extract conditions that are PRESENT, ACTIVE, HISTORICAL, or INCIDENTAL — never negated ones.

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

=== ICD-10-CM OFFICIAL CODING GUIDELINES (MUST FOLLOW) ===

1. SYMPTOM EXCLUSION:
   - Do NOT code symptoms when the underlying diagnosis explaining that symptom is already coded.
   - Do NOT code abnormal lab findings as standalone diagnoses when the condition causing them is coded.
   - Include symptom codes ONLY when no underlying diagnosis in the final code list explains them.

2. MANDATORY SEQUENCING ("Code First" / "Code Also" / "Use Additional Code"):
   - When ICD-10-CM requires dual or multiple coding for a condition, you MUST include ALL required codes.
   - Etiology/manifestation pairs require BOTH codes — the underlying disease AND the manifestation.
   - Conditions with behavioral or functional manifestations require the underlying disease code PLUS the manifestation code.
   - Sepsis and severe sepsis/septic shock require the full causal chain: the systemic infection, the localized source, and the severity code.
   - Always follow the sequencing order specified by coding conventions.

3. SPECIFICITY — ALWAYS CODE TO THE HIGHEST LEVEL OF DETAIL:
   - Use the most specific code available based on the documentation.
   - Include laterality (left/right/bilateral) when documented.
   - Include anatomical site when documented.
   - Include severity, stage, or type when documented.
   - Include episode of care (initial encounter, subsequent encounter, sequela) when applicable.
   - Combination codes: When a single code captures both the condition and its complication or manifestation, use the combination code instead of two separate codes.

4. WOUND AND INJURY CODING:
   - Code wounds, ulcers, and skin lesions with maximum specificity: site, laterality, depth, and stage.
   - Infected wounds require BOTH the wound code AND the infection code.
   - Verify laterality carefully — confirm left vs right from the clinical documentation.

5. LAB VALUE CODING:
   - Only code abnormal lab values as diagnoses when they are clinically significant AND acknowledged by the provider.
   - Do NOT code lab artifacts (e.g., hemolyzed specimens, lipemic samples, contaminated cultures).
   - If the note states a result is spurious, artifactual, or unreliable, do NOT code it.
   - Transient lab abnormalities that normalized without treatment may not warrant a code.

6. HISTORICAL vs ACTIVE CONDITIONS:
   - Do NOT code conditions that are purely historical with no current clinical relevance:
     * Conditions documented only as distant history with no ongoing treatment or monitoring
     * Healed injuries, resolved infections, or completed treatments from years prior
     * Prior surgeries — code the presence of implants/devices if applicable, but NOT the original condition that led to the surgery
   - DO code historical conditions when they remain clinically relevant:
     * Any condition the patient is CURRENTLY taking medication for
     * Any condition being actively monitored with labs, imaging, or specialist visits
     * Any condition that impacts current functional status, fall risk, or care planning
     * Personal history codes (Z85-Z87 series) when the history affects current management
     * Sequelae of prior conditions when residual effects persist (e.g., neurological deficits from a prior event)
     * Chronic progressive conditions documented in PMH that do not resolve (e.g., degenerative diseases, bone density loss, vision disorders, cognitive disorders)

7. COMPLETENESS CHECK:
   - After reviewing the candidate codes, scan the ENTIRE clinical note for documented conditions that may not appear in the candidate list.
   - Check PMH, problem lists, medication lists, and discharge diagnoses for conditions with clear documentation.
   - You may add codes not in the candidate list if they have clear documentation in the note.
   - Ensure medication-implied conditions are captured: if a patient is on a medication that treats a specific condition, and no other reason for that medication is documented, code the implied condition.

8. CONFIDENCE THRESHOLD:
   - Only include codes with DEFINITE or PROBABLE documentation.
   - Do NOT code conditions described as "possible", "suspected", "concern for", "rule out", or "cannot exclude" UNLESS the condition was actively treated.
   - For inpatient stays: uncertain diagnoses may be coded as if confirmed per ICD-10-CM inpatient guidelines.
   - If documentation is speculative AND no treatment was directed at it, do NOT include it.

9. EXCLUDES NOTES AND CODE CONFLICTS:
   - Do not assign two codes that are mutually exclusive per ICD-10-CM Excludes1 notes.
   - Do not assign redundant codes when one code fully encompasses another.

10. EXTERNAL CAUSE AND STATUS CODES:
    - Include external cause codes when the documentation describes how an injury or condition occurred.
    - Include status codes (presence of implants, devices, transplants) when documented.

=== CLASSIFICATION RULES ===

1. DESIGNATION (STRICT — READ CAREFULLY):
   - "Primary": The SINGLE principal diagnosis that MOST DIRECTLY drove or occasioned the admission. This is the ONE condition that is the primary reason the patient needs care. Almost always exactly ONE code is Primary.
   - A second Primary is allowed ONLY if two conditions were truly co-equal, inseparable reasons for admission (extremely rare). A third Primary is virtually never appropriate.
   - "Secondary": EVERYTHING else — all other active diagnoses, complications during the stay, chronic comorbidities, and incidental findings.
   - Complications that occurred DURING the hospital stay are ALWAYS Secondary — they are complications, not the reason for admission.
   - Sequencing codes required by convention are Secondary — they exist for coding compliance, not because they independently drove the admission.
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
# BATCH HELPERS
# -------------------------------------------------------------

def _split_note_into_batches(note: str, batch_size: int = 10) -> list[str]:
    """Split a note with page markers into batches of `batch_size` pages.
    If no page markers found, split by character count."""
    import re
    # Split on page markers: ---[filename pN]---
    parts = re.split(r'(---\[.+? p\d+\]---)', note)
    # Reconstruct pages: marker + content pairs
    pages = []
    for i, part in enumerate(parts):
        if re.match(r'---\[.+? p\d+\]---', part):
            content = parts[i + 1] if i + 1 < len(parts) else ""
            pages.append(part + "\n" + content)

    if not pages:
        # No page markers — split by rough char limit (~15K per batch)
        char_limit = 15000
        batches = []
        for i in range(0, len(note), char_limit):
            batches.append(note[i:i + char_limit])
        return batches

    # Group pages into batches
    batches = []
    for i in range(0, len(pages), batch_size):
        batch = "\n\n".join(pages[i:i + batch_size])
        batches.append(batch)
    return batches


def _deduplicate_conditions(all_conditions: list[ExtractedCondition],
                            similarity_threshold: float = 0.85) -> list[ExtractedCondition]:
    """Deduplicate conditions using semantic similarity (embedding-based).

    Two conditions are considered duplicates if their embedding cosine similarity
    exceeds `similarity_threshold`. Keeps the first occurrence (which has
    more context from earlier batches).
    """
    if not all_conditions:
        return []

    initialize_system()

    # Build embedding queries for all conditions
    queries = [build_embedding_query(c) for c in all_conditions]
    vectors = _embedder.encode(queries, convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(vectors)

    # Greedy dedup: iterate in order, skip any condition too similar to an already-kept one
    kept_indices = []
    kept_vectors = []
    for i, vec in enumerate(vectors):
        is_dup = False
        if kept_vectors:
            import numpy as np
            kept_mat = np.stack(kept_vectors)
            sims = kept_mat @ vec  # cosine similarities (vectors are normalized)
            if sims.max() >= similarity_threshold:
                is_dup = True
        if not is_dup:
            kept_indices.append(i)
            kept_vectors.append(vec)

    return [all_conditions[i] for i in kept_indices]


# -------------------------------------------------------------
# ORCHESTRATION PIPELINE
# -------------------------------------------------------------

def process_patient_note(note, top_k_per_condition=5, similarity_threshold=0.45,
                          on_progress=None, page_chunks=None, batch_size=10):
    """
    Main pipeline orchestrator with batched processing for large notes.

    Architecture:
    - For notes with page markers: split into batches of `batch_size` pages
    - Each batch: refine + extract conditions (separate LLM calls)
    - Merge + deduplicate conditions across batches
    - FAISS code matching on merged conditions
    - Final LLM ranking on candidate codes
    """
    initialize_system()
    logs = []

    # Split into batches
    batches = _split_note_into_batches(note, batch_size)
    num_batches = len(batches)
    logs.append(f"Note split into {num_batches} batch(es) of up to {batch_size} pages")

    # Steps 0+1 per batch: Refine and extract conditions
    all_conditions = []
    all_refined_text_parts = []
    refined = None

    for batch_idx, batch_text in enumerate(batches):
        batch_label = f"Batch {batch_idx + 1}/{num_batches}"
        logs.append(f"\n{batch_label}: {len(batch_text)} chars")

        # Step 0: Refine
        batch_refined_text = batch_text
        try:
            batch_refined = refine_clinical_note(batch_text)
            batch_refined_text = batch_refined.merged_note
            if batch_idx == 0:
                refined = batch_refined  # Keep first batch's refinement for metadata
            logs.append(f"  {batch_label} refined: {len(batch_refined.sections)} sections")
        except Exception as e:
            logger.warning("%s refinement failed: %s, using raw text", batch_label, e)
            logs.append(f"  {batch_label} refinement failed, using raw text")
        all_refined_text_parts.append(batch_refined_text)

        # Step 1: Parse sections + extract conditions (with cross-batch context)
        sections = parse_sections(batch_refined_text)
        already_names = [c.condition for c in all_conditions] if all_conditions else None
        try:
            extracted = extract_all_conditions(batch_refined_text, sections,
                                               already_extracted=already_names)
            all_conditions.extend(extracted.conditions)
            logs.append(f"  {batch_label} extracted: {len(extracted.conditions)} new conditions")
        except Exception as e:
            logger.warning("%s extraction failed: %s", batch_label, e)
            logs.append(f"  {batch_label} extraction failed: {e}")

        if on_progress:
            on_progress(0.1 + 0.3 * (batch_idx + 1) / num_batches)

    if not all_conditions:
        logs.append("ERROR: No conditions extracted from any batch.")
        if on_progress: on_progress(1.0)
        return {"status": "failed", "logs": logs, "error": "No conditions extracted from any batch"}

    # Deduplicate conditions across batches
    before_dedup = len(all_conditions)
    all_conditions = _deduplicate_conditions(all_conditions)
    logs.append(f"\nMerged conditions: {before_dedup} total -> {len(all_conditions)} unique")

    # Merge refined text for final ranking context
    merged_refined = "\n\n".join(all_refined_text_parts)
    # Cap merged text for the final ranker to avoid token limits
    MAX_RANKER_CHARS = 100000
    if len(merged_refined) > MAX_RANKER_CHARS:
        logs.append(f"Truncating merged text from {len(merged_refined)} to {MAX_RANKER_CHARS} chars for final ranking")
        merged_refined = merged_refined[:MAX_RANKER_CHARS]

    # Step 1c: Embed-back validation
    logs.append("\nStep 1c: Validating extraction quality via embedding check...")
    validated_conditions = validate_extractions(all_conditions, _code_index, _embedder)
    if on_progress: on_progress(0.45)

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

    # Cap candidates to avoid output token overflow
    MAX_CANDIDATES = 80
    if len(candidates) > MAX_CANDIDATES:
        logs.append(f"Capping candidates from {len(candidates)} to {MAX_CANDIDATES} (by similarity score)")
        candidates = candidates[:MAX_CANDIDATES]

    # Step 3: Final LLM ranking (single LLM call, with retry on shorter note)
    logs.append(f"\nStep 3: LLM final ranking of {len(candidates)} candidates...")
    final_report = None
    for attempt, note_limit in enumerate([MAX_RANKER_CHARS, 50000, 25000]):
        ranker_text = merged_refined[:note_limit]
        try:
            final_report = step_3_final_ranker(ranker_text, candidates)
            if attempt > 0:
                logs.append(f"  Succeeded on attempt {attempt + 1} with {note_limit} char note")
            break
        except Exception as e:
            logs.append(f"  Attempt {attempt + 1} failed ({note_limit} chars): {e}")
            logger.warning("Step 3 attempt %d failed: %s", attempt + 1, e)

    if final_report is None:
        logs.append("ERROR: Final ranking failed after all retry attempts")
        if on_progress: on_progress(1.0)
        return {"status": "failed", "logs": logs, "error": "Final ranking failed after retries"}

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
        "extracted_conditions": [c.model_dump() for c in all_conditions],
        "refinement": refined.model_dump() if refined else None,
        "raw_note": note,
    }
