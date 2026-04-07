"""
ICD-10-CM Pipeline Cross-Verification Evaluation.

Verifies pipeline ICD-10 codes against MDS expert confirmed/rejected conditions.
No intermediate gold standard — directly checks if each pipeline code corresponds
to an expert-confirmed, expert-rejected, or unreviewed condition.

Usage:
    python evaluate.py                          # Full evaluation
    python evaluate.py --skip-inference         # Recompute from cache
    python evaluate.py --patient Patient1       # Single patient
    python evaluate.py --debug                  # Verbose output
"""

import os
import sys
import json
import argparse
import logging
import time
from pathlib import Path
from typing import Literal
from pydantic import BaseModel
from dotenv import load_dotenv

import fitz  # PyMuPDF
import numpy as np
import faiss

load_dotenv()

logger = logging.getLogger("evaluate")
DEBUG = False

# --- Constants ---
BASE_DIR = Path(__file__).parent
PATIENTS_DIR = BASE_DIR / "Patients_with_MDS_Experts" / "Patients_with_MDS_Experts"
EVAL_DIR = BASE_DIR / "eval_results"
MAX_CHARS_PER_PAGE = 10000
MAX_NOTE_CHARS = 200000
CACHE_FILE = EVAL_DIR / "predictions_cache.json"
REPORT_FILE = EVAL_DIR / "eval_report.json"

MODEL_VERIFY = "gemini-3-pro-preview"        # Gemini 3 Pro for cross-verification (needs nuanced clinical judgment)
MODEL_PARSE = "gemini-3.1-flash-lite-preview"  # Flash Lite for expert PDF parsing (straightforward extraction)


# =====================================================================
# Pydantic models
# =====================================================================

class ExpertCondition(BaseModel):
    condition_name: str
    mds_code: str
    status: Literal["Confirmed", "Rejected", "Possible"]
    section: str
    nta_points: int | None = None
    evidence_quotes: list[str] = []

class ExpertReport(BaseModel):
    patient_name: str
    conditions: list[ExpertCondition]

class CodeVerification(BaseModel):
    icd10_code: str
    icd10_description: str
    verdict: Literal["confirmed", "rejected", "not_reviewed"]
    matched_expert_condition: str | None = None
    reasoning: str

class ConditionCoverage(BaseModel):
    condition_name: str
    mds_code: str
    covered: bool
    covering_codes: list[str] = []
    reasoning: str

class CrossVerificationReport(BaseModel):
    code_verifications: list[CodeVerification]
    condition_coverages: list[ConditionCoverage]


# =====================================================================
# Step 1: Parse MDS Expert PDFs
# =====================================================================

def extract_pdf_text(pdf_path: str) -> str:
    doc = fitz.open(pdf_path)
    text = "".join(page.get_text("text") + "\n" for page in doc)
    doc.close()
    return text


def parse_expert_pdf(pdf_path: str) -> ExpertReport:
    from google import genai
    text = extract_pdf_text(pdf_path)
    if not text.strip():
        return ExpertReport(patient_name="Unknown", conditions=[])

    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
    response = client.models.generate_content(
        model=MODEL_PARSE,
        contents=f"""Parse this MDS expert report into structured conditions.

For each condition entry, extract:
- condition_name: The condition description (e.g., "Asthma, COPD", "Anemia")
- mds_code: The MDS code (e.g., "I6200", "I0200", "M1040A")
- status: "Confirmed", "Rejected", or "Possible"
- section: Which section it appears under - "PDPM", "MDS Section I", "Quality", or "High Risk Meds"
- nta_points: NTA points if mentioned (integer or null)
- evidence_quotes: Key evidence quotes from the report

IMPORTANT:
- "Confirmed PDPM Conditions" and "Confirmed MDS Section I Conditions" -> status = "Confirmed"
- "Rejected PDPM Conditions" and "Rejected MDS Section I Conditions" -> status = "Rejected"
- "Possible PDPM Conditions For Further Clinical Review" -> status = "Possible"
- Extract the patient name from the report header

Report text:
{text}""",
        config={
            "temperature": 0.0,
            "response_mime_type": "application/json",
            "response_schema": ExpertReport,
        },
    )
    return response.parsed


# =====================================================================
# Step 2: FAISS Retrieval (secondary benchmark)
# =====================================================================

def map_conditions_to_icd10_faiss(
    conditions: list[ExpertCondition],
    top_k: int = 5,
    similarity_threshold: float = 0.45,
) -> list[dict]:
    from indexer import build_or_load_code_index
    from sentence_transformers import SentenceTransformer

    confirmed = [c for c in conditions if c.status == "Confirmed"]
    if not confirmed:
        return []

    code_index, code_list, _ = build_or_load_code_index()
    embedder = SentenceTransformer("FremyCompany/BioLORD-2023")

    results = []
    for cond in confirmed:
        query_vec = embedder.encode([cond.condition_name], convert_to_numpy=True).astype("float32")
        faiss.normalize_L2(query_vec)
        scores, indices = code_index.search(query_vec, top_k)

        matches = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or score < similarity_threshold:
                continue
            code_info = code_list[idx]
            matches.append({
                "code": code_info["Specific_Code"].replace(".", ""),
                "description": code_info["Specific_Description"],
                "similarity": float(score),
            })

        results.append({
            "condition_name": cond.condition_name,
            "mds_code": cond.mds_code,
            "faiss_codes": [m["code"] for m in matches],
            "faiss_details": matches,
        })
    return results


# =====================================================================
# Step 3: Run pipeline on discharge notes
# =====================================================================

def get_patient_discharge_pdfs(patient_dir: Path) -> list[Path]:
    exclude_dirs = {"mds_expert", "snf small prod report"}
    pdfs = []
    for f in patient_dir.iterdir():
        if f.is_file() and f.suffix.lower() == ".pdf":
            pdfs.append(f)
        elif f.is_dir() and f.name.lower() not in exclude_dirs:
            for sub_f in f.iterdir():
                if sub_f.is_file() and sub_f.suffix.lower() == ".pdf":
                    pdfs.append(sub_f)
    return sorted(pdfs)


def extract_note_from_pdfs(pdf_paths: list[Path]) -> tuple[str, list]:
    from pdf_extractor import extract_text_from_pdf, merge_chunks_to_note, PageChunk

    all_chunks = []
    for pdf_path in pdf_paths:
        txt_cache = pdf_path.with_suffix(".txt")

        if txt_cache.exists() and txt_cache.stat().st_size > 0:
            print(f"           {pdf_path.name}: cached")
            cached_text = txt_cache.read_text(encoding="utf-8")
            import re
            page_parts = re.split(r'---\[.+? p(\d+)\]---\n?', cached_text)
            if len(page_parts) > 1:
                for i in range(1, len(page_parts), 2):
                    page_num = int(page_parts[i])
                    text = page_parts[i + 1].strip() if i + 1 < len(page_parts) else ""
                    if text:
                        all_chunks.append(PageChunk(text=text, source_file=pdf_path.name, page_number=page_num))
            elif cached_text.strip():
                all_chunks.append(PageChunk(text=cached_text.strip(), source_file=pdf_path.name, page_number=1))
            # Filter oversized
            before = len([c for c in all_chunks if c.source_file == pdf_path.name])
            all_chunks = [c for c in all_chunks if c.source_file != pdf_path.name or len(c.text) <= MAX_CHARS_PER_PAGE]
            skipped = before - len([c for c in all_chunks if c.source_file == pdf_path.name])
            if skipped:
                print(f"           {pdf_path.name}: skipped {skipped} oversized pages")
        else:
            with open(pdf_path, "rb") as f:
                file_bytes = f.read()
            doc = fitz.open(stream=file_bytes, filetype="pdf")
            total_pages = len(doc)
            scanned = sum(1 for i in range(total_pages) if not doc[i].get_text("text").strip())
            doc.close()

            print(f"           {pdf_path.name}: {total_pages} pages ({scanned} scanned, OCR running...)")
            start = time.time()
            chunks = extract_text_from_pdf(file_bytes, pdf_path.name)
            good_chunks = [c for c in chunks if len(c.text) <= MAX_CHARS_PER_PAGE]
            skipped = len(chunks) - len(good_chunks)
            if skipped:
                print(f"           {pdf_path.name}: skipped {skipped} oversized pages")
            all_chunks.extend(good_chunks)
            print(f"           {pdf_path.name}: {len(good_chunks)} pages in {time.time() - start:.0f}s")
            txt_cache.write_text(merge_chunks_to_note(good_chunks), encoding="utf-8")

    merged = merge_chunks_to_note(all_chunks) if all_chunks else ""
    if len(merged) > MAX_NOTE_CHARS:
        merged = merged[:MAX_NOTE_CHARS]
    return merged, all_chunks


def run_pipeline_on_patient(patient_dir: Path, top_k: int, threshold: float) -> dict:
    from pipeline import process_patient_note, initialize_system
    initialize_system()

    pdf_paths = get_patient_discharge_pdfs(patient_dir)
    if not pdf_paths:
        return {"status": "failed", "error": "No discharge PDFs found"}

    print(f"           {len(pdf_paths)} PDF(s) to process")
    start = time.time()
    note, page_chunks = extract_note_from_pdfs(pdf_paths)
    if not note.strip():
        return {"status": "failed", "error": "No text extracted from PDFs"}

    print(f"           Note: {len(note):,} chars (extracted in {time.time() - start:.0f}s)")
    print(f"           Running pipeline (batched)...")
    pipeline_start = time.time()
    result = process_patient_note(note, top_k_per_condition=top_k,
                                   similarity_threshold=threshold, page_chunks=page_chunks)
    print(f"           Pipeline done in {time.time() - pipeline_start:.0f}s")
    return result


# =====================================================================
# Step 4: Cross-verification (single LLM call per patient)
# =====================================================================

def cross_verify_pipeline_codes(
    pipeline_report: list[dict],
    expert_conditions: list[ExpertCondition],
) -> CrossVerificationReport:
    """Verify each pipeline code against expert confirmed/rejected conditions."""
    from google import genai

    # Build pipeline codes list for prompt
    codes_text = "\n".join(
        f"- {c['exact_code']}: {c['description']}"
        for c in pipeline_report
    )

    # Build expert conditions lists
    confirmed = [c for c in expert_conditions if c.status == "Confirmed" and c.section != "High Risk Meds"]
    rejected = [c for c in expert_conditions if c.status == "Rejected"]

    confirmed_text = "\n".join(
        f"- [{c.mds_code}] {c.condition_name}"
        + (f"\n  Evidence: {'; '.join(c.evidence_quotes[:2])}" if c.evidence_quotes else "")
        for c in confirmed
    )

    rejected_text = "\n".join(
        f"- [{c.mds_code}] {c.condition_name}"
        + (f"\n  Reason: {'; '.join(c.evidence_quotes[:1])}" if c.evidence_quotes else "")
        for c in rejected
    )

    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

    response = client.models.generate_content(
        model=MODEL_VERIFY,
        contents=f"""You are a certified medical coding expert. Cross-verify a pipeline's ICD-10-CM codes
against an MDS expert's clinical review.

PIPELINE CODES (what the coding pipeline produced):
{codes_text}

EXPERT CONFIRMED CONDITIONS (expert verified these are present):
{confirmed_text}

EXPERT REJECTED CONDITIONS (expert determined these do NOT apply):
{rejected_text}

TASK 1 - CODE VERIFICATION:
For each pipeline code, determine:
- "confirmed": The code represents a condition the expert CONFIRMED. A match means the ICD-10 code
  is a reasonable representation of that condition, even if more or less specific.
  Example: Expert confirmed "Heart Failure" and pipeline has I50.32 (Chronic diastolic HF) = confirmed.
- "rejected": The code represents a condition the expert explicitly REJECTED, AND the clinical
  concept is truly the same. Be VERY CAREFUL with rejections:
  CRITICAL — read the expert's NOTE/REASON for rejection before deciding:
  * Expert rejected "Septicemia" with note "Resident has diagnosis of sepsis, no septicemia"
    → This means sepsis IS present but septicemia is NOT. Codes A41.x (sepsis) and R65.21
    (septic shock) are NOT rejected — they represent sepsis, which the expert confirms exists.
  * Expert rejected "Morbid Obesity" with note "has obesity, no morbid obesity"
    → Regular obesity codes (E66.x) are NOT rejected, only morbid obesity (E66.01).
  * Expert rejected "Renal Failure" with note "AKI resolved"
    → AKI codes (N17.x) may still be valid for coding resolved conditions during the stay.
    Only reject if the expert says the condition never existed at all.
  * Expert rejected "Wound Infection (other than foot)" with note "No record of wound infection"
    → Foot wound infection codes ARE still valid if confirmed separately.
  * Only mark "rejected" when the expert says the condition DOES NOT EXIST or NEVER EXISTED.
    If the expert rejected a BROADER category but the pipeline coded a NARROWER specific
    condition that IS documented, that is NOT a rejection — use "confirmed" or "not_reviewed".
- "not_reviewed": The code is for a condition the expert did not evaluate, OR the match is ambiguous.

TASK 2 - CONDITION COVERAGE:
For each expert-confirmed condition, determine if ANY pipeline code reasonably covers it.
A condition is covered if at least one pipeline code represents that clinical concept,
even with different specificity. List the covering code(s).

Be generous with matching on the confirmed side — focus on clinical meaning, not exact code specificity.
Be STRICT with rejections — only reject when the clinical concept is unambiguously the same.""",
        config={
            "temperature": 0.0,
            "response_mime_type": "application/json",
            "response_schema": CrossVerificationReport,
        },
    )
    return response.parsed


# =====================================================================
# Step 5: Metrics computation
# =====================================================================

def compute_cross_metrics(cross_report: CrossVerificationReport) -> dict:
    """Compute metrics from a cross-verification report."""
    verifications = cross_report.code_verifications
    coverages = cross_report.condition_coverages

    total_codes = len(verifications)
    confirmed_count = sum(1 for v in verifications if v.verdict == "confirmed")
    rejected_count = sum(1 for v in verifications if v.verdict == "rejected")
    not_reviewed_count = sum(1 for v in verifications if v.verdict == "not_reviewed")

    total_conditions = len(coverages)
    covered_count = sum(1 for c in coverages if c.covered)
    missed_count = total_conditions - covered_count

    confirmation_rate = confirmed_count / total_codes if total_codes else 0
    rejection_rate = rejected_count / total_codes if total_codes else 0
    not_reviewed_rate = not_reviewed_count / total_codes if total_codes else 0
    coverage_rate = covered_count / total_conditions if total_conditions else 0

    # Detail lists
    rejected_codes = [
        {"code": v.icd10_code, "description": v.icd10_description,
         "matched_condition": v.matched_expert_condition, "reasoning": v.reasoning}
        for v in verifications if v.verdict == "rejected"
    ]
    missed_conditions = [
        {"condition": c.condition_name, "mds_code": c.mds_code, "reasoning": c.reasoning}
        for c in coverages if not c.covered
    ]

    return {
        "total_codes": total_codes,
        "confirmed": confirmed_count,
        "rejected": rejected_count,
        "not_reviewed": not_reviewed_count,
        "confirmation_rate": round(confirmation_rate, 4),
        "rejection_rate": round(rejection_rate, 4),
        "not_reviewed_rate": round(not_reviewed_rate, 4),
        "total_conditions": total_conditions,
        "covered": covered_count,
        "missed": missed_count,
        "coverage_rate": round(coverage_rate, 4),
        "rejected_codes": rejected_codes,
        "missed_conditions": missed_conditions,
    }


def aggregate_cross_metrics(all_metrics: dict) -> dict:
    """Aggregate cross-verification metrics across patients."""
    n = len(all_metrics)
    if not n:
        return {}

    total_codes = sum(m["total_codes"] for m in all_metrics.values())
    total_confirmed = sum(m["confirmed"] for m in all_metrics.values())
    total_rejected = sum(m["rejected"] for m in all_metrics.values())
    total_not_reviewed = sum(m["not_reviewed"] for m in all_metrics.values())
    total_conditions = sum(m["total_conditions"] for m in all_metrics.values())
    total_covered = sum(m["covered"] for m in all_metrics.values())

    return {
        "total_patients": n,
        "total_codes": total_codes,
        "total_conditions": total_conditions,
        "confirmation_rate": round(total_confirmed / total_codes, 4) if total_codes else 0,
        "rejection_rate": round(total_rejected / total_codes, 4) if total_codes else 0,
        "not_reviewed_rate": round(total_not_reviewed / total_codes, 4) if total_codes else 0,
        "coverage_rate": round(total_covered / total_conditions, 4) if total_conditions else 0,
        "mean_confirmation_rate": round(sum(m["confirmation_rate"] for m in all_metrics.values()) / n, 4),
        "mean_rejection_rate": round(sum(m["rejection_rate"] for m in all_metrics.values()) / n, 4),
        "mean_coverage_rate": round(sum(m["coverage_rate"] for m in all_metrics.values()) / n, 4),
        "all_rejected_codes": [rc for m in all_metrics.values() for rc in m["rejected_codes"]],
        "all_missed_conditions": [mc for m in all_metrics.values() for mc in m["missed_conditions"]],
    }


# =====================================================================
# Step 6: Report generation
# =====================================================================

def print_report(aggregate: dict, per_patient: dict):
    print("\n" + "=" * 70)
    print("   ICD-10-CM PIPELINE CROSS-VERIFICATION REPORT")
    print("=" * 70)
    print(f"\nPatients evaluated: {aggregate['total_patients']}")
    print(f"Total pipeline codes verified: {aggregate['total_codes']}")
    print(f"Total expert conditions checked: {aggregate['total_conditions']}")

    print(f"\n--- Pipeline Code Verification ---")
    print(f"  Confirmation Rate: {aggregate['confirmation_rate']:.1%}  (codes matching expert-confirmed conditions)")
    print(f"  Rejection Rate:    {aggregate['rejection_rate']:.1%}  (codes matching expert-REJECTED conditions)")
    print(f"  Not-Reviewed Rate: {aggregate['not_reviewed_rate']:.1%}  (conditions expert didn't evaluate)")

    print(f"\n--- Expert Condition Coverage ---")
    print(f"  Coverage Rate: {aggregate['coverage_rate']:.1%}  (expert-confirmed conditions covered by pipeline)")

    if aggregate["all_rejected_codes"]:
        print(f"\n--- Rejected Code Matches (errors) ---")
        for rc in aggregate["all_rejected_codes"]:
            print(f"  {rc['code']}: {rc['matched_condition']} — {rc['reasoning']}")

    if aggregate["all_missed_conditions"]:
        print(f"\n--- Missed Conditions ---")
        for mc in aggregate["all_missed_conditions"]:
            print(f"  [{mc['mds_code']}] {mc['condition']} — {mc['reasoning']}")

    print(f"\n--- Per-Patient Breakdown ---")
    for patient_id, m in per_patient.items():
        print(f"\n  {patient_id}:")
        print(f"    Codes: {m['total_codes']}  |  Confirmed: {m['confirmed']}  |  Rejected: {m['rejected']}  |  Not-reviewed: {m['not_reviewed']}")
        print(f"    Confirmation: {m['confirmation_rate']:.0%}  |  Rejection: {m['rejection_rate']:.0%}  |  Coverage: {m['coverage_rate']:.0%} ({m['covered']}/{m['total_conditions']})")
        if m["rejected_codes"]:
            for rc in m["rejected_codes"]:
                print(f"    [REJECTED] {rc['code']}: {rc['matched_condition']}")
        if m["missed_conditions"]:
            for mc in m["missed_conditions"]:
                print(f"    [MISSED] {mc['condition']}")

    print("\n" + "=" * 70)


def save_report(aggregate: dict, per_patient: dict, cross_details: dict, output_path: Path):
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "aggregate_metrics": aggregate,
        "per_patient_metrics": per_patient,
        "cross_verification_details": cross_details,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2, default=str)


# =====================================================================
# Step 7: AI-generated analysis
# =====================================================================

def generate_ai_analysis(aggregate: dict, per_patient: dict, cross_details: dict) -> str:
    from google import genai

    summary = {
        "aggregate_metrics": aggregate,
        "per_patient_metrics": per_patient,
        "cross_verification_details": cross_details,
    }
    summary_json = json.dumps(summary, indent=2, default=str)

    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

    response = client.models.generate_content(
        model=MODEL_VERIFY,
        contents=f"""You are a medical coding quality analyst. Analyze these cross-verification results
for an ICD-10-CM coding pipeline evaluated against MDS expert clinical reviews.

METHODOLOGY:
- Each pipeline ICD-10 code was verified against the MDS expert's confirmed/rejected conditions
- "confirmed" = pipeline code matches an expert-confirmed condition (good)
- "rejected" = pipeline code matches an expert-REJECTED condition (error)
- "not_reviewed" = condition wasn't evaluated by the expert (neutral — may be valid from other docs)
- Coverage rate = % of expert-confirmed conditions covered by at least one pipeline code

Write a detailed markdown report:

## Executive Summary
- Overall pipeline quality assessment
- Key metrics: confirmation rate, rejection rate, coverage rate

## Per-Patient Analysis
- Best/worst performing patients and patterns
- What types of conditions are consistently covered or missed

## Error Analysis
- Any rejected code matches — are these real errors or debatable clinical judgments?
- Pattern in missed conditions — medication codes? rare conditions?

## Not-Reviewed Codes Analysis
- Are the "not_reviewed" codes likely valid clinical findings from the source documents?
- Or are they noise/over-coding?

## Recommendations
- Specific improvements to the pipeline
- Priority ranking by expected impact

Data:
{summary_json}""",
        config={"temperature": 0.2},
    )
    return response.text


# =====================================================================
# Main orchestrator
# =====================================================================

def discover_patients(patient_filter: str | None = None) -> list[Path]:
    if not PATIENTS_DIR.exists():
        return []
    patients = sorted([d for d in PATIENTS_DIR.iterdir() if d.is_dir()])
    if patient_filter:
        patients = [d for d in patients if d.name == patient_filter]
    return patients


def find_expert_pdf(patient_dir: Path) -> Path | None:
    for d in patient_dir.iterdir():
        if d.is_dir() and d.name.lower() == "mds_expert":
            pdfs = [f for f in d.iterdir() if f.suffix.lower() == ".pdf"]
            return pdfs[0] if pdfs else None
    return None


def _save_cache(cache: dict):
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f, indent=2, default=str)


def _process_single_patient(patient_dir, args, cache, cache_lock):
    """Process a single patient. Returns (patient_id, metrics, cross_detail, cache_updates) or None."""
    import threading

    patient_id = patient_dir.name
    patient_start = time.time()
    cache_updates = {}

    print(f"\n--- {patient_id} ---")

    # --- Step 1: Parse expert PDF ---
    expert_pdf = find_expert_pdf(patient_dir)
    if not expert_pdf:
        print(f"  {patient_id}: No MDS expert PDF found, skipping.")
        return None

    cache_key_expert = f"{patient_id}_expert"
    with cache_lock:
        cached_expert = cache.get(cache_key_expert)
    if cached_expert:
        expert_report = ExpertReport(**cached_expert)
    else:
        print(f"  {patient_id} [1] Parsing expert PDF...", end=" ", flush=True)
        step_start = time.time()
        expert_report = parse_expert_pdf(str(expert_pdf))
        cache_updates[cache_key_expert] = expert_report.model_dump()
        print(f"done ({time.time() - step_start:.0f}s)")

    confirmed = [c for c in expert_report.conditions if c.status == "Confirmed"]
    rejected = [c for c in expert_report.conditions if c.status == "Rejected"]
    possible = [c for c in expert_report.conditions if c.status == "Possible"]
    print(f"  {patient_id}: {len(confirmed)} confirmed, {len(rejected)} rejected, {len(possible)} possible")

    # --- Step 2: FAISS retrieval ---
    cache_key_faiss = f"{patient_id}_faiss_k{args.top_k}_t{args.threshold}"
    with cache_lock:
        cached_faiss = cache.get(cache_key_faiss)
    if not cached_faiss:
        print(f"  {patient_id} [2] FAISS retrieval...", end=" ", flush=True)
        step_start = time.time()
        faiss_results = map_conditions_to_icd10_faiss(
            expert_report.conditions, top_k=args.top_k, similarity_threshold=args.threshold)
        cache_updates[cache_key_faiss] = faiss_results
        print(f"done ({time.time() - step_start:.0f}s)")

    # --- Step 3: Run pipeline ---
    cache_key_pipeline = f"{patient_id}_pipeline_k{args.top_k}_t{args.threshold}"
    with cache_lock:
        cached_pipeline = cache.get(cache_key_pipeline)
    if cached_pipeline:
        pipeline_result = cached_pipeline
        print(f"  {patient_id} [3] Pipeline: cached ({len(pipeline_result.get('report', []))} codes)")
    else:
        print(f"  {patient_id} [3] Running pipeline...")
        step_start = time.time()
        pipeline_result = run_pipeline_on_patient(patient_dir, args.top_k, args.threshold)
        cache_updates[cache_key_pipeline] = json.loads(json.dumps(pipeline_result, default=str))
        print(f"  {patient_id} [3] Pipeline done in {time.time() - step_start:.0f}s")

    if pipeline_result.get("status") != "success":
        print(f"  {patient_id}: FAILED: {pipeline_result.get('error', 'unknown')[:80]}")
        # Still save cache updates
        with cache_lock:
            cache.update(cache_updates)
            _save_cache(cache)
        return None

    pipeline_report = pipeline_result.get("report", [])

    # --- Step 4: Cross-verify ---
    cache_key_xv = f"{patient_id}_cross_verify"
    with cache_lock:
        cached_xv = cache.get(cache_key_xv)
    if cached_xv:
        cross_report = CrossVerificationReport(**cached_xv)
    else:
        print(f"  {patient_id} [4] Cross-verifying...", end=" ", flush=True)
        step_start = time.time()
        cross_report = cross_verify_pipeline_codes(pipeline_report, expert_report.conditions)
        cache_updates[cache_key_xv] = cross_report.model_dump()
        print(f"done ({time.time() - step_start:.0f}s)")

    # Save all cache updates
    with cache_lock:
        cache.update(cache_updates)
        _save_cache(cache)

    # Compute metrics
    metrics = compute_cross_metrics(cross_report)
    print(f"  {patient_id}: Confirmed: {metrics['confirmation_rate']:.0%} | "
          f"Rejected: {metrics['rejection_rate']:.0%} | "
          f"Coverage: {metrics['coverage_rate']:.0%} | "
          f"{time.time() - patient_start:.0f}s")

    return (patient_id, metrics, cross_report.model_dump())


def run_evaluation(args):
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import threading

    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    patients = discover_patients(args.patient)
    if not patients:
        print("No patients found.")
        return

    workers = min(args.parallel, len(patients))
    print(f"Evaluating {len(patients)} patient(s) with {workers} worker(s)...")

    cache = {}
    if args.no_cache:
        print("(cache disabled — running everything fresh)")
    elif CACHE_FILE.exists():
        with open(CACHE_FILE) as f:
            cache = json.load(f)

    cache_lock = threading.Lock()
    per_patient_metrics = {}
    cross_details = {}

    if workers == 1:
        # Sequential — simpler output
        for patient_dir in patients:
            result = _process_single_patient(patient_dir, args, cache, cache_lock)
            if result:
                pid, metrics, xv_detail = result
                per_patient_metrics[pid] = metrics
                cross_details[pid] = xv_detail
    else:
        # Parallel
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(_process_single_patient, pd, args, cache, cache_lock): pd.name
                for pd in patients
            }
            for future in as_completed(futures):
                pid = futures[future]
                try:
                    result = future.result()
                    if result:
                        pid, metrics, xv_detail = result
                        per_patient_metrics[pid] = metrics
                        cross_details[pid] = xv_detail
                except Exception as e:
                    print(f"  {pid}: ERROR — {e}")

    # --- Aggregate and report ---
    if not per_patient_metrics:
        print("\nNo patients evaluated successfully.")
        return

    aggregate = aggregate_cross_metrics(per_patient_metrics)
    print_report(aggregate, per_patient_metrics)

    # Save reports
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    history_dir = EVAL_DIR / "history"
    history_dir.mkdir(parents=True, exist_ok=True)

    save_report(aggregate, per_patient_metrics, cross_details, REPORT_FILE)
    save_report(aggregate, per_patient_metrics, cross_details,
                history_dir / f"eval_report_{timestamp}.json")

    # --- AI analysis ---
    print(f"\nGenerating AI analysis...", end=" ", flush=True)
    step_start = time.time()
    ai_report = generate_ai_analysis(aggregate, per_patient_metrics, cross_details)
    ai_report_path = EVAL_DIR / "ai_analysis_report.md"
    ai_report_path.write_text(ai_report, encoding="utf-8")
    (history_dir / f"ai_analysis_{timestamp}.md").write_text(ai_report, encoding="utf-8")
    print(f"done ({time.time() - step_start:.0f}s)")

    print(f"\nReports saved to: {EVAL_DIR}/")
    print(f"  eval_report.json      — metrics + cross-verification details")
    print(f"  ai_analysis_report.md — AI analysis")
    print(f"  history/              — timestamped copies")

    print(f"\n{'='*60}")
    print(ai_report)
    print(f"{'='*60}")


def main():
    global DEBUG
    parser = argparse.ArgumentParser(description="ICD-10-CM Pipeline Cross-Verification Evaluation")
    parser.add_argument("--patient", type=str, default=None, help="Evaluate single patient (e.g., Patient1)")
    parser.add_argument("--top-k", type=int, default=5, help="FAISS top_k_per_condition")
    parser.add_argument("--threshold", type=float, default=0.45, help="FAISS similarity threshold")
    parser.add_argument("--skip-inference", action="store_true", help="Recompute metrics from cached results only")
    parser.add_argument("--no-cache", action="store_true", help="Ignore all cached results, run everything fresh")
    parser.add_argument("--parallel", type=int, default=1, help="Number of patients to process in parallel (default: 1)")
    parser.add_argument("--debug", action="store_true", help="Show verbose output")
    args = parser.parse_args()

    DEBUG = args.debug
    if DEBUG:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    else:
        logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
        for name in ["httpx", "httpcore", "sentence_transformers", "google", "medic", "urllib3"]:
            logging.getLogger(name).setLevel(logging.WARNING)

    run_evaluation(args)


if __name__ == "__main__":
    main()
