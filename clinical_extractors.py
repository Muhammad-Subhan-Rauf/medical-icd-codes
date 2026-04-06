"""
Clinical context extractors: demographics, hospital stay verification, medications.
Each runs as a separate Gemini call and can execute in parallel.
"""

import os
import logging
from pydantic import BaseModel
from google import genai
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger("medic.clinical_extractors")

MODEL_FAST = "gemini-2.5-flash"

_client = None


def _get_client():
    global _client
    if _client is None:
        api_key = os.environ.get("GEMINI_API_KEY")
        _client = genai.Client(api_key=api_key)
    return _client


# ---------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------

class PatientDemographics(BaseModel):
    patient_name: str | None = None
    date_of_birth: str | None = None
    payor_type: str | None = None
    referring_provider: str | None = None
    hospital_name: str | None = None


class HospitalStayVerification(BaseModel):
    admission_date: str | None = None
    discharge_date: str | None = None
    inpatient_midnights: int | None = None
    qualifies_3_midnight: bool | None = None
    observation_days_noted: bool = False
    evidence_quotes: list[str] = []


class Medication(BaseModel):
    name: str
    category: str  # Anticoagulant, Antibiotic, Antidiabetic, Opioid, etc.
    evidence_quote: str


class MedicationReport(BaseModel):
    medications: list[Medication] = []


# ---------------------------------------------------------------
# Extractor Functions
# ---------------------------------------------------------------

def extract_demographics(note: str) -> PatientDemographics:
    """Extract patient demographics from clinical note."""
    try:
        client = _get_client()
        prompt = """Extract patient demographics from this clinical note. Return null for any field not found.

Fields to extract:
- patient_name: Full name of the patient
- date_of_birth: Date of birth in any format found
- payor_type: Insurance/payor (e.g., Medicare, Medicaid, Commercial)
- referring_provider: Name and credentials of the referring/attending physician
- hospital_name: Name of the hospital or facility

CLINICAL NOTE:
""" + note

        response = client.models.generate_content(
            model=MODEL_FAST,
            contents=prompt,
            config={"response_mime_type": "application/json", "response_schema": PatientDemographics, "temperature": 0.0}
        )
        return PatientDemographics.model_validate_json(response.text)
    except Exception as e:
        logger.error("Demographics extraction failed: %s", e)
        return PatientDemographics()


def extract_hospital_stay(note: str) -> HospitalStayVerification:
    """Extract and verify 3-midnight qualifying hospital stay."""
    try:
        client = _get_client()
        prompt = """Analyze this clinical note to verify the qualifying hospital stay for Medicare Part A SNF coverage.

Extract:
- admission_date: The hospital admission date (any format found in the note)
- discharge_date: The hospital discharge date (any format found in the note)
- inpatient_midnights: Count the number of inpatient midnights between admission and discharge. Each calendar midnight the patient was an inpatient counts as 1 midnight.
- qualifies_3_midnight: true if inpatient_midnights >= 3, false otherwise
- observation_days_noted: true if the note mentions observation status or observation days
- evidence_quotes: Extract the exact verbatim quotes from the note that document the admission date, discharge date, and any observation status mentions

IMPORTANT: Only count inpatient midnights. Observation days do NOT count toward the 3-midnight requirement.

CLINICAL NOTE:
""" + note

        response = client.models.generate_content(
            model=MODEL_FAST,
            contents=prompt,
            config={"response_mime_type": "application/json", "response_schema": HospitalStayVerification, "temperature": 0.0}
        )
        return HospitalStayVerification.model_validate_json(response.text)
    except Exception as e:
        logger.error("Hospital stay extraction failed: %s", e)
        return HospitalStayVerification()


def extract_medications(note: str) -> MedicationReport:
    """Extract and categorize medications from clinical note."""
    try:
        client = _get_client()
        prompt = """Extract ALL medications mentioned in this clinical note. For each medication, provide:
- name: The medication name (generic or brand as documented)
- category: One of: Anticoagulant, Antibiotic, Antidiabetic, Opioid, Antihypertensive, Diuretic, Cardiac, Respiratory, Psychiatric, Analgesic, Steroid, Immunosuppressant, Other
- evidence_quote: The exact verbatim quote from the note where this medication is mentioned

Include medications from all sections: active medications, discharge medications, medication reconciliation, and medications mentioned in the history or treatment plan.

CLINICAL NOTE:
""" + note

        response = client.models.generate_content(
            model=MODEL_FAST,
            contents=prompt,
            config={"response_mime_type": "application/json", "response_schema": MedicationReport, "temperature": 0.0}
        )
        return MedicationReport.model_validate_json(response.text)
    except Exception as e:
        logger.error("Medication extraction failed: %s", e)
        return MedicationReport()


def extract_all_clinical_context(note: str) -> dict:
    """Run all 3 clinical extractors in parallel. Returns combined dict."""
    results = {
        "demographics": PatientDemographics(),
        "hospital_stay": HospitalStayVerification(),
        "medications": MedicationReport(),
    }

    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {
            executor.submit(extract_demographics, note): "demographics",
            executor.submit(extract_hospital_stay, note): "hospital_stay",
            executor.submit(extract_medications, note): "medications",
        }
        for future in as_completed(futures):
            key = futures[future]
            try:
                results[key] = future.result()
            except Exception as e:
                logger.error("Clinical extractor '%s' failed: %s", key, e)

    return results
