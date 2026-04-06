# Medical ICD-10 Coding Pipeline

An AI-powered medical coding system that automatically extracts clinical conditions from patient notes and maps them to ICD-10-CM codes. Built for Medicare SNF/PDPM workflows, it combines Google's Gemini LLM with FAISS vector search over 71,000+ ICD-10-CM codes to produce auditable, evidence-backed coding reports.

## Features

- **PDF ingestion** with page-level source tracking and Gemini Vision OCR fallback for scanned documents
- **Structured condition extraction** using Pydantic models — body site, laterality, severity, and episode of care
- **FAISS semantic search** over the full ICD-10-CM code set (~71K codes) using BioLORD-2023 medical embeddings
- **Medicare PDPM enrichment** — PDPM clinical categories, NTA comorbidity scoring, SLP comorbidity flags, and MDS field mapping
- **Streamlit dashboard** with real-time pipeline progress, downloadable CSV/JSON reports, and full agent logic trace

## Pipeline Architecture

The pipeline executes in 4 stages — 3 LLM calls and 1 programmatic FAISS search:

```
Raw Note / PDF  ──>  Step 0: Refine  ──>  Step 1: Extract  ──>  Step 2: FAISS Match  ──>  Step 3: Rank  ──>  Report
                      (LLM call 1)         (LLM call 2)         (programmatic)           (LLM call 3)
```

### Step 0 — Clinical Note Refinement (LLM)

The raw clinical note is preprocessed by Gemini to:
- Expand medical abbreviations (e.g., `HTN` → `hypertension`, `DM2` → `type 2 diabetes mellitus`)
- Structure the note into standard clinical sections (HPI, PMH, Medications, Labs, Assessment/Plan, etc.)
- Normalize terminology to align with ICD-10-CM conventions
- Infer implied conditions from medications and lab values (e.g., `metformin` → type 2 diabetes, `HbA1c > 6.5%` → diabetes)

Output: A structured `RefinedClinicalNote` with sections, inferred conditions, expanded abbreviations, and warnings.

### Step 1 — Condition Extraction (LLM)

Gemini extracts every medical condition, diagnosis, and comorbidity from the refined note. Each condition is returned as a structured object with:
- **condition** — standard medical name aligned to ICD-10-CM chapter terminology
- **category** — `acute`, `chronic`, `historical`, or `incidental`
- **body_site**, **laterality**, **severity**, **episode** — structured fields for code specificity
- **source_section** — which clinical section the condition was found in

An embed-back validation step then checks each extracted condition against the FAISS index to flag any that don't align well with ICD-10 terminology.

### Step 2 — FAISS Code Matching (Programmatic)

No LLM call — purely vector search. Each extracted condition is:
1. Converted to an embedding query from its structured fields
2. Encoded using the [BioLORD-2023](https://huggingface.co/FremyCompany/BioLORD-2023) biomedical sentence transformer
3. Searched against a FAISS `IndexFlatIP` index of 71K ICD-10-CM code embeddings (cosine similarity)
4. Filtered by a configurable similarity threshold (default 0.45)

Result: A deduplicated list of candidate ICD-10-CM codes ranked by similarity score.

### Step 3 — LLM Final Ranking & Classification (LLM)

Gemini acts as a CDI specialist and certified coder, reviewing all candidates against the clinical note to:
- **Accept or reject** each code based on clinical evidence
- **Classify** each code: Primary vs. Secondary designation, Active vs. Resolved status, Acute vs. Chronic acuity
- **Extract verbatim evidence quotes** from the note for each code
- **Enforce ICD-10-CM conventions** — symptom exclusion rules, sepsis sequencing, wound coding specificity
- **Add missing codes** for common chronic conditions documented in the note but not in the candidate list
- **Assign confidence scores** (0–100) based on documentation strength

### Post-Processing — Medicare PDPM Enrichment

After the LLM ranking, each accepted code is enriched with CMS PDPM metadata:
- **PDPM Clinical Category** from the FY2026 CMS crosswalk
- **NTA Comorbidity** scoring — maps codes to NTA categories, MDS fields, and computes aggregate NTA points (capped at 8 per CMS rules)
- **SLP Comorbidity** flags for speech-language pathology classification
- **Document references** — if PDFs were uploaded, evidence quotes are traced back to specific pages

## Tech Stack

| Component | Technology |
|---|---|
| LLM | Google Gemini 2.5 Flash |
| Embeddings | BioLORD-2023 (768-dim, biomedical) |
| Vector Search | FAISS (IndexFlatIP, cosine similarity) |
| Structured Output | Pydantic models + Gemini JSON mode |
| PDF Extraction | PyMuPDF + Gemini Vision OCR fallback |
| Frontend | Streamlit |
| CMS Data | PDPM ICD-10-CM Mappings FY2026 |

## Setup

### Prerequisites

- Python 3.10+
- A [Google Gemini API key](https://aistudio.google.com/apikey)

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Muhammad-Subhan-Rauf/medical-icd-codes.git
   cd medical-icd-codes
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv

   # Windows
   venv\Scripts\activate

   # macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up your API key:**

   Copy the example env file and add your Gemini API key:
   ```bash
   cp .env.example .env
   ```
   Edit `.env` and replace the placeholder with your actual key:
   ```
   GEMINI_API_KEY=your_actual_gemini_api_key
   ```

### Running the App

```bash
streamlit run app.py
```

On first launch, the system will build two FAISS indices (~71K code-level + ~20K group-level embeddings). This takes a few minutes and is cached for subsequent runs.

### Usage

1. **Upload PDFs** — drag and drop clinical PDFs (discharge summaries, medication lists, etc.) or paste a clinical note directly into the text area
2. **Configure** — adjust "Codes per Condition" and "Similarity Threshold" in the sidebar
3. **Analyze** — click "Analyze & Match Code" to run the pipeline
4. **Review** — the dashboard shows primary diagnosis, NTA comorbidities, active/resolved diagnosis lists, PDPM compliance checks, and the full agent logic trace
5. **Download** — export results as CSV or JSON

## Project Structure

```
├── app.py                  # Streamlit frontend and dashboard
├── pipeline.py             # Main pipeline orchestrator (4-stage architecture)
├── indexer.py              # FAISS index builder for ICD-10-CM codes
├── clinical_extractors.py  # Parallel clinical context extractors (demographics, medications, hospital stay)
├── nta_scoring.py          # NTA comorbidity scoring and MDS field mapping
├── pdf_extractor.py        # PDF text extraction with OCR fallback
├── ICD10codes.csv          # Full ICD-10-CM code set (~71K codes)
├── mappings/               # CMS PDPM FY2026 crosswalk CSVs
│   ├── PDPM-ICD10-Mappings-FY2026-Clinical-Categories-by-Dx.csv
│   ├── PDPM-ICD10-Mappings-FY2026-NTA-Comorbidity.csv
│   ├── PDPM-ICD10-Mappings-FY2026-SLP-Comorbidity.csv
│   └── PDPM-ICD10-Mappings-FY2026-Overview.csv
├── requirements.txt
└── .env.example
```
