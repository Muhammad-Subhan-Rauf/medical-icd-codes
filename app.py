import streamlit as st
import os
import time
import json
import pandas as pd
from dotenv import load_dotenv
from indexer import build_or_load_index
from pipeline import process_patient_note
from pdf_extractor import extract_text_from_multiple_pdfs

# Set page config
st.set_page_config(
    page_title="Gemini RAG Medical Coder",
    page_icon="🏥",
    layout="wide"
)

# Load environment variables
load_dotenv()

# Check API key
if not os.environ.get("GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEY") == "your_gemini_api_key_here":
    st.error("Please set your GEMINI_API_KEY in the .env file.")
    st.stop()

@st.cache_resource(show_spinner="Initializing system and loading FAISS index... This may take a few minutes if building for the first time.")
def load_system():
    build_or_load_index()
    return True

st.title("🏥 Gemini RAG + Iterative Medical Coder")
st.markdown("""
This app implements a high-performance RAG pipeline utilizing Google's `gemini-2.5-flash` model, `Pydantic` structured outputs, and `FAISS` vector search.
""")

try:
    load_system()
except Exception as e:
    st.error(f"Failed to load the index: {e}")
    st.stop()

# Sidebar Configuration
with st.sidebar:
    st.header("🔬 Pipeline Configuration")
    top_k_per_condition = st.slider(
        "Codes per Condition",
        min_value=1,
        max_value=10,
        value=5,
        help="How many candidate ICD-10 codes to retrieve per extracted condition via FAISS search."
    )
    similarity_threshold = st.slider(
        "Similarity Threshold",
        min_value=0.30,
        max_value=0.80,
        value=0.45,
        step=0.05,
        help="Minimum cosine similarity to accept a code match. Lower = more codes (may include false positives). Higher = fewer but more precise codes."
    )
    st.info("💡 Pipeline: LLM refines note → LLM extracts conditions (structured, section-aware) → Embed-back validation → FAISS matches codes → LLM ranks. 3 LLM calls total.")

# Layout
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Patient Clinical Note")

    # PDF Upload
    uploaded_pdfs = st.file_uploader(
        "Upload clinical PDFs (discharge summary, medication list, etc.)",
        type=["pdf"],
        accept_multiple_files=True
    )

    # Extract text from PDFs if uploaded
    pdf_text = ""
    page_chunks = []
    if uploaded_pdfs:
        pdf_text, page_chunks = extract_text_from_multiple_pdfs(uploaded_pdfs)
        total_pages = len(page_chunks)
        st.success(f"Loaded {len(uploaded_pdfs)} PDF(s), {total_pages} pages extracted.")

    patient_note = st.text_area(
        "Enter or edit clinical note here:",
        value=pdf_text,
        height=300,
        placeholder="e.g. Patient presents with severe fever, abdominal pain..."
    )

    analyze_btn = st.button("Analyze & Match Code", type="primary", use_container_width=True)

with col2:
    st.subheader("Agent Logic Trace")

    # Placeholders for timer and progress
    timer_placeholder = st.empty()
    progress_placeholder = st.empty()
    log_area = st.empty()

    if analyze_btn and patient_note:
        start_time = time.time()

        # Define progress callback
        def update_ui_progress(percent_val):
            progress_placeholder.progress(percent_val, text=f"Processing Pipeline... ({int(percent_val*100)}%)")
            elapsed = time.time() - start_time
            timer_placeholder.markdown(f"⏱️ **Elapsed Time:** `{elapsed:.2f}s`")

        with st.spinner("Executing Pipeline (Refine → Extract → FAISS Match → LLM Rank)..."):
            pipeline_result = process_patient_note(
                patient_note,
                top_k_per_condition=top_k_per_condition,
                similarity_threshold=similarity_threshold,
                on_progress=update_ui_progress,
                page_chunks=page_chunks if page_chunks else None
            )

            end_time = time.time()
            total_duration = end_time - start_time

            # Final timer update
            timer_placeholder.markdown(f"✅ **Total Analysis Time:** `{total_duration:.2f}s`")
            progress_placeholder.empty()

        # Persist results in session state so downloads don't clear them
        st.session_state['pipeline_result'] = pipeline_result
        st.session_state['page_chunks'] = page_chunks

# ---------------------------------------------------------------
# RESULTS DASHBOARD (full width, below the two-column layout)
# ---------------------------------------------------------------
# Retrieve persisted results from session state
pipeline_result = st.session_state.get('pipeline_result')
stored_page_chunks = st.session_state.get('page_chunks', [])

if pipeline_result and pipeline_result.get("status") == "success":
    st.markdown("---")

    report = pipeline_result['report']
    primary_codes = [c for c in report if c['designation'] == "Primary"]
    secondary_codes = [c for c in report if c['designation'] == "Secondary"]
    active_codes = [c for c in report if c['status'] == "Active"]
    resolved_codes = [c for c in report if c['status'] == "Resolved"]
    nta_codes = [c for c in report if c.get('is_nta')]
    slp_codes = [c for c in report if c.get('SLP_Comorbidity')]
    nta_summary = pipeline_result.get('nta_summary', {})

    # ---------------------------------------------------------------
    # SUMMARY METRICS BAR
    # ---------------------------------------------------------------
    st.subheader("🔍 Medicare PDPM / SNF Dashboard")
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Billable Conditions", len(report))
    m2.metric("Primary", len(primary_codes))
    m3.metric("Secondary", len(secondary_codes))
    m4.metric("NTA Comorbidities", nta_summary.get('total_categories', 0))
    m5.metric("SLP Comorbidities", len(slp_codes))
    m6.metric("Total NTA Points", nta_summary.get('total_nta_points', 0))

    # ---------------------------------------------------------------
    # DOWNLOAD REPORT
    # ---------------------------------------------------------------
    report_df = pd.DataFrame([
        {
            "ICD-10 Code": c['exact_code'],
            "Description": c['description'],
            "Designation": c['designation'],
            "Status": c['status'],
            "Acuity": c['acuity'],
            "PDPM Category": c.get('PDPM_Category', ''),
            "NTA Comorbidity": c.get('NTA_Comorbidity_Name', '') or '',
            "NTA MDS Field": c.get('NTA_MDS_Field', '') or '',
            "SLP Comorbidity": c.get('SLP_Comorbidity', False),
            "Confidence": c['probability_score'],
            "Clinical Evidence": c['clinical_evidence_quote'],
            "Document Reference": c['document_reference'],
            "Reasoning": c['reasoning'],
        }
        for c in report
    ])

    dl1, dl2 = st.columns(2)
    with dl1:
        st.download_button(
            "Download CSV",
            data=report_df.to_csv(index=False),
            file_name="icd10_coding_report.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with dl2:
        st.download_button(
            "Download JSON",
            data=json.dumps(report, indent=2),
            file_name="icd10_coding_report.json",
            mime="application/json",
            use_container_width=True,
        )

    # ---------------------------------------------------------------
    # PRIMARY DIAGNOSIS DRIVER
    # ---------------------------------------------------------------
    st.markdown("### 🎯 Primary Diagnosis Driver (MDS Item I0020B)")
    if primary_codes:
        for item in primary_codes:
            with st.container(border=True):
                nta_tag = f"NTA({item.get('NTA_Comorbidity_Name', '')})" if item.get('is_nta') else ""
                mds_tag = f"MDS: {item.get('NTA_MDS_Field', '')}" if item.get('NTA_MDS_Field') else ""
                pdpm_impact = ", ".join(filter(None, [nta_tag, mds_tag]))

                st.markdown(f"#### `{item['exact_code']}` — {item['description']}")
                cols = st.columns([1, 1, 1])
                cols[0].markdown(f"**Confidence:** {item['probability_score']}%")
                cols[1].markdown(f"**Status:** {item['status']}")
                cols[2].markdown(f"**Acuity:** {item['acuity']}")
                if pdpm_impact:
                    st.markdown(f"**PDPM Impact:** {pdpm_impact}")
                st.markdown(f"**PDPM Clinical Category:** {item['PDPM_Category']}")
                st.info(f"**Clinical Evidence:** {item['document_reference']}: \"{item['clinical_evidence_quote']}\"")
                st.markdown(f"**Rationale:** {item['reasoning']}")
    else:
        st.warning("No primary diagnosis identified.")

    # ---------------------------------------------------------------
    # NTA COMORBIDITY DETAIL
    # ---------------------------------------------------------------
    if nta_summary.get('itemized'):
        st.markdown("### 💊 NTA Comorbidities")
        st.markdown(f"**{nta_summary['total_categories']} comorbidity categories** | **{nta_summary['total_nta_points']} aggregate NTA points**")

        for cat in nta_summary['itemized']:
            codes_in_cat = [c for c in report if c['exact_code'] in cat['icd10_codes']]
            with st.expander(f"**{cat['comorbidity_name']}** | MDS: {cat['mds_field']}", expanded=True):
                for i, c in enumerate(codes_in_cat):
                    with st.container(border=True):
                        st.markdown(f"`{c['exact_code']}` — {c['description']}")
                        st.caption(f'Evidence: "{c["clinical_evidence_quote"]}" ({c["reasoning"]})')

    # ---------------------------------------------------------------
    # ACTIVE DIAGNOSIS LIST
    # ---------------------------------------------------------------
    st.markdown("### 📋 Active Diagnosis List")
    active_primary = [c for c in active_codes if c['designation'] == 'Primary']
    active_secondary = [c for c in active_codes if c['designation'] == 'Secondary']
    active_nta = [c for c in active_codes if c.get('is_nta')]
    active_slp = [c for c in active_codes if c.get('SLP_Comorbidity')]

    st.markdown(
        f"**{len(active_codes)} Billable Conditions** | "
        f"**{len(active_primary)} Primary** | "
        f"**{len(active_secondary)} Secondary** | "
        f"**{len(active_nta)} NTA** | "
        f"**{len(active_slp)} SLP**"
    )

    if active_codes:
        for item in active_codes:
            nta_label = f"NTA, MDS: {item.get('NTA_MDS_Field', '')}" if item.get('is_nta') else ""
            classification = f"{item['acuity']} | {item['designation']}"
            with st.expander(f"**{item['exact_code']}** | {item['description']} — {classification}", expanded=True):
                with st.container(border=True):
                    cols = st.columns([1, 1])
                    cols[0].markdown(f"**PDPM Category:** {item['PDPM_Category']}")
                    if nta_label:
                        cols[1].markdown(f"**Classification:** {nta_label}")
                    st.info(f"**Evidence:** {item['document_reference']}: \"{item['clinical_evidence_quote']}\" ({item['reasoning']})")
    else:
        st.write("No active diagnoses identified.")

    # ---------------------------------------------------------------
    # RESOLVED DIAGNOSIS LIST
    # ---------------------------------------------------------------
    st.markdown("### 📚 Resolved Diagnosis List")
    resolved_primary = [c for c in resolved_codes if c['designation'] == 'Primary']
    resolved_secondary = [c for c in resolved_codes if c['designation'] == 'Secondary']
    resolved_nta = [c for c in resolved_codes if c.get('is_nta')]
    resolved_slp = [c for c in resolved_codes if c.get('SLP_Comorbidity')]

    st.markdown(
        f"**{len(resolved_codes)} Billable Conditions** | "
        f"**{len(resolved_primary)} Primary** | "
        f"**{len(resolved_secondary)} Secondary** | "
        f"**{len(resolved_nta)} NTA** | "
        f"**{len(resolved_slp)} SLP**"
    )

    if resolved_codes:
        for item in resolved_codes:
            nta_label = f"NTA, MDS: {item.get('NTA_MDS_Field', '')}" if item.get('is_nta') else ""
            classification = f"{item['acuity']} | {item['designation']}"
            with st.expander(f"**{item['exact_code']}** | {item['description']} — {classification}", expanded=False):
                with st.container(border=True):
                    cols = st.columns([1, 1])
                    cols[0].markdown(f"**PDPM Category:** {item['PDPM_Category']}")
                    if nta_label:
                        cols[1].markdown(f"**Classification:** {nta_label}")
                    st.info(f"**Evidence:** {item['document_reference']}: \"{item['clinical_evidence_quote']}\" ({item['reasoning']})")
    else:
        st.write("No resolved diagnoses identified.")

    # ---------------------------------------------------------------
    # PDPM QUALITY MEASURE DEFENSE & COMPLIANCE
    # ---------------------------------------------------------------
    qm_warnings = pipeline_result.get('qm_warnings', [])
    st.markdown("### 🛡️ PDPM Quality Measure Defense & Compliance")

    if primary_codes:
        primary = primary_codes[0]
        st.markdown(f"**Primary Diagnosis Verification:** {primary['exact_code']} - {primary['description']}")
        st.markdown(f"**PDPM Clinical Category:** {primary['PDPM_Category']}")
        if primary.get('PDPM_Category') not in ('Unmapped', None):
            st.success("Primary diagnosis is mapped to a valid PDPM clinical category.")
        else:
            st.warning("⚠️ Primary diagnosis is not mapped. Verify it matches the SNF admission reason.")

    for warning in qm_warnings:
        st.warning(f"⚠️ {warning}")

    # ---------------------------------------------------------------
    # AGENT LOGIC TRACE
    # ---------------------------------------------------------------
    with st.expander("🛠️ View Agent Logic Trace"):
        log_text = ""
        for msg in pipeline_result.get("logs", []):
            log_text += f"{msg}\n"
        st.code(log_text, language="log")

    # ---------------------------------------------------------------
    # PDFs ANALYZED
    # ---------------------------------------------------------------
    if stored_page_chunks:
        st.markdown("---")
        unique_files = list(dict.fromkeys(c.source_file for c in stored_page_chunks))
        st.markdown("**PDFs Analyzed:**")
        for i, f in enumerate(unique_files, 1):
            pages = [c.page_number for c in stored_page_chunks if c.source_file == f]
            st.markdown(f"{i}. {f} ({len(pages)} pages)")

elif pipeline_result and pipeline_result.get("status") != "success":
    st.markdown("---")
    st.error("### ❌ No matches discovered for this clinical note.")
    if pipeline_result:
        with st.expander("🛠️ View Agent Logic Trace"):
            log_text = ""
            for msg in pipeline_result.get("logs", []):
                log_text += f"{msg}\n"
            st.code(log_text, language="log")
