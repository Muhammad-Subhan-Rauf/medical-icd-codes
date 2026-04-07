"""
ICD-10-CM Rule Database.

Parses the CMS FY2026 Tabular List XML to extract per-code rules:
- codeAlso, codeFirst, useAdditionalCode
- excludes1, excludes2
- includes, inclusionTerm
- notes

Used by the pipeline to inject code-specific rules into the LLM ranker prompt,
and for programmatic post-validation of final code sets.
"""

import os
import pickle
import xml.etree.ElementTree as ET
import logging

logger = logging.getLogger("medic.icd10_rules")

RULES_XML = os.path.join(os.path.dirname(__file__), "icd10_rules_data", "icd10cm-tabular-2026.xml")
RULES_CACHE = os.path.join(os.path.dirname(__file__), "icd10_rules_data", "rules_db.pkl")

# Rule types we extract from the XML
RULE_TAGS = {
    "codeAlso", "codeFirst", "useAdditionalCode",
    "excludes1", "excludes2",
    "includes", "inclusionTerm", "notes",
}

_rules_db = None


def _extract_notes(element) -> list[str]:
    """Extract all note texts from an element."""
    notes = []
    for note in element.iter("note"):
        if note.text and note.text.strip():
            notes.append(note.text.strip())
    return notes


def _parse_diag(diag_element, inherited_rules: dict) -> dict:
    """Parse a single diag element and return {code: rules_dict}.

    Rules are inherited from parent elements (chapter → section → category → code).
    """
    results = {}

    name_elem = diag_element.find("name")
    if name_elem is None or not name_elem.text:
        return results

    code = name_elem.text.strip().replace(".", "")
    desc_elem = diag_element.find("desc")
    desc = desc_elem.text.strip() if desc_elem is not None and desc_elem.text else ""

    # Start with inherited rules
    rules = {
        "description": desc,
        "code_also": list(inherited_rules.get("code_also", [])),
        "code_first": list(inherited_rules.get("code_first", [])),
        "use_additional": list(inherited_rules.get("use_additional", [])),
        "excludes1": list(inherited_rules.get("excludes1", [])),
        "excludes2": list(inherited_rules.get("excludes2", [])),
        "includes": list(inherited_rules.get("includes", [])),
        "inclusion_terms": [],
        "notes": [],
    }

    # Extract this element's own rules
    for child in diag_element:
        if child.tag == "codeAlso":
            rules["code_also"].extend(_extract_notes(child))
        elif child.tag == "codeFirst":
            rules["code_first"].extend(_extract_notes(child))
        elif child.tag == "useAdditionalCode":
            rules["use_additional"].extend(_extract_notes(child))
        elif child.tag == "excludes1":
            rules["excludes1"].extend(_extract_notes(child))
        elif child.tag == "excludes2":
            rules["excludes2"].extend(_extract_notes(child))
        elif child.tag == "includes":
            rules["includes"].extend(_extract_notes(child))
        elif child.tag == "inclusionTerm":
            rules["inclusion_terms"].extend(_extract_notes(child))
        elif child.tag == "notes":
            rules["notes"].extend(_extract_notes(child))

    results[code] = rules

    # Recurse into child diag elements, passing this element's rules as inherited
    child_inherited = {
        "code_also": rules["code_also"],
        "code_first": rules["code_first"],
        "use_additional": rules["use_additional"],
        "excludes1": rules["excludes1"],
        "excludes2": rules["excludes2"],
    }
    for child_diag in diag_element.findall("diag"):
        results.update(_parse_diag(child_diag, child_inherited))

    return results


def build_rules_db(xml_path: str = RULES_XML) -> dict:
    """Parse the CMS ICD-10-CM Tabular List XML into a rules database.

    Returns: dict mapping code (no dots) → rules dict
    """
    global _rules_db
    if _rules_db is not None:
        return _rules_db

    # Try loading from cache
    if os.path.exists(RULES_CACHE):
        logger.info("Loading rules DB from cache...")
        with open(RULES_CACHE, "rb") as f:
            _rules_db = pickle.load(f)
        logger.info("Rules DB loaded: %d codes", len(_rules_db))
        return _rules_db

    logger.info("Parsing CMS Tabular List XML: %s", xml_path)
    tree = ET.parse(xml_path)
    root = tree.getroot()

    rules_db = {}

    for chapter in root.findall("chapter"):
        # Extract chapter-level rules
        chapter_rules = {"code_also": [], "code_first": [], "use_additional": [],
                         "excludes1": [], "excludes2": []}
        for child in chapter:
            if child.tag == "codeAlso":
                chapter_rules["code_also"].extend(_extract_notes(child))
            elif child.tag == "codeFirst":
                chapter_rules["code_first"].extend(_extract_notes(child))
            elif child.tag == "useAdditionalCode":
                chapter_rules["use_additional"].extend(_extract_notes(child))
            elif child.tag == "excludes1":
                chapter_rules["excludes1"].extend(_extract_notes(child))
            elif child.tag == "excludes2":
                chapter_rules["excludes2"].extend(_extract_notes(child))

        for section in chapter.findall("section"):
            # Extract section-level rules
            section_rules = {k: list(v) for k, v in chapter_rules.items()}
            for child in section:
                if child.tag == "codeAlso":
                    section_rules["code_also"].extend(_extract_notes(child))
                elif child.tag == "codeFirst":
                    section_rules["code_first"].extend(_extract_notes(child))
                elif child.tag == "useAdditionalCode":
                    section_rules["use_additional"].extend(_extract_notes(child))
                elif child.tag == "excludes1":
                    section_rules["excludes1"].extend(_extract_notes(child))
                elif child.tag == "excludes2":
                    section_rules["excludes2"].extend(_extract_notes(child))

            for diag in section.findall("diag"):
                rules_db.update(_parse_diag(diag, section_rules))

    # Cache
    with open(RULES_CACHE, "wb") as f:
        pickle.dump(rules_db, f)
    logger.info("Rules DB built and cached: %d codes", len(rules_db))

    _rules_db = rules_db
    return _rules_db


def lookup_rules(code: str) -> dict | None:
    """Look up rules for a specific ICD-10-CM code (with or without dots)."""
    db = build_rules_db()
    normalized = code.replace(".", "").strip().upper()
    return db.get(normalized)


def get_rules_text(code: str) -> str:
    """Get a formatted text summary of rules for a code, suitable for LLM prompts."""
    rules = lookup_rules(code)
    if not rules:
        return ""

    lines = []
    if rules["code_also"]:
        lines.append("  Code Also: " + "; ".join(rules["code_also"]))
    if rules["code_first"]:
        lines.append("  Code First: " + "; ".join(rules["code_first"]))
    if rules["use_additional"]:
        lines.append("  Use Additional: " + "; ".join(rules["use_additional"]))
    if rules["excludes1"]:
        lines.append("  Excludes1 (never use with): " + "; ".join(rules["excludes1"]))
    if rules["excludes2"]:
        lines.append("  Excludes2 (code separately): " + "; ".join(rules["excludes2"]))

    return "\n".join(lines)


def enrich_candidates(candidates: list[dict]) -> list[dict]:
    """Attach code-specific rules to each candidate dict.

    Adds a 'coding_rules' key with formatted rule text.
    """
    db = build_rules_db()
    for candidate in candidates:
        code = candidate.get("exact_code", "").replace(".", "").upper()
        rules_text = get_rules_text(code)
        candidate["coding_rules"] = rules_text
    return candidates


def validate_final_codes(codes: list[str]) -> list[dict]:
    """Programmatic post-validation of a final code set.

    Checks for:
    - Excludes1 conflicts (mutually exclusive codes both present)
    - Missing Code Also / Use Additional companion codes

    Returns list of warning dicts: {type, code, message}
    """
    import re
    db = build_rules_db()
    code_set = {c.replace(".", "").upper() for c in codes}
    warnings = []

    for code in code_set:
        rules = db.get(code)
        if not rules:
            continue

        # Check Excludes1 — extract code references from note text
        for note in rules.get("excludes1", []):
            # Extract codes like "G30.-", "F02.80", "I60-I69"
            referenced_codes = re.findall(r'[A-Z]\d[\dA-Z]{0,5}', note)
            for ref in referenced_codes:
                ref_norm = ref.replace(".", "").upper()
                # Check if any code in the final set starts with this reference
                for final_code in code_set:
                    if final_code != code and (final_code == ref_norm or
                            (ref_norm.endswith("-") and final_code.startswith(ref_norm[:-1]))):
                        warnings.append({
                            "type": "excludes1",
                            "code": code,
                            "conflict": final_code,
                            "message": f"Excludes1: {code} and {final_code} should not be coded together ({note})",
                        })

        # Check Code Also / Use Additional — warn if companion seems missing
        for note in rules.get("code_also", []) + rules.get("use_additional", []):
            referenced_codes = re.findall(r'([A-Z]\d[\dA-Z]{0,4})', note)
            for ref in referenced_codes:
                ref_norm = ref.replace(".", "").upper()
                # Check if any code in the set covers this reference
                found = any(fc.startswith(ref_norm[:3]) for fc in code_set if fc != code)
                if not found and len(ref_norm) >= 3:
                    warnings.append({
                        "type": "missing_companion",
                        "code": code,
                        "companion_hint": ref_norm,
                        "message": f"Code {code} may require companion code ({note})",
                    })

    # Deduplicate warnings
    seen = set()
    unique = []
    for w in warnings:
        key = (w["type"], w["code"], w.get("conflict", w.get("companion_hint", "")))
        if key not in seen:
            seen.add(key)
            unique.append(w)

    return unique


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Build the database
    db = build_rules_db()
    print(f"Total codes in rules DB: {len(db)}")

    # Test lookups
    test_codes = ["G31.83", "R65.21", "A41.9", "E11.9", "J44.9", "L03.116"]
    for code in test_codes:
        print(f"\n{'='*50}")
        print(f"  {code}")
        rules = lookup_rules(code)
        if rules:
            print(f"  Description: {rules['description']}")
            rules_text = get_rules_text(code)
            if rules_text:
                print(rules_text)
            else:
                print("  (no specific rules)")
        else:
            print("  NOT FOUND")
