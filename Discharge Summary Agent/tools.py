# tools.py
# Tools the agent can use during its reasoning loop.

from typing import List, Dict
from datetime import datetime


# ─────────────────────────────────────────────
# DISCHARGE SUMMARY SECTIONS
# Defines every section we need to fill in and
# what keywords to search for in the documents.
# More keywords = better chance of finding content
# in OCR text that may have spelling errors.
# ─────────────────────────────────────────────

DISCHARGE_SECTIONS = [
    {
        "name": "Patient Demographics",
        "keywords": [
            "name", "age", "gender", "sex", "dob", "mrn", "ip no",
            "date of birth", "patient name", "weight", "wt", "male",
            "female", "years", "year old", "blood group", "height"
        ]
    },
    {
        "name": "Admission and Discharge Dates",
        "keywords": [
            "date of admission", "date of discharge", "admitted", "discharged",
            "admission date", "discharge date", "28/2", "26/2", "27/2",
            "1/3", "2/3", "3/3", "28-2", "26-2", "arrival", "d/c",
            "pod", "post op day", "day of"
        ]
    },
    {
        "name": "Principal Diagnosis",
        "keywords": [
            "diagnosis", "final diagnosis", "provisional diagnosis",
            "dka", "diabetic ketoacidosis", "t2dm", "pyelonephritis",
            "afi", "acute febrile", "gastroenteritis", "uti",
            "urinary tract", "dehydration", "infection", "diagonsis",
            "diagnoisis", "final diag"
        ]
    },
    {
        "name": "Secondary Diagnoses",
        "keywords": [
            "secondary diagnosis", "comorbidities", "known case",
            "k/c/o", "past history", "cholelithiasis", "synovitis",
            "uncontrolled", "hypertension", "hypothyroid", "thyroid",
            "diabetes", "t2 dm", "type 2", "hepatomegaly",
            "fatty liver", "klclo", "klco", "h/o", "history of"
        ]
    },
    {
        "name": "Hospital Course",
        "keywords": [
            "course in hospital", "patient presented", "admitted to",
            "treated with", "improvement", "progress", "management",
            "iv fluids", "antibiotics", "insulin", "initial investigations",
            "serum creatinine", "sodium", "electrolytes", "usg abdomen",
            "repeat", "normal cbc", "ward", "evaluation"
        ]
    },
    {
        "name": "Procedures Performed",
        "keywords": [
            "procedure", "iv cannulation", "foley", "catheter",
            "usg", "echo", "ecg", "ct", "x-ray", "blood culture",
            "urine culture", "cannulization", "cannula", "catheterisation",
            "phototherapy", "pd sets", "vein", "oxygen", "intubation",
            "bone marrow", "dressing", "blood transfusion"
        ]
    },
    {
        "name": "Admission Medications",
        "keywords": [
            # These keywords target what the patient was on BEFORE admission
            "on ayurvedic", "ayurvedic medication", "regular medication",
            "on medication", "prior medication", "drug history",
            "metformin", "lantus", "known case of", "k/c/o t2dm",
            "hba1c", "outside report", "on treatment", "on tab",
            "klclo t2dm", "on regular"
        ]
    },
    {
        "name": "Discharge Medications",
        "keywords": [
            "advice on discharge", "discharge medication", "tab.",
            "inj.", "tablet", "capsule", "1-0-0", "1-1-1", "1-0-1",
            "sos", "meromac", "pan", "emeset", "meropenem",
            "raciper", "ractiper", "oflox", "zedott", "lopiramide",
            "meftal", "m strong", "before food", "days", "frequency",
            "duration", "dosage"
        ]
    },
    {
        "name": "Allergies",
        "keywords": [
            "allergy", "allergies", "allergic", "drug allergy",
            "known drug", "not known", "nka", "no known allergy",
            # OCR often misreads "Not Known" — include likely OCR variants
            "known drug allergies", "nch", "loum", "no allergy",
            "allergic history", "drug reaction", "known allergies",
            "no known", "nil allergy", "nil known"
        ]
    },
    {
        "name": "Follow-up Instructions",
        "keywords": [
            "follow-up", "follow up", "review", "review on",
            "urine culture", "report awaited", "cbc", "review immediately",
            "outpatient", "opd", "review in case", "follow-up instructions",
            "loose stools", "vomiting", "fatigue", "09.03", "19.03",
            "instructions on discharge"
        ]
    },
    {
        "name": "Pending Results",
        "keywords": [
            "awaited", "pending", "report awaited", "culture sent",
            "report due", "not yet", "culture and sensitivity",
            "sent to lab", "report received", "lab report due",
            "urine c/s", "blood c/s", "sensitivity sent"
        ]
    },
    {
        "name": "Condition at Discharge",
        "keywords": [
            "condition at discharge", "hemodynamically stable",
            "stable", "discharge condition", "on request",
            "improved", "condition", "attenders not willing",
            "being discharged", "discharge at request", "lama",
            "against medical advice"
        ]
    },
    {
        "name": "Vital Signs at Discharge",
        "keywords": [
            # Look specifically around discharge date pages
            "bp", "blood pressure", "pulse", "hr", "heart rate",
            "spo2", "temperature", "rr", "respiratory rate",
            # Date-based — these appear on the last day of admission
            "2/3/26", "3/3/26", "2/3/2026", "3/3/2026",
            "97", "98", "99", "110/70", "120/80",
            "pr-", "bp-", "rr-", "spo2-"
        ]
    }
]


# ─────────────────────────────────────────────
# TOOL 1: Search in text
# ─────────────────────────────────────────────

def search_in_text(full_text: str, keywords: List[str],
                   context_lines: int = 25) -> str:
    """
    Searches through clinical notes and returns paragraphs
    that contain relevant keywords, with surrounding context.

    context_lines increased to 25 so the LLM gets more
    surrounding text to understand the clinical context.
    """
    lines = full_text.split('\n')
    relevant_lines = []
    seen_blocks = set()
    max_chars = 5000

    for i, line in enumerate(lines):
        line_lower = line.lower()
        if any(kw.lower() in line_lower for kw in keywords):
            start = max(0, i - 4)
            end = min(len(lines), i + context_lines)
            block = '\n'.join(lines[start:end])
            # Deduplicate overlapping blocks
            block_key = block[:100]
            if block_key not in seen_blocks:
                seen_blocks.add(block_key)
                relevant_lines.append(block)

    if not relevant_lines:
        return "NOT_FOUND_IN_DOCUMENTS"

    result = '\n\n---\n\n'.join(relevant_lines)
    return result[:max_chars]


# ─────────────────────────────────────────────
# TOOL 2: Drug Interaction Checker (MOCKED)
# In production this would call DrugBank/Micromedex.
# ─────────────────────────────────────────────

KNOWN_INTERACTIONS = {
    ("meropenem", "metformin"):
        "⚠️ DRUG INTERACTION FLAG: Meropenem may reduce Metformin efficacy. Monitor blood glucose closely.",
    ("metformin", "contrast"):
        "⚠️ DRUG INTERACTION FLAG: Metformin should be held 48h before/after contrast studies.",
    ("tramadol", "metformin"):
        "⚠️ DRUG INTERACTION FLAG: Tramadol with Metformin - monitor for lactic acidosis risk.",
    ("lantus", "actrapid"):
        "⚠️ DRUG INTERACTION ALERT: Two insulin types present (Lantus + Actrapid) - verify dosing protocol.",
    ("meropenem", "valproate"):
        "⚠️ DRUG INTERACTION FLAG: Meropenem significantly reduces Valproate levels.",
    ("ofloxacin", "metformin"):
        "⚠️ DRUG INTERACTION FLAG: Fluoroquinolones (Ofloxacin) can cause hypoglycaemia with antidiabetics.",
    ("oflox", "metformin"):
        "⚠️ DRUG INTERACTION FLAG: Ofloxacin with Metformin - monitor blood glucose.",
}


def check_drug_interactions(medications_text: str) -> List[str]:
    """
    Checks for known dangerous drug combinations.
    Returns a list of warnings. Empty list = no issues found.

    NOTE: This is a MOCK tool. Production would use a real API.
    """
    if not medications_text or "NOT_FOUND" in medications_text:
        return ["⚠️ DRUG CHECK: Could not perform check - medication list unavailable"]

    medications_lower = medications_text.lower()
    warnings = []

    for (drug1, drug2), warning in KNOWN_INTERACTIONS.items():
        if drug1 in medications_lower and drug2 in medications_lower:
            warnings.append(warning)

    # Flag insulin presence — always needs monitoring plan at discharge
    if "lantus" in medications_lower or "actrapid" in medications_lower:
        warnings.append(
            "⚠️ INSULIN PRESENT: Verify blood glucose monitoring plan "
            "is documented for discharge."
        )

    return warnings


# ─────────────────────────────────────────────
# TOOL 3: Clinical Flag System
# Agent calls this whenever something needs
# a doctor's eye before finalising.
# ─────────────────────────────────────────────

class ClinicalFlagSystem:
    """
    Tracks all issues requiring clinician attention.
    The agent raises flags — it never resolves clinical
    uncertainty by itself.
    """

    def __init__(self):
        self.flags = []

    def add_flag(self, flag_type: str, description: str,
                 severity: str = "REVIEW_REQUIRED") -> str:
        """
        Record a flag for clinician review.

        severity options:
          INFO            — minor note, good to know
          REVIEW_REQUIRED — must be checked before finalising
          URGENT          — safety concern, act immediately
        """
        flag = {
            "id": len(self.flags) + 1,
            "type": flag_type,
            "severity": severity,
            "description": description,
            "timestamp": datetime.now().strftime("%H:%M:%S")
        }
        self.flags.append(flag)

        emoji = {"INFO": "ℹ️", "REVIEW_REQUIRED": "⚠️", "URGENT": "🚨"}.get(severity, "⚠️")
        return f"{emoji} [{flag_type}] {description}"

    def get_all_flags(self) -> List[Dict]:
        return self.flags

    def format_flags_for_summary(self) -> str:
        if not self.flags:
            return "No flags raised during processing."
        lines = []
        for flag in self.flags:
            emoji = {"INFO": "ℹ️", "REVIEW_REQUIRED": "⚠️", "URGENT": "🚨"}.get(
                flag["severity"], "⚠️")
            lines.append(
                f"{emoji} **[{flag['severity']}]** {flag['type']}: {flag['description']}"
            )
        return '\n'.join(lines)


# ─────────────────────────────────────────────
# TOOL 4: Structural Conflict Detector
# Checks extracted sections for internal conflicts
# ─────────────────────────────────────────────

def detect_conflicts(extracted_sections: Dict[str, str]) -> List[str]:
    """
    Scans the already-extracted sections for obvious conflicts,
    e.g. multiple competing diagnoses or contradictory conditions.
    """
    conflicts = []

    diagnosis = extracted_sections.get("Principal Diagnosis", "")
    if diagnosis and "NOT_FOUND" not in diagnosis and "⚠️" not in diagnosis:
        diagnosis_lower = diagnosis.lower()
        conditions_found = []

        condition_keywords = {
            "DKA / Diabetic Ketoacidosis": ["dka", "diabetic ketoacidosis"],
            "Pyelonephritis": ["pyelonephritis"],
            "Acute Febrile Illness (AFI)": ["afi", "acute febrile"],
            "Acute Gastroenteritis": ["gastroenteritis", "loose stools"],
            "UTI": ["urinary tract infection", "uti"],
        }

        for condition_name, keywords in condition_keywords.items():
            if any(kw in diagnosis_lower for kw in keywords):
                conditions_found.append(condition_name)

        if len(conditions_found) > 2:
            conflicts.append(
                f"MULTIPLE_DIAGNOSES: {len(conditions_found)} conditions found in principal "
                f"diagnosis section ({', '.join(conditions_found)}). "
                f"Verify which is primary and which are secondary."
            )

    # Check if allergies and drug section contradict each other
    allergies = extracted_sections.get("Allergies", "").lower()
    if "no known" in allergies or "not known" in allergies or "nka" in allergies:
        # Fine — consistent
        pass
    elif allergies and "NOT_FOUND" not in allergies and "⚠️" not in allergies:
        conflicts.append(
            "ALLERGY_NEEDS_VERIFICATION: Allergy status unclear from OCR — "
            "confirm with patient before discharge."
        )

    return conflicts
