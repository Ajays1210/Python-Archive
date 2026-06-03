# agent.py
# The main agent — this is where all the "thinking" happens.
#
# HOW IT WORKS (ReAct pattern):
#   THINK  → decide what information is needed next
#   ACT    → search the documents or call a tool
#   OBSERVE → look at the result and decide what to do
#   REPEAT  until all sections are done or max steps reached
#
# CORE SAFETY RULE:
#   This agent NEVER invents clinical facts.
#   If something is not in the documents it flags it
#   and asks the clinician to fill it in.

import ollama
import json
from datetime import datetime
from typing import Dict, List

from pdf_reader import extract_pdfs_from_folder, categorize_documents
from tools import (
    DISCHARGE_SECTIONS,
    search_in_text,
    check_drug_interactions,
    ClinicalFlagSystem,
    detect_conflicts
)


class DischargeAgent:
    """
    Reads patient records and produces a structured discharge
    summary draft for clinician review.
    """

    def __init__(self, model: str = "llama3.2:3b", max_steps: int = 30):
        self.model = model
        self.max_steps = max_steps
        self.step_count = 0
        self.trace = []
        self.flags = ClinicalFlagSystem()
        self.extracted_info = {}
        self.all_text = ""
        self.pdf_texts = {}

        print(f"Agent initialized | Model: {model} | Max steps: {max_steps}")

    # ─────────────────────────────────────────────
    # STEP LOGGER
    # ─────────────────────────────────────────────

    def log_step(self, step_type: str, reasoning: str,
                 action: str, result: str):
        """Records every step: what the agent thought, did, and saw."""
        result_preview = (
            str(result)[:300] + "..."
            if len(str(result)) > 300
            else str(result)
        )
        entry = {
            "step_number": self.step_count,
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "step_type": step_type,
            "reasoning": reasoning,
            "action": action,
            "result_preview": result_preview
        }
        self.trace.append(entry)

        print(f"\n{'─' * 50}")
        print(f"[Step {self.step_count}] {step_type}")
        print(f"  💭 Reason : {reasoning}")
        print(f"  🔧 Action : {action}")
        print(f"  📋 Result : {result_preview[:150]}")

    # ─────────────────────────────────────────────
    # LLM QUERY with retry
    # ─────────────────────────────────────────────

    def query_llm(self, prompt: str,
                  system_prompt: str = None,
                  max_retries: int = 2) -> str:
        """
        Sends a prompt to Ollama and returns the response.
        Retries up to max_retries times before giving up.
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        for attempt in range(max_retries + 1):
            try:
                response = ollama.chat(
                    model=self.model,
                    messages=messages,
                    options={
                        "temperature": 0.05,  # Low = factual, not creative
                        "num_ctx": 4096,
                    }
                )
                return response['message']['content']

            except Exception as e:
                if attempt < max_retries:
                    print(f"  ⚠️  LLM attempt {attempt + 1} failed, retrying...")
                else:
                    self.flags.add_flag(
                        "LLM_FAILURE",
                        f"LLM did not respond after {max_retries} retries: {str(e)[:100]}",
                        "REVIEW_REQUIRED"
                    )
                    return f"LLM_ERROR: {str(e)}"

    # ─────────────────────────────────────────────
    # SECTION EXTRACTOR (first attempt)
    # ─────────────────────────────────────────────

    def extract_section(self, section_name: str, relevant_text: str) -> str:
        """
        Asks the LLM to extract one specific section from the
        most relevant part of the document.

        The system prompt enforces the no-fabrication rule.
        """
        if not relevant_text or relevant_text == "NOT_FOUND_IN_DOCUMENTS":
            return "NOT_FOUND"

        system_prompt = (
            "You are a clinical documentation assistant preparing a discharge summary draft.\n\n"
            "CRITICAL RULES — follow every one of these:\n"
            "1. ONLY extract information explicitly written in the provided text.\n"
            "2. NEVER guess, infer, or invent any clinical information.\n"
            "3. The text comes from OCR of handwritten notes — minor spelling errors are expected.\n"
            "   Extract the information even if a word is slightly misspelled.\n"
            "4. If the information is clearly absent, respond with exactly: NOT_FOUND\n"
            "5. If information is partially present, include what you found and append:\n"
            "   [INCOMPLETE - FLAG FOR REVIEW]\n"
            "6. For dates look for patterns: dd/mm/yy or dd/mm/yyyy\n"
            "7. For vitals look for: BP, PR, HR, RR, SPO2, Temp followed by numbers\n"
            "8. Copy exact values (numbers, drug names, dates) — do not rephrase them."
        )

        prompt = (
            f"Extract ONLY the following section from the clinical notes: {section_name}\n\n"
            "Rules:\n"
            "- Found clearly → extract it\n"
            "- Partially found → extract what is there + [INCOMPLETE - FLAG FOR REVIEW]\n"
            "- Not present at all → respond: NOT_FOUND\n"
            "- NEVER invent information\n\n"
            "=== CLINICAL NOTES ===\n"
            f"{relevant_text}\n"
            "=== END ===\n\n"
            f"Extract {section_name}:"
        )

        return self.query_llm(prompt, system_prompt)

    # ─────────────────────────────────────────────
    # FALLBACK EXTRACTOR (second attempt)
    # Tries a wider search when first attempt fails
    # ─────────────────────────────────────────────

    def extract_section_fallback(self, section_name: str) -> str:
        """
        When the targeted search returns NOT_FOUND, try again
        using a broader slice of the full document text.

        This catches cases where OCR garbled the keywords
        so the first search missed the relevant passage.
        """
        # Use first 3000 chars + a middle chunk to cover more of the document
        broad_text = self.all_text[:3000] + "\n\n" + self.all_text[10000:13000]

        prompt = (
            f"Search carefully in the text below for: {section_name}\n\n"
            "IMPORTANT:\n"
            "- This text is from OCR of handwritten medical notes.\n"
            "  Words may be misspelled or partially garbled.\n"
            "- Extract anything that looks relevant even if spelling is off.\n"
            "- If truly nothing found, respond exactly: NOT_FOUND\n"
            "- NEVER invent or guess information.\n\n"
            "=== TEXT ===\n"
            f"{broad_text}\n"
            "=== END ===\n\n"
            f"What does the text say about {section_name}? "
            "(If nothing, say NOT_FOUND):"
        )

        return self.query_llm(prompt)

    # ─────────────────────────────────────────────
    # MEDICATION RECONCILIATION
    # ─────────────────────────────────────────────

    def reconcile_medications(self, admission_meds: str,
                               discharge_meds: str) -> str:
        """
        Compares admission vs discharge medications.
        Any change without a documented reason is flagged.
        """
        prompt = (
            "You are reviewing medication changes for a hospital discharge summary.\n\n"
            "Compare the ADMISSION medications vs DISCHARGE medications.\n\n"
            "List each medication under one of these categories:\n"
            "1. ✅ CONTINUED: present in both admission and discharge\n"
            "2. ➕ NEW: added at discharge, not on admission\n"
            "3. ➖ STOPPED: was on admission but not at discharge\n"
            "4. 🔄 CHANGED: same drug but different dose or frequency\n"
            "5. ⚠️ UNDOCUMENTED CHANGE: change with no clear reason — FLAG THESE\n\n"
            "RULES:\n"
            "- Only report what is explicitly stated.\n"
            "- Do NOT guess or infer reasons for changes.\n"
            "- If admission medications are unclear, say so.\n\n"
            "=== ADMISSION MEDICATIONS ===\n"
            f"{admission_meds}\n\n"
            "=== DISCHARGE MEDICATIONS ===\n"
            f"{discharge_meds}\n"
            "=== END ===\n\n"
            "Medication Reconciliation:"
        )
        return self.query_llm(prompt)

    # ─────────────────────────────────────────────
    # CONFLICT CHECKER
    # ─────────────────────────────────────────────

    def check_for_conflicts(self) -> List[str]:
        """
        Scans the document for conflicting clinical information,
        e.g. different diagnoses on different pages.
        """
        conflicts = []

        # Sample beginning + middle of document (LLM context limit)
        sample = (
            self.all_text[:3000]
            + "\n...(middle section)...\n"
            + self.all_text[5000:8000]
        )

        prompt = (
            "Review these clinical notes and list ANY conflicting information.\n\n"
            "Look for:\n"
            "- Different diagnoses on different pages\n"
            "- Different dates for the same event\n"
            "- Medication names that differ between notes\n"
            "- Lab values that contradict each other\n\n"
            "If NO conflicts found, respond exactly: NO_CONFLICTS_DETECTED\n\n"
            "=== CLINICAL NOTES SAMPLE ===\n"
            f"{sample}\n"
            "=== END ===\n\n"
            "Conflicts found:"
        )

        result = self.query_llm(prompt)

        if (result
                and "NO_CONFLICTS_DETECTED" not in result
                and "LLM_ERROR" not in result
                and len(result.strip()) > 15):
            conflicts.append(result[:600])

        return conflicts

    # ─────────────────────────────────────────────
    # MAIN AGENT LOOP
    # ─────────────────────────────────────────────

    def run(self, patient_folder: str) -> str:
        """
        Main execution loop following the ReAct pattern:
          1. Ingest all PDFs (with OCR if needed)
          2. Extract each required section
          3. Fallback search for anything not found
          4. Medication reconciliation
          5. Conflict detection
          6. Drug interaction check
          7. Compile the final summary draft
        """

        print("\n" + "=" * 60)
        print("  DISCHARGE SUMMARY AGENT — STARTING")
        print("=" * 60)

        # ── STEP 1: Read PDFs ──────────────────────────────────────

        self.step_count += 1
        self.log_step(
            "PLAN",
            "Starting document ingestion — reading all PDFs",
            "extract_pdfs_from_folder",
            "Beginning PDF extraction..."
        )

        try:
            self.pdf_texts = extract_pdfs_from_folder(patient_folder)

            if not self.pdf_texts:
                self.flags.add_flag(
                    "MISSING_DOCUMENTS",
                    "No PDF documents found in patient folder.",
                    "URGENT"
                )
                return self._compile_summary()

            # Merge all text into one searchable string
            self.all_text = "\n\n".join(
                f"=== DOCUMENT: {name} ===\n{text}"
                for name, text in self.pdf_texts.items()
            )

            doc_categories = categorize_documents(self.pdf_texts)

            self.log_step(
                "OBSERVE",
                "PDFs read and categorised successfully",
                "extract_pdfs_from_folder",
                (
                    f"Docs: {len(self.pdf_texts)} | "
                    f"Total chars: {len(self.all_text)} | "
                    f"Categories: {doc_categories}"
                )
            )

        except Exception as e:
            self.flags.add_flag(
                "DOCUMENT_READ_ERROR",
                f"Failed to read documents: {str(e)}",
                "URGENT"
            )
            self.log_step("ERROR", "Document read failed",
                          "extract_pdfs_from_folder", str(e))
            return self._compile_summary()

        # ── STEP 2: Extract each section ──────────────────────────

        print(f"\n{'=' * 60}")
        print(f"  EXTRACTING {len(DISCHARGE_SECTIONS)} SECTIONS")
        print(f"{'=' * 60}")

        for section in DISCHARGE_SECTIONS:

            # Hard cap — agent cannot loop forever
            if self.step_count >= self.max_steps:
                self.flags.add_flag(
                    "STEP_LIMIT_REACHED",
                    f"Hit max step limit ({self.max_steps}). "
                    f"Remaining sections were not processed.",
                    "REVIEW_REQUIRED"
                )
                print(f"\n⚠️  MAX STEPS REACHED — stopping early")
                break

            self.step_count += 1

            # THINK: find relevant text
            self.log_step(
                "THINK",
                f"Searching for: '{section['name']}'",
                "search_in_text",
                f"Keywords: {section['keywords'][:4]}"
            )

            relevant_text = search_in_text(self.all_text, section["keywords"])

            # ACT: ask LLM to extract
            self.log_step(
                "ACT",
                f"Sending {len(relevant_text)} chars to LLM for: '{section['name']}'",
                "extract_section",
                f"Text length: {len(relevant_text)} chars"
            )

            result = self.extract_section(section["name"], relevant_text)

            # ── OBSERVE: handle the result ─────────────────────────

            if not result or "LLM_ERROR" in result:
                # LLM completely failed
                self.flags.add_flag(
                    "EXTRACTION_FAILED",
                    f"LLM failed to extract '{section['name']}'",
                    "REVIEW_REQUIRED"
                )
                self.extracted_info[section["name"]] = (
                    "⚠️ EXTRACTION FAILED — Clinician must complete this section"
                )

            elif "NOT_FOUND" in result:
                # First attempt found nothing — try the fallback
                self.log_step(
                    "THINK",
                    f"'{section['name']}' not found in targeted search — trying broad fallback",
                    "extract_section_fallback",
                    "Scanning wider document slice"
                )

                fallback_result = self.extract_section_fallback(section["name"])

                if (fallback_result
                        and "NOT_FOUND" not in fallback_result
                        and "LLM_ERROR" not in fallback_result
                        and len(fallback_result.strip()) > 15):
                    # Fallback found something — mark as low confidence
                    self.extracted_info[section["name"]] = (
                        fallback_result.strip()
                        + "\n\n[EXTRACTED VIA FALLBACK — VERIFY CAREFULLY]"
                    )
                    self.flags.add_flag(
                        "LOW_CONFIDENCE_EXTRACTION",
                        f"'{section['name']}' found only via fallback search "
                        f"(OCR quality may be poor).",
                        "REVIEW_REQUIRED"
                    )
                else:
                    # Genuinely not in the documents
                    self.extracted_info[section["name"]] = (
                        "⚠️ NOT FOUND IN DOCUMENTS — Clinician must complete this section"
                    )
                    self.flags.add_flag(
                        "MISSING_DATA",
                        f"'{section['name']}' not found after two search attempts.",
                        "REVIEW_REQUIRED"
                    )

            elif "INCOMPLETE" in result or "FLAG FOR REVIEW" in result:
                # Partial data — keep it but flag it
                self.extracted_info[section["name"]] = result
                self.flags.add_flag(
                    "INCOMPLETE_DATA",
                    f"'{section['name']}' is incomplete — some fields missing.",
                    "REVIEW_REQUIRED"
                )

            else:
                # Clean extraction
                self.extracted_info[section["name"]] = result

            self.log_step(
                "OBSERVE",
                f"Section '{section['name']}' complete",
                "extract_section",
                result
            )

        # ── STEP 3: Medication Reconciliation ─────────────────────

        if self.step_count < self.max_steps:
            self.step_count += 1
            self.log_step(
                "THINK",
                "Comparing admission vs discharge medications for safety",
                "reconcile_medications",
                "Starting reconciliation"
            )

            admission_meds = self.extracted_info.get(
                "Admission Medications", "NOT FOUND IN DOCUMENTS"
            )
            discharge_meds = self.extracted_info.get(
                "Discharge Medications", "NOT FOUND IN DOCUMENTS"
            )

            reconciliation = self.reconcile_medications(
                admission_meds, discharge_meds
            )
            self.extracted_info["Medication Changes (Reconciliation)"] = reconciliation

            if any(
                word in reconciliation.lower()
                for word in ["undocumented", "no documented reason",
                             "unclear", "cannot determine"]
            ):
                self.flags.add_flag(
                    "MEDICATION_RECONCILIATION",
                    "Medication changes detected without clear documented reasons.",
                    "REVIEW_REQUIRED"
                )

            self.log_step(
                "OBSERVE", "Medication reconciliation complete",
                "reconcile_medications", reconciliation[:200]
            )

        # ── STEP 4: Conflict Detection ─────────────────────────────

        if self.step_count < self.max_steps:
            self.step_count += 1
            self.log_step(
                "THINK",
                "Checking documents for contradictory information",
                "check_for_conflicts",
                "Scanning..."
            )

            llm_conflicts = self.check_for_conflicts()
            for c in llm_conflicts:
                self.flags.add_flag(
                    "CONFLICTING_INFORMATION",
                    f"Conflicting data detected: {c[:200]}",
                    "REVIEW_REQUIRED"
                )

            structural_conflicts = detect_conflicts(self.extracted_info)
            for c in structural_conflicts:
                self.flags.add_flag("STRUCTURAL_CONFLICT", c, "REVIEW_REQUIRED")

            total_conflicts = len(llm_conflicts) + len(structural_conflicts)
            self.log_step(
                "OBSERVE",
                f"Conflict check complete — {total_conflicts} conflict(s) found",
                "check_for_conflicts",
                f"LLM conflicts: {len(llm_conflicts)} | "
                f"Structural: {len(structural_conflicts)}"
            )

        # ── STEP 5: Drug Interaction Check ────────────────────────

        if self.step_count < self.max_steps:
            self.step_count += 1
            self.log_step(
                "THINK",
                "Checking discharge medications for dangerous interactions",
                "check_drug_interactions",
                "Running mocked drug interaction tool"
            )

            all_meds_text = (
                self.extracted_info.get("Admission Medications", "")
                + " "
                + self.extracted_info.get("Discharge Medications", "")
            )

            interactions = check_drug_interactions(all_meds_text)

            if interactions:
                for interaction in interactions:
                    self.flags.add_flag(
                        "DRUG_INTERACTION",
                        interaction,
                        "URGENT"
                    )
                self.extracted_info["Drug Interaction Alerts"] = "\n".join(interactions)
            else:
                self.extracted_info["Drug Interaction Alerts"] = (
                    "No known interactions detected "
                    "(mocked check — verify with pharmacist before discharge)."
                )

            self.log_step(
                "OBSERVE",
                f"Drug check complete — {len(interactions)} alert(s)",
                "check_drug_interactions",
                str(interactions[:2]) if interactions else "None"
            )

        # ── STEP 6: Compile summary ────────────────────────────────

        self.step_count += 1
        self.log_step(
            "COMPLETE",
            "All steps done — compiling discharge summary draft",
            "compile_summary",
            f"Steps: {self.step_count} | Flags: {len(self.flags.get_all_flags())}"
        )

        return self._compile_summary()

    # ─────────────────────────────────────────────
    # SUMMARY COMPILER
    # ─────────────────────────────────────────────

    def _compile_summary(self) -> str:
        """
        Formats all extracted sections into a readable
        discharge summary draft with safety disclaimers.
        """
        lines = []

        # ── Header ────────────────────────────────────────────────
        lines.append("# DISCHARGE SUMMARY — AI DRAFT")
        lines.append("")
        lines.append("> ⚠️  **THIS IS AN AI-GENERATED DRAFT FOR CLINICIAN REVIEW ONLY**")
        lines.append("> ⚠️  **DO NOT USE AS A FINAL CLINICAL DOCUMENT**")
        lines.append("> ⚠️  **ALL INFORMATION MUST BE VERIFIED BY THE TREATING PHYSICIAN**")
        lines.append("")
        lines.append(f"- **Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        lines.append(f"- **AI Model:** {self.model}")
        lines.append(f"- **Agent steps taken:** {self.step_count}")
        lines.append(f"- **Flags for clinician review:** {len(self.flags.get_all_flags())}")
        lines.append("")
        lines.append("---")
        lines.append("")

        # ── Flags first — clinician must see these immediately ─────
        all_flags = self.flags.get_all_flags()
        if all_flags:
            urgent = [f for f in all_flags if f["severity"] == "URGENT"]
            review = [f for f in all_flags if f["severity"] == "REVIEW_REQUIRED"]
            info   = [f for f in all_flags if f["severity"] == "INFO"]

            lines.append("## 🚨 FLAGS FOR CLINICIAN REVIEW")
            lines.append("")
            if urgent:
                lines.append("### 🚨 URGENT — Act before discharge")
                for flag in urgent:
                    lines.append(f"- 🚨 **[URGENT]** *{flag['type']}*: {flag['description']}")
                lines.append("")
            if review:
                lines.append("### ⚠️ REVIEW REQUIRED — Check before finalising")
                for flag in review:
                    lines.append(f"- ⚠️ **[REVIEW]** *{flag['type']}*: {flag['description']}")
                lines.append("")
            if info:
                lines.append("### ℹ️ INFO")
                for flag in info:
                    lines.append(f"- ℹ️ *{flag['type']}*: {flag['description']}")
                lines.append("")

            lines.append("---")
            lines.append("")

        # ── Clinical sections in logical order ─────────────────────
        section_order = [
            "Patient Demographics",
            "Admission and Discharge Dates",
            "Principal Diagnosis",
            "Secondary Diagnoses",
            "Allergies",
            "Vital Signs at Discharge",
            "Condition at Discharge",
            "Hospital Course",
            "Procedures Performed",
            "Admission Medications",
            "Discharge Medications",
            "Medication Changes (Reconciliation)",
            "Drug Interaction Alerts",
            "Pending Results",
            "Follow-up Instructions",
        ]

        for section_name in section_order:
            if section_name in self.extracted_info:
                lines.append(f"## {section_name}")
                lines.append("")
                lines.append(self.extracted_info[section_name])
                lines.append("")
                lines.append("---")
                lines.append("")

        # Any extra sections not in the ordered list
        for section_name, content in self.extracted_info.items():
            if section_name not in section_order:
                lines.append(f"## {section_name}")
                lines.append("")
                lines.append(content)
                lines.append("")
                lines.append("---")
                lines.append("")

        # ── Run statistics ─────────────────────────────────────────
        lines.append("## Agent Run Statistics")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        lines.append(f"| Total steps taken | {self.step_count} |")
        lines.append(f"| Max steps cap | {self.max_steps} |")
        lines.append(f"| Sections extracted | {len(self.extracted_info)} |")
        lines.append(f"| Flags raised | {len(all_flags)} |")
        lines.append(f"| Documents processed | {len(self.pdf_texts)} |")
        lines.append("")
        lines.append(
            "> *This document was produced by an AI agent. "
            "The agent is designed to never fabricate clinical information. "
            "Every field marked NOT FOUND or flagged must be completed "
            "by the clinician before this document is finalised.*"
        )

        return "\n".join(lines)

    def save_trace(self, output_path: str):
        """Saves the complete step-by-step agent trace to JSON."""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(self.trace, f, indent=2, ensure_ascii=False)
            print(f"\n✅ Trace saved: {output_path}")
        except Exception as e:
            print(f"\n❌ Could not save trace: {e}")
