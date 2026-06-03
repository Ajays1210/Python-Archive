# agent.py
# True ReAct agent loop: Think -> Act -> Observe -> Repeat
# The agent decides which section to extract next, when to call tools,
# and when to finish, based on what it has already found.

import ollama
import json
from datetime import datetime
from typing import Dict, List, Optional, Set

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
    ReAct agent that reads patient records and produces a structured
    discharge summary draft for clinician review.
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

        # Which sections still need to be extracted
        self.remaining_sections = {s["name"] for s in DISCHARGE_SECTIONS}
        # Which special actions are still pending
        self.pending_actions = {
            "medication_reconciliation": True,
            "conflict_detection": True,
            "drug_interaction_check": True
        }

        print(f"Agent initialized | Model: {model} | Max steps: {max_steps}")

    # ------------------------------------------------------------------
    # STEP LOGGER (unchanged)
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # LLM QUERY with retry (unchanged)
    # ------------------------------------------------------------------
    def query_llm(self, prompt: str,
                  system_prompt: str = None,
                  max_retries: int = 2) -> str:
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
                        "temperature": 0.05,
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

    # ------------------------------------------------------------------
    # SECTION EXTRACTOR (unchanged logic, but now callable as tool)
    # ------------------------------------------------------------------
    def extract_section(self, section_name: str, relevant_text: str) -> str:
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

    def extract_section_fallback(self, section_name: str) -> str:
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

    # ------------------------------------------------------------------
    # TOOL: Extract a specific section
    # ------------------------------------------------------------------
    def _tool_extract_section(self, section_name: str) -> str:
        """Perform extraction for a given section, with fallback."""
        # Find keywords for this section
        section_def = next((s for s in DISCHARGE_SECTIONS if s["name"] == section_name), None)
        if not section_def:
            return f"ERROR: Unknown section '{section_name}'"

        keywords = section_def["keywords"]
        relevant_text = search_in_text(self.all_text, keywords)

        result = self.extract_section(section_name, relevant_text)

        if not result or "LLM_ERROR" in result:
            self.flags.add_flag(
                "EXTRACTION_FAILED",
                f"LLM failed to extract '{section_name}'",
                "REVIEW_REQUIRED"
            )
            self.extracted_info[section_name] = "⚠️ EXTRACTION FAILED — Clinician must complete this section"
            return "FAILED"

        elif "NOT_FOUND" in result:
            # Try fallback
            fallback_result = self.extract_section_fallback(section_name)
            if (fallback_result and "NOT_FOUND" not in fallback_result
                    and "LLM_ERROR" not in fallback_result
                    and len(fallback_result.strip()) > 15):
                self.extracted_info[section_name] = (
                    fallback_result.strip()
                    + "\n\n[EXTRACTED VIA FALLBACK — VERIFY CAREFULLY]"
                )
                self.flags.add_flag(
                    "LOW_CONFIDENCE_EXTRACTION",
                    f"'{section_name}' found only via fallback search (OCR quality may be poor).",
                    "REVIEW_REQUIRED"
                )
                return "EXTRACTED_LOW_CONFIDENCE"
            else:
                self.extracted_info[section_name] = (
                    "⚠️ NOT FOUND IN DOCUMENTS — Clinician must complete this section"
                )
                self.flags.add_flag(
                    "MISSING_DATA",
                    f"'{section_name}' not found after two search attempts.",
                    "REVIEW_REQUIRED"
                )
                return "NOT_FOUND"

        elif "INCOMPLETE" in result or "FLAG FOR REVIEW" in result:
            self.extracted_info[section_name] = result
            self.flags.add_flag(
                "INCOMPLETE_DATA",
                f"'{section_name}' is incomplete — some fields missing.",
                "REVIEW_REQUIRED"
            )
            return "INCOMPLETE"

        else:
            self.extracted_info[section_name] = result
            return "EXTRACTED"

    # ------------------------------------------------------------------
    # TOOL: Medication reconciliation
    # ------------------------------------------------------------------
    def _tool_reconcile_medications(self) -> str:
        admission_meds = self.extracted_info.get("Admission Medications", "NOT FOUND IN DOCUMENTS")
        discharge_meds = self.extracted_info.get("Discharge Medications", "NOT FOUND IN DOCUMENTS")
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
        result = self.query_llm(prompt)
        self.extracted_info["Medication Changes (Reconciliation)"] = result
        if any(word in result.lower() for word in ["undocumented", "no documented reason", "unclear", "cannot determine"]):
            self.flags.add_flag(
                "MEDICATION_RECONCILIATION",
                "Medication changes detected without clear documented reasons.",
                "REVIEW_REQUIRED"
            )
        return result

    # ------------------------------------------------------------------
    # TOOL: Conflict detection
    # ------------------------------------------------------------------
    def _tool_detect_conflicts(self) -> List[str]:
        conflicts = []
        sample = self.all_text[:3000] + "\n...(middle)...\n" + self.all_text[5000:8000]
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
        if result and "NO_CONFLICTS_DETECTED" not in result and "LLM_ERROR" not in result and len(result.strip()) > 15:
            conflicts.append(result[:600])
        structural = detect_conflicts(self.extracted_info)
        conflicts.extend(structural)
        for c in conflicts:
            self.flags.add_flag("CONFLICTING_INFORMATION", c[:200], "REVIEW_REQUIRED")
        return conflicts

    # ------------------------------------------------------------------
    # TOOL: Drug interaction check
    # ------------------------------------------------------------------
    def _tool_check_drug_interactions(self) -> List[str]:
        all_meds_text = (
            self.extracted_info.get("Admission Medications", "")
            + " "
            + self.extracted_info.get("Discharge Medications", "")
        )
        interactions = check_drug_interactions(all_meds_text)
        if interactions:
            for interaction in interactions:
                self.flags.add_flag("DRUG_INTERACTION", interaction, "URGENT")
            self.extracted_info["Drug Interaction Alerts"] = "\n".join(interactions)
        else:
            self.extracted_info["Drug Interaction Alerts"] = (
                "No known interactions detected (mocked check — verify with pharmacist before discharge)."
            )
        return interactions

    # ------------------------------------------------------------------
    # PLANNING: Decide next action using LLM
    # ------------------------------------------------------------------
    def _decide_next_action(self) -> Dict:
        """
        Ask the LLM what to do next based on current state.
        Returns a dict with keys: 'action_type' and 'target' (if applicable).
        Possible action types:
          - extract_section : target = section name
          - reconcile_medications
          - detect_conflicts
          - check_interactions
          - finalize
        """
        # Build a summary of what's already extracted
        extracted_list = [s for s in self.remaining_sections if s not in self.remaining_sections]
        done_str = "\n".join([f"- {s}: {self.extracted_info.get(s, '')[:100]}..." for s in extracted_list]) if extracted_list else "None yet."

        remaining_str = "\n".join([f"- {s}" for s in self.remaining_sections]) if self.remaining_sections else "None."

        pending_actions_str = "\n".join([f"- {a}" for a, pending in self.pending_actions.items() if pending])

        prompt = f"""
You are a clinical agent planning the next step to complete a discharge summary.

Current state:
- Already extracted sections (first 100 chars each):
{done_str}

- Sections still needed:
{remaining_str}

- Special actions still pending:
{pending_actions_str}

Available actions:
1. extract_section [section_name]  (extract one missing section)
2. reconcile_medications           (compare admission vs discharge meds)
3. detect_conflicts                (check for contradictory info)
4. check_interactions              (check for drug interactions)
5. finalize                        (all done, compile summary)

Rules:
- Only propose actions that are still needed.
- If no sections remain and all special actions are done, choose finalize.
- Prefer extracting sections before running special actions, but if admission/discharge meds are both extracted, you may reconcile earlier.

Respond ONLY with a JSON object like:
{{"action_type": "extract_section", "target": "Patient Demographics"}}
or
{{"action_type": "reconcile_medications"}}
etc.
"""
        response = self.query_llm(prompt, system_prompt="You are a helpful clinical AI planning assistant. Output only valid JSON.")
        try:
            # Clean response (in case LLM adds extra text)
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            if json_start != -1 and json_end > json_start:
                decision = json.loads(response[json_start:json_end])
            else:
                decision = json.loads(response)
            # Validate action_type
            if decision.get("action_type") not in ["extract_section", "reconcile_medications", "detect_conflicts", "check_interactions", "finalize"]:
                raise ValueError("Invalid action_type")
            if decision["action_type"] == "extract_section" and "target" not in decision:
                raise ValueError("Missing target for extract_section")
            return decision
        except Exception as e:
            print(f"  ⚠️  Failed to parse LLM decision: {e}\nResponse: {response[:200]}")
            # Fallback: if any sections remain, pick first; else if pending actions, pick first; else finalize
            if self.remaining_sections:
                return {"action_type": "extract_section", "target": next(iter(self.remaining_sections))}
            for action, pending in self.pending_actions.items():
                if pending:
                    return {"action_type": action}
            return {"action_type": "finalize"}

    # ------------------------------------------------------------------
    # MAIN REACT LOOP
    # ------------------------------------------------------------------
    def run(self, patient_folder: str) -> str:
        print("\n" + "=" * 60)
        print("  DISCHARGE SUMMARY AGENT — STARTING (ReAct loop)")
        print("=" * 60)

        # ---- Initial ingestion ----
        self.step_count += 1
        self.log_step("PLAN", "Reading all PDFs from patient folder", "extract_pdfs_from_folder", "Starting...")
        try:
            self.pdf_texts = extract_pdfs_from_folder(patient_folder)
            if not self.pdf_texts:
                self.flags.add_flag("MISSING_DOCUMENTS", "No PDF documents found.", "URGENT")
                return self._compile_summary()
            self.all_text = "\n\n".join(
                f"=== DOCUMENT: {name} ===\n{text}"
                for name, text in self.pdf_texts.items()
            )
            doc_categories = categorize_documents(self.pdf_texts)
            self.log_step("OBSERVE", "PDFs read and categorized", "extract_pdfs_from_folder",
                          f"Docs: {len(self.pdf_texts)} | Chars: {len(self.all_text)} | Categories: {doc_categories}")
        except Exception as e:
            self.flags.add_flag("DOCUMENT_READ_ERROR", f"Failed to read documents: {str(e)}", "URGENT")
            self.log_step("ERROR", "Document read failed", "extract_pdfs_from_folder", str(e))
            return self._compile_summary()

        # ---- ReAct loop ----
        while self.step_count < self.max_steps:
            self.step_count += 1

            # 1. THINK: decide next action
            decision = self._decide_next_action()
            action = decision["action_type"]
            target = decision.get("target", None)

            self.log_step("THINK", f"Decided next action: {action}" + (f" on '{target}'" if target else ""),
                          action, "Planning")

            # 2. ACT: perform the action
            if action == "extract_section":
                if target not in self.remaining_sections:
                    self.log_step("ACT", f"Section '{target}' already extracted, skipping", "skip", "")
                    continue
                self.log_step("ACT", f"Extracting section '{target}'", "extract_section", "")
                result_status = self._tool_extract_section(target)
                if result_status in ["EXTRACTED", "EXTRACTED_LOW_CONFIDENCE", "INCOMPLETE", "NOT_FOUND", "FAILED"]:
                    self.remaining_sections.discard(target)
                self.log_step("OBSERVE", f"Extraction of '{target}' complete: {result_status}",
                              "extract_section", self.extracted_info.get(target, "")[:200])

            elif action == "reconcile_medications":
                if not self.pending_actions.get("medication_reconciliation", False):
                    self.log_step("ACT", "Medication reconciliation already done, skipping", "skip", "")
                else:
                    self.log_step("ACT", "Running medication reconciliation", "reconcile_medications", "")
                    result = self._tool_reconcile_medications()
                    self.pending_actions["medication_reconciliation"] = False
                    self.log_step("OBSERVE", "Medication reconciliation complete", "reconcile_medications", result[:200])

            elif action == "detect_conflicts":
                if not self.pending_actions.get("conflict_detection", False):
                    self.log_step("ACT", "Conflict detection already done, skipping", "skip", "")
                else:
                    self.log_step("ACT", "Checking for conflicts", "detect_conflicts", "")
                    conflicts = self._tool_detect_conflicts()
                    self.pending_actions["conflict_detection"] = False
                    self.log_step("OBSERVE", f"Conflict detection complete: {len(conflicts)} found",
                                  "detect_conflicts", str(conflicts)[:200])

            elif action == "check_interactions":
                if not self.pending_actions.get("drug_interaction_check", False):
                    self.log_step("ACT", "Drug interaction check already done, skipping", "skip", "")
                else:
                    self.log_step("ACT", "Checking drug interactions", "check_drug_interactions", "")
                    interactions = self._tool_check_drug_interactions()
                    self.pending_actions["drug_interaction_check"] = False
                    self.log_step("OBSERVE", f"Drug interaction check complete: {len(interactions)} alerts",
                                  "check_drug_interactions", str(interactions)[:200])

            elif action == "finalize":
                self.log_step("COMPLETE", "All tasks done, compiling final summary", "finalize", "")
                break

            else:
                self.log_step("ERROR", f"Unknown action '{action}', falling back to finalize", "fallback", "")
                break

            # After each action, check if we are done
            if not self.remaining_sections and not any(self.pending_actions.values()):
                self.log_step("COMPLETE", "All sections extracted and special actions done", "auto_finalize", "")
                break

        # If loop ended due to step cap, flag it
        if self.step_count >= self.max_steps and (self.remaining_sections or any(self.pending_actions.values())):
            self.flags.add_flag(
                "STEP_LIMIT_REACHED",
                f"Hit max step limit ({self.max_steps}). Remaining work not completed.",
                "REVIEW_REQUIRED"
            )
            print(f"\n⚠️  MAX STEPS REACHED — stopping early")

        return self._compile_summary()

    # ------------------------------------------------------------------
    # SUMMARY COMPILER (unchanged, but referenced)
    # ------------------------------------------------------------------
    def _compile_summary(self) -> str:
        lines = []
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

        for section_name, content in self.extracted_info.items():
            if section_name not in section_order:
                lines.append(f"## {section_name}")
                lines.append("")
                lines.append(content)
                lines.append("")
                lines.append("---")
                lines.append("")

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
        lines.append("> *This document was produced by an AI agent. The agent is designed to never fabricate clinical information. Every field marked NOT FOUND or flagged must be completed by the clinician before this document is finalised.*")

        return "\n".join(lines)

    def save_trace(self, output_path: str):
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(self.trace, f, indent=2, ensure_ascii=False)
            print(f"\n✅ Trace saved: {output_path}")
        except Exception as e:
            print(f"\n❌ Could not save trace: {e}")git add "Discharge Summary Agent/agent.py"