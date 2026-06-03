# Discharge Summary Agent

An agentic AI system that reads raw patient record PDFs
and produces a structured discharge summary draft for clinician review.
Built with Python + Ollama (llama3.2:3b running locally).

---

## How to Run

### Prerequisites

1. **Ollama** — download from https://ollama.com and install.
   Then pull the model:
   ```
   ollama pull llama3.2:3b
   ```

2. **Tesseract OCR** (Windows) — download from:
   https://github.com/UB-Mannheim/tesseract/wiki
   Install to `C:\Program Files\Tesseract-OCR\`

3. **Poppler** (Windows) — download from:
   https://github.com/oschwartz10612/poppler-windows/releases
   Extract to `C:\poppler-26.02.0\`

4. **Python packages**:
   ```
   pip install -r requirements.txt
   ```

### Running the agent

1. Start Ollama (open the app or run `ollama serve`)
2. Put patient PDF(s) in the `patient_data/` folder
3. Run:
   ```
   python main.py
   ```

### Output files (saved to `output/`)

| File | Description |
|------|-------------|
| `discharge_summary.md` | The structured discharge summary draft |
| `agent_trace.json` | Every step the agent took (reasoning + result) |
| `flags_for_review.json` | All issues flagged for clinician attention |

---

## Agent Loop Design

The agent uses the **ReAct pattern** (Reason → Act → Observe):

```
PLAN   → Read all PDFs (OCR if scanned)
  ↓
THINK  → Pick the next section to extract
  ↓
ACT    → Search document for relevant text, send to LLM
  ↓
OBSERVE → Did we get a clean result?
  ├── YES → Store it, move to next section
  ├── PARTIAL → Store with [INCOMPLETE] flag
  └── NOT FOUND → Run fallback broad search, then flag if still missing
  ↓
REPEAT until all 13 sections are done or max_steps cap is hit
  ↓
Medication reconciliation → Conflict detection → Drug interaction check
  ↓
Compile final summary with all flags
```

The agent is **not a fixed pipeline** — it re-plans after every step.
If a section is not found by the targeted keyword search, it
automatically tries a broader fallback scan before giving up.

---

## No-Fabrication Guardrail

This is the most important safety property of the system.

**How it is enforced:**

1. **System prompt** — every LLM call includes an explicit instruction:
   *"NEVER guess, infer, or invent any clinical information.
   If the information is not present, respond with exactly: NOT_FOUND."*

2. **Two-attempt strategy** — if the first targeted search returns
   NOT_FOUND, the agent tries a broader fallback. If that also returns
   NOT_FOUND, the section is marked:
   `⚠️ NOT FOUND IN DOCUMENTS — Clinician must complete this section`

3. **Low-confidence labelling** — any extraction that required the
   fallback is labelled `[EXTRACTED VIA FALLBACK — VERIFY CAREFULLY]`
   so the clinician knows to scrutinise it.

4. **Draft-only output** — the summary header contains three visible
   warnings that this is an AI draft and must not be used as a
   final clinical document.

5. **Temperature = 0.05** — the LLM is set to near-zero temperature
   so it stays factual and does not generate creative content.

---

## Failure and Conflict Handling

| Situation | What the agent does |
|-----------|---------------------|
| LLM call fails | Retries up to 2 times; flags section as EXTRACTION_FAILED |
| Section not found | Runs fallback search; flags as MISSING_DATA if still absent |
| PDF has no text (scanned) | Automatically switches to Tesseract OCR |
| OCR fails | Returns whatever standard extraction got; flags the issue |
| Max steps reached | Stops cleanly; flags remaining sections as unprocessed |
| Conflicting diagnoses | LLM detects conflict; flagged as CONFLICTING_INFORMATION |
| Undocumented medication change | Flagged as MEDICATION_RECONCILIATION |
| Drug interaction detected | Flagged as URGENT in both summary and flags file |

---

## Why Each Tool Exists

| Tool | Purpose |
|------|---------|
| `search_in_text` | Finds the most relevant paragraph for each section without sending the full document to the LLM (which would exceed context limits) |
| `check_drug_interactions` | Mocked safety check — in production would call DrugBank/Micromedex API |
| `ClinicalFlagSystem` | Accumulates every issue that needs clinician attention in one place |
| `detect_conflicts` | Structural check on already-extracted sections for logical contradictions |

---

## Limitations

- **Small model** — llama3.2:3b has a 3-billion parameter count.
  A larger model (8B+) would extract more accurately from
  garbled OCR text.

- **OCR quality on handwriting** — Tesseract was designed for
  printed text. Heavily cursive handwriting is partially misread,
  which is why some sections come back with OCR errors.

- **Single PDF** — the current setup processes one PDF containing
  all patient records. A multi-patient folder structure would
  need a loop in `main.py`.

- **Mocked drug checker** — the interaction database has only
  5 known pairs. Production requires a real API.

- **No Part 2 (learning loop)** — the simulated doctor-edit
  feedback mechanism was not implemented due to time constraints.

---

## What I Would Do With More Time

1. Use **llama3:8b** or a medical-specific model for better extraction
2. Pre-process images before OCR (deskew, denoise, binarize)
   to improve accuracy on handwritten notes
3. Implement the **Part 2 learning loop**:
   - Simulate a doctor reviewer that applies consistent edits
   - Measure edit distance between draft and corrected version
   - Use correction memory injected into future prompts
4. Add a proper **drug interaction API** (DrugBank or openFDA)
5. Build a simple web UI so clinicians can review and edit in-browser
6. Add **multi-patient batch processing**
