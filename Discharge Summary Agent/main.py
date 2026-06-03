# main.py
# Entry point — run this file to start the agent.
# PyCharm: right-click → Run  or press the green ▶ button.

import os
import json
from agent import DischargeAgent


def main():

    # ── Configuration ──────────────────────────────────────────
    PATIENT_FOLDER = "patient_data"   # Put patient PDFs here
    OUTPUT_FOLDER  = "output"         # Results are saved here
    MODEL          = "llama3.2:3b"    # Ollama model to use
    MAX_STEPS      = 30               # Hard cap on agent iterations

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # ── Check Ollama is running ────────────────────────────────
    print("Checking Ollama connection...")
    try:
        import ollama

        models     = ollama.list()
        model_list = (
            models.get('models', [])
            if isinstance(models, dict)
            else getattr(models, 'models', [])
        )

        available = []
        for m in model_list:
            if isinstance(m, dict):
                available.append(m.get('model') or m.get('name', ''))
            else:
                available.append(
                    getattr(m, 'model', '') or getattr(m, 'name', '')
                )

        print(f"✅ Ollama running | Models available: {available}")

        if not any(MODEL in m for m in available):
            print(f"\n⚠️  Model '{MODEL}' not found.")
            print(f"   Run this to download it:  ollama pull {MODEL}")
            return

    except Exception as e:
        print(f"❌ Cannot connect to Ollama: {e}")
        print("   Make sure Ollama is open (look for the llama icon in your system tray).")
        return

    # ── Check patient data folder ──────────────────────────────
    if not os.path.exists(PATIENT_FOLDER):
        print(f"\n❌ Folder '{PATIENT_FOLDER}' not found.")
        print("   Create it and put your patient PDFs inside.")
        return

    pdf_files = [f for f in os.listdir(PATIENT_FOLDER) if f.lower().endswith('.pdf')]
    if not pdf_files:
        print(f"\n❌ No PDFs found in '{PATIENT_FOLDER}/'")
        return

    print(f"✅ Found {len(pdf_files)} PDF(s) in {PATIENT_FOLDER}/")

    # ── Run the agent ──────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"  STARTING DISCHARGE SUMMARY AGENT")
    print(f"  Model: {MODEL} | Max Steps: {MAX_STEPS}")
    print(f"{'=' * 60}")

    agent   = DischargeAgent(model=MODEL, max_steps=MAX_STEPS)
    summary = agent.run(PATIENT_FOLDER)

    # ── Save all outputs ───────────────────────────────────────

    # 1. Discharge summary (markdown)
    summary_path = os.path.join(OUTPUT_FOLDER, "discharge_summary.md")
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(summary)
    print(f"\n✅ Discharge summary → {summary_path}")

    # 2. Full agent trace (JSON)
    trace_path = os.path.join(OUTPUT_FOLDER, "agent_trace.json")
    agent.save_trace(trace_path)

    # 3. Flags for clinician review (JSON)
    flags_path = os.path.join(OUTPUT_FOLDER, "flags_for_review.json")
    with open(flags_path, 'w', encoding='utf-8') as f:
        json.dump(agent.flags.get_all_flags(), f, indent=2, ensure_ascii=False)
    print(f"✅ Flags           → {flags_path}")

    # ── Final summary to terminal ──────────────────────────────
    print("\n" + "=" * 60)
    print("  AGENT FINISHED")
    print("=" * 60)
    print(f"\n  Steps taken  : {agent.step_count}")
    print(f"  Flags raised : {len(agent.flags.get_all_flags())}")
    print(f"\n  Output files:")
    print(f"    📄 {summary_path}")
    print(f"    📋 {trace_path}")
    print(f"    🚩 {flags_path}")

    print("\n" + "─" * 60)
    print("SUMMARY PREVIEW (first 1200 chars):")
    print("─" * 60)
    print(summary[:1200])
    print("...")


if __name__ == "__main__":
    main()
