---
name: promote
description: Checkpoint C — the user promotes a system that passed its holdout and book review; closes research on it and starts deployment (MQL5/ONNX port + parity). User-invoked only.
argument-hint: "<system-slug>"
disable-model-invocation: true
---

# Promote a system

1. Verify:
   - The ledger records a **decisive holdout PASS**: `quantlab.ledger.holdout_state(book, system)`
     has `passed` true and `killed` false, and the latest `holdout_exam` has `status == "PASS"`.
   - `NOT_DECISIVE` is **not** a pass (DESIGN §4.4). The system stays `holdout_pending` until a
     re-exam on more data (`/unlock-holdout`) ends PASS. The user cannot override this.
   - `FAIL` means `killed`: never promote.
   - The portfolio review recommends inclusion, or the user explicitly overrides it (record that).
   If any check fails, stop and explain.
2. Label the issue `promoted` and close it (research done). Update the Obsidian card via
   **report-builder** (status: promoted).
3. Delegate to **mql5-porter** for the EA / ONNX port and the parity protocol. Report parity results;
   the system is "ready for MT5" only when parity passes **and** the user confirms.
4. Record the book membership change for future `/decay-review` runs.
