---
name: promote
description: Checkpoint C — the user promotes a system that passed its holdout and book review; closes research on it and starts deployment (MQL5/ONNX port + parity). User-invoked only.
argument-hint: "<system-slug>"
disable-model-invocation: true
---

# Promote a system

1. Verify: holdout PASS recorded in the ledger, portfolio review recommends inclusion (or the user
   explicitly overrides — record that). Otherwise stop and explain.
2. Label the issue `promoted` and close it (research done). Update the Obsidian card via
   **report-builder** (status: promoted).
3. Delegate to **mql5-porter** for the EA / ONNX port and the parity protocol. Report parity results;
   the system is "ready for MT5" only when parity passes **and** the user confirms.
4. Record the book membership change for future `/decay-review` runs.
