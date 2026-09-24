---
name: scout
description: Run the quant-scout to propose a batch of new trading ideas as hypothesis cards for the user to approve (checkpoint A). Only when the user explicitly asks.
argument-hint: "[book: fbs|b3] [theme or constraint, e.g. 'mean reversion on metals H4', 'filters for existing system #12'] [n=5]"
disable-model-invocation: true
---

# Scout an idea batch

1. Delegate to the **quant-scout** agent with: `$ARGUMENTS` (default: FBS book, 5 ideas, any
   component type), a pointer to `research/DESIGN.md`, and the list of already-tested ideas
   (ledger + closed issues + `DocumentationVault/systems/`).
2. The scout drafts cards in `research/systems/_drafts/`. Review them yourself against the card
   template (every field filled, rules unambiguous, prior ranges justified, sources cited,
   provenance honest). Send back incomplete cards once.
3. Present the batch to the user as a ranked table: name · book · component · mechanism (1 line) ·
   expected trades/yr · cost headroom · source. **Checkpoint A**: ask which to approve.
4. For approved cards only: create GitHub issues (`gh issue create --label hypothesis`) and note
   the approval in each issue body ("Approved for testing by the user on <date>"). Rejected drafts
   are deleted, not kept as issues.
