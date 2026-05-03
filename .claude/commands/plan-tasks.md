Audit the AlphaForge $ARGUMENTS repository and open GitHub issues for any gaps found between the documentation and the current implementation.

## Your inputs

- Argument: "$ARGUMENTS" — must be either `backend` or `frontend`
- If the argument is `backend`, work in `/home/douglaso/Finances/AlphaForge/AlphaForgeBackend/` against repo `DougOscar/AlphaForgeBackend`
- If the argument is `frontend`, work in `/home/douglaso/Finances/AlphaForge/AlphaForgeFrontend/` against repo `DougOscar/AlphaForgeFrontend`
- Documentation source of truth: `/home/douglaso/Finances/AlphaForge/AlphaForgeDoc/`

## Step-by-step

### 1. Read the documentation

Read the following files to understand what the platform is supposed to do:
- `AlphaForgeDoc/00-overview/Bounded Context Map.md`
- `AlphaForgeDoc/00-overview/Dependency Rule.md`
- `AlphaForgeDoc/00-overview/Happy Path Flows.md`
- `AlphaForgeDoc/00-overview/Domain Invariants.md`
- `AlphaForgeDoc/09-application-layer/Application Layer.md`
- `AlphaForgeDoc/09-application-layer/Infrastructure Layer.md`
- All subdirectory docs relevant to the chosen repo (e.g., `01-market-data/`, `02-strategy/`, `03-backtesting-engine/`, etc.)

### 2. Inspect the repository

For **backend**: scan every Python package (`shared_kernel/`, `market_data/`, `strategy/`, `backtesting_engine/`, `alpha_research/`, `performance_analytics/`, `optimization/`, `robustness/`, `infrastructure/`). For each package, read the source files and note:
- What is fully implemented vs what is a stub (`raise NotImplementedError`)
- What is mentioned in the docs but has no corresponding code at all
- What is implemented but missing tests

For **frontend**: scan the repo structure and note what UI screens / API integrations are expected by the documentation but absent.

### 3. Fetch existing issues

Run: `gh issue list --repo DougOscar/AlphaForge[Backend|Frontend] --state open --limit 100`

Build a set of issue titles to avoid creating duplicates.

### 4. Identify gaps

Cross-reference documentation requirements against what exists in code and what already has an open issue. A gap is any documented feature, use case, repository, value object, domain service, or infrastructure adapter that:
- Is not yet implemented (stub or missing entirely), AND
- Does not already have an open issue

### 5. Present the plan

Before opening any issues, print a numbered list of the gaps you found with a one-line description and proposed issue title for each. Then ask the user:

> "I found N gaps. Shall I open all of them, or would you like to select which ones to create?"

Wait for user confirmation before proceeding.

### 6. Open issues

For each approved gap, create a GitHub issue:

```bash
gh issue create \
  --repo DougOscar/AlphaForge[Backend|Frontend] \
  --title "<title>" \
  --label "enhancement" \
  --body "<body>"
```

Issue body format:
```
## Context
<One paragraph linking this to the documented feature or use case. Reference the relevant doc file by name.>

## What needs to be done
<Bullet list of the concrete work items — files to create, classes to implement, interfaces to satisfy.>

## Acceptance criteria
- [ ] <specific, testable criterion>
- [ ] All existing tests pass
- [ ] ruff and mypy clean
```

### 7. Report

After creating all issues, print a summary table:

| # | Issue title | GitHub issue number |
|---|-------------|---------------------|
| 1 | ...         | #N                  |
