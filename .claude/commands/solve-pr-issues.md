Resolve all outstanding problems on an open pull request: fix merge conflicts, address every code review comment, verify quality gates, then push. If any finding has no safe automated fix, update the comment and label the issue for manual review.

## Your inputs

Arguments: "$ARGUMENTS" — format: `<repo> <pr-number>`
- `repo` is either `backend` or `frontend`
- `pr-number` is the GitHub PR number (e.g. `30`)

Resolve repo:
- `backend` → repo handle `DougOscar/AlphaForgeBackend`, working dir `/home/douglaso/Finances/AlphaForge/AlphaForgeBackend/`
- `frontend` → repo handle `DougOscar/AlphaForgeFrontend`, working dir `/home/douglaso/Finances/AlphaForge/AlphaForgeFrontend/`

---

## Step 1 — Load the PR

```bash
gh pr view <pr-number> --repo <repo> --json number,title,headRefName,baseRefName,body,closingIssuesReferences
```

Note the head branch, base branch, and any issue numbers from `Closes #N` / `Fixes #N` in the PR body.

Fetch all open (unresolved) review threads:

```bash
gh api graphql -f query='
{
  repository(owner:"DougOscar", name:"AlphaForge[Backend|Frontend]") {
    pullRequest(number: <pr-number>) {
      headRefOid
      reviewThreads(first: 50) {
        nodes {
          id
          isResolved
          comments(first: 3) {
            nodes { id body author { login } }
          }
        }
      }
    }
  }
}'
```

Collect every thread where `isResolved: false`. These are the findings to fix.

---

## Step 2 — Checkout and rebase

```bash
git fetch origin
git checkout <head-branch>
git rebase origin/<base-branch>
```

### If rebase produces conflicts

For each conflicted file:

1. Read the conflict markers (`<<<<<<<`, `=======`, `>>>>>>>`) carefully.
2. Understand what both sides are doing — do not blindly pick one side.
3. Resolve to a result that preserves the intent of **both** sides.
4. Never drop code that fixes a bug or enforces an invariant; if unsure, keep both and leave a comment.

After resolving all conflicts:

```bash
git add <resolved-files>
git rebase --continue
```

If a rebase conflict cannot be safely resolved automatically, abort (`git rebase --abort`), document the conflict in a PR comment, and proceed to Step 6 for the affected finding.

---

## Step 3 — Analyse every review finding

For each unresolved thread from Step 1:

1. **Classify the finding** — identify which file and line it refers to.
2. **Determine solvability**:
   - **Automatable** — a concrete code fix is clear from the comment. Proceed with Step 4.
   - **Needs investigation** — the fix requires understanding surrounding context. Read the relevant files first.
   - **No safe automated fix** — the comment describes a design decision, trade-off, or requirement that needs human judgement. Proceed to Step 6 for this finding.

Read every relevant source file before editing. Do not guess at the fix from the comment alone.

---

## Step 4 — Apply fixes

Fix each automatable finding. While applying each fix, enforce all Clean Architecture and code quality rules:

**Clean Architecture / DDD rules (backend):**
- Domain layer (`domain/`) must have zero imports from `infrastructure/`, `application/`, or any other BC's domain.
- No BC imports another BC's domain model. Use `shared_kernel/` primitives for cross-BC types.
- Use cases in `<bc>/application/` orchestrate domain objects — no business logic.
- Stubs (`raise NotImplementedError`) must be replaced — never leave them in implemented code.

**Code quality (backend):**
- Use `Decimal` for all price values and financial quantities, never `float`.
- Type-annotate all public functions and methods.
- Avoid `type: ignore` — fix the underlying type issue instead.
- No bare `except:` clauses — catch specific exception types.
- No magic numbers — use named constants.

After applying each fix, run the quality gates immediately to confirm no regression:

```bash
.venv/bin/pytest tests/ -q
.venv/bin/ruff check .
.venv/bin/mypy <affected_package>
```

If a fix introduces new failures, diagnose and correct them before moving to the next finding. Do not accumulate broken state.

---

## Step 5 — Double-check: scan for new problems

After all fixes are applied and gates pass, perform a self-review of every file you changed:

1. Re-read the diff: `git diff origin/<base-branch>`
2. Check against all six review criteria from `/review-pr`:
   - **Code efficiency** — no redundant computation in loops, no dead variables.
   - **Performance** — no regex where `str.replace` suffices, no linear scans on data that should be sets.
   - **Security** — no path traversal, no injection risks, no swallowed exceptions.
   - **Magic numbers** — no bare numeric literals with non-obvious business meaning.
   - **Naming** — names match domain vocabulary; mutable module-level constants wrapped in `MappingProxyType` or typed as `Mapping`.
   - **Architecture** — no cross-layer or cross-BC imports; no `float` on financial values.
3. If you find a new issue, fix it now and rerun the quality gates.
4. If the new issue has no safe automated fix, add it to the list for Step 6.

---

## Step 6 — Escalate findings that cannot be auto-fixed

For each finding (from review comments or your own Step 5 scan) that has no safe automated fix:

### 6a — Reply to the review thread

```bash
gh api repos/DougOscar/AlphaForge[Backend|Frontend]/pulls/<pr-number>/comments \
  --method POST \
  --field commit_id="<head-sha>" \
  --field in_reply_to=<comment-id> \
  --field body="$(cat <<'EOF'
**Needs manual review.**

<Explain what the problem is, why it cannot be safely auto-fixed, what options exist, and which option you recommend. Include a concrete code snippet for each option if possible.>
EOF
)"
```

### 6b — Add "help wanted" label to the linked issue

For each issue number found in step 1 (`Closes #N`):

```bash
gh label create "help wanted" --repo <repo> --color "e4e669" --description "Needs human decision" --force
gh issue edit <N> --repo <repo> --add-label "help wanted"
```

---

## Step 7 — Resolve fixed threads

For every review thread whose finding you successfully fixed in Step 4 or Step 5, mark it resolved:

```bash
gh api graphql -f query='
mutation {
  resolveReviewThread(input: { threadId: "<thread-id>" }) {
    thread { isResolved }
  }
}'
```

---

## Step 8 — Commit and push

Stage only the files you modified:

```bash
git add <file1> <file2> ...
git commit -m "$(cat <<'EOF'
fix(<scope>): address PR #<pr-number> review findings

<One paragraph: what was fixed and why. List each finding by its review
comment summary. Note any findings escalated for manual review.>

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
git push origin <head-branch>
```

---

## Step 9 — Final quality gate run

```bash
.venv/bin/pytest tests/ -q
.venv/bin/ruff check .
.venv/bin/mypy <affected_package>
```

All three must be clean before finishing. If any gate fails at this point, fix and commit again.

---

## Step 10 — Confirm

Print a summary:

```
PR #<N> updated
===============
Merge conflicts resolved : <count or "none">
Review findings fixed    : <count>
Findings escalated       : <count> (manual review required)
New problems found & fixed: <count>
Quality gates            : pytest ✓  ruff ✓  mypy ✓

Fixed threads resolved on GitHub: <count>
Escalated threads commented     : <count>
```

If any findings were escalated, append:
> "⚠ <count> finding(s) require manual review. 'help wanted' label added to issue #N."
