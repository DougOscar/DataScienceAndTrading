Implement a GitHub issue that has been labeled "Refined & Ready": create a branch, write the code and tests, verify quality gates, commit, and open a PR.

## Your inputs

Arguments: "$ARGUMENTS" — format: `<repo> <issue-number>`
- `repo` is either `backend` or `frontend`
- `issue-number` is the GitHub issue number

Resolve repo:
- `backend` → repo handle `DougOscar/AlphaForgeBackend`, working dir `/home/douglaso/Finances/AlphaForge/AlphaForgeBackend/`
- `frontend` → repo handle `DougOscar/AlphaForgeFrontend`, working dir `/home/douglaso/Finances/AlphaForge/AlphaForgeFrontend/`

## Step-by-step

### 1. Read and validate the issue

```bash
gh issue view <issue-number> --repo DougOscar/AlphaForge[Backend|Frontend]
```

**Verify the issue has the "Refined & Ready" label.** If it does not, stop and tell the user:
> "Issue #N does not have the 'Refined & Ready' label. Run `/refine-issue <repo> <issue-number>` first."

Find the implementation plan in the issue comments (posted by `/refine-issue`). This is the authoritative specification — follow it precisely.

### 1b — Cross-check merged PRs for prior solutions

Before writing any code, check whether any part of this issue has already been solved in a previous PR. This prevents re-introducing a bug that was already fixed or duplicating work.

```bash
# List recent merged PRs to spot any that touch the same bounded context or files
gh pr list --repo <repo> --state merged --limit 20 --json number,title,mergedAt,files \
  | python3 -c "
import sys, json
prs = json.load(sys.stdin)
for pr in prs:
    print(f\"#{pr['number']} {pr['title']}\")
    for f in pr.get('files', []):
        print(f\"  {f['path']}\")
"
```

For any merged PR that touches the same files or bounded context as this issue:

```bash
gh pr diff <merged-pr-number> --repo <repo>
```

Record any patterns, fixes, or decisions from merged PRs that must be preserved or built upon. Do **not** re-introduce code that was explicitly removed in a prior PR. If a prior PR solved the same problem differently than the implementation plan describes, follow the prior PR's approach and note the divergence.

### 2. Create the feature branch

Determine the base branch (default: `main`). Then:

```bash
git checkout main
git pull origin main
git checkout -b feature/<issue-number>-<kebab-case-title>
```

Derive the branch name from the issue title: lowercase, replace spaces/special chars with hyphens, strip leading/trailing hyphens. Example: "Add CSV adapter for MT5 format" → `feature/15-add-csv-adapter-for-mt5-format`.

### 2b — Move issue to "Developing" on the project board

```bash
ISSUE_NODE_ID=$(gh issue view <issue-number> --repo <repo> --json id -q .id)
ITEM_ID=$(gh api graphql \
  -f query='query($id:ID!){node(id:$id){...on Issue{projectItems(first:5){nodes{id project{id}}}}}}' \
  -f id="$ISSUE_NODE_ID" \
  --jq '.data.node.projectItems.nodes[]|select(.project.id=="PVT_kwHOCNlJes4BURsS")|.id' 2>/dev/null)

[ -n "$ITEM_ID" ] && gh api graphql \
  -f query='mutation($p:ID!,$i:ID!,$f:ID!,$v:String!){updateProjectV2ItemFieldValue(input:{projectId:$p,itemId:$i,fieldId:$f,value:{singleSelectOptionId:$v}}){projectV2Item{id}}}' \
  -f p="PVT_kwHOCNlJes4BURsS" -f i="$ITEM_ID" \
  -f f="PVTSSF_lAHOCNlJes4BURsSzhBa_lk" -f v="38c4234e"
```

### 3. Implement

Follow the implementation plan from the issue comment exactly. While coding, enforce these rules:

**Clean Architecture / DDD rules (backend):**
- Domain layer (`domain/`) must have zero imports from `infrastructure/`, `application/`, or any other BC's domain.
- Repository interfaces go in `<bc>/domain/repositories.py`; implementations go in `infrastructure/`.
- Use cases in `<bc>/application/` orchestrate domain objects — no business logic.
- No BC imports another BC's domain model. Use `shared_kernel/` primitives for cross-BC types.
- New value objects must enforce their invariants in `__init__` (raise `ValueError` on violation).
- Stubs (`raise NotImplementedError`) must be replaced — never leave them in implemented code.

**Code quality (backend):**
- Use `Decimal` for all price values, never `float`.
- Type-annotate all public functions and methods.
- Avoid `type: ignore` comments — fix the underlying type issue instead.
- No bare `except:` clauses — catch specific exception types.
- No magic numbers — use named constants.

### 4. Write tests

Create or extend test files in `tests/unit/<bounded_context>/`. Follow the existing test structure:
- One test class per logical concern (e.g., `TestHappyPath`, `TestInvalidInput`, `TestDomainInvariant`).
- Use `tmp_path` for any file I/O.
- Test every acceptance criterion listed in the implementation plan.
- Test every domain invariant the new code is supposed to enforce.

### 5. Run quality gates

From the working directory, run all three gates and fix any failures before committing:

```bash
.venv/bin/pytest tests/ -q
.venv/bin/ruff check .
.venv/bin/mypy <affected_package>
```

Do not proceed to commit if any gate fails. Fix the root cause — do not suppress errors with `# type: ignore` or `# noqa` unless the suppression is genuinely unavoidable and is explained with a comment.

### 6. Commit

Stage only the files you created or modified:

```bash
git add <file1> <file2> ...
git commit -m "$(cat <<'EOF'
<type>(<scope>): <short description> (#<issue-number>)

<Body: what was implemented and why. Reference the implementation plan.
One paragraph, present tense.>

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

Commit type: `feat` for new functionality, `fix` for bug fixes, `refactor` for restructuring, `test` for test-only changes.

### 7. Push and open a PR

```bash
git push -u origin feature/<issue-number>-<kebab-title>
```

```bash
gh pr create \
  --repo DougOscar/AlphaForge[Backend|Frontend] \
  --title "<type>(<scope>): <description>" \
  --body "$(cat <<'EOF'
Closes #<issue-number>

## Summary
- <bullet: what was added/changed>
- <bullet>

## Architecture compliance
- [ ] Domain layer has no infrastructure imports
- [ ] Repository interfaces in `domain/`; implementations in `infrastructure/`
- [ ] No cross-BC domain model imports
- [ ] Domain invariants enforced in value object constructors

## Test plan
- [ ] <test scenario 1>
- [ ] <test scenario 2>
- [ ] All existing tests pass
- [ ] ruff clean
- [ ] mypy clean

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

### 8. Confirm

Print:
> "PR opened: <pr-url> — closes #<issue-number>"
