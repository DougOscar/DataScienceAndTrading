Perform a thorough code review of a GitHub pull request. Post inline comments for every finding and close any previously resolved review threads.

## Your inputs

Arguments: "$ARGUMENTS" — format: `<repo> <issue-number>`
- `repo` is either `backend` or `frontend`
- `issue-number` is the GitHub issue number

Resolve repo:
- `backend` → repo handle `DougOscar/AlphaForgeBackend`, working dir `/home/douglaso/Finances/AlphaForge/AlphaForgeBackend/`
- `frontend` → repo handle `DougOscar/AlphaForgeFrontend`, working dir `/home/douglaso/Finances/AlphaForge/AlphaForgeFrontend/`

## Review criteria

Evaluate every changed file against all six areas below. Do not skip any area.

### 1. Code efficiency
- Redundant computation inside loops (e.g., same expression evaluated multiple times per iteration)
- Unnecessary copies of collections (e.g., building a list just to compare it)
- Dead code or variables that are assigned but never read

### 2. Performance
- Use of `re.sub` or other regex for simple string operations where `str.replace` suffices
- Linear searches on data structures that should be sets or dicts
- Sort + compare patterns that can be replaced by a single O(n) pass
- Repeated I/O or system calls that could be batched or cached

### 3. Security and vulnerabilities
- Path traversal: file paths accepted from the caller without being validated against a safe root directory
- Injection risks: any user-controlled string interpolated into shell commands, SQL, or file paths
- `type: ignore` suppressions that hide real type errors (these can mask logic bugs)
- Broad `except Exception` or bare `except:` that swallow unexpected errors silently

### 4. Magic numbers and constants
- Bare numeric literals that encode a non-obvious business rule or configuration value
- Hard-coded string literals that appear in multiple places
- Any number or string whose purpose is not immediately self-evident from context

### 5. Naming conventions
- Variable, function, and class names that are too terse (`x`, `d`, `tmp`, `actual`)
- Names that don't match the domain vocabulary defined in `AlphaForgeDoc/`
- Inconsistencies between similar constructs (e.g., one method says `get_` and another says `fetch_`)
- Mutable module-level constants that should be `frozenset` or `tuple`

### 6. Architecture compliance (AlphaForge-specific)
- Domain layer importing from `infrastructure/` or `application/` — **hard violation**
- Business logic in a use case (application layer) — **hard violation**
- One bounded context importing another BC's domain model — **hard violation**
- `float` used for price values instead of `Decimal`
- Repository interfaces defined outside `domain/repositories.py`
- `raise NotImplementedError` left in code that should be implemented

## Step-by-step

### 1. Fetch the diff and existing threads

```bash
gh pr diff <pr-number>
```

Also fetch any existing review threads to check which ones are already resolved:
```bash
gh api graphql -f query='{ repository(owner:"DougOscar", name:"AlphaForgeBackend") { pullRequest(number: <pr-number>) { reviewThreads(first: 30) { nodes { id isResolved comments(first:1) { nodes { body } } } } } } }'
```

### 2. Resolve already-fixed threads

For each thread that is `isResolved: false` but whose finding has been addressed in the current diff, resolve it:
```bash
gh api graphql -f query='mutation { resolveReviewThread(input: { threadId: "<id>" }) { thread { isResolved } } }'
```

### 3. Review the diff

Read every changed file carefully. For each finding, record:
- File path and line number
- Which of the six criteria it violates
- Severity: 🔴 blocks merge / 🟠 should fix / 🟡 minor
- Concrete fix with a code snippet

### 4. Post the review

Open the review with a summary table:

```bash
gh api repos/DougOscar/AlphaForgeBackend/pulls/<pr-number>/reviews \
  --method POST \
  --field commit_id="<head-sha>" \
  --field event="COMMENT" \
  --field body="<summary>"
```

Post an inline comment for each finding:

```bash
gh api repos/DougOscar/AlphaForgeBackend/pulls/<pr-number>/comments \
  --method POST \
  --field commit_id="<head-sha>" \
  --field path="<file>" \
  --field position=<diff-position> \
  --field body="<finding>"
```

If a finding is on an unchanged line (not in the diff), include it in the summary review body with an explicit file:line reference instead.

### 4b — Move linked issue(s) to "Code Review" on the project board

For each issue that the PR closes (look for `Closes #N` / `Fixes #N` in the PR body), move it to "Code Review":

```bash
ISSUE_NODE_ID=$(gh issue view <N> --repo DougOscar/AlphaForgeBackend --json id -q .id)
ITEM_ID=$(gh api graphql \
  -f query='query($id:ID!){node(id:$id){...on Issue{projectItems(first:5){nodes{id project{id}}}}}}' \
  -f id="$ISSUE_NODE_ID" \
  --jq '.data.node.projectItems.nodes[]|select(.project.id=="PVT_kwHOCNlJes4BURsS")|.id' 2>/dev/null)

[ -n "$ITEM_ID" ] && gh api graphql \
  -f query='mutation($p:ID!,$i:ID!,$f:ID!,$v:String!){updateProjectV2ItemFieldValue(input:{projectId:$p,itemId:$i,fieldId:$f,value:{singleSelectOptionId:$v}}){projectV2Item{id}}}' \
  -f p="PVT_kwHOCNlJes4BURsS" -f i="$ITEM_ID" \
  -f f="PVTSSF_lAHOCNlJes4BURsSzhBa_lk" -f v="985559f8"
```

### 5. Summary format

The top-level review body must include:

```markdown
## Code Review

| # | Severity | Criteria | Location | Finding |
|---|----------|----------|----------|---------|
| 1 | 🔴 | Security | file.py:42 | Short description |
...

<total count> finding(s). <resolved count> previously-open thread(s) resolved.
```

If the PR is clean across all six criteria, post:
> "No findings. All six review criteria passed. PR looks good to merge."
