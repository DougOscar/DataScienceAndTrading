Create a detailed implementation plan for a GitHub issue, post it as a comment, and label the issue "Refined & Ready".

## Your inputs

Arguments: "$ARGUMENTS" — format: `<repo> <issue-number>`
- `repo` is either `backend` or `frontend`
- `issue-number` is the GitHub issue number (e.g. `15`)

Parse arguments:
- repo = first token of "$ARGUMENTS"
- issue_number = second token of "$ARGUMENTS"

Resolve repo:
- `backend` → repo handle `DougOscar/AlphaForgeBackend`, working dir `/home/douglaso/Finances/AlphaForge/AlphaForgeBackend/`
- `frontend` → repo handle `DougOscar/AlphaForgeFrontend`, working dir `/home/douglaso/Finances/AlphaForge/AlphaForgeFrontend/`

Documentation root: `/home/douglaso/Finances/AlphaForge/AlphaForgeDoc/`

## Step-by-step

### 1. Read the issue

```bash
gh issue view <issue-number> --repo DougOscar/AlphaForge[Backend|Frontend]
```

Note the title, description, and any existing comments.

### 2. Read relevant documentation

Identify which bounded context(s) and layer(s) the issue touches. Read the matching docs in `AlphaForgeDoc/`:
- Bounded context doc (e.g., `01-market-data/Market Data BC.md`)
- Entities, value objects, and domain events docs for that context
- `09-application-layer/Application Layer.md` if a use case is involved
- `09-application-layer/Infrastructure Layer.md` if an adapter is involved
- `00-overview/Dependency Rule.md` and `00-overview/Domain Invariants.md` always

### 3. Inspect the relevant code

Read the existing source files in the affected bounded context package(s). Understand:
- What already exists and can be reused
- What interfaces need implementing
- What domain invariants the new code must uphold
- What the test structure looks like for this BC

### 4. Write the implementation plan

Structure the plan as follows:

```markdown
## Implementation Plan

### Overview
<Two to three sentences: what this issue delivers and which layer(s) it touches.>

### Architecture notes
<Which bounded context(s), which layers (Domain / Application / Infrastructure). Call out any dependency-rule constraints or domain invariants that apply.>

### Files to create or modify

| File | Action | What to do |
|------|--------|-----------|
| `path/to/file.py` | Create / Modify | Description |

### Implementation steps

1. **Step title** — detail
2. **Step title** — detail
...

### Tests to write

| Test file | Test class | Scenarios to cover |
|-----------|-----------|-------------------|
| `tests/unit/...` | `TestXxx` | scenario1, scenario2 |

### Acceptance criteria
- [ ] Specific, testable criterion
- [ ] All existing tests still pass
- [ ] `.venv/bin/ruff check .` → clean
- [ ] `.venv/bin/mypy <affected_package>` → clean
```

### 5. Post the plan and label the issue

Post the plan as a comment:
```bash
gh issue comment <issue-number> \
  --repo DougOscar/AlphaForge[Backend|Frontend] \
  --body "<plan markdown>"
```

Add the "Refined & Ready" label:
```bash
gh issue edit <issue-number> \
  --repo DougOscar/AlphaForge[Backend|Frontend] \
  --add-label "Refined & Ready"
```

### 5b — Move issue to "Refining" on the project board

```bash
ISSUE_NODE_ID=$(gh issue view <issue-number> --repo <repo> --json id -q .id)
ITEM_ID=$(gh api graphql \
  -f query='query($id:ID!){node(id:$id){...on Issue{projectItems(first:5){nodes{id project{id}}}}}}' \
  -f id="$ISSUE_NODE_ID" \
  --jq '.data.node.projectItems.nodes[]|select(.project.id=="PVT_kwHOCNlJes4BURsS")|.id' 2>/dev/null)

[ -n "$ITEM_ID" ] && gh api graphql \
  -f query='mutation($p:ID!,$i:ID!,$f:ID!,$v:String!){updateProjectV2ItemFieldValue(input:{projectId:$p,itemId:$i,fieldId:$f,value:{singleSelectOptionId:$v}}){projectV2Item{id}}}' \
  -f p="PVT_kwHOCNlJes4BURsS" -f i="$ITEM_ID" \
  -f f="PVTSSF_lAHOCNlJes4BURsSzhBa_lk" -f v="28d72a04"
```

### 6. Confirm

Print a one-line confirmation:
> "Issue #N labeled 'Refined & Ready'. Implementation plan posted: <issue-url>"
