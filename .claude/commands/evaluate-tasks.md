Evaluate all open issues in an AlphaForge repository: assign priority scores, complexity estimates, and flag inter-issue dependencies. Update titles, labels, and post relationship comments.

## Your inputs

Arguments: "$ARGUMENTS" — must be either `backend` or `frontend`

Resolve repo:
- `backend` → `DougOscar/AlphaForgeBackend`, working dir `/home/douglaso/Finances/AlphaForge/AlphaForgeBackend/`
- `frontend` → `DougOscar/AlphaForgeFrontend`, working dir `/home/douglaso/Finances/AlphaForge/AlphaForgeFrontend/`

Documentation root: `/home/douglaso/Finances/AlphaForge/AlphaForgeDoc/`

---

## Step 1 — Ensure labels exist

Create any missing priority and complexity labels. Use `--force` to skip errors if the label already exists:

```bash
gh label create "p0" --repo <repo> --color "b60205" --description "Urgent — fix immediately" --force
gh label create "p1" --repo <repo> --color "d93f0b" --description "High priority" --force
gh label create "p2" --repo <repo> --color "fbca04" --description "Medium priority" --force
gh label create "p3" --repo <repo> --color "0e8a16" --description "Low priority" --force
gh label create "Easy"      --repo <repo> --color "c2e0c6" --description "< 1 day" --force
gh label create "Medium"    --repo <repo> --color "fef2c0" --description "1–3 days" --force
gh label create "Hard"      --repo <repo> --color "f9d0c4" --description "3–7 days" --force
gh label create "Very Hard" --repo <repo> --color "e11d48" --description "1+ weeks" --force
```

---

## Step 2 — Fetch all open issues

```bash
gh issue list --repo <repo> --state open --limit 100 --json number,title,body,labels,comments
```

Strip any existing `[pN]` or complexity tokens from each title before analysis so re-running the skill is idempotent.

---

## Step 3 — Read the documentation

Read these files to build an accurate picture of the platform's architectural dependencies:

- `AlphaForgeDoc/00-overview/Bounded Context Map.md` — data flow between BCs
- `AlphaForgeDoc/00-overview/Happy Path Flows.md` — end-to-end use-case sequences
- `AlphaForgeDoc/00-overview/Dependency Rule.md` — what must exist before what
- `AlphaForgeDoc/09-application-layer/Application Layer.md` — use case list
- `AlphaForgeDoc/09-application-layer/Infrastructure Layer.md` — adapter list

---

## Step 4 — Evaluate every issue

For each issue apply the priority rubric and complexity rubric below. Record your reasoning so it can be displayed in the preview.

### Priority rubric

| Score | Label | When to assign |
|-------|-------|----------------|
| p0 | Urgent | Active bug causing wrong results, data corruption, or security vulnerability in production-relevant code. |
| p1 | High | Foundation that multiple other issues depend on; without it a significant workflow cannot be tested end-to-end. Shared Kernel primitives, core domain models, primary infrastructure adapters, test infrastructure, and CI belong here. |
| p2 | Medium | Important feature but not a hard prerequisite for anything else currently open. Most use cases, adapters, and domain services that are self-contained. |
| p3 | Low | Optional, quality-of-life, or late-pipeline feature. DevOps extras, secondary analytics, optional robustness tooling, the domain event bus. |

**Tie-breaking rule:** if two issues are equally urgent, the one that unblocks more other issues wins the higher priority.

### Complexity rubric

| Label | Effort | Signals |
|-------|--------|---------|
| Easy | < 1 day | Single class or function, fully specified by docs, no cross-BC interaction, straightforward test cases. |
| Medium | 1–3 days | A handful of classes across one or two files, touches one layer, moderate test surface. |
| Hard | 3–7 days | Spans multiple files or layers, non-trivial domain logic, significant test matrix, or requires careful design decisions. |
| Very Hard | 1+ weeks | Cross-cutting concern, multiple BCs coordinated, complex algorithms (e.g. search heuristics, Monte Carlo), or deep architectural decisions. |

### Dependency detection

Two issues are **related** when:
- One issue's implementation is a direct prerequisite for the other (blocking dependency).
- They operate on the same aggregate, value object, or domain service.
- A doc explicitly links their bounded contexts (see Bounded Context Map).

Label the relationship type:
- `blocks` — issue A must be completed before issue B can begin.
- `related` — same domain area, can be developed in parallel but should be reviewed together.

---

## Step 5 — Present the evaluation for approval

Print a table of all proposed changes before touching GitHub. Format:

```
EVALUATION PREVIEW
==================

#  | Current title (stripped)                          | Priority | Complexity | Relationships
---|---------------------------------------------------|----------|------------|-----------------------------------
2  | Implement QualityDiagnostics domain service       | p1       | Medium     | blocks #3
3  | Implement IngestMarketDataUC and DiagnoseQualityUC| p1       | Medium     | blocked by #2, blocks #7
...
```

After the table, print a one-paragraph summary of the overall dependency graph and the critical path (the longest chain of blocking dependencies).

Then ask:
> "Apply these changes to all N issues? (yes / no / edit)"
> - `yes` — proceed with all changes as shown
> - `no` — abort, no changes made
> - `edit` — pause so the user can specify overrides, then re-present the table

Wait for the user's response before continuing.

---

## Step 6 — Apply changes

For each issue, execute the following in order:

### 6a — Update the title

Strip any existing `[pN]` and complexity tokens, then prepend the new ones:

```bash
gh issue edit <number> \
  --repo <repo> \
  --title "[<priority>] [<complexity>] <clean-title>"
```

### 6b — Add labels

```bash
gh issue edit <number> \
  --repo <repo> \
  --add-label "<priority>,<complexity>"
```

### 6c — Post a dependency comment (only for issues that have relationships)

```bash
gh issue comment <number> \
  --repo <repo> \
  --body "$(cat <<'EOF'
## Issue Relationships

| Type | Issue | Title |
|------|-------|-------|
| blocks   | #N | Title |
| blocked by | #M | Title |
| related  | #K | Title |

> Posted by `/evaluate-tasks`. Re-run the skill to refresh.
EOF
)"
```

Only post this comment if the issue has at least one relationship. Do not post an empty relationships section.

### 6d — Sync Priority and Complexity to the project board

This is the most important step of step 6 — labels alone are not sufficient. Update the **Priority** and **Complexity** single-select fields on the Alpha Forge project board for every issue. Re-running must be idempotent.

```bash
# Resolve the issue's project item ID
ISSUE_NODE_ID=$(gh issue view <number> --repo <repo> --json id -q .id)
ITEM_ID=$(gh api graphql \
  -f query='query($id:ID!){node(id:$id){...on Issue{projectItems(first:5){nodes{id project{id}}}}}}' \
  -f id="$ISSUE_NODE_ID" \
  --jq '.data.node.projectItems.nodes[]|select(.project.id=="PVT_kwHOCNlJes4BURsS")|.id' 2>/dev/null)

# If the issue is not yet in the project, add it first
[ -z "$ITEM_ID" ] && ITEM_ID=$(gh api graphql \
  -f query='mutation($p:ID!,$c:ID!){addProjectV2ItemById(input:{projectId:$p,contentId:$c}){item{id}}}' \
  -f p="PVT_kwHOCNlJes4BURsS" -f c="$ISSUE_NODE_ID" \
  --jq '.data.addProjectV2ItemById.item.id')

# Helper — update one project field
update_field() {
  gh api graphql \
    -f query='mutation($p:ID!,$i:ID!,$f:ID!,$v:String!){updateProjectV2ItemFieldValue(input:{projectId:$p,itemId:$i,fieldId:$f,value:{singleSelectOptionId:$v}}){projectV2Item{id}}}' \
    -f p="PVT_kwHOCNlJes4BURsS" -f i="$ITEM_ID" -f f="$1" -f v="$2" > /dev/null
}
```

**Priority field** — field ID `PVTSSF_lAHOCNlJes4BURsSzhBbACA`:

| Label | Option ID |
|-------|-----------|
| `p0`  | `eea3704f` |
| `p1`  | `57edfac3` |
| `p2`  | `09a3cb11` |
| `p3`  | `675b0ab9` |

```bash
update_field "PVTSSF_lAHOCNlJes4BURsSzhBbACA" "<priority-option-id>"
```

**Complexity field** — field ID `PVTSSF_lAHOCNlJes4BURsSzhBbAGc`:

| Label       | Option ID  |
|-------------|------------|
| `Easy`      | `74c185e5` |
| `Medium`    | `179365a1` |
| `Hard`      | `33befe8c` |
| `Very Hard` | `fd9db862` |

```bash
update_field "PVTSSF_lAHOCNlJes4BURsSzhBbAGc" "<complexity-option-id>"
```

---

## Step 7 — Print the final summary

After all updates are applied, print:

```
EVALUATION COMPLETE
===================
Issues updated : N
Labels applied : N × (1 priority + 1 complexity)
Dependency comments posted: N

Critical path (longest blocking chain):
  #A [p1] → #B [p1] → #C [p2] → ... → #Z [p3]

Issues with no dependencies (can start immediately):
  #X, #Y, #Z
```
