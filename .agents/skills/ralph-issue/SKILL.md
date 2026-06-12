---
name: ralph-issue
description: "Convert a GitHub issue (with Agent Brief) into scripts/ralph/prd.json for the Ralph loop. Use when preparing an issue for autonomous execution. Triggers on: ralph-issue, ralph issue, convert issue to prd.json, prepare issue for ralph, /ralph-issue."
user-invocable: true
---

# Ralph Issue → prd.json

Bridge **GitHub issues** to **Ralph execution** for stock-analysis.

**Repo:** `trovlunde/stock_analyzer`  
**Issue index:** `docs/issues/README.md`  
**PRD context:** `docs/prd-stack-tooling-improvements.md`

---

## The Job

Given a GitHub issue number (e.g. `2`):

1. Fetch issue title, body, labels, and comments
2. Find the **Agent Brief** comment (authoritative spec) — see triage `AGENT-BRIEF.md` format
3. Cross-read the matching PRD section and `docs/stack-improvement-plan.md` if linked
4. Split into **3–6 Ralph-sized stories**
5. Write `scripts/ralph/prd.json`
6. Show a summary for human approval **before** suggesting `ralph.sh`

**Do NOT run `ralph.sh` or implement code** — only produce `prd.json`.

---

## Fetching the issue

Try in order:

```bash
gh issue view <N> --repo trovlunde/stock_analyzer --json title,body,labels,comments,state
```

If `gh` is unavailable, use GitHub MCP or fetch:
`https://github.com/trovlunde/stock_analyzer/issues/<N>`

Fall back to `docs/issues/README.md` + PRD sections for context, but **warn** if no Agent Brief is on the issue.

---

## Eligibility

| OK for Ralph | Not for Ralph (stop and advise) |
|--------------|----------------------------------|
| `ready-for-agent` label or clear Agent Brief | `needs-triage`, `needs-info`, `wontfix` |
| Phase 1–2 implementation issues (#2–#9) | Spikes / ADRs (#10–#13) |
| Bug fixes with pytest-verifiable criteria | Design-only choices without a decided option (#4 NN strategy) |

If the issue lacks an Agent Brief, **stop** and tell the user to run `/triage` first.

---

## Output: `scripts/ralph/prd.json`

```json
{
  "project": "stock-analysis",
  "branchName": "ralph/issue-<N>-<short-kebab-name>",
  "description": "[Issue title] — GitHub trovlunde/stock_analyzer#<N>",
  "githubIssue": "https://github.com/trovlunde/stock_analyzer/issues/<N>",
  "userStories": [
    {
      "id": "US-001",
      "title": "[Small story title]",
      "description": "As a [user], I want [feature] so that [benefit]",
      "acceptanceCriteria": [
        "Specific verifiable criterion from Agent Brief",
        "uv run pytest passes"
      ],
      "priority": 1,
      "passes": false,
      "notes": "GitHub: trovlunde/stock_analyzer#<N>"
    }
  ]
}
```

`githubIssue` is optional but recommended for traceability.

---

## Story splitting rules

Same as the `ralph` skill:

- **One story = one Ralph iteration** (2–3 sentences max)
- Order: audit/discover → fix → test → docs/integration
- Every story: `"uv run pytest passes"` as final criterion
- Tests use mocks/fixtures — no live `yfinance` in criteria
- Flask/UI stories: add browser or manual verification criterion

### Example split for issue #2 (cache contract)

| ID | Title |
|----|-------|
| US-001 | Audit `prepare_classification_data_cache` return paths (fundamental classifier) |
| US-002 | Normalize cache hit and miss to return `(X, y)` only |
| US-003 | Add regression test: cache miss → write → hit → read |
| US-004 | Verify batch training caller unpacks two values (mocked ticker) |

---

## Archive before overwrite

If `scripts/ralph/prd.json` exists with a **different** `branchName`:

1. Copy `prd.json` + `progress.txt` → `scripts/ralph/archive/YYYY-MM-DD-<old-branch-suffix>/`
2. Reset `progress.txt` header (keep **Codebase Patterns** section if present)

---

## After writing prd.json

Print for the user:

1. Story count and titles
2. Branch name: `ralph/issue-<N>-...`
3. Suggested max iterations: `stories + 2` (e.g. 4 stories → 6 iterations)
4. Remind: review `prd.json`, then run commands from `AGENTS.md` § GitHub issue → Ralph

---

## Checklist

- [ ] Issue fetched; Agent Brief found (or user warned)
- [ ] Issue eligible for Ralph
- [ ] 3–6 stories, dependency-ordered
- [ ] Each story has `uv run pytest passes`
- [ ] `notes` links to GitHub issue on every story
- [ ] Previous `prd.json` archived if branch changed
- [ ] Written to `scripts/ralph/prd.json`
- [ ] Summary shown; user asked to approve before `ralph.sh`
