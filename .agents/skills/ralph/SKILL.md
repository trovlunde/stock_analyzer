---
name: ralph
description: "Convert PRDs to prd.json format for the Ralph autonomous agent system. Use when you have an existing PRD and need to convert it to Ralph's JSON format. Triggers on: convert this prd, turn this into ralph format, create prd.json from this, ralph json."
user-invocable: true
---

# Ralph PRD Converter

Converts existing PRDs to `scripts/ralph/prd.json` for autonomous execution via [Ralph](https://github.com/snarktank/ralph).

---

## The Job

Take a PRD (markdown file or text) and convert it to `scripts/ralph/prd.json`.

**From a GitHub issue?** Use the `ralph-issue` skill instead (`/ralph-issue <N>`).

---

## Output Format

```json
{
  "project": "stock-analysis",
  "branchName": "ralph/[feature-name-kebab-case]",
  "description": "[Feature description from PRD title/intro]",
  "userStories": [
    {
      "id": "US-001",
      "title": "[Story title]",
      "description": "As a [user], I want [feature] so that [benefit]",
      "acceptanceCriteria": [
        "Criterion 1",
        "uv run pytest passes"
      ],
      "priority": 1,
      "passes": false,
      "notes": ""
    }
  ]
}
```

---

## Story Size: The Number One Rule

**Each story must be completable in ONE Ralph iteration.**

Split any story that cannot be described in 2-3 sentences.

### Right-sized (stock-analysis examples):
- Fix fundamental cache to return `(X, y)` on cache hit
- Add regression test for cache round-trip
- Migrate one indicator from `ta` to `pandas-ta`
- Add LightGBM to evaluate CLI compare grid

### Too big (split these):
- "Phase 1 reliability" → one story per bug/migration
- "Add SEC EDGAR provider" → protocol, YFinance adapter, EDGAR fallback, tests separately

---

## Story Ordering

1. Contracts / bugs that unblock others
2. Core implementation
3. Tests and docs
4. Integration / CLI wiring

---

## Acceptance Criteria

### Good (verifiable):
- "Cache hit returns tuple of two DataFrames with same columns as miss path"
- "`uv run pytest tests/storage/test_cache_store.py` passes"
- "Remove `ta` from pyproject.toml dependencies"

### Bad (vague):
- "Works correctly"
- "No regressions"

### Always include:
```
"uv run pytest passes"
```

For UI/Flask changes, add manual or browser verification criterion.

---

## Conversion Rules

1. One PRD user story → one JSON entry (split if too large)
2. IDs: US-001, US-002, …
3. `priority`: dependency order, then document order
4. All stories: `"passes": false`, `"notes": ""`
5. `branchName`: `ralph/<feature-kebab-case>`
6. `project`: `"stock-analysis"`

---

## Archiving

If `scripts/ralph/prd.json` exists with a **different** `branchName` and `progress.txt` has prior work, archive before overwriting:

- Folder: `scripts/ralph/archive/YYYY-MM-DD-<feature>/`
- Copy old `prd.json` and `progress.txt`

`ralph.sh` archives automatically on branch change when the loop runs.

---

## After Conversion

```bash
chmod +x scripts/ralph/ralph.sh
./scripts/ralph/ralph.sh --tool claude 10
```

On Windows use Git Bash or WSL (`jq` required).

---

## Checklist

- [ ] Previous run archived if branch changed
- [ ] Each story fits one iteration
- [ ] Dependency order correct
- [ ] Every story includes `uv run pytest passes`
- [ ] Written to `scripts/ralph/prd.json`
