# Agent instructions (stock-analysis)

Guidance for human developers and autonomous agents (Ralph, Cursor, Claude Code).

## Project layout

| Path | Purpose |
|------|---------|
| `stock_analysis/` | Main Python package |
| `tests/` | pytest suite (`uv run pytest`) |
| `docs/` | PRDs, improvement plans, ADRs |
| `scripts/ralph/` | [Ralph](https://github.com/snarktank/ralph) autonomous loop (prd.json, progress.txt) |
| `tasks/` | Markdown PRDs before Ralph JSON conversion |

## Commands

```bash
uv sync
uv run pytest
uv run stock-analysis-evaluate compare --ticker ^GSPC --holdout-months 12
uv run python -m stock_analysis.jobs.daily --dry-run
```

## Conventions

- Python ≥3.12; dependencies via **uv** only
- Temporal holdout for classifier evaluation (train past, test recent)
- SQLite `CacheStore` with namespaced keys for cached fundamentals/market data
- Paper trading: daily job, `SignalStrategy` registry — do not break `DecisionResult` / `FillResult` contracts

## Ralph workflow

Based on [snarktank/ralph](https://github.com/snarktank/ralph) (bash loop, fresh context per iteration). See also [JeredBlu's setup guide](https://github.com/JeredBlu/guides/blob/main/Ralph_Wiggum_Guide.md) for sandbox and safety practices.

### Path A — GitHub issue (recommended for stack tooling)

One issue = one Ralph run. Issue index: `docs/issues/README.md`. Recommended order: #2 → #3 → #5 → #4.

| Step | Skill / action |
|------|----------------|
| 1. Triage | `/triage` → `ready-for-agent` + Agent Brief on issue |
| 2. Compile | `/ralph-issue 2` → `scripts/ralph/prd.json` |
| 3. Review | Human approves story list |
| 4. Execute | `./scripts/ralph/ralph.sh --tool claude <N>` |
| 5. Land | PR closes issue |

### Path B — New feature (no GitHub issue yet)

1. `prd` skill → `tasks/prd-<feature>.md`
2. `ralph` skill → `scripts/ralph/prd.json`
3. Run the loop (below)

### Run the loop

Git Bash or WSL on Windows; requires `jq` and Claude Code:

```bash
chmod +x scripts/ralph/ralph.sh
./scripts/ralph/ralph.sh --tool claude 10   # always set max iterations
```

Memory between iterations: git commits, `scripts/ralph/progress.txt`, `scripts/ralph/prd.json`.

Each story must fit one context window. Split large PRDs/issues before converting.

### GitHub issue → Ralph — command cheat sheet

Prerequisites (once):

```bash
uv sync
# Git Bash or WSL:
#   brew install jq          # macOS
#   apt install jq             # Linux
#   winget install jqlang.jq   # Windows (optional, for ralph.sh)
#   winget install GitHub.cli  # optional, for gh commands
gh auth login                  # if using gh
```

Per issue (example: #2):

```bash
# 1. In Cursor / Claude Code (not terminal):
#    /triage → move issue #2 to ready-for-agent (post Agent Brief)
#    /ralph-issue 2

# 2. Review generated task list
cat scripts/ralph/prd.json | jq '.userStories[] | {id, title, priority, passes}'

# 3. Run Ralph (Git Bash/WSL, from repo root)
./scripts/ralph/ralph.sh --tool claude 8

# 4. Monitor
cat scripts/ralph/progress.txt
git log --oneline -10
uv run pytest

# 5. After <promise>COMPLETE</promise> — push branch and open PR
git push -u origin HEAD
gh pr create --title "fix: fundamental cache return contract (closes #2)" --body "$(cat <<'EOF'
## Summary
- Fixes inconsistent prepare_classification_data_cache return shape
- Adds regression tests

## Test plan
- [x] uv run pytest

Closes #2
EOF
)"
```

**Windows (Git Bash):** Claude sandbox is not supported — `ralph.sh` auto-uses `--dangerously-skip-permissions`. Prefer **WSL2** for sandbox support. If still stuck, check for a Claude permission prompt in another terminal window, or read `scripts/ralph/.iteration-1.log` while the loop runs.

### Safety and cost (recommended)

- **Sandbox:** `.claude/settings.json` enables Claude Code sandbox so `ralph.sh` does not need `--dangerously-skip-permissions`
- **Max iterations:** always pass a limit (10–20 for testing)
- **No push:** Ralph commits locally; `git push` requires explicit approval in sandbox settings
- **Monitor:** `scripts/ralph/progress.txt`, git log, and `uv run pytest` between runs

### Optional: Playwright MCP for Flask UI stories

Only needed when Ralph stories touch `app.py` or rendered templates. Add `.mcp.json` with `@playwright/mcp` and a `screenshots/` folder — not required for backend/ML work.

### File mapping (vs JeredBlu guide)

| JeredBlu guide | This repo |
|----------------|-----------|
| `plan.md` + embedded JSON | `scripts/ralph/prd.json` (snarktank format) |
| `activity.md` | `scripts/ralph/progress.txt` |
| `PROMPT.md` | `scripts/ralph/CLAUDE.md` |
| Claude plugin `/ralph` | Not used — bash loop gives true fresh context per iteration |
