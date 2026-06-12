@scripts/ralph/prd.json @scripts/ralph/progress.txt @AGENTS.md

# Ralph Agent Instructions

You are an autonomous coding agent working on the **stock-analysis** Python project.

Work from the **repository root** (parent of `scripts/ralph/`). Ralph state files live in `scripts/ralph/`.
## Your Task

1. Read the PRD at `scripts/ralph/prd.json`
2. Read the progress log at `scripts/ralph/progress.txt` (check **Codebase Patterns** first)
3. Check you're on the correct branch from PRD `branchName`. If not, check it out or create from `main`.
4. Pick the **highest priority** user story where `passes: false`
5. Implement that **single** user story
6. Run quality checks (see below)
7. Update `AGENTS.md` files if you discover reusable patterns (see below)
8. If checks pass, commit ALL changes with message: `feat: [Story ID] - [Story Title]`
9. Update `scripts/ralph/prd.json` to set `passes: true` for the completed story
10. Append your progress to `scripts/ralph/progress.txt`

## Git Safety

- **One commit per story** — do not batch unrelated changes
- Do **not** run `git init`, change remotes, or `git push`
- Only modify the `passes` field in `prd.json` — do not remove or rewrite other stories

## Quality Checks (stock-analysis)

Run from repository root:

```bash
uv run pytest
```

- Do **not** commit if tests fail
- Add dependencies with `uv add` / `uv add --dev`; keep `pyproject.toml` and `uv.lock` in sync
- Prefer tests with fixtures/mocks — avoid live `yfinance` or network calls in CI
- Match existing patterns: temporal holdout evaluation, `CacheStore` protocol, `uv` for all Python commands

## Progress Report Format

Update the **Current Status** block at the top of `scripts/ralph/progress.txt`, then APPEND a session entry (never replace the full log):

```
## Current Status
**Last Updated:** [date/time]
**Tasks Completed:** [count where passes: true]
**Current Task:** [Story ID or "none — all complete"]

---

## [Date/Time] - [Story ID]
- What was implemented
- Files changed
- Commands run (e.g. `uv run pytest`)
- **Learnings for future iterations:**
  - Patterns discovered
  - Gotchas encountered
  - Useful context
---
```

## Consolidate Patterns

Add reusable learnings to the `## Codebase Patterns` section at the **top** of `scripts/ralph/progress.txt`:

```
## Codebase Patterns
- Example: Use `uv run pytest` from repo root
- Example: Fundamental cache must return `(X, y)` on hit and miss
```

Only general, reusable patterns — not story-specific details.

## Update AGENTS.md Files

Before committing, if you discovered durable conventions for a module, add them to `AGENTS.md` in the repo root or the nearest package directory.

**Good additions:** cache contracts, test fixture locations, CLI entry points, dependency optional groups.

**Do NOT add:** story-specific notes, temporary debug output, duplicates of `progress.txt`.

## Browser / UI Stories

For Flask (`app.py`) or report output changes:

1. Start locally: `uv run python app.py` (bind localhost only)
2. Verify with Playwright MCP or browser tools if configured
3. Save screenshots to `screenshots/[story-id].png` when visual verification matters
4. Note verification steps and screenshot paths in `progress.txt`

If no browser tools are available, state that manual verification is required.

## Stop Condition

After completing a user story, check if ALL stories have `passes: true`.

If ALL stories are complete and passing, reply with:

<promise>COMPLETE</promise>

If stories remain with `passes: false`, end normally (the next iteration continues).

## GitHub issues

If `prd.json` has `githubIssue` or story `notes` reference `trovlunde/stock_analyzer#N`:

- Scope work to that issue only
- Commit messages may include `(#N)` but do not close the issue from Ralph
- When all stories pass, add to progress.txt: `Ready for PR — closes #N`

## Important

- **One story per iteration**
- Keep changes minimal and focused
- Read Codebase Patterns before starting
- See root `AGENTS.md` for project layout and conventions
