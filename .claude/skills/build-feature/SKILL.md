---
name: build-feature
description: Run the build loop for one or more features (builder, then reviewer, then tester) using this project's subagents. Use when asked to build the next feature, a specific feature id, or to keep building.
---

# Build loop

You are the lead. You coordinate; subagents do the heavy reading and writing so your own context stays small.

Target: `$ARGUMENTS` — a feature id like `F03`, `next`, or `parallel 3` (up to 3 independent features at once). Default: `next`.

## 0. Orient (keep this cheap)
- Read the top 30 lines of `docs/progress.md` and `docs/features.json`.
- If `docs/SPEC.md` doesn't exist, stop and run the planner first ("Use the planner agent to plan: <idea>"). Ask the user for the idea if you don't have it.
- Make sure the working tree is clean and docs are committed. Builders work in worktrees that branch from the default branch, so anything uncommitted is invisible to them.

## 1. Pick work
- Candidates: `passes: false`, every `depends_on` already passing, lowest `priority` first.
- Build in parallel only if the user asked for it AND the candidates' `area` lists don't overlap. Never more than 3 builders at once.
- When unsure, go one at a time. Parallel writers touching the same files cost more to untangle than they save.

## 2. Build
- Spawn one `builder` per feature (in parallel when allowed). Give each: the feature id, its acceptance criteria, its area, and "commit on your branch and report the branch name."
- Don't paste file contents into prompts; builders read what they need.

## 3. Review, before merging
- For each finished feature, spawn a `reviewer` with the feature id and branch name. It gets a fresh context on purpose: don't summarize the builder's reasoning for it.
- VERDICT: FAIL → resume the SAME builder (send it a message, so it keeps its context and cache) with the findings. Max 2 fix rounds per feature.

## 4. Merge, then test
- `git merge --no-ff <branch>` into main. On a conflict, resume the builder to rebase and resolve.
- Spawn the `tester` with the feature id. VERDICT: FAIL → resume the builder with the report, then review and test again. This counts toward the 2-round limit.

## 5. Close out
- Pass: set `passes: true` in `docs/features.json`, add a 1–2 line entry at the top of `docs/progress.md`, commit.
- Still failing after 2 rounds: stop. Write what's blocking in the feature's `notes`, revert the merge if main is broken (`git revert -m 1 <merge-commit>`), and tell the user. Don't loop forever.
- For every review or test finding, append one line to `docs/findings.log`: `date | feature | reviewer/tester | severity | short description`. /retro learns from this file.

## Report to the user
10 lines or fewer: what passed, what failed and why, what's next. Suggest `/retro` after every 3–5 finished features.
