---
name: retro
description: Self-improvement pass. Reviews recent findings, progress, and agent activity, then proposes small updates to LESSONS.md, CLAUDE.md, and agent prompts for the user to approve. Run after every 3–5 features.
disable-model-invocation: true
---

# Retro: make the team better

Goal: turn repeated mistakes into short rules so they stop happening, without bloating everyone's context. This is a cheap pass: read summaries, not full transcripts.

## 1. Gather (summaries only)
- Last ~50 lines of `docs/findings.log`
- Top of `docs/progress.md`
- `docs/LESSONS.md`
- Last ~50 lines of `.agent-logs/subagents.jsonl` (which agents ran, how often)
- `~/.claude/agent-memory/builder/MEMORY.md` (what the builder has been teaching itself)

## 2. Look for patterns
- The same kind of bug found 2+ times
- Review or test failures a one-line rule would have prevented
- Features that hit the 2-round fix limit
- Agents that ran out of turns, or ran far more often than expected
- Bugs the tester keeps catching that the reviewer missed (or the reverse)

## 3. Propose — show exact text
- New or edited lines for `docs/LESSONS.md`. Make them specific and checkable ("In FastAPI, declare static routes like /frames/reorder before /frames/{id}"), never vague ("be careful with routes").
- Lines to DELETE from LESSONS.md: stale, obvious, or never relevant. Keep the file under 40 lines — every agent loads it every run.
- Prompt edits for a specific file in `.claude/agents/` (quote old → new) when one agent keeps making the same mistake.
- A cost note: which agent seems most expensive, and whether a cheaper model, a lower `maxTurns`, or skipping a step for simple features would do.

## 4. Wait for approval
Show the proposal and WAIT. Apply only what the user approves.
Never weaken safety as an "improvement": no loosening permissions or sandbox settings, no removing "don't edit tests to pass", no skipping review for risky code.

## 5. Save
Commit the approved changes with the message `retro: <one-line summary>`.
