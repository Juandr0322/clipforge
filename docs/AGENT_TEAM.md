# Agent Team Starter Kit (Claude Code)

A small, safe, token-aware team of AI agents for building an app with Claude Code.
Drop these files into a project, edit two things, and run three commands.

```
You
 └─ Lead (your main Claude Code session, Sonnet) ── coordinates, holds the plan, merges
     ├─ planner   (Opus)    idea → docs/SPEC.md + docs/features.json      runs once
     ├─ Explore   (Haiku)   cheap read-only lookups (code + docs)           any time
     ├─ builder ×1–3 (Sonnet, own git worktree)   one feature each           parallel only if files don't overlap
     ├─ reviewer  (Opus, fresh context, read-only)  reviews the diff        before merge
     └─ tester    (Sonnet + Playwright, read-only)  uses the running app    after merge
 /retro (you approve) ── turns repeated mistakes into rules in docs/LESSONS.md
```

## What's in the box

| File | What it does |
|---|---|
| `CLAUDE.md` | Short project rules every agent loads. Pulls in `docs/LESSONS.md`. |
| `.claude/agents/*.md` | The five agents: role, tools, model, turn limit. |
| `.claude/skills/build-feature/` | `/build-feature` — the build → review → merge → test loop. |
| `.claude/skills/retro/` | `/retro` — the self-improvement pass (you approve every change). |
| `.claude/settings.json` | Sandbox, permission rules, secret protection, concurrency limits, hooks. |
| `.claude/hooks/filter-test-output.sh` | Shows agents only test failures; full log goes to a file. Big token saver. |
| `.claude/hooks/log-subagent.sh` | One line per finished agent, so /retro can see who runs most. |
| `docs/features.json` | Feature list with acceptance criteria and pass/fail status. |
| `docs/progress.md` | Running log so any fresh agent can catch up in seconds. |
| `docs/LESSONS.md` | Learned rules (max 40 lines). This is the "self-improving" memory. |
| `docs/findings.log` | Every review/test finding, one line each. Input for /retro. |

## Setup (about 15 minutes)

**1. Prerequisites**
- Latest Claude Code (`claude update`), git, Node.js.
- macOS, Linux, or Windows **via WSL2** (the sandbox doesn't run on native Windows).
- Linux/WSL2 only: `sudo apt-get install bubblewrap socat`
- For the tester's browser: `npx playwright install chromium`

**2. Copy the kit into your project root** and commit it:
```bash
git init            # if it isn't a repo yet
git branch -M main  # the build loop assumes your default branch is "main"
git add -A && git commit -m "Add agent team kit"
```

**3. Edit two things**
- `CLAUDE.md`: app name, and the real install/dev/test/lint commands.
- `.claude/settings.json` → `sandbox.network.allowedDomains`: add any other package registries or APIs your build needs. Keep the list short. Broad entries like `github.com` give a compromised agent a way to send data out.

**4. Start Claude Code and lock it down**
```bash
claude
```
- Accept the "trust this folder" prompt (hooks and the tester's Playwright server need it).
- Run `/model sonnet` for the lead. (Opus is only used where it pays off: planner and reviewer.)
- Run `/sandbox` → choose **auto-allow**. Then ask Claude to run `touch ~/sandbox-probe`. It should FAIL. If it succeeds, the sandbox is off; re-run `/sandbox` and check the Dependencies tab.
- In a second terminal, run `claude agents` to confirm all five agents loaded.

**5. Plan (the cheapest place to fix mistakes)**
```
Use the planner agent to plan: <your 1–4 sentence app idea>
```
Read `docs/SPEC.md` and `docs/features.json` yourself. Fix anything wrong now — a bad plan multiplies into every later step. Then commit.

**6. Build**
```
/build-feature next          # one feature at a time (start here)
/build-feature F04           # a specific feature
/build-feature parallel 3    # up to 3 features at once, only if their files don't overlap
```

**7. Improve**
- After every 3–5 features: `/retro`. It proposes new rules and prompt tweaks; you approve them.
- Once a week: `/insights` for a report on where sessions go wrong.

**8. Watch spend**
- `/usage` (session cost and what used it), `/context` (what's filling the window).
- `/clear` between unrelated tasks. A long session re-sends its whole history every message.
- If your app calls the Claude API, give it its own API key in a Console workspace with a low spend limit.

## Tuning knobs

| If… | Do this |
|---|---|
| You hit plan limits | Set `model: sonnet` in `planner.md` and `reviewer.md`. |
| Simple features feel over-processed | Skip the tester for tiny UI tweaks; keep the reviewer. |
| Explore answers are too shallow | Delete `.claude/agents/explore.md` to go back to the built-in Explore. |
| A tool won't work in the sandbox (e.g. Docker) | Add the narrowest possible pattern to `sandbox.excludedCommands`. Those commands then run with full access. |
| Builders keep colliding | Ask the planner to re-split `area` so features touch different folders, or stay sequential. |

## Running unattended (later, carefully)

Only once the loop works well while you watch. Run it inside a container or VM (the official Claude Code dev container, which has a default-deny firewall, is the easy path), never on your main machine with permissions skipped. Cap the cost:
```bash
claude -p "/build-feature next" --max-budget-usd 5
```

## Why it's built this way (one line each)
- **One lead, flat tree.** Subagents can't spawn more subagents here (`CLAUDE_CODE_MAX_SUBAGENT_SPAWN_DEPTH=1`), so nothing fans out behind your back.
- **Parallel reading, careful writing.** Parallel writers make clashing decisions; builders only run in parallel on non-overlapping files, each in its own worktree.
- **Fresh-context reviewer.** A reviewer that didn't write the code, and didn't see the builder's long context, catches more.
- **Tester uses the real app.** Code that "looks done" often isn't; clicking through it finds stubs and broken wiring.
- **Cheap models for cheap jobs.** Haiku scouts, Sonnet builds, Opus plans and judges.
- **Files as memory.** progress.md + features.json + git let any fresh agent catch up without re-reading the whole repo.
- **Self-improvement with a human gate.** Agents that grade and edit themselves unsupervised can learn to game their own checks.
