#!/usr/bin/env bash
# SubagentStop hook: append one short line per finished subagent to .agent-logs/subagents.jsonl.
# /retro reads this to see which agents run most often. Each line is capped at ~400 chars.
dir="${CLAUDE_PROJECT_DIR:-.}/.agent-logs"
mkdir -p "$dir"
jq -c --arg t "$(date -u +%FT%TZ)" \
  '{t: $t, agent: (.agent_type // "unknown"), raw: (tostring | .[0:400])}' \
  >> "$dir/subagents.jsonl" 2>/dev/null
exit 0
