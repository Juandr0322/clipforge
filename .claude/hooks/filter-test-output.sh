#!/usr/bin/env bash
# PreToolUse hook (Bash): run plain test commands with the full log saved to a file,
# and show the agent only the failures plus a short tail.
# A 2,000-line test log becomes ~30 lines in the agent's context.
#
# Only rewrites simple commands like `npm test`, `pnpm run test -- auth`, `pytest tests/`.
# Anything with pipes, chaining, redirects, or substitutions is left untouched,
# so this hook can never auto-approve something like `npm test && rm -rf ...`.

input=$(cat)
cmd=$(printf '%s' "$input" | jq -r '.tool_input.command // empty')

# Leave multi-line commands alone.
if [[ "$cmd" == *$'\n'* ]]; then
  echo '{}'
  exit 0
fi

re_pkg='^(npm|pnpm|yarn) (run )?test( [^;&|<>`$(){}\\]*)?$'
re_other='^(pytest|python -m pytest|python3 -m pytest|go test|npx vitest|npx jest)( [^;&|<>`$(){}\\]*)?$'

if [[ "$cmd" =~ $re_pkg || "$cmd" =~ $re_other ]]; then
  log='.agent-logs/last-test.log'
  wrapped="mkdir -p .agent-logs && ($cmd) > $log 2>&1; s=\$?; grep -E -i -A 4 '(fail|error)' $log | head -60; echo '--- last lines ---'; tail -n 8 $log; echo \"[full log: $log | exit code \$s]\"; exit \$s"
  printf '%s' "$input" | jq --arg c "$wrapped" \
    '{hookSpecificOutput: {hookEventName: "PreToolUse", permissionDecision: "allow", updatedInput: (.tool_input + {command: $c})}}'
else
  echo '{}'
fi
