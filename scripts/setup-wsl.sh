#!/usr/bin/env bash
# One-command setup for running ClipForge and its agent team on Windows (inside WSL2 Ubuntu).
#
# Run it from inside the cloned repo, in the Ubuntu window (NOT PowerShell):
#     bash scripts/setup-wsl.sh
#
# What it does (safe to run again; finished steps are skipped):
#   1. Installs the Linux tools (git, ffmpeg, Python, jq, gh, and bubblewrap + socat for the sandbox)
#   2. Lets the sandbox work on Ubuntu 24.04+ (AppArmor rule for bubblewrap only)
#   3. Installs Claude Code
#   4. Creates the Python environment and installs ClipForge + the test browser
#   5. Asks for your Anthropic API key (hidden while you type) and saves it to .env, readable only by you
#   6. Checks that everything works, including the sandbox
#
# It never prints your key, never touches your Windows drive, and never pushes anything.

set -euo pipefail

say()  { printf '\n\033[1;34m==> %s\033[0m\n' "$*"; }
ok()   { printf '    \033[32mOK\033[0m  %s\n' "$*"; }
warn() { printf '    \033[33m!!\033[0m  %s\n' "$*"; }
die()  { printf '\n\033[31mSTOPPED: %s\033[0m\n' "$*" >&2; exit 1; }

# ---------- 0. Safety checks before changing anything ----------
[[ "$(uname -s)" == "Linux" ]] || die "Run this inside the Ubuntu (WSL) window, not PowerShell."
[[ $EUID -ne 0 ]] || die "Don't run this with sudo. Run: bash scripts/setup-wsl.sh (it asks for your password when needed)."

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
[[ -f CLAUDE.md && -f requirements.txt && -f .claude/settings.json ]] || die "Run this from inside the clipforge repo."
case "$REPO" in
  /mnt/*) die "The repo is on your Windows drive ($REPO). Clone it into your Linux home instead: cd ~ && git clone ..." ;;
esac
if grep -qi microsoft /proc/version 2>/dev/null; then ok "Running inside WSL"; else warn "This doesn't look like WSL; continuing as plain Linux."; fi

say "Step 1/6: Linux tools (asks for your Ubuntu password once)"
PKGS=(git curl ca-certificates ffmpeg python3 python3-pip python3-venv jq gh bubblewrap socat)
MISSING=()
for p in "${PKGS[@]}"; do dpkg -s "$p" >/dev/null 2>&1 || MISSING+=("$p"); done
if ((${#MISSING[@]})); then
  sudo apt-get update -qq
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq "${MISSING[@]}"
fi
ok "Installed: ${PKGS[*]}"

say "Step 2/6: Let the sandbox start (Ubuntu 24.04+ AppArmor rule)"
userns="$(sysctl -n kernel.apparmor_restrict_unprivileged_userns 2>/dev/null || echo 0)"
if [[ "$userns" == "1" && ! -f /etc/apparmor.d/bwrap ]]; then
  # Grants user namespaces to bwrap itself only, not to the commands it runs.
  sudo tee /etc/apparmor.d/bwrap >/dev/null <<'EOF'
abi <abi/4.0>,
include <tunables/global>

profile bwrap /usr/bin/bwrap flags=(unconfined) {
  userns,
  include if exists <local/bwrap>
}
EOF
  sudo systemctl reload apparmor 2>/dev/null || sudo apparmor_parser -r /etc/apparmor.d/bwrap
  ok "Added AppArmor rule for bubblewrap"
else
  ok "No AppArmor change needed"
fi
if bwrap --ro-bind / / --dev /dev --proc /proc true 2>/dev/null; then
  ok "Sandbox engine (bubblewrap) works"
else
  die "bubblewrap can't start, so the sandbox won't work. Copy this message to Claude for help."
fi

say "Step 3/6: Claude Code"
export PATH="$HOME/.local/bin:$PATH"
if ! command -v claude >/dev/null 2>&1; then
  curl -fsSL https://claude.ai/install.sh | bash || die "Couldn't download Claude Code. Check your internet connection and run this script again."
fi
# shellcheck disable=SC2016  # write the literal text; it expands when .bashrc runs
grep -q '.local/bin' "$HOME/.bashrc" 2>/dev/null || echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$HOME/.bashrc"
command -v claude >/dev/null 2>&1 || die "Claude Code didn't install. Copy the lines above to Claude."
ok "Claude Code $(claude --version 2>/dev/null | head -1)"

say "Step 4/6: Python environment for ClipForge (a few minutes)"
[[ -d .venv ]] || python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install -q --upgrade pip
python -m pip install -q -r requirements.txt -r requirements-dev.txt
ok "ClipForge is installed in .venv"
if python -m playwright install --with-deps chromium >/dev/null 2>&1; then
  ok "Test browser (Chromium) installed"
else
  warn "Test browser didn't download (only needed for website tests). Rerun this script later to retry."
fi

say "Step 5/6: Your Anthropic API key (for ClipForge's highlight detection)"
[[ -f .env ]] || cp .env.example .env
chmod 600 .env
if grep -qE '^ANTHROPIC_API_KEY=.+' .env; then
  ok "A key is already saved in .env (not shown)"
else
  echo "    Paste your key and press Enter. Nothing will show while you paste; that's normal."
  echo "    Press Enter with nothing to skip (you can rerun this script later)."
  key=""
  read -rsp "    API key: " key </dev/tty || true
  echo
  if [[ -n "$key" ]]; then
    [[ "$key" =~ ^[A-Za-z0-9_-]+$ ]] || die "That doesn't look like an API key (unexpected characters). Nothing was saved."
    tmp="$(mktemp)"
    awk -v k="$key" 'BEGIN{d=0} /^ANTHROPIC_API_KEY=/{print "ANTHROPIC_API_KEY=" k; d=1; next} {print} END{if(!d) print "ANTHROPIC_API_KEY=" k}' .env > "$tmp"
    install -m 600 "$tmp" .env && rm -f "$tmp"
    unset key
    ok "Key saved to .env (only you can read it; git ignores it)"
  else
    warn "Skipped. ClipForge will only run in mock mode until you add a key."
  fi
fi
# A key exported in your shell would make Claude Code bill your API account instead of your plan.
if grep -qs 'ANTHROPIC_API_KEY' "$HOME/.bashrc" "$HOME/.profile" || [[ -n "${ANTHROPIC_API_KEY:-}" ]]; then
  warn "ANTHROPIC_API_KEY is set in your shell. Remove it from ~/.bashrc/~/.profile; ClipForge reads it from .env."
fi

say "Step 6/6: Checks"
if [[ -z "$(git config --global user.name || true)" ]]; then
  warn "git doesn't know your name yet (agents commit as you). Run once:"
  echo '        git config --global user.name "Your Name"'
  echo '        git config --global user.email "your-github-noreply-email"'
else
  ok "git user: $(git config --global user.name)"
fi
ffmpeg -version >/dev/null 2>&1 && ok "ffmpeg works"
python -c "import cv2, sys; v=cv2.__version__; sys.exit(0 if v.startswith('4.') else 1)" && ok "OpenCV 4.x (OpenCV 5 would break face detection)"
jq -e '.sandbox.enabled and .sandbox.failIfUnavailable and (.sandbox.allowUnsandboxedCommands|not)' .claude/settings.json >/dev/null \
  && ok "Agent safety settings: sandbox on, no unsandboxed escape, refuses to run if the sandbox is broken"
if python -m pytest tests/test_settings.py -q >/dev/null 2>&1; then ok "Quick test run passed"; else warn "Quick test run failed; tell Claude."; fi

say "Done. What's left for you:"
cat <<'EOF'
    1. Close this window and open a new Ubuntu window (so the `claude` command is found).
    2. Run:   cd ~/clipforge && claude
       Sign in with your browser, then choose "Yes" when asked to trust this folder.
    3. In Claude, type:   /sandbox     (it should say the sandbox is on)
    4. Then ask:   Use the build-feature skill to build the next features, in parallel where they touch different files.
EOF
