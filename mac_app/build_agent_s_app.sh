#!/bin/bash
# Build "Agent S.app" in ~/Applications so Agent S can be started from the Dock.
#
# Opening the app asks what Agent S should do, then runs it in a Terminal
# window with the claude_subscription settings (your Claude Max login, no API
# key, confirmation dialog before every action). It runs inside Terminal on
# purpose: Terminal already has the Accessibility and Screen Recording
# permissions Agent S needs, and you can watch the log and stop it with Ctrl+C.
#
# Usage (from the repo):  bash mac_app/build_agent_s_app.sh [--dock]
#   --dock   also add the app to the Dock (restarts the Dock once)

set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
APP="$HOME/Applications/Agent S.app"
ICON_SRC="$REPO/mac_app/agent_s_icon.png"

if [[ "$(uname)" != "Darwin" ]]; then
  echo "This builds a macOS app; run it on your Mac." >&2
  exit 1
fi
if [[ ! -x "$REPO/.venv/bin/python" ]]; then
  echo "No Python environment at $REPO/.venv. Set it up first (see mac_feasibility/RUN_AGENT_S.md)." >&2
  exit 1
fi

# Escape the repo path for an AppleScript string literal.
REPO_AS="${REPO//\\/\\\\}"
REPO_AS="${REPO_AS//\"/\\\"}"

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

cat > "$WORK/agent_s.applescript" <<APPLESCRIPT
set repoPath to "$REPO_AS"
set promptText to "What should Agent S do on your Mac?" & return & return & ¬
	"It uses your Claude Max login (no API key) and asks before every action."
set dialogResult to display dialog promptText default answer "" with title "Agent S" ¬
	buttons {"Cancel", "Start"} default button "Start" cancel button "Cancel"
set taskText to text returned of dialogResult
if taskText is "" then return

set cmd to "cd " & quoted form of repoPath & ¬
	" && { conda deactivate 2>/dev/null || true; }" & ¬
	" && source .venv/bin/activate" & ¬
	" && python -m gui_agents.s3.cli_app" & ¬
	" --provider claude_subscription --ground_provider claude_subscription" & ¬
	" --max_steps 15 --max_model_calls 60" & ¬
	" --task " & quoted form of taskText

tell application "Terminal"
	activate
	do script cmd
end tell
APPLESCRIPT

mkdir -p "$HOME/Applications"
rm -rf "$APP"
osacompile -o "$APP" "$WORK/agent_s.applescript"

# Icon: build an .icns from the square PNG and swap it in for the default applet icon.
if [[ -f "$ICON_SRC" ]]; then
  ICONSET="$WORK/agent_s.iconset"
  mkdir -p "$ICONSET"
  for size in 16 32 128 256 512; do
    sips -z "$size" "$size" "$ICON_SRC" --out "$ICONSET/icon_${size}x${size}.png" >/dev/null
    sips -z $((size * 2)) $((size * 2)) "$ICON_SRC" --out "$ICONSET/icon_${size}x${size}@2x.png" >/dev/null
  done
  iconutil -c icns "$ICONSET" -o "$APP/Contents/Resources/applet.icns"
  touch "$APP"
fi

echo "Built: $APP"

if [[ "${1:-}" == "--dock" ]]; then
  defaults write com.apple.dock persistent-apps -array-add \
    "<dict><key>tile-data</key><dict><key>file-data</key><dict><key>_CFURLString</key><string>$APP</string><key>_CFURLStringType</key><integer>0</integer></dict></dict></dict>"
  killall Dock
  echo "Added to the Dock."
else
  echo "To keep it in the Dock: open it once, then right-click its Dock icon > Options > Keep in Dock."
  open -R "$APP"
fi
