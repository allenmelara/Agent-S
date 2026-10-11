#!/bin/bash
# Build "Agent S Menu Bar.app" in ~/Applications. Opening it puts the Agent S
# robot in the menu bar (top right), with New task…, Stop task and an
# Approvals setting. Tasks still run in a Terminal window, like the Dock app.
#
# Usage (from the repo):  bash mac_app/build_menubar_app.sh [--login]
#   --login   also open it automatically when you log in

set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
APP="$HOME/Applications/Agent S Menu Bar.app"
ICON_SRC="$REPO/mac_app/agent_s_icon.png"

if [[ "$(uname)" != "Darwin" ]]; then
  echo "This builds a macOS app; run it on your Mac." >&2
  exit 1
fi
if [[ ! -x "$REPO/.venv/bin/python" ]]; then
  echo "No Python environment at $REPO/.venv. Set it up first (see mac_feasibility/RUN_AGENT_S.md)." >&2
  exit 1
fi
if ! "$REPO/.venv/bin/python" -c "import rumps" 2>/dev/null; then
  echo "Installing rumps (menu-bar library) into .venv..."
  "$REPO/.venv/bin/python" -m pip install -q rumps
fi

REPO_AS="${REPO//\\/\\\\}"
REPO_AS="${REPO_AS//\"/\\\"}"

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

# The app only starts menubar.py in the background and exits; menubar.py
# keeps a single copy running (~/.agent_s/menubar.pid).
cat > "$WORK/menubar.applescript" <<APPLESCRIPT
set repoPath to "$REPO_AS"
do shell script "cd " & quoted form of repoPath & ¬
	" && nohup .venv/bin/python mac_app/menubar.py > /dev/null 2>&1 &"
APPLESCRIPT

mkdir -p "$HOME/Applications"
rm -rf "$APP"
osacompile -o "$APP" "$WORK/menubar.applescript"

if [[ -f "$ICON_SRC" ]]; then
  ICONSET="$WORK/agent_s.iconset"
  mkdir -p "$ICONSET"
  for size in 16 32 128 256 512; do
    sips -z "$size" "$size" "$ICON_SRC" --out "$ICONSET/icon_${size}x${size}.png" >/dev/null
    sips -z $((size * 2)) $((size * 2)) "$ICON_SRC" --out "$ICONSET/icon_${size}x${size}@2x.png" >/dev/null
  done
  iconutil -c icns "$ICONSET" -o "$APP/Contents/Resources/applet.icns"
  rm -f "$APP/Contents/Resources/Assets.car"
  /usr/libexec/PlistBuddy -c "Delete :CFBundleIconName" "$APP/Contents/Info.plist" 2>/dev/null || true
fi
# No Dock icon while it starts the background process.
/usr/libexec/PlistBuddy -c "Add :LSUIElement bool true" "$APP/Contents/Info.plist" 2>/dev/null || true
# Name it, so macOS permission prompts say "Agent S Menu Bar" instead of "applet".
for key in CFBundleName CFBundleDisplayName; do
  /usr/libexec/PlistBuddy -c "Set :$key Agent S Menu Bar" "$APP/Contents/Info.plist" 2>/dev/null ||
    /usr/libexec/PlistBuddy -c "Add :$key string Agent S Menu Bar" "$APP/Contents/Info.plist"
done
codesign --force --sign - "$APP" 2>/dev/null || true
touch "$APP"

echo "Built: $APP"

if [[ "${1:-}" == "--login" ]]; then
  osascript -e "tell application \"System Events\" to make login item at end with properties {path:\"$APP\", hidden:true}" >/dev/null
  echo "Added to Login Items: it will start when you log in."
fi

open "$APP"
echo "The Agent S robot should now be in the menu bar (top right)."
