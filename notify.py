"""Native macOS notifications — EXACTLY per Amir's proven pattern
(~/Claude/Projects/trading-journal/docs/macos-notifications-prompt.md):

  * osascript `display notification`, text passed as ARGV (no quoting/escaping bugs)
  * sound SEPARATELY via `afplay` on a built-in system sound (decoupled from osascript)
  * non-blocking Popen, wrapped so it NEVER raises into the caller
  * NOT terminal-notifier (blocked/unreliable on modern macOS)

Delivery identity is "Script Editor" (com.apple.ScriptEditor2): the user must enable it in
System Settings → Notifications (style "Alerts" recommended) — it appears in that list only
AFTER the first fire. Focus/DND silently hides everything (no error): if "I got nothing",
check the moon icon first, then the Script Editor permission.

Used by alerts.py for grade-A EP/RVOL events. Gate: config.SIG_NOTIFY_MACOS.
"""

from __future__ import annotations

import os
import subprocess


def notify(title: str, message: str, subtitle: str = "", sound: str = "Ping") -> None:
    """Fire a notification + sound. Non-blocking; never raises."""
    script = ("on run argv\n"
              "display notification (item 1 of argv) with title (item 2 of argv) "
              "subtitle (item 3 of argv)\n"
              "end run")
    try:
        subprocess.Popen(
            ["osascript", "-e", script, "--", str(message), str(title), str(subtitle or "")],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass
    p = f"/System/Library/Sounds/{sound}.aiff"      # Basso, Ping, Glass, Tink, Submarine, ...
    try:
        if os.path.exists(p):
            subprocess.Popen(["afplay", p], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass
