#!/usr/bin/env python3
"""Watch a REMOTE directory for new FITS files and pull each completed one down
to a local directory, with the password embedded in the script.

Auth: pexpect. ssh/scp refuse a password on stdin (they read /dev/tty), so
pexpect runs them inside a pseudo-terminal, waits for the "password:" prompt,
and types the password. Handles a first-time host-key prompt too.

  Install:  pip install pexpect

Race safety: a file is pulled only after its remote size is unchanged across
one poll interval, so a frame still being written isn't copied mid-write.
"""

import time
from pathlib import Path

import pexpect

# --- config ---------------------------------------------------------------
HOST       = "waikoko-new"
USER       = "nirc8"
PASSWORD   = "ka08hoku2025!"                     # <-- real password here
REMOTE_DIR = "/sdata907/nirc8/2026jun29"               # remote dir to watch
LOCALDIR   = Path("/home/mcisse/nirc2_temp/")   # local dir to copy into
PATTERN    = "*.fits"
INTERVAL   = 3.0                            # seconds between checks
# --------------------------------------------------------------------------

TARGET = f"{USER}@{HOST}"
SSH_OPTS = ["-o", "StrictHostKeyChecking=accept-new"]


def _auth(child):
    """Answer a host-key prompt and/or the password prompt."""
    i = child.expect([r"continue connecting", r"[Pp]assword:", pexpect.EOF])
    if i == 0:                       # first-time host key
        child.sendline("yes")
        child.expect(r"[Pp]assword:")
        child.sendline(PASSWORD)
    elif i == 1:                     # password prompt
        child.sendline(PASSWORD)
    # i == 2: command already finished (e.g. key auth) -> nothing to send


def ssh_capture(remote_cmd, timeout=30):
    """Run a remote command, return its stdout (empty string on failure)."""
    child = pexpect.spawn("ssh", [*SSH_OPTS, TARGET, remote_cmd],
                          encoding="utf-8", timeout=timeout)
    try:
        _auth(child)
        child.expect(pexpect.EOF)
        return child.before.strip()
    except (pexpect.TIMEOUT, pexpect.EOF):
        return ""
    finally:
        child.close()


def scp_pull(remote_path, timeout=600):
    child = pexpect.spawn("scp", [*SSH_OPTS, f"{TARGET}:{remote_path}",
                                  str(LOCALDIR)],
                          encoding="utf-8", timeout=timeout)
    try:
        _auth(child)
        child.expect(pexpect.EOF)
    except (pexpect.TIMEOUT, pexpect.EOF):
        pass
    child.close()
    return child.exitstatus == 0


def latest_remote():
    """Return (path, size) of the newest matching remote file, or (None, 0)."""
    out = ssh_capture(
        f'f=$(ls -1t {REMOTE_DIR}/{PATTERN} 2>/dev/null | head -1); '
        f'[ -n "$f" ] && stat -c "%s %n" "$f"'
    )
    # the captured text may include an echoed newline; take the last real line
    line = out.splitlines()[-1] if out else ""
    if not line or " " not in line:
        return None, 0
    size, path = line.split(" ", 1)
    return path.strip(), int(size)


LOCALDIR.mkdir(parents=True, exist_ok=True)
copied = set()
sizes = {}
while True:
    try:
        path, size = latest_remote()
        if path:
            name = Path(path).name
            if name not in copied:
                if sizes.get(path) == size:
                    # size held steady across one interval -> write finished
                    if scp_pull(path):
                        copied.add(name)
                        sizes.pop(path, None)
                        print(f"pulled {name}")
                else:
                    sizes[path] = size   # still growing; recheck next cycle
        time.sleep(INTERVAL)
    except KeyboardInterrupt:
        break
