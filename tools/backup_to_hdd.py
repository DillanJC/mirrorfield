# -*- coding: utf-8 -*-
"""Weekly backup: mirrorfield repo (full git history) + pocket docs + AI memory
-> dated zip on the E: HDD (a separate physical disk from the C: NVMe).
Keeps the newest 4 backups; deletes older ones. Run manually or via the
scheduled task 'mirrorfield-weekly-backup'."""
import os
import zipfile
from datetime import date
from pathlib import Path

DEST_DIR = Path(r"E:\mirrorfield-backups")
EXCLUDE_DIRS = {".venv", "__pycache__", ".pytest_cache", "node_modules"}
POCKET = [
    Path(r"C:\Users\User\Downloads\ai_safety_forum_positions.md"),
    Path(r"C:\Users\User\Downloads\DEPTH_MAP.md"),
    Path(r"C:\Users\User\Downloads\MIRRORFIELD_STATE.md"),
    Path(r"C:\Users\User\Downloads\LOCAL_AI_GUIDE.md"),
    Path(r"C:\Users\User\Downloads\dillan-creative-index.md"),
]
MEMORY = Path(r"C:\Users\User\.claude\projects\C--Users-User-geometric-safety-features-Experiment\memory")
REPO = Path(r"C:\Users\User\mirrorfield")
KEEP = 4


def main():
    DEST_DIR.mkdir(parents=True, exist_ok=True)
    out = DEST_DIR / f"mirrorfield_backup_{date.today().isoformat()}.zip"
    count = 0
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=6,
                         strict_timestamps=False) as z:
        for dirpath, dirnames, filenames in os.walk(REPO):
            dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIRS]
            for f in filenames:
                p = Path(dirpath) / f
                z.write(p, "mirrorfield/" + str(p.relative_to(REPO)))
                count += 1
        for p in POCKET:
            if p.exists():
                z.write(p, "pocket_docs/" + p.name); count += 1
        if MEMORY.exists():
            for p in MEMORY.glob("*.md"):
                z.write(p, "ai_memory/" + p.name); count += 1
    print(f"{out} — {count} files, {out.stat().st_size/1024/1024:.1f} MB")
    # rotate: keep newest KEEP
    backups = sorted(DEST_DIR.glob("mirrorfield_backup_*.zip"))
    for old in backups[:-KEEP]:
        old.unlink()
        print(f"rotated out: {old.name}")


if __name__ == "__main__":
    main()
