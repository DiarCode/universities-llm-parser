#!/usr/bin/env python3
import argparse
import concurrent.futures as cf
import re
import subprocess
from pathlib import Path
from typing import List, Tuple

ACRONYMS = {"usa": "USA", "uk": "UK", "uae": "UAE",
            "eu": "EU", "prc": "PRC", "drc": "DRC"}


def to_country_name(stem: str) -> str:
    # replace separators with spaces
    s = re.sub(r"[_\-]+", " ", stem.strip())
    # keep pure acronyms as-is; otherwise title-case word by word
    words = []
    for w in s.split():
        lc = w.lower()
        words.append(ACRONYMS.get(lc, w if w.isupper() else w.capitalize()))
    return " ".join(words)


def build_cmd(country: str, file_path: Path) -> List[str]:
    return [
        "poetry", "run", "python", "country_universities_llm.py",
        "--country", country,
        "--universities-file", str(file_path),
    ]


def run_one(file_path: Path, dry_run: bool = False) -> Tuple[Path, int]:
    country = to_country_name(file_path.stem)
    cmd = build_cmd(country, file_path)
    if dry_run:
        print("DRY-RUN:", " ".join(f'"{c}"' if " " in c else c for c in cmd))
        return (file_path, 0)
    print(f"→ {country}  [{file_path}]")
    res = subprocess.run(cmd, capture_output=False)
    return (file_path, res.returncode)


def main():
    p = argparse.ArgumentParser(
        description="Run country_universities_llm.py for all *.txt in a folder.")
    p.add_argument("--dir", default="uni_names",
                   help="Directory with *.txt files (default: ./uni_names)")
    p.add_argument("--jobs", "-j", type=int, default=1,
                   help="Parallel jobs (default: 1)")
    p.add_argument("--dry-run", action="store_true",
                   help="Print commands without executing")
    p.add_argument("--stop-on-error", action="store_true",
                   help="Stop at first failing command")
    args = p.parse_args()

    base = Path(args.dir)
    if not base.exists():
        raise SystemExit(f"Directory not found: {base}")

    files = sorted([p for p in base.glob("*.txt") if p.is_file()])
    if not files:
        raise SystemExit(f"No .txt files found in: {base}")

    failures = []
    if args.jobs > 1 and not args.dry_run:
        with cf.ThreadPoolExecutor(max_workers=args.jobs) as ex:
            futures = {ex.submit(run_one, f, args.dry_run): f for f in files}
            for fut in cf.as_completed(futures):
                _, rc = fut.result()
                if rc != 0:
                    failures.append(futures[fut])
                    if args.stop_on_error:
                        break
    else:
        for f in files:
            _, rc = run_one(f, args.dry_run)
            if rc != 0:
                failures.append(f)
                if args.stop_on_error:
                    break

    if failures:
        print("\nSome runs failed:")
        for f in failures:
            print(" -", f)
        raise SystemExit(1)
    print("\nAll done ✔")


if __name__ == "__main__":
    main()
