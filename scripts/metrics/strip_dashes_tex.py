import argparse
import re
import shutil
import sys
from pathlib import Path

# Order matters: longest LaTeX ligature first, and unicode before ASCII.
# Every replacement produces a comma, a full stop, or a plain hyphen - never a
# dash ligature and never a bare " - " floating in the middle of a sentence.
DASHES = "—–‒―"          # em, en, figure, horizontal bar


def clean(text: str) -> str:
    # 1. LaTeX dash ligatures used as punctuation: " --- " and " -- "
    #    A dash used parenthetically becomes a comma; that reads more plainly
    #    and is what the marker expects in running prose.
    text = re.sub(r"\s*-{3}\s*", ", ", text)
    text = re.sub(r"(?<=\s)-{2}(?=\s)", ",", text)

    # 2. unicode em/en dashes, same treatment
    text = re.sub(rf"\s*[{DASHES}]\s*", ", ", text)

    # 3. tidy the damage: ", ," -> ",", " ,"-> ",", ",." -> "."
    text = re.sub(r",\s*,", ",", text)
    text = re.sub(r"\s+,", ",", text)
    text = re.sub(r",\s*([.;:])", r"\1", text)
    text = re.sub(r",\s*\)", ")", text)
    text = re.sub(r"\(\s*,\s*", "(", text)

    # 4. a comma immediately before a sentence end or a LaTeX row break is wrong
    text = re.sub(r",(\s*\\\\)", r"\1", text)

    # 5. collapse runs of spaces, but never touch newlines or indentation
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text


def report(before: str, after: str) -> None:
    for name, pat in (("--- (LaTeX em)", r"-{3}"),
                      ("-- (LaTeX en)", r"(?<=\s)-{2}(?=\s)"),
                      ("unicode em/en", rf"[{DASHES}]")):
        b = len(re.findall(pat, before))
        a = len(re.findall(pat, after))
        print(f"  {name:16s} {b:4d} -> {a}")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Remove em/en dashes from a .tex or .txt file without "
                    "damaging hyphenated words.")
    ap.add_argument("path")
    ap.add_argument("--write", action="store_true",
                    help="edit in place (a .bak copy is kept)")
    args = ap.parse_args()

    src = Path(args.path)
    if not src.exists():
        print(f"no such file: {src}", file=sys.stderr)
        return 1

    before = src.read_text(encoding="utf8")
    after = clean(before)

    print(f"{src.name}: dash counts")
    report(before, after)

    if not args.write:
        print("\ndry run - nothing written. Re-run with --write to apply.")
        return 0

    backup = src.with_suffix(src.suffix + ".bak_dashes")
    shutil.copy(src, backup)
    src.write_text(after, encoding="utf8")
    print(f"\nwritten. backup at {backup.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
