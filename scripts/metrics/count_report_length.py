import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TEX = ROOT / "main.tex"

# IEEEtran conference, 10pt two-column: measured ~1050 words fill one full page
# of body text. Floats are charged separately below.
WORDS_PER_PAGE = 1050
PAGE_COST = {"figure": 0.33, "table": 0.18}   # fraction of a page each float eats

SPEC_MIN, SPEC_MAX = 8, 10


def strip_tex(src: str) -> str:
    """Remove everything that is not body prose the marker reads as text."""
    # drop the references section entirely (spec: page limit excludes references)
    src = re.split(r"\\section\*?\{References\}", src)[0]
    # drop the preamble
    src = src.split(r"\begin{document}", 1)[-1]
    # drop comment lines and trailing comments (but keep escaped \%)
    src = re.sub(r"(?m)(?<!\\)%.*$", "", src)
    # drop floats - they are counted separately, not as prose
    src = re.sub(r"\\begin\{(figure|table)\*?\}.*?\\end\{\1\*?\}", "", src, flags=re.S)
    # drop math
    src = re.sub(r"\$[^$]*\$", " ", src)
    # unwrap common commands, keeping their visible argument
    src = re.sub(r"\\(emph|textbf|textit|texttt|textsc)\{([^{}]*)\}", r"\2", src)
    # drop remaining control sequences and braces
    src = re.sub(r"\\[A-Za-z@]+\*?", " ", src)
    src = re.sub(r"[{}]", " ", src)
    return src


def main() -> int:
    if not TEX.exists():
        print(f"error: {TEX} not found", file=sys.stderr)
        return 1

    raw = TEX.read_text(encoding="utf8")
    body = strip_tex(raw)

    words = re.findall(r"[A-Za-z][A-Za-z\-']*", body)
    n_words = len(words)

    counts = {k: len(re.findall(rf"\\begin\{{{k}\*?\}}", raw)) for k in PAGE_COST}
    float_pages = sum(counts[k] * PAGE_COST[k] for k in counts)
    text_pages = n_words / WORDS_PER_PAGE
    est = text_pages + float_pages

    n_refs = len(re.findall(r"^@", (ROOT / "references.bib").read_text(encoding="utf8"), re.M))
    n_cites = len(set(re.findall(r"\\cite\{([^}]*)\}", raw)))

    print("=" * 58)
    print("  CA2 report length check   (main.tex)")
    print("=" * 58)
    print(f"  body words (excl. references and floats) : {n_words:>6}")
    print(f"  figures                                  : {counts['figure']:>6}")
    print(f"  tables                                   : {counts['table']:>6}")
    print(f"  references in .bib                       : {n_refs:>6}")
    print(f"  distinct \\cite keys used                 : {n_cites:>6}")
    print("-" * 58)
    print(f"  text pages   {text_pages:>5.1f}")
    print(f"  float pages  {float_pages:>5.1f}")
    print(f"  ESTIMATED    {est:>5.1f} pages   (spec: {SPEC_MIN}-{SPEC_MAX})")
    print("-" * 58)

    if est < SPEC_MIN:
        short = (SPEC_MIN - est) * WORDS_PER_PAGE
        print(f"  UNDER the {SPEC_MIN}-page minimum -> 5-point penalty risk.")
        print(f"  Need roughly {short:.0f} more words, or add figures.")
    elif est > SPEC_MAX:
        over = (est - SPEC_MAX) * WORDS_PER_PAGE
        print(f"  OVER the {SPEC_MAX}-page limit -> 5-point penalty risk.")
        print(f"  Cut roughly {over:.0f} words.")
    else:
        print(f"  Within the {SPEC_MIN}-{SPEC_MAX} page window.")

    print()
    print("  Estimate only. LaTeX float placement shifts real page count by")
    print("  +/- 0.5 page, so confirm on Overleaf before submitting.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
