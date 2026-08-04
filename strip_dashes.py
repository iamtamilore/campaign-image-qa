"""
strip_dashes.py — remove em-dashes (—) and en-dashes (–) from a text or .odt file.
Replaces them (and the LaTeX --- / -- forms) with a plain hyphen, and tidies spacing.

Usage:
  python strip_dashes.py myfile.txt              # writes myfile.txt in place
  python strip_dashes.py final_submission.odt    # edits the .odt (CLOSE it in LibreOffice first!)
  python strip_dashes.py in.txt out.txt          # write to a new file instead
"""
import sys, re, shutil, zipfile, os
from pathlib import Path

def clean(text: str) -> str:
    text = text.replace("—", "-")   # — em-dash
    text = text.replace("–", "-")   # – en-dash
    text = text.replace("---", "-")      # LaTeX em-dash
    text = text.replace("--", "-")       # LaTeX en-dash
    text = re.sub(r" *- *", " - ", text) # normalise " - " spacing
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text

def do_txt(src: Path, dst: Path):
    dst.write_text(clean(src.read_text(encoding="utf-8")), encoding="utf-8")
    print("cleaned ->", dst)

def do_odt(src: Path, dst: Path):
    if src != dst:
        shutil.copy(src, dst)
    tmp = dst.with_suffix(".odt.tmp")
    with zipfile.ZipFile(dst) as zin:
        names = zin.namelist()
        data = {n: zin.read(n) for n in names}
    xml = data["content.xml"].decode("utf-8")
    data["content.xml"] = clean(xml).encode("utf-8")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zout:
        # mimetype MUST be first and STORED (uncompressed)
        if "mimetype" in data:
            zout.writestr("mimetype", data.pop("mimetype"), zipfile.ZIP_STORED)
        for n in names:
            if n in data:
                zout.writestr(n, data[n])
    os.replace(tmp, dst)
    print("cleaned .odt ->", dst)

def main():
    if len(sys.argv) < 2:
        print(__doc__); sys.exit(1)
    src = Path(sys.argv[1])
    dst = Path(sys.argv[2]) if len(sys.argv) > 2 else src
    if not src.exists():
        print("no such file:", src); sys.exit(1)
    if src.suffix.lower() == ".odt":
        do_odt(src, dst)
    else:
        do_txt(src, dst)

if __name__ == "__main__":
    main()
