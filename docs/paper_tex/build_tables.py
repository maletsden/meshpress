"""Extract each table env from main.tex into standalone Table_N.tex + Table_N.pdf for EM upload."""
from __future__ import annotations
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).parent
TEX = ROOT / "main.tex"
OUT = ROOT / "tables_out"
OUT.mkdir(exist_ok=True)

PREAMBLE = r"""\documentclass[11pt,a4paper]{article}
\usepackage[a4paper,margin=2cm,landscape]{geometry}
\usepackage{booktabs}
\usepackage{multirow}
\usepackage{array}
\usepackage{amsmath,amssymb}
\providecommand{\toprule}{\hline\hline}
\providecommand{\midrule}{\hline}
\providecommand{\botrule}{\hline\hline}
\pagestyle{empty}
\begin{document}
\setlength{\parindent}{0pt}
"""
POSTAMBLE = r"\end{document}" + "\n"

# Strip the \section{sec:...} cross-refs (replace with literal text), since standalone has no refs.
def sanitize(block: str) -> str:
    block = re.sub(r"~\\cite\{[^}]+\}", "", block)
    block = re.sub(r"Section~\\ref\{[^}]+\}", "the manuscript", block)
    block = re.sub(r"Figure~\\ref\{[^}]+\}", "the corresponding figure", block)
    block = re.sub(r"Equation~\\eqref\{[^}]+\}", "the predictor equation", block)
    block = re.sub(r"\\ref\{[^}]+\}", "[ref]", block)
    block = re.sub(r"\\label\{[^}]+\}", "", block)
    # sidewaystable -> table (so a single page suffices)
    block = block.replace(r"\begin{sidewaystable}", r"\begin{table}[h]")
    block = block.replace(r"\end{sidewaystable}", r"\end{table}")
    # \textheight inside sideways tabular* -> \textwidth
    block = block.replace(r"{\textheight}", r"{\textwidth}")
    return block

def extract_tables(text: str) -> list[str]:
    # Match either table or sidewaystable env, including content.
    pattern = re.compile(
        r"\\begin\{(sidewaystable|table)\}.*?\\end\{\1\}",
        re.DOTALL,
    )
    return [m.group(0) for m in pattern.finditer(text)]

def main():
    text = TEX.read_text(encoding="utf-8")
    blocks = extract_tables(text)
    print(f"found {len(blocks)} tables")
    for i, block in enumerate(blocks, 1):
        clean = sanitize(block)
        counter = f"\\setcounter{{table}}{{{i - 1}}}\n"
        doc = PREAMBLE + counter + clean + "\n" + POSTAMBLE
        f = OUT / f"Table{i}.tex"
        f.write_text(doc, encoding="utf-8")
        # compile
        for _ in range(1):
            r = subprocess.run(
                ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", f.name],
                cwd=OUT, capture_output=True, text=True,
            )
            if r.returncode != 0:
                print(f"Table{i}: FAIL")
                print(r.stdout[-1200:])
                break
        else:
            pass
        if (OUT / f"Table{i}.pdf").exists():
            print(f"Table{i}.pdf OK")
        else:
            print(f"Table{i}.pdf MISSING")

if __name__ == "__main__":
    main()
