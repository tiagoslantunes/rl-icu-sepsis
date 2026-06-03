# Report

LaTeX source for the project report (NOVA IMS style).

## Compile
- **Overleaf:** upload the `report/` folder; set `report.tex` as the main file; compile with pdfLaTeX.
- **Local:** `cd report && pdflatex report.tex && pdflatex report.tex` (run twice for the table of contents and cross-references).

Figures are bundled in `report/figures/` (copied from the repository `plots/`), so
the folder compiles standalone. All numbers are reproducible from the committed
result files and the `run_*.py` scripts.

> Fill in the group name and the remaining members on the title page before submitting.
