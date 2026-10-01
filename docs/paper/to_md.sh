#!/bin/bash
# Write paper.md, a Markdown copy of paper.tex for reading on GitHub or any
# Markdown viewer: figures as SVG, tables inlined from results/, citations
# resolved from refs.bib. paper.tex stays the source; rerun this after editing it.
#   docs/paper/to_md.sh
set -eo pipefail
cd "$(dirname "$0")"
pdftocairo -svg figs/overview.pdf figs/overview.svg
# Inline the \input tables, drop the grouped header rows Markdown tables
# cannot hold, and point figures at their SVG copies.
perl -pe 's/\\input\{([^}]+)\}/`cat $1`/ge' paper.tex \
  | grep -v -e '\\cmidrule' -e '\\multicolumn' \
  | sed 's/\.pdf}/.svg}/' \
  | pandoc -f latex -t gfm --citeproc --bibliography refs.bib --lua-filter md.lua --wrap=none -o paper.md
echo "wrote $(wc -l < paper.md) lines to docs/paper/paper.md"
