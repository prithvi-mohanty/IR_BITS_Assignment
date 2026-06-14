#!/usr/bin/env bash
# Build the IR Assignment 1 submission bundle:
#   1. Verify all 11 screenshots exist in screenshots/
#   2. Generate Report.pdf from Report_Outline.md via pandoc + weasyprint
#   3. Zip the submission folder with the correct exclusions
#
# Usage:  ./make_submission.sh
# Output: ../IR_AIMLCZG537_Group64_Assignment1.zip
#
# Requires: pandoc, weasyprint (pip install weasyprint)

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

ZIP_NAME="IR_AIMLCZG537_Group64_Assignment1.zip"
ZIP_PATH="$REPO_DIR/../$ZIP_NAME"

REQUIRED_SHOTS=(
  "01_document_upload.png"
  "02_preprocessing_output.png"
  "03_inverted_index.png"
  "04_stemming_lemmatization_table.png"
  "05_jaccard_experiment.png"
  "06_phrase_query_results.png"
  "07_bst_btree_benchmark.png"
  "08_tolerant_wildcard.png"
  "09_tolerant_spelling.png"
  "10_tolerant_phonetic.png"
  "11_virtual_lab.png"
)

echo "==> Step 1/3: Verifying screenshots/"
missing=0
for f in "${REQUIRED_SHOTS[@]}"; do
  if [[ ! -s "screenshots/$f" ]]; then
    echo "    MISSING: screenshots/$f"
    missing=$((missing+1))
  else
    size=$(wc -c < "screenshots/$f" | tr -d ' ')
    echo "    ok      : screenshots/$f ($size bytes)"
  fi
done
if (( missing > 0 )); then
  echo
  echo "ERROR: $missing screenshot(s) missing. Take them in the Streamlit app and drop them into screenshots/."
  exit 1
fi

echo
echo "==> Step 2/3: Generating Report.pdf via pandoc + weasyprint"
if ! command -v pandoc >/dev/null; then
  echo "ERROR: pandoc not installed. Run: brew install pandoc"
  exit 1
fi
if ! python -c "import weasyprint" >/dev/null 2>&1; then
  echo "ERROR: weasyprint not installed. Run: pip install weasyprint"
  exit 1
fi
pandoc Report_Outline.md \
  --pdf-engine=weasyprint \
  --metadata title="IR Assignment 1 — Group 64" \
  -o Report.pdf
echo "    Report.pdf -> $(wc -c < Report.pdf | tr -d ' ') bytes"

echo
echo "==> Step 3/3: Building submission zip"
rm -f "$ZIP_PATH"
PARENT_DIR="$(cd "$REPO_DIR/.." && pwd)"
REPO_BASENAME="$(basename "$REPO_DIR")"
cd "$PARENT_DIR"
zip -r "$ZIP_NAME" "$REPO_BASENAME" \
  --exclude "$REPO_BASENAME/.git/*" \
  --exclude "$REPO_BASENAME/.git" \
  --exclude "$REPO_BASENAME/.gitignore" \
  --exclude "$REPO_BASENAME/.claude/*" \
  --exclude "$REPO_BASENAME/.idea/*" \
  --exclude "$REPO_BASENAME/.venv/*" \
  --exclude "$REPO_BASENAME/__pycache__/*" \
  --exclude "$REPO_BASENAME/.DS_Store" \
  --exclude "$REPO_BASENAME/**/.DS_Store" \
  --exclude "$REPO_BASENAME/*.sh" \
  --exclude "$REPO_BASENAME/Report_Outline.md" \
  --exclude "$REPO_BASENAME/IR_Assignment1_*.docx" \
  --exclude "$REPO_BASENAME/data/*" \
  > /dev/null

echo "    zip -> $ZIP_PATH ($(wc -c < "$ZIP_PATH" | tr -d ' ') bytes)"
echo
echo "==> Contents:"
unzip -l "$ZIP_PATH" | tail -30
echo
echo "DONE. Upload $ZIP_PATH to Taxila before 15 June 23:59."
