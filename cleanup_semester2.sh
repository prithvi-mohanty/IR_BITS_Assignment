#!/bin/bash
# Removes IR assignment files that were moved to PycharmProjects
# Run once from Terminal: bash ~/PycharmProjects/IR_BITS_Assignment/cleanup_semester2.sh

set -e
IR_DIR="$HOME/Downloads/Semester 2/IR"

echo "Removing duplicates from: $IR_DIR"
rm -f "$IR_DIR/app.py"       && echo "  ✓ app.py"
rm -f "$IR_DIR/README.md"    && echo "  ✓ README.md"
rm -rf "$IR_DIR/__pycache__" && echo "  ✓ __pycache__/"
rm -rf "$IR_DIR/code"        && echo "  ✓ code/"
echo "Done."
