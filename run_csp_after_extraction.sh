#!/bin/zsh
# Wait for csp_harmonized_epochs.py to finish, verify epoch counts, then run the full CSP LOSO.
while pgrep -f csp_harmonized_epochs.py > /dev/null; do sleep 60; done
n=$(ls results/harmonized/epoch_cache/*.npz | wc -l); echo "cached subjects: $n"
[ "$n" -eq 54 ] || { echo "expected 54 cached subjects, got $n; not running LOSO"; exit 1; }
venv/bin/python csp_harmonized_loso.py --jobs 4 > results/harmonized/csp_harmonized_run.log 2>&1
tail -8 results/harmonized/csp_harmonized_run.log | grep -v Warning
