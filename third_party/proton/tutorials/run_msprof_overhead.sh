#!/bin/bash
# Measure proton instrumentation overhead using msprof op.
# Extracts Task Duration from stdout — no CSV parsing needed.
#
# Usage: bash run_msprof_overhead.sh [-e 1|2]

set -euo pipefail

EXAMPLE="${1:-2}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

eval "$(conda shell.bash hook)"
conda activate Triton

# Clean stale OPPROF dirs
find "$SCRIPT_DIR" -maxdepth 1 -type d -name 'OPPROF_*' -exec rm -rf {} + 2>/dev/null || true

echo "=== msprof op (example=${EXAMPLE}) ==="

# Run msprof, capture output to temp file
tmpfile=$(mktemp /tmp/msprof_overhead_XXXXXX.txt)
msprof op --aic-metrics=BasicInfo --launch-count=10 \
    python "$SCRIPT_DIR/03_overhead.py" -e "$EXAMPLE" 2>&1 | tee "$tmpfile"

echo ""
echo "=== Kernel Timing ==="

# Extract Task Duration from msprof stdout
# Format:
#   Op Name: gemm_kernel_plain
#       ...
#       Task Duration(us): 2469.789307
plain_us=$(grep -A5 'Op Name: gemm_kernel_plain$' "$tmpfile" | grep 'Task Duration' | head -1 | grep -oP '[\d.]+')
instr_us=$(grep -A5 'Op Name: gemm_kernel_instrumented$' "$tmpfile" | grep 'Task Duration' | head -1 | grep -oP '[\d.]+')

if [ -z "$plain_us" ]; then
    # Try without exact end-of-line match (may have trailing chars)
    plain_us=$(grep -A5 'Op Name: gemm_kernel_plain' "$tmpfile" | grep 'Task Duration' | head -1 | grep -oP '[\d.]+')
fi
if [ -z "$instr_us" ]; then
    instr_us=$(grep -A5 'Op Name: gemm_kernel_instrumented' "$tmpfile" | grep 'Task Duration' | head -1 | grep -oP '[\d.]+')
fi

echo "  gemm_kernel_plain:          ${plain_us:-NOT FOUND} us"
echo "  gemm_kernel_instrumented:   ${instr_us:-NOT FOUND} us"

if [ -n "$plain_us" ] && [ -n "$instr_us" ]; then
    python3 -c "
plain = float('$plain_us')
instr = float('$instr_us')
diff  = instr - plain
pct   = (diff / plain) * 100
print(f'  Overhead (kernel-only):     {diff:+.3f} us  ({pct:+.1f}%)')
"
fi

# Cleanup
find "$SCRIPT_DIR" -maxdepth 1 -type d -name 'OPPROF_*' -exec rm -rf {} + 2>/dev/null || true
rm -f "$tmpfile"
echo ""
echo "Done."
