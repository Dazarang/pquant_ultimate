#!/usr/bin/env bash
# gate.sh -- Verification gate for autoresearch experiments.
# Checks immutability, runs experiment, extracts metric, enforces sanity.
#
# Exit codes:
#   0 = experiment ran and produced a score
#   1 = gate violation (immutable file changed, timeout, sanity fail)
#
# Outputs on success:
#   Last line: COMPOSITE_SCORE=X.XXXX  (extracted by run.sh)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

TIMEOUT_SECONDS=2700  # 45 minutes max
LOG_FILE="research/run.log"

# ------------------------------------------------------------------
# 1. Immutability checks -- lib/ and gate.sh must not be modified
# ------------------------------------------------------------------
PROTECTED_FILES=(
    "lib/data.py"
    "lib/eval.py"
    "lib/features.py"
    "lib/__init__.py"
    "research/gate.sh"
    "research/utils/baseline.py"
    "research/utils/model_wrappers.py"
)

for f in "${PROTECTED_FILES[@]}"; do
    if git diff --name-only HEAD -- "$f" | grep -q .; then
        echo "GATE VIOLATION: $f was modified. Reverting."
        git checkout HEAD -- "$f"
        exit 1
    fi
done

echo "Immutability check: PASSED"

# ------------------------------------------------------------------
# 2. Run experiment with timeout
# ------------------------------------------------------------------
echo "Running experiment (timeout: ${TIMEOUT_SECONDS}s)..."

if perl -e "alarm $TIMEOUT_SECONDS; exec @ARGV" uv run python research/experiment.py > "$LOG_FILE" 2>&1; then
    echo "Experiment completed."
else
    EXIT_CODE=$?
    if [ "$EXIT_CODE" -eq 124 ]; then
        echo "GATE VIOLATION: Experiment exceeded ${TIMEOUT_SECONDS}s timeout."
    else
        echo "GATE VIOLATION: Experiment crashed (exit code $EXIT_CODE)."
        echo "--- Last 30 lines of log ---"
        tail -n 30 "$LOG_FILE"
    fi
    exit 1
fi

# ------------------------------------------------------------------
# 3. Extract metric
# ------------------------------------------------------------------
SCORE_LINE=$(grep "^COMPOSITE_SCORE=" "$LOG_FILE" | tail -1)
PASSED_LINE=$(grep "^PASSED=" "$LOG_FILE" | tail -1)

if [ -z "$SCORE_LINE" ]; then
    echo "GATE VIOLATION: No COMPOSITE_SCORE found in output."
    tail -n 20 "$LOG_FILE"
    exit 1
fi

SCORE="${SCORE_LINE#COMPOSITE_SCORE=}"
PASSED="${PASSED_LINE#PASSED=}"

echo "Score: $SCORE"
echo "Passed tiers: $PASSED"

# ------------------------------------------------------------------
# 4. Sanity checks
# ------------------------------------------------------------------

# Must pass at least tier 1
if [ "$PASSED" = "False" ]; then
    echo "GATE: Tiers not passed. Score=$SCORE"
    # Still output score for logging, but mark as failed
    echo "GATE_STATUS=FAILED"
    echo "$SCORE_LINE"
    exit 1
fi

echo "GATE_STATUS=PASSED"
echo "$SCORE_LINE"
exit 0
