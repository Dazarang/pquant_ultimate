#!/usr/bin/env bash
# run_codex.sh -- Autoresearch outer loop (Codex variant).
#
# Usage:
#   ./research/run_codex.sh              # Run on current branch (default 20 iterations)
#   ./research/run_codex.sh 50           # Run 50 iterations
#   ./research/run_codex.sh 20 --branch  # Run on a new autoresearch/ branch
#
# Each iteration:
#   1. Codex edits experiment.py and/or features_lab.py
#   2. gate.sh verifies and runs
#   3. If score improves: commit. If not: revert + log to COMBAT_LOG.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

MAX_ITERS="${1:-20}"

# Optionally create research branch
STARTING_BRANCH=$(git branch --show-current)
if [ "${2:-}" = "--branch" ] || [ "${1:-}" = "--branch" ]; then
    BRANCH="autoresearch/$(date +%Y%m%d_%H%M%S)"
    git checkout -b "$BRANCH"
    echo "Created branch: $BRANCH (from $STARTING_BRANCH)"
else
    BRANCH="$STARTING_BRANCH"
fi

RESEARCH_LOG="research/RESEARCH_LOG.md"
COMBAT_LOG="research/COMBAT_LOG.md"
RESULTS_TSV="research/results.tsv"
BEST_SCORE_FILE="research/.best_score"
CONSECUTIVE_FAILS=0
MAX_CONSECUTIVE_FAILS=$(( MAX_ITERS / 2 < 3 ? 3 : MAX_ITERS / 2 ))
CONSECUTIVE_KNOWLEDGE=0
MAX_CONSECUTIVE_KNOWLEDGE=5

# Initialize files if needed
[ -f "$RESEARCH_LOG" ] || echo "# Research Log" > "$RESEARCH_LOG"
[ -f "$COMBAT_LOG" ] || printf "# Combat Log\n\nWhat worked, what failed, and why. Read this BEFORE starting a new iteration.\n" > "$COMBAT_LOG"
[ -f "$RESULTS_TSV" ] || printf "iter\tscore\tstatus\tdescription\n" > "$RESULTS_TSV"
[ -f "$BEST_SCORE_FILE" ] || echo "-999" > "$BEST_SCORE_FILE"

BEST_SCORE=$(cat "$BEST_SCORE_FILE")

# Get short description of what changed from git diff
get_change_description() {
    local diff
    diff=$(git diff -- research/experiment.py research/features_lab.py)
    local desc
    desc=$(echo "$diff" | grep "^+" | grep -v "^+++" | head -5 | sed 's/^+//' | tr '\n' ' ' | sed 's/[[:space:]]\+/ /g' | cut -c1-80 || true)
    if [ -z "$desc" ]; then
        desc="unknown change"
    fi
    echo "$desc" | tr '\t' ' '
}

echo "============================================================"
echo "AUTORESEARCH (Codex)"
echo "============================================================"
echo "Best score: $BEST_SCORE"
echo "Max iterations: $MAX_ITERS"
echo ""

run_codex_attempt() {
    local iter=$1
    local prompt="You are an autonomous ML researcher. Your goal: maximize the composite score for PivotLow (stock bottom) prediction.

CURRENT BEST SCORE: $BEST_SCORE
ITERATION: $iter / $MAX_ITERS

READ FIRST:
- research/COMBAT_LOG.md (what has been tried, what failed, what worked)
- research/experiment.py (current best experiment)
- research/features_lab.py (current custom features)
- research/program.md (full instructions and constraints)

You may ONLY edit:
- research/experiment.py (model, hyperparams, features, stocks, thresholds)
- research/features_lab.py (add new backward-looking features)

Do NOT edit anything in lib/, research/gate.sh, or research/baseline.py.

Test ONE hypothesis. Multiple changes are fine if they serve a single testable idea. Commit nothing -- gate.sh handles that.
After editing, stop. The outer loop will run gate.sh.

If COMBAT_LOG.md shows your idea was already tried, pick something different."

    codex --full-auto exec "$prompt" 2>/dev/null
}

for ((i=1; i<=MAX_ITERS; i++)); do
    echo ""
    echo "============================================================"
    echo "ITERATION $i / $MAX_ITERS  |  Best: $BEST_SCORE"
    echo "============================================================"

    echo "Dispatching Codex agent..."
    run_codex_attempt "$i" || true

    # --- Check if anything changed ---
    if ! git diff --quiet -- research/experiment.py research/features_lab.py; then
        DESCRIPTION=$(get_change_description)
        echo "Changes detected: $DESCRIPTION"
        echo "Running gate..."
    else
        echo "No changes to experiment.py or features_lab.py. Knowledge iteration."
        CONSECUTIVE_KNOWLEDGE=$((CONSECUTIVE_KNOWLEDGE + 1))

        echo "" >> "$RESEARCH_LOG"
        echo "### Iteration $i -- KNOWLEDGE" >> "$RESEARCH_LOG"
        echo "No code changes." >> "$RESEARCH_LOG"

        if [ "$CONSECUTIVE_KNOWLEDGE" -ge "$MAX_CONSECUTIVE_KNOWLEDGE" ]; then
            echo "STOP: $MAX_CONSECUTIVE_KNOWLEDGE consecutive knowledge-only iterations. Search space may be exhausted."
            break
        fi
        continue
    fi

    CONSECUTIVE_KNOWLEDGE=0

    # --- Run gate ---
    GATE_OUTPUT=""
    if GATE_OUTPUT=$(bash research/gate.sh 2>&1); then
        SCORE=$(echo "$GATE_OUTPUT" | grep "^COMPOSITE_SCORE=" | tail -1 | cut -d= -f2)
        echo "Gate PASSED. Score: $SCORE"

        IMPROVED=$(python3 -c "print('yes' if float('$SCORE') > float('$BEST_SCORE') else 'no')")

        if [ "$IMPROVED" = "yes" ]; then
            DELTA=$(python3 -c "print(f'{float(\"$SCORE\") - float(\"$BEST_SCORE\"):+.4f}')")
            echo "NEW BEST: $SCORE ($DELTA)"

            git add research/experiment.py research/features_lab.py
            git commit -m "$(cat <<EOF
research(codex): iter $i, score=$SCORE ($DELTA)
EOF
)"
            BEST_SCORE="$SCORE"
            echo "$BEST_SCORE" > "$BEST_SCORE_FILE"
            CONSECUTIVE_FAILS=0

            printf "%d\t%s\t%s\t%s\n" "$i" "$SCORE" "keep" "$DESCRIPTION" >> "$RESULTS_TSV"

            echo "" >> "$RESEARCH_LOG"
            echo "### Iteration $i -- IMPROVED ($DELTA)" >> "$RESEARCH_LOG"
            echo "Score: $SCORE | $DESCRIPTION" >> "$RESEARCH_LOG"
            echo "Commit: $(git rev-parse --short HEAD)" >> "$RESEARCH_LOG"
        else
            DELTA=$(python3 -c "print(f'{float(\"$SCORE\") - float(\"$BEST_SCORE\"):+.4f}')")
            echo "No improvement: $SCORE ($DELTA). Reverting."

            printf "%d\t%s\t%s\t%s\n" "$i" "$SCORE" "discard" "$DESCRIPTION" >> "$RESULTS_TSV"

            echo "" >> "$COMBAT_LOG"
            echo "### Iteration $i -- REVERTED ($DELTA)" >> "$COMBAT_LOG"
            echo "Score: $SCORE vs best $BEST_SCORE" >> "$COMBAT_LOG"
            echo "Change: $DESCRIPTION" >> "$COMBAT_LOG"
            DIFF=$(git diff -- research/experiment.py research/features_lab.py | head -60)
            echo '```diff' >> "$COMBAT_LOG"
            echo "$DIFF" >> "$COMBAT_LOG"
            echo '```' >> "$COMBAT_LOG"

            git checkout -- research/experiment.py research/features_lab.py
            CONSECUTIVE_FAILS=$((CONSECUTIVE_FAILS + 1))

            echo "" >> "$RESEARCH_LOG"
            echo "### Iteration $i -- REVERTED ($DELTA)" >> "$RESEARCH_LOG"
            echo "Score: $SCORE | $DESCRIPTION" >> "$RESEARCH_LOG"
        fi
    else
        echo "Gate FAILED."
        echo "$GATE_OUTPUT" | tail -10

        FAIL_REASON=$(echo "$GATE_OUTPUT" | grep "GATE VIOLATION:" | head -1 || true)

        printf "%d\t%s\t%s\t%s\n" "$i" "NA" "crash" "$DESCRIPTION" >> "$RESULTS_TSV"

        echo "" >> "$COMBAT_LOG"
        echo "### Iteration $i -- GATE FAILED" >> "$COMBAT_LOG"
        echo "Reason: $FAIL_REASON" >> "$COMBAT_LOG"
        echo "Change: $DESCRIPTION" >> "$COMBAT_LOG"
        DIFF=$(git diff -- research/experiment.py research/features_lab.py | head -60)
        echo '```diff' >> "$COMBAT_LOG"
        echo "$DIFF" >> "$COMBAT_LOG"
        echo '```' >> "$COMBAT_LOG"
        TRACEBACK=$(echo "$GATE_OUTPUT" | tail -5)
        echo "Traceback:" >> "$COMBAT_LOG"
        echo '```' >> "$COMBAT_LOG"
        echo "$TRACEBACK" >> "$COMBAT_LOG"
        echo '```' >> "$COMBAT_LOG"

        git checkout -- research/experiment.py research/features_lab.py
        CONSECUTIVE_FAILS=$((CONSECUTIVE_FAILS + 1))

        echo "" >> "$RESEARCH_LOG"
        echo "### Iteration $i -- GATE FAILED" >> "$RESEARCH_LOG"
        echo "Reason: $FAIL_REASON" >> "$RESEARCH_LOG"
        echo "Change: $DESCRIPTION" >> "$RESEARCH_LOG"
    fi

    # --- Stop conditions ---
    if [ "$CONSECUTIVE_FAILS" -ge "$MAX_CONSECUTIVE_FAILS" ]; then
        echo "STOP: $MAX_CONSECUTIVE_FAILS consecutive failures. Circuit breaker."
        break
    fi

    # --- Trim combat log to last 30 entries ---
    ENTRY_COUNT=$(grep -c "^### " "$COMBAT_LOG" 2>/dev/null || echo 0)
    if [ "$ENTRY_COUNT" -gt 30 ]; then
        HEADER=$(head -3 "$COMBAT_LOG")
        KEEP_FROM=$(grep -n "^### " "$COMBAT_LOG" | tail -30 | head -1 | cut -d: -f1)
        { echo "$HEADER"; echo ""; tail -n +"$KEEP_FROM" "$COMBAT_LOG"; } > "$COMBAT_LOG.tmp"
        mv "$COMBAT_LOG.tmp" "$COMBAT_LOG"
    fi

    # --- Update plot ---
    uv run python research/plot.py 2>/dev/null || true
done

echo ""
echo "============================================================"
echo "AUTORESEARCH (Codex) COMPLETE"
echo "============================================================"
echo "Branch: $BRANCH"
echo "Final best score: $BEST_SCORE"
echo "See: $RESEARCH_LOG"
echo "See: $COMBAT_LOG"
echo "See: $RESULTS_TSV"
echo "Plot: research/progress.png"
if [ "$BRANCH" != "$STARTING_BRANCH" ]; then
    echo ""
    echo "To merge winning results: git checkout $STARTING_BRANCH && git merge $BRANCH"
fi

uv run python research/plot.py 2>/dev/null || true
