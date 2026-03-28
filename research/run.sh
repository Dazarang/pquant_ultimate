#!/usr/bin/env bash
# run.sh -- Autoresearch outer loop.
#
# Usage:
#   ./research/run.sh              # Run on current branch (default 20 iterations)
#   ./research/run.sh 50           # Run 50 iterations
#   ./research/run.sh 20 --branch  # Run on a new autoresearch/ branch
#   ./research/run.sh manual       # Manual mode (you edit, it gates)
#
# Each iteration:
#   1. Agent edits experiment.py and/or features_lab.py
#   2. gate.sh verifies and runs
#   3. If score improves: commit. If not: revert + log to COMBAT_LOG.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

MAX_ITERS="${1:-20}"
MANUAL_MODE=false
if [ "${1:-}" = "manual" ]; then
    MANUAL_MODE=true
    MAX_ITERS=999
fi

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
MAX_CONSECUTIVE_FAILS=10
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
    # Extract added lines (skip diff headers), take first meaningful ones
    local desc
    desc=$(echo "$diff" | grep "^+" | grep -v "^+++" | head -5 | sed 's/^+//' | tr '\n' ' ' | sed 's/[[:space:]]\+/ /g' | cut -c1-80)
    if [ -z "$desc" ]; then
        desc="unknown change"
    fi
    # Strip tabs to avoid TSV corruption
    echo "$desc" | tr '\t' ' '
}

echo "============================================================"
echo "AUTORESEARCH"
echo "============================================================"
echo "Best score: $BEST_SCORE"
echo "Max iterations: $MAX_ITERS"
echo "Mode: $([ "$MANUAL_MODE" = true ] && echo 'manual' || echo 'agent')"
echo ""

run_claude_attempt() {
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

Make ONE focused change. Commit nothing -- gate.sh handles that.
After editing, stop. The outer loop will run gate.sh.

If COMBAT_LOG.md shows your idea was already tried, pick something different."

    env -u CLAUDECODE claude -p --dangerously-skip-permissions --effort max --model opus "$prompt"
}

for ((i=1; i<=MAX_ITERS; i++)); do
    echo ""
    echo "============================================================"
    echo "ITERATION $i / $MAX_ITERS  |  Best: $BEST_SCORE"
    echo "============================================================"

    # --- Agent or manual edit ---
    if [ "$MANUAL_MODE" = true ]; then
        echo "Edit research/experiment.py and/or research/features_lab.py, then press Enter."
        read -r
    else
        echo "Dispatching Claude Code agent..."
        run_claude_attempt "$i" || true
    fi

    # --- Check if anything changed ---
    if ! git diff --quiet -- research/experiment.py research/features_lab.py; then
        DESCRIPTION=$(get_change_description)
        echo "Changes detected: $DESCRIPTION"
        echo "Running gate..."
    else
        echo "No changes to experiment.py or features_lab.py. Knowledge iteration."
        CONSECUTIVE_KNOWLEDGE=$((CONSECUTIVE_KNOWLEDGE + 1))

        # Log
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
        # Gate passed -- extract score
        SCORE=$(echo "$GATE_OUTPUT" | grep "^COMPOSITE_SCORE=" | tail -1 | cut -d= -f2)
        echo "Gate PASSED. Score: $SCORE"

        # Compare to best
        IMPROVED=$(python3 -c "print('yes' if float('$SCORE') > float('$BEST_SCORE') else 'no')")

        if [ "$IMPROVED" = "yes" ]; then
            DELTA=$(python3 -c "print(f'{float(\"$SCORE\") - float(\"$BEST_SCORE\"):+.4f}')")
            echo "NEW BEST: $SCORE ($DELTA)"

            # Commit winning experiment
            git add research/experiment.py research/features_lab.py
            git commit -m "$(cat <<EOF
research: iter $i, score=$SCORE ($DELTA)
EOF
)"
            BEST_SCORE="$SCORE"
            echo "$BEST_SCORE" > "$BEST_SCORE_FILE"
            CONSECUTIVE_FAILS=0

            # TSV log
            printf "%d\t%s\t%s\t%s\n" "$i" "$SCORE" "keep" "$DESCRIPTION" >> "$RESULTS_TSV"

            # Markdown log
            echo "" >> "$RESEARCH_LOG"
            echo "### Iteration $i -- IMPROVED ($DELTA)" >> "$RESEARCH_LOG"
            echo "Score: $SCORE | $DESCRIPTION" >> "$RESEARCH_LOG"
            echo "Commit: $(git rev-parse --short HEAD)" >> "$RESEARCH_LOG"
        else
            DELTA=$(python3 -c "print(f'{float(\"$SCORE\") - float(\"$BEST_SCORE\"):+.4f}')")
            echo "No improvement: $SCORE ($DELTA). Reverting."

            # TSV log
            printf "%d\t%s\t%s\t%s\n" "$i" "$SCORE" "discard" "$DESCRIPTION" >> "$RESULTS_TSV"

            # Combat log (before revert)
            echo "" >> "$COMBAT_LOG"
            echo "### Iteration $i -- REVERTED ($DELTA)" >> "$COMBAT_LOG"
            echo "Score: $SCORE vs best $BEST_SCORE" >> "$COMBAT_LOG"
            echo "Change: $DESCRIPTION" >> "$COMBAT_LOG"
            DIFF=$(git diff -- research/experiment.py research/features_lab.py | head -60)
            echo '```diff' >> "$COMBAT_LOG"
            echo "$DIFF" >> "$COMBAT_LOG"
            echo '```' >> "$COMBAT_LOG"

            # Revert
            git checkout -- research/experiment.py research/features_lab.py
            CONSECUTIVE_FAILS=$((CONSECUTIVE_FAILS + 1))

            # Markdown log
            echo "" >> "$RESEARCH_LOG"
            echo "### Iteration $i -- REVERTED ($DELTA)" >> "$RESEARCH_LOG"
            echo "Score: $SCORE | $DESCRIPTION" >> "$RESEARCH_LOG"
        fi
    else
        # Gate failed
        echo "Gate FAILED."
        echo "$GATE_OUTPUT" | tail -10

        # TSV log
        printf "%d\t%s\t%s\t%s\n" "$i" "0" "crash" "$DESCRIPTION" >> "$RESULTS_TSV"

        # Revert
        git checkout -- research/experiment.py research/features_lab.py
        CONSECUTIVE_FAILS=$((CONSECUTIVE_FAILS + 1))

        # Markdown log
        echo "" >> "$RESEARCH_LOG"
        echo "### Iteration $i -- GATE FAILED" >> "$RESEARCH_LOG"
        FAIL_REASON=$(echo "$GATE_OUTPUT" | grep "GATE VIOLATION:" | head -1)
        echo "Reason: $FAIL_REASON" >> "$RESEARCH_LOG"
        echo "Change: $DESCRIPTION" >> "$RESEARCH_LOG"
    fi

    # --- Stop conditions ---
    if [ "$CONSECUTIVE_FAILS" -ge "$MAX_CONSECUTIVE_FAILS" ]; then
        echo "STOP: $MAX_CONSECUTIVE_FAILS consecutive failures. Circuit breaker."
        break
    fi

    # --- Update plot ---
    uv run python research/plot.py 2>/dev/null || true
done

echo ""
echo "============================================================"
echo "AUTORESEARCH COMPLETE"
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

# Final plot
uv run python research/plot.py 2>/dev/null || true
