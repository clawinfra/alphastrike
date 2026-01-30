#!/bin/bash
# AlphaStrike Autonomous Agent Loop
# Usage: ./scripts/ralph/loop.sh [max_iterations]

set -e

MAX_ITERATIONS=${1:-25}
ITERATION=0
PROJECT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"

cd "$PROJECT_DIR"

echo "========================================"
echo "AlphaStrike Autonomous Agent Loop"
echo "Max iterations: $MAX_ITERATIONS"
echo "Project: $PROJECT_DIR"
echo "========================================"

# Check prerequisites
if ! command -v claude &> /dev/null; then
    echo "Error: claude CLI not found"
    exit 1
fi

if ! command -v uv &> /dev/null; then
    echo "Error: uv not found"
    exit 1
fi

# Initialize git if needed
if [ ! -d ".git" ]; then
    git init
    git add .
    git commit -m "Initial commit: project structure and documentation"
fi

# Create branch if needed
BRANCH_NAME=$(cat prd.json | grep -o '"branchName": *"[^"]*"' | cut -d'"' -f4)
if [ -n "$BRANCH_NAME" ]; then
    git checkout -B "$BRANCH_NAME" 2>/dev/null || git checkout "$BRANCH_NAME"
fi

while [ $ITERATION -lt $MAX_ITERATIONS ]; do
    ITERATION=$((ITERATION + 1))
    echo ""
    echo "========================================"
    echo "Iteration $ITERATION of $MAX_ITERATIONS"
    echo "Time: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "========================================"

    # Check if all stories complete
    REMAINING=$(cat prd.json | grep -c '"passes": false' || echo "0")
    if [ "$REMAINING" -eq 0 ]; then
        echo "All stories complete! Exiting."
        break
    fi
    echo "Remaining stories: $REMAINING"

    # Run claude with agent instructions
    OUTPUT=$(claude --print "Read scripts/ralph/CLAUDE.md for instructions. Then check handoff.json if it exists. Otherwise read prd.json and implement the highest priority story where passes=false. Work autonomously until the story is complete or you need to hand off." 2>&1) || true

    # Check for completion signal
    if echo "$OUTPUT" | grep -q "<promise>COMPLETE</promise>"; then
        echo "Agent signaled COMPLETE!"
        break
    fi

    # Check for handoff signal
    if echo "$OUTPUT" | grep -q "<handoff>CONTEXT_THRESHOLD</handoff>"; then
        echo "Agent requested context handoff. Starting fresh instance..."
        sleep 2
        continue
    fi

    # Small delay between iterations
    sleep 5
done

echo ""
echo "========================================"
echo "Loop completed after $ITERATION iterations"
echo "========================================"

# Final status
echo ""
echo "Final story status:"
cat prd.json | grep -E '"id"|"title"|"passes"' | head -60
