#!/usr/bin/env bash
# Overnight agent runner — keeps the research agent alive until stopped.
# - Uses workspace config (max_cycles: -1 = run forever)
# - Auto-restarts if the agent exits (crash, API error, OOM, etc.)
# - Waits 30s between restarts to avoid hammering the API
# - All output to a rotating log
#
# Usage:
#   nohup bash run_agent_overnight.sh &        # start
#   pkill -f run_agent_overnight.sh            # stop
#
# Designed to survive the parent shell exiting (setsid + nohup + disown).

set -u

PROJECT="/home/bigboss/code/depth_estimation_unify_theory"
AGENT_DIR="/home/bigboss/code/auto_research_agent"
PYTHON="/home/bigboss/miniconda3/bin/python"
LOG="/home/bigboss/code/depth_estimation_unify_theory/overnight_runner.log"
MAX_RESTARTS=200          # safety cap (~many days of restarts)
RESTART_DELAY=30          # seconds between restarts

echo "[$(date '+%F %T')] === Overnight runner started (PID $$) ===" >> "$LOG"
echo "[$(date '+%F %T')] project=$PROJECT" >> "$LOG"
echo "[$(date '+%F %T')] log=$LOG" >> "$LOG"

restarts=0
while [ "$restarts" -lt "$MAX_RESTARTS" ]; do
    # Reset cycle counter only on the very first start, NOT on restarts —
    # we want the agent to resume from where it left off after a crash.
    if [ "$restarts" -eq 0 ]; then
        printf "0" > "$PROJECT/.cycle_counter"
        echo "[$(date '+%F %T')] first start: cycle counter reset to 0" >> "$LOG"
    fi

    echo "[$(date '+%F %T')] starting agent (attempt $((restarts+1))/$MAX_RESTARTS)" >> "$LOG"

    # Run agent with workspace config (max_cycles=-1 forever).
    # bash -l so it inherits ~/.bashrc API keys (placed before interactive guard).
    cd "$AGENT_DIR" || { echo "[$(date '+%F %T')] FATAL: cannot cd $AGENT_DIR" >> "$LOG"; sleep 60; restarts=$((restarts+1)); continue; }

    bash -lc "$PYTHON -u api.py run --project '$PROJECT'" >> "$LOG" 2>&1
    rc=$?

    echo "[$(date '+%F %T')] agent exited (rc=$rc), waiting ${RESTART_DELAY}s before restart..." >> "$LOG"

    # Record current state snapshot for morning inspection
    {
        echo "--- state snapshot after exit ---"
        cat "$PROJECT/state.json" 2>/dev/null | python3 -c "import sys,json; d=json.load(sys.stdin); print('cycle:',d.get('cycle'),'status:',d.get('status'),'decision:',d.get('last_decision','')[:100],'error:',d.get('last_error','')[:120])" 2>/dev/null
    } >> "$LOG"

    sleep "$RESTART_DELAY"
    restarts=$((restarts+1))
done

echo "[$(date '+%F %T')] reached MAX_RESTARTS ($MAX_RESTARTS), stopping runner" >> "$LOG"
