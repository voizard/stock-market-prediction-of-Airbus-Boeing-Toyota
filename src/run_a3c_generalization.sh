#!/bin/bash
# A3C generality test: six sector-diverse US stocks x 3 arms (baseline / nstep10 /
# nstep10boot), 20 seeds (100-119), 75k steps, train <= 2024-05-31, test >= 2024-06-01.
#
# This is the run that produced results/results_{lmt,cat,jpm,ko,nvda,xom}_a3c*.json.
# Same protocol as the AIR/TM replication (src/run_a3c_repl2.sh in the working
# checkout); only the tickers differ.
#
# NOTE ON PATHS: the training needs the working checkout, because env_runner.py and
# the course modules (aif_environment / aif_analysis) live there and are not part of
# this repository. Run this script from that checkout, e.g.:
#     cd <checkout>/aif-latex/aif_run && bash <repo>/src/run_a3c_generalization.sh
# The tickers must be registered in env_runner.CSV (see src/env_runner.py) and the
# 5Y CSVs must exist in the data directory (see src/data_update/ and README.md).
#
# The script is resume-safe: train_eval_a3c.py skips agents that already have a
# checkpoint and evaluates the test window for every agent that has no test block yet.
# Launched in three waves of six arms (two stocks each) to stay inside 64 CPUs / RAM.

RUN=/data/.openclaw/workspace/aif-latex/aif_run          # working checkout (env_runner, course)
TRAIN=/data/.openclaw/workspace/stock-market-prediction-of-Airbus-Boeing-Toyota/src/train_eval_a3c.py
cd "$RUN" || exit 1
: > "$RUN/batch8_status.txt"
echo "batch8 start $(date -u)" >> "$RUN/batch8_status.txt"

launch () {  # $1=ticker $2=label $3=logfile ; rest = env assignments
  local tk=$1 lbl=$2 log=$3; shift 3
  ( timeout 18000 env N_AGENTS=20 SEED_BASE=100 TIMESTEPS=75000 "$@" \
      python3 "$TRAIN" "$tk" >> "$RUN/$log" 2>&1
    echo "$lbl rc=$?" >> "$RUN/batch8_status.txt" ) &
}

wave () {  # $@ = two tickers -> six arms
  for tk in "$@"; do
    launch "$tk" "${tk}_base"        "log_${tk}_a3c_base20.txt"
    launch "$tk" "${tk}_nstep10"     "log_${tk}_a3c_nstep10.txt"     A3C_NSTEP=10 A3C_TAG=nstep10
    launch "$tk" "${tk}_nstep10boot" "log_${tk}_a3c_nstep10boot.txt" A3C_NSTEP=10 A3C_BOOT=5 A3C_BOOT_BLOCK=20 A3C_TAG=nstep10boot
  done
  wait
}

wave LMT CAT
wave JPM KO
wave NVDA XOM

echo "ALL DONE $(date -u)" >> "$RUN/batch8_status.txt"
cat "$RUN/batch8_status.txt"
echo "===== ANALYSIS (6 new stocks) ====="
python3 "$(dirname "$TRAIN")/compare_a3c_repl.py" LMT CAT JPM KO NVDA XOM
echo "===== ANALYSIS (all nine) ====="
python3 "$(dirname "$TRAIN")/compare_a3c_repl.py" BA AIR TM LMT CAT JPM KO NVDA XOM
