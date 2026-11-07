#!/usr/bin/env bash
set -e
python3 run_solution.py "$@" 2>/dev/null || python run_solution.py "$@"
