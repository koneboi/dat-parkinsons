#!/usr/bin/env bash
set -e
export PYTHONPATH=.
PY=.venv/bin/python
$PY - <<'EOF'
import sys, os
sys.path.insert(0, os.getcwd())
from pathlib import Path
from src.exp_preproc import build_cache
for variant, cube, center in [("str160","160","striatum"),
                               ("str120","120","striatum"),
                               ("brain160","160","brain")]:
    build_cache(Path("cache_mips_exp")/variant, Path("data/niftis"),
                int(cube), center)
print("ALL CACHE DONE")
EOF