#!/usr/bin/env bash
# Usage: bash relax_pipeline.sh raw.pdb
# Optional environment settings: PYTHON, PH, PLATFORM, THREADS,
# MAX_ITERATIONS, SAVE_EVERY, TOLERANCE, RUNS_DIR.
# Requires the five generic Python scripts and preparation_checks.py beside it.
# A fresh run directory is always created. Exit 2 means a stage needs review.
set -euo pipefail

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    cat <<'HELP'
Usage: bash relax_pipeline.sh raw.pdb

Caps each continuous standard-protein chain with ACE/NME, assigns protonation
with PROPKA, adds hydrogens, minimizes hydrogens, then minimizes all atoms.
Any input hydrogens are discarded and reassigned; OXT is replaced by the cap.
Already-capped inputs and broken chains are rejected by the capping stage.
Default: pH 7.0, ff14SB/GBn2, 0.15 M implicit ionic strength, 298.15 K screening,
GPU auto-selection with CPU fallback, force tolerance 1 kJ/mol/nm,
2000 iterations per minimization, animation frames every 5 iterations.
These are provisional modeling settings, not a validated biological protocol.

Outputs: a unique directory under pipeline_runs/ beside the input, containing
intermediates, protonation reports, logs, relaxed.pdb, relaxed_frames.pdb and
relaxed.json. Success is printed only when every stage passes. Exit 2 means
review is required; existing candidates/reports are retained and later stages
are not run. Re-running creates a fresh directory (does not resume).

Optional environment variables:
  PYTHON           Python interpreter (default: python, or python3 if absent)
  PH               target pH (default: 7.0)
  PLATFORM         auto, CPU, CUDA, HIP, Metal (default: auto; minimization only)
  THREADS          CPU threads (default: 2)
  MAX_ITERATIONS   maximum iterations per minimization (default: 2000)
  SAVE_EVERY       optimization-frame interval (default: 5)
  TOLERANCE        component RMS force target in kJ/mol/nm (default: 1.0)
  RUNS_DIR         override parent directory for runs

Example on a GPU pod (fail rather than fall back to CPU during minimization):
  PLATFORM=CUDA bash relax_pipeline.sh raw.pdb
HELP
    exit 0
fi
if [[ $# -ne 1 ]]; then
    printf 'Usage: bash %s raw.pdb\n' "$0" >&2
    exit 1
fi
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
if [[ -z "${PYTHON:-}" ]]; then
    if command -v python >/dev/null 2>&1; then PYTHON=python; else PYTHON=python3; fi
fi
if ! command -v "$PYTHON" >/dev/null 2>&1; then
    printf 'ERROR: Python interpreter not found: %s\n' "$PYTHON" >&2
    exit 1
fi
PH=${PH:-7.0}
PLATFORM=${PLATFORM:-auto}
THREADS=${THREADS:-2}
MAX_ITERATIONS=${MAX_ITERATIONS:-2000}
SAVE_EVERY=${SAVE_EVERY:-5}
TOLERANCE=${TOLERANCE:-1.0}

printf '[Preflight] Checking input, dependencies, and settings...\n'
INPUT=$("$PYTHON" - "$1" "$PH" "$PLATFORM" "$THREADS" "$MAX_ITERATIONS" "$SAVE_EVERY" "$TOLERANCE" <<'PY'
from pathlib import Path
import sys, math
p=Path(sys.argv[1]).expanduser().resolve()
if not p.is_file() or p.suffix.lower()!='.pdb':
    raise SystemExit('ERROR: supply an existing .pdb file')
try:
    import openmm, numpy, propka.run
except ImportError as exc:
    raise SystemExit(f'ERROR: missing dependency: {exc}. Install OpenMM, numpy, and propka==3.5.1 into the selected Python interpreter.')
ph=float(sys.argv[2]);tol=float(sys.argv[7])
if not math.isfinite(ph) or not 0<=ph<=14: raise SystemExit('ERROR: PH must be between 0 and 14')
if sys.argv[3] not in ('auto','CPU','CUDA','HIP','Metal'):raise SystemExit('ERROR: invalid PLATFORM')
if any(int(x)<=0 for x in sys.argv[4:7]):raise SystemExit('ERROR: thread/iteration/frame settings must be positive integers')
if not math.isfinite(tol) or tol<=0:raise SystemExit('ERROR: invalid TOLERANCE')
print(p)
PY
)
for file in cap_generic.py assign_protonation_generic.py add_hydrogens_generic.py relax_hydrogens_generic.py relax_full_structure_generic.py preparation_checks.py; do
    if [[ ! -f "$SCRIPT_DIR/$file" ]]; then
        printf 'ERROR: required script is missing: %s/%s\n' "$SCRIPT_DIR" "$file" >&2
        exit 1
    fi
done
INPUT_DIR=$(dirname -- "$INPUT")
INPUT_NAME=$(basename -- "$INPUT")
STEM=${INPUT_NAME%.*}
RUNS_DIR=${RUNS_DIR:-"$INPUT_DIR/pipeline_runs"}
mkdir -p -- "$RUNS_DIR"
RUNS_DIR=$(cd -- "$RUNS_DIR" && pwd)
RUN_DIR=$(mktemp -d "$RUNS_DIR/${STEM}_$(date -u +%Y%m%dT%H%M%SZ)_XXXXXX")
LOG="$RUN_DIR/pipeline.log"
log() { printf '%s\n' "$*" | tee -a "$LOG"; }
CURRENT_STAGE=initialization
finish() {
    local rc=$?
    trap - EXIT
    if [[ $rc -ne 0 ]]; then
        printf '\nSTOPPED at %s (exit %s). Outputs retained: %s\n' "$CURRENT_STAGE" "$rc" "$RUN_DIR" | tee -a "$LOG" >&2 || true
    fi
}
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

log "Run directory: $RUN_DIR"
log "Input: $INPUT"
log "Settings: pH=$PH; ff14SB + GBn2; salt=0.15 M; dielectric=1/78.3; screening temperature=298.15 K"
log "Minimization: platform=$PLATFORM; CPU threads=$THREADS; iteration cap=$MAX_ITERATIONS; RMS tolerance=$TOLERANCE; frames every $SAVE_EVERY iterations"
log 'Preparation policy: replace existing hydrogens, cap all chains, then assign protonation from the capped assembly.'

# Keep the exact submitted structure for provenance. All stages use this copy.
cp -- "$INPUT" "$RUN_DIR/raw_input.pdb"
"$PYTHON" - "$INPUT" "$RUN_DIR" "$PH" "$PLATFORM" "$THREADS" "$MAX_ITERATIONS" "$SAVE_EVERY" "$TOLERANCE" "$SCRIPT_DIR" <<'PY'
from pathlib import Path
import hashlib,json,sys,datetime,importlib.metadata
source,run=Path(sys.argv[1]),Path(sys.argv[2]);scripts=Path(sys.argv[9])
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
data=dict(source=str(source),input_sha256=sha(run/'raw_input.pdb'),
          started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
          ph=float(sys.argv[3]),platform=sys.argv[4],threads=int(sys.argv[5]),
          max_iterations=int(sys.argv[6]),save_every=int(sys.argv[7]),tolerance=float(sys.argv[8]),
          python=sys.executable,versions={x:importlib.metadata.version(x) for x in ('OpenMM','numpy','propka')},
          scripts={name:sha(scripts/name) for name in ('relax_pipeline.sh','cap_generic.py','assign_protonation_generic.py','add_hydrogens_generic.py','relax_hydrogens_generic.py','relax_full_structure_generic.py','preparation_checks.py')})
(run/'run_manifest.json').write_text(json.dumps(data,indent=2)+'\n')
PY

run_stage() {
    local label=$1
    shift
    CURRENT_STAGE=$label
    log ""
    log "[$label] Starting at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    if "$PYTHON" -u "$@" 2>&1 | tee -a "$LOG"; then
        log "[$label] Passed."
    else
        local codes=("${PIPESTATUS[@]}")
        local rc=${codes[0]}
        if [[ $rc -eq 0 ]]; then rc=${codes[1]}; fi
        if [[ $rc -eq 2 ]]; then
            log "[$label] NEEDS REVIEW. Read the stage report before proceeding."
        else
            log "[$label] FAILED (exit $rc). See the error above."
        fi
        exit "$rc"
    fi
}

run_stage '1/5 Capping' "$SCRIPT_DIR/cap_generic.py" \
    --input "$RUN_DIR/raw_input.pdb" --output "$RUN_DIR/capped.pdb" --strip-hydrogens
run_stage '2/5 Protonation assignment' "$SCRIPT_DIR/assign_protonation_generic.py" \
    --input "$RUN_DIR/capped.pdb" --ph "$PH" --output-prefix "$RUN_DIR/protonation"
log '[Protonation] These are provisional assignments. Uncertain sites and alternative cases are recorded in the reports; the baseline is used below.'
run_stage '3/5 Hydrogen addition' "$SCRIPT_DIR/add_hydrogens_generic.py" \
    --input "$RUN_DIR/capped.pdb" --output "$RUN_DIR/hydrogenated.pdb" \
    --ph "$PH" --variants "$RUN_DIR/protonation_variants.json"
run_stage '4/5 Hydrogen relaxation' "$SCRIPT_DIR/relax_hydrogens_generic.py" \
    --input "$RUN_DIR/hydrogenated.pdb" --output "$RUN_DIR/hydrogen_relaxed.pdb" \
    --platform "$PLATFORM" --threads "$THREADS" --max-iterations "$MAX_ITERATIONS" \
    --save-every "$SAVE_EVERY" --tolerance "$TOLERANCE"
run_stage '5/5 Full-atom relaxation' "$SCRIPT_DIR/relax_full_structure_generic.py" \
    --input "$RUN_DIR/hydrogen_relaxed.pdb" --output "$RUN_DIR/relaxed.pdb" \
    --platform "$PLATFORM" --threads "$THREADS" --max-iterations "$MAX_ITERATIONS" \
    --save-every "$SAVE_EVERY" --tolerance "$TOLERANCE"
CURRENT_STAGE='Final verification'
"$PYTHON" - "$RUN_DIR" <<'PY'
from pathlib import Path
import json,sys,hashlib
p=Path(sys.argv[1])
for name,expected in [('hydrogen_relaxed','hydrogen_minimization_passed'),('relaxed','full_atom_minimization_passed')]:
    d=json.loads((p/(name+'.json')).read_text())
    if d.get('status')!=expected or not d.get('converged'):raise SystemExit('ERROR: final report does not confirm convergence: '+name)
    for field in ('output','trajectory'):
        if hashlib.sha256(Path(d[field]).read_bytes()).hexdigest()!=d[field+'_sha256']:
            raise SystemExit('ERROR: output hash mismatch: '+field)
PY
log ''
log 'SUCCESS: all five stages passed. This is energy minimization, not a thermodynamic stability test.'
log "Relaxed structure: $RUN_DIR/relaxed.pdb"
log "Animation: $RUN_DIR/relaxed_frames.pdb"
log "Report: $RUN_DIR/relaxed.json"
log "Full log: $LOG"
