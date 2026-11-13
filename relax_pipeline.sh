#!/usr/bin/env bash
# Usage: bash relax_pipeline.sh raw.pdb
#
# Caps chains (ACE/NME), assigns protonation (PROPKA), adds hydrogens, removes
# severe clashes if any, minimizes hydrogens, minimizes all atoms, then runs a
# final geometry check. Each run gets a fresh directory under pipeline_runs/
# beside the input. Exit 2 means a stage needs review; later stages don't run.
#
# Optional environment variables (default):
#   PYTHON          interpreter (python, else python3)
#   PH              target pH (7.0)
#   PLATFORM        auto, CPU, CUDA, HIP or Metal (auto)
#   THREADS         CPU threads (2)
#   MAX_ITERATIONS  iterations per minimization (2000)
#   SAVE_EVERY      iterations between animation frames (5)
#   TOLERANCE       RMS force target in kJ/mol/nm (10)
#   RUNS_DIR        parent directory for runs (pipeline_runs/ beside the input)
#
# Example on a GPU pod (fail instead of falling back to CPU):
#   PLATFORM=CUDA bash relax_pipeline.sh raw.pdb
set -euo pipefail

# --help prints the comment block above.
usage() { awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "$0"; }
if [[ ${1:-} == -h || ${1:-} == --help ]]; then usage; exit 0; fi
if [[ $# -ne 1 ]]; then usage >&2; exit 1; fi
if [[ ! -f $1 ]]; then echo "ERROR: no such file: $1" >&2; exit 1; fi

SCRIPT_DIR=$(cd -- "$(dirname -- "$0")" && pwd)
PYTHON=${PYTHON:-$(command -v python || echo python3)}
PH=${PH:-7.0}
PLATFORM=${PLATFORM:-auto}
THREADS=${THREADS:-2}
MAX_ITERATIONS=${MAX_ITERATIONS:-2000}
SAVE_EVERY=${SAVE_EVERY:-5}
TOLERANCE=${TOLERANCE:-10}
# Checked here so a typo fails now rather than after stages 1-3.
case $PLATFORM in auto|CPU|CUDA|HIP|Metal) ;; *) echo "ERROR: invalid PLATFORM: $PLATFORM" >&2; exit 1 ;; esac
# Settings shared by the three minimization stages.
MINIMIZE=(--platform "$PLATFORM" --threads "$THREADS" --max-iterations "$MAX_ITERATIONS"
          --save-every "$SAVE_EVERY" --tolerance "$TOLERANCE")
# Each stage's report records its own script hash; this records the pipeline's.
PIPELINE_SHA256=$("$PYTHON" -c 'import hashlib, sys; print(hashlib.sha256(open(sys.argv[1], "rb").read()).hexdigest())' "$0")

NAME=$(basename -- "$1")
INPUT=$(cd -- "$(dirname -- "$1")" && pwd)/$NAME
RUNS_DIR=${RUNS_DIR:-$(dirname -- "$INPUT")/pipeline_runs}
mkdir -p -- "$RUNS_DIR"
RUNS_DIR=$(cd -- "$RUNS_DIR" && pwd)
RUN_DIR=$(mktemp -d "$RUNS_DIR/${NAME%.*}_$(date -u +%Y%m%dT%H%M%SZ)_XXXXXX")
LOG=$RUN_DIR/pipeline.log
log() { printf '%s\n' "$*" | tee -a "$LOG"; }

log "Run directory: $RUN_DIR"
log "Input: $INPUT"
log "Pipeline script sha256: $PIPELINE_SHA256"
log "Settings: pH=$PH; platform=$PLATFORM; threads=$THREADS; iteration cap=$MAX_ITERATIONS; RMS tolerance=$TOLERANCE kJ/mol/nm; frames every $SAVE_EVERY iterations"
# Keep the exact submitted structure for provenance. All stages use this copy.
cp -- "$INPUT" "$RUN_DIR/raw_input.pdb"

run_stage() {
    local label=$1 script=$SCRIPT_DIR/$2
    shift 2
    # Python exits 2 for a missing script, which would otherwise read as "needs review".
    [[ -f $script ]] || { log "[$label] FAILED: missing script $script"; exit 1; }
    log ""
    log "[$label] Starting at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    if "$PYTHON" -u "$script" "$@" 2>&1 | tee -a "$LOG"; then
        log "[$label] Passed."
    else
        local rc=$?  # pipefail makes this Python's exit code, not tee's
        if [[ $rc -eq 2 ]]; then
            log "[$label] NEEDS REVIEW. Read the stage report before proceeding."
        else
            log "[$label] FAILED (exit $rc). See the error above."
        fi
        exit "$rc"
    fi
}

run_stage '1/6 Capping' cap_generic.py \
    --input "$RUN_DIR/raw_input.pdb" --output "$RUN_DIR/capped.pdb" --strip-hydrogens
run_stage '2/6 Protonation assignment' assign_protonation_generic.py \
    --input "$RUN_DIR/capped.pdb" --ph "$PH" --output-prefix "$RUN_DIR/protonation"
run_stage '3/6 Hydrogen addition' add_hydrogens_generic.py \
    --input "$RUN_DIR/capped.pdb" --output "$RUN_DIR/hydrogenated.pdb" \
    --ph "$PH" --variants "$RUN_DIR/protonation_variants.json"
run_stage '4/6 Conditional clash cleanup' remove_clashes_generic.py \
    --input "$RUN_DIR/hydrogenated.pdb" --output "$RUN_DIR/clash_cleaned.pdb" "${MINIMIZE[@]}"
log '[Cleanup] If repair ran, PROPKA assignments came from overlapping geometry; reassess affected protonation states on the cleaned structure if they matter.'
run_stage '5/6 Hydrogen relaxation' relax_hydrogens_generic.py \
    --input "$RUN_DIR/clash_cleaned.pdb" --output "$RUN_DIR/hydrogen_relaxed.pdb" "${MINIMIZE[@]}"
run_stage '6/6 Full-atom relaxation' relax_full_structure_generic.py \
    --input "$RUN_DIR/hydrogen_relaxed.pdb" --output "$RUN_DIR/relaxed.pdb" "${MINIMIZE[@]}"
run_stage 'Final geometry check' validate_clash_output.py \
    --input "$RUN_DIR/relaxed.pdb" --cleanup-report "$RUN_DIR/clash_cleaned.json" \
    --output "$RUN_DIR/geometry_gate.json"

log ""
log "SUCCESS: all stages passed. This is energy minimization, not a thermodynamic stability test."
log "Outputs in $RUN_DIR: relaxed.pdb (structure), relaxed_frames.pdb (animation), relaxed.json (report), geometry_gate.json, pipeline.log"
