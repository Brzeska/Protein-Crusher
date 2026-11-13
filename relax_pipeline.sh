#!/usr/bin/env bash
# Usage: bash relax_pipeline.sh raw.pdb
#
# Relaxes one structure under several sampled protonation states.
#
#   Setup (once):
#     1. Cap chains with ACE/NME.
#     2. Draw NSAMPLES protonation microstates from Henderson-Hasselbalch
#        fractions at pH PH (sample_protonation.py).
#   Then, for each sample, in its own sample_NNN/ folder:
#     add hydrogens -> remove severe clashes if any -> relax hydrogens
#       -> relax all atoms -> final geometry check
#
# Each run gets a fresh directory under pipeline_runs/ beside the input.
# If a sample fails or needs review, the remaining samples still run; the
# summary at the end lists every sample's outcome (also in samples_summary.tsv).
# Samples are probability-weighted draws: average their results equally.
#
# Exit status: 0 every sample passed; 2 a stage needs review (setup, or a
# sample, with none failed); 1 a stage failed.
#
# Optional environment variables (default):
#   PYTHON          interpreter (python, else python3)
#   PH              target pH (7.0)
#   NSAMPLES        protonation samples to draw and relax (10)
#   SEED            random seed for protonation sampling (2026)
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

# ---- Settings ---------------------------------------------------------------
SCRIPT_DIR=$(cd -- "$(dirname -- "$0")" && pwd)
PYTHON=${PYTHON:-$(command -v python || echo python3)}
PH=${PH:-7.0}
NSAMPLES=${NSAMPLES:-10}
SEED=${SEED:-2026}
PLATFORM=${PLATFORM:-auto}
THREADS=${THREADS:-2}
MAX_ITERATIONS=${MAX_ITERATIONS:-2000}
SAVE_EVERY=${SAVE_EVERY:-5}
TOLERANCE=${TOLERANCE:-10}

# Check settings now, so a typo fails before any work is done.
case $PLATFORM in auto|CPU|CUDA|HIP|Metal) ;; *) echo "ERROR: invalid PLATFORM: $PLATFORM" >&2; exit 1 ;; esac
if [[ ! $NSAMPLES =~ ^[1-9][0-9]*$ ]]; then echo "ERROR: NSAMPLES must be a positive integer: $NSAMPLES" >&2; exit 1; fi
if [[ ! $SEED =~ ^[0-9]+$ ]]; then echo "ERROR: SEED must be a non-negative integer: $SEED" >&2; exit 1; fi

# Settings shared by the three minimization stages.
MINIMIZE=(--platform "$PLATFORM" --threads "$THREADS" --max-iterations "$MAX_ITERATIONS"
          --save-every "$SAVE_EVERY" --tolerance "$TOLERANCE")

# ---- Run directory and logging ----------------------------------------------
NAME=$(basename -- "$1")
INPUT=$(cd -- "$(dirname -- "$1")" && pwd)/$NAME
RUNS_DIR=${RUNS_DIR:-$(dirname -- "$INPUT")/pipeline_runs}
mkdir -p -- "$RUNS_DIR"
RUNS_DIR=$(cd -- "$RUNS_DIR" && pwd)
RUN_DIR=$(mktemp -d "$RUNS_DIR/${NAME%.*}_$(date -u +%Y%m%dT%H%M%SZ)_XXXXXX")
LOG=$RUN_DIR/pipeline.log
SUMMARY=$RUN_DIR/samples_summary.tsv

log() { printf '%s\n' "$*" | tee -a "$LOG"; }
now() { date -u +%Y-%m-%dT%H:%M:%SZ; }

# Each stage's report records its own script hash; this records the pipeline's.
PIPELINE_SHA256=$("$PYTHON" -c 'import hashlib, sys; print(hashlib.sha256(open(sys.argv[1], "rb").read()).hexdigest())' "$0")

log "Run directory: $RUN_DIR"
log "Input: $INPUT"
log "Pipeline script sha256: $PIPELINE_SHA256"
log "Settings: pH=$PH; samples=$NSAMPLES; seed=$SEED; platform=$PLATFORM; threads=$THREADS; iteration cap=$MAX_ITERATIONS; RMS tolerance=$TOLERANCE kJ/mol/nm; frames every $SAVE_EVERY iterations"
# Keep the exact submitted structure for provenance. All stages use this copy.
cp -- "$INPUT" "$RUN_DIR/raw_input.pdb"

# ---- Stage runner -----------------------------------------------------------
# Runs one Python stage and logs its output. Returns the script's exit code:
# 0 passed, 2 needs review, anything else failed.
run_stage() {
    local label=$1 script=$SCRIPT_DIR/$2
    shift 2
    # Python exits 2 for a missing script, which would otherwise read as "needs review".
    if [[ ! -f $script ]]; then log "[$label] FAILED: missing script $script"; return 1; fi
    log ""
    log "[$label] Starting at $(now)"
    local rc=0
    "$PYTHON" -u "$script" "$@" 2>&1 | tee -a "$LOG" || rc=$?  # pipefail: rc is Python's, not tee's
    case $rc in
        0) log "[$label] Passed." ;;
        2) log "[$label] NEEDS REVIEW. Read the stage report." ;;
        *) log "[$label] FAILED (exit $rc). See the error above." ;;
    esac
    return "$rc"
}

# Runs every per-sample stage in one sample folder. Stops at the first stage
# that does not pass and returns that stage's exit code.
relax_sample() {
    local dir=$1
    local name
    name=$(basename -- "$dir")
    run_stage "$name 1/5 Hydrogen addition" add_hydrogens_generic.py \
        --input "$RUN_DIR/capped.pdb" --output "$dir/hydrogenated.pdb" \
        --ph "$PH" --variants "$dir/variants.json" || return
    run_stage "$name 2/5 Conditional clash cleanup" remove_clashes_generic.py \
        --input "$dir/hydrogenated.pdb" --output "$dir/clash_cleaned.pdb" "${MINIMIZE[@]}" || return
    run_stage "$name 3/5 Hydrogen relaxation" relax_hydrogens_generic.py \
        --input "$dir/clash_cleaned.pdb" --output "$dir/hydrogen_relaxed.pdb" "${MINIMIZE[@]}" || return
    run_stage "$name 4/5 Full-atom relaxation" relax_full_structure_generic.py \
        --input "$dir/hydrogen_relaxed.pdb" --output "$dir/relaxed.pdb" "${MINIMIZE[@]}" || return
    run_stage "$name 5/5 Final geometry check" validate_clash_output.py \
        --input "$dir/relaxed.pdb" --cleanup-report "$dir/clash_cleaned.json" \
        --output "$dir/geometry_gate.json" || return
}

# ---- Setup: runs once; a problem here stops the whole run -------------------
run_stage 'Setup 1/2 Capping' cap_generic.py \
    --input "$RUN_DIR/raw_input.pdb" --output "$RUN_DIR/capped.pdb" --strip-hydrogens || exit
run_stage 'Setup 2/2 Protonation sampling' sample_protonation.py \
    --input "$RUN_DIR/capped.pdb" --ph "$PH" --samples "$NSAMPLES" --seed "$SEED" \
    --output-dir "$RUN_DIR" || exit

# ---- Samples: a problem in one sample does not stop the others --------------
passed=0 review=0 failed=0
printf 'sample\tstatus\tfolder\n' > "$SUMMARY"
for dir in "$RUN_DIR"/sample_*/; do
    [[ -d $dir ]] || continue  # the pattern stays literal if nothing matched
    dir=${dir%/}
    rc=0
    relax_sample "$dir" || rc=$?
    case $rc in
        0) status=passed;       passed=$((passed + 1)) ;;
        2) status=needs_review; review=$((review + 1)) ;;
        *) status=failed;       failed=$((failed + 1)) ;;
    esac
    printf '%s\t%s\t%s\n' "$(basename -- "$dir")" "$status" "$dir" >> "$SUMMARY"
done

# ---- Summary ----------------------------------------------------------------
log ""
log "Samples: $passed passed, $review need review, $failed failed (of $NSAMPLES)."
while IFS=$'\t' read -r sample status _; do
    [[ $sample == sample ]] || log "  $sample: $status"
done < "$SUMMARY"
log "Each sample folder holds relaxed.pdb (structure), relaxed_frames.pdb (animation), relaxed.json (report), geometry_gate.json."
log "Protonation draws: $RUN_DIR/protonation_report.json. Log: $LOG"
log "This is energy minimization, not a thermodynamic stability test."

if (( failed > 0 )); then exit 1; fi
if (( review > 0 )); then exit 2; fi
log "SUCCESS: every sample passed."
