# Protein Crusher

Run the preparation and implicit-solvent minimization pipeline on a GPU host:

```bash
python -m pip install -r requirements-cuda.txt
PLATFORM=CUDA bash relax_pipeline.sh raw.pdb
```

The pipeline caps chains, assigns protonation with PROPKA, adds hydrogens,
conditionally removes heavy-atom clashes, relaxes hydrogens, and relaxes all
atoms. Each invocation creates a new run directory; exit 2 means review is
required and later stages do not run. Candidate structures and reports remain
available for inspection. See `bash relax_pipeline.sh --help` for settings.

## Conditional clash removal

`remove_clashes_generic.py` runs between hydrogen addition and hydrogen
relaxation. It screens heavy-atom pairs with the existing geometric contact
criterion (Bondi-radius overlap greater than 0.40 Å, excluding bonded and 1-3
neighbors; 1-4 contacts are reported separately), but only **severe** overlaps,
greater than 1.0 Å by default (`--severe-overlap`), trigger repair. The 0.40 Å
screen also catches ordinary contacts such as hydrogen-bonded N/O pairs near
2.6 Å, so those are reported but not acted on. For reference, 5O3L's worst
heavy-atom overlap is 0.93 Å (hydrogen relaxation handles it), while every
RibbonFold model checked has overlaps above 2.4 Å; 1.0 Å means C–C closer than
2.4 Å or O–O closer than 2.04 Å. This is a geometric screen, not a MolProbity
clashscore.

When no severe overlaps exist, the input is copied **byte-for-byte**,
without parameterization, a compute context, coordinate perturbation, or
minimization. Both existing relaxation algorithms are unchanged. Clean-input
results retain their previous numerical path; stochastic hydrogen placement
and different hardware still prevent a blanket bitwise reproducibility claim.

For clashing inputs, cleanup uses a temporary clone of the ff14SB/GBn2 system:

- Retain harmonic bonds, angles, and periodic/RB torsions; remove normal LJ,
  electrostatics, and implicit solvent. Reject unsupported force classes.
- Use `4*epsilon*(1/u^2-1/u)`, where `u=alpha+(r/sigma)^6`, with arithmetic
  sigma and geometric epsilon mixing. Every original nonbonded exception,
  including scaled 1-4 pairs, is excluded. This is the proposed soft-core LJ
  **including its attractive well**, not a purely repulsive potential.
- Restrain Cα positions with `0.5*k*displacement^2`, default
  `k=1000 kJ/mol/nm^2`, and peptide ω with `k*(1+cos(omega))`, default
  `k=100 kJ/mol`. Actual peptide bonds, including ACE/NME caps, define ω.
- Hold L-chirality at every gated stereocenter (Cα except Gly, Ile/Thr Cβ) with
  `k*min(0, V/2.0-1)^2`, default `k=1000 kJ/mol` (`--chiral-k`), where `V` is
  the same signed volume in Å³ that the chirality gate checks. It is zero for
  normal centers (about 2.4–2.8 Å³) and rises as a center flattens or inverts.
  ff14SB has no chirality term, and with 1-4 and electrostatic terms removed a
  deep overlap can otherwise push a substituent through the plane and invert
  the center (observed on a RibbonFold model before this restraint).
- Minimize at α = 0.5, 0.2, 0.05, 0.01, 0.001. Each round gets the pipeline's
  iteration budget and tolerance. Require finite energies/forces, nonincreasing
  energy within each round, and final-round force convergence. Intermediate
  rounds may reach the iteration cap while the continuation proceeds.
- Break near-exact coincidences (<0.001 nm) with a seeded, small perturbation
  before minimization. A finite energy alone does not define a radial force
  direction at zero separation. Perturbed atom indices are recorded.

Only PDB coordinates are passed onward; the existing relaxation scripts build
fresh real-force-field systems, so temporary restraints and soft forces cannot
leak into those stages. Cleanup records rounds, geometry before/after, hashes,
backend details, displacement, and optimization frames in `clash_cleaned.*`.
These frames are optimization steps, not molecular dynamics.

For direct use (on an already hydrogenated input):

```bash
python remove_clashes_generic.py --input hydrogenated.pdb \
  --output clash_cleaned.pdb --platform CUDA
```

Use `--help` for α, restraint, severe-overlap, iteration, and seed options. The shell pipeline
uses the documented defaults for α and restraints.

## Acceptance checks

Repaired PDB coordinates are reloaded and checked before hydrogen relaxation:
severe heavy-atom overlaps, all harmonic bond deviations (>0.10 Å), peptide C–N
lengths (outside 1.20–1.45 Å), ω more than 60° from trans, stereochemistry,
and bonds passing through residue rings. Peptides 30–60° from trans are listed
as `twisted_trans_peptides` but pass at this point: they are on the trans side
of the ~90° barrier, and full relaxation flattens them downhill. The final
relaxed structure is checked with the usual 30° ω limit
again by `validate_clash_output.py`, now including hydrogen overlaps. Both
checks fail only on severe overlaps, using the threshold recorded in the cleanup
report; smaller overlaps are listed as close contacts, not failures. Plausible
hydrogen bonds and 1-4 contacts remain separately classified. The
existing full-relaxation stage also gates all harmonic bond lengths.

Chirality checks assume standard L-proteins and cover Cα (except glycine) and
the additional Ile/Thr Cβ centers, rejecting inverted or nearly planar centers
(ordered signed triple product ≤0.5 Å³). **All peptides are targeted to trans**;
intentional cis peptides and D-amino acids need a different policy.

Ring checks triangulate the residue rings in His/Phe/Tyr/Trp/Pro and test
nonincident covalent bond segments against the surfaces, including coplanar
intersections. Missing/degenerate rings fail. This detects geometric ring
piercings; it is not a proof of correct topology, unknotting, or biological
validity. Extra geometry gates are skipped on the clean-input branch to retain
its previous acceptance behavior. Reports state the applicable thresholds.

PROPKA still runs **before** cleanup. Its estimates near severe overlaps can be
unreliable. Protonation states are retained and this limitation is recorded;
reassess protonation and regenerate hydrogens on the cleaned structure when
those sites matter. Cleanup does not automatically restart the preparation
pipeline or silently change protonation.

## Lightweight verification

```bash
OPENMM_CPU_THREADS=1 python -m unittest discover -s tests -p test_clash_cleanup.py -v
bash -n relax_pipeline.sh
```

These tests use tiny synthetic geometries and two/three-particle OpenMM
systems, including one three-particle continuation, plus mocked pipeline stages.
They do not run a protein minimization. Production GPU behavior and repair
quality on clashing protein assemblies still require validation on a GPU host.

OpenMM API reference:
https://docs.openmm.org/latest/api-python/generated/openmm.openmm.CustomNonbondedForce.html
