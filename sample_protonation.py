#!/usr/bin/env python3
"""Sample protonation microstates from Henderson-Hasselbalch fractions.

Example:
    python sample_protonation.py --input capped.pdb --ph 7.0 --samples 10 --output-dir run/

What it does
    1. Finds every ionizable side chain in an ACE/NME-capped structure.
    2. Gives each site a protonated fraction from a fixed model pKa:

           f_protonated = 1 / (1 + 10**(pH - pKa))

    3. Draws N microstates. In each draw, every site is protonated with
       probability f_protonated, independently of every other site.
    4. Writes one variants file per draw (sample_001/variants.json, ...) for
       add_hydrogens_generic.py, plus protonation_report.json in --output-dir.

Each draw is already weighted by its probability, so results from the samples
should be averaged with equal weight.

Assumptions
    - pKa values are free amino-acid model values. A protein environment can
      shift a real pKa by a pH unit or more.
    - Sites are independent. In a fibril, the same residue in neighboring
      layers is about 4.8 A away, and charging one makes its neighbors harder
      to charge. This script ignores that coupling.
    - Each His's neutral tautomer (HID or HIE) is chosen once from the input
      geometry (see choose_histidine_tautomer) and reused in every sample.
      The input's His ring orientation is trusted; flipped rings (a common
      cryo-EM/X-ray ambiguity) are not tested.
    - Tyr and Arg stay protonated: add_hydrogens_generic.py cannot build
      deprotonated Tyr (TYM) or neutral Arg. If the pH makes either
      deprotonated form more than 1% likely, the script exits with status 2.
    - Disulfide-bonded Cys (CYX) are not titrated.
    - Chain ends must be ACE/NME caps, so there are no titratable termini.

Exit status: 0 samples written, 2 needs review (see message), 1 error.
"""
import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from openmm import unit
from openmm.app import PDBFile, element

import preparation_checks as checks


# Model pKa values for free amino-acid side chains (standard textbook values).
MODEL_PKA = {
    'ASP': 3.9,
    'GLU': 4.1,
    'HIS': 6.0,
    'CYS': 8.3,
    'TYR': 10.1,
    'LYS': 10.5,
    'ARG': 12.5,
}

# Residue names (Amber/OpenMM) for each site's two states:
# (protonated, deprotonated). For His, "protonated" is HIP (H on both ring
# nitrogens). Deprotonated His is neutral, and which neutral tautomer it uses
# is decided per residue by choose_histidine_tautomer.
STATE_NAMES = {
    'ASP': ('ASH', 'ASP'),
    'GLU': ('GLH', 'GLU'),
    'HIS': ('HIP', 'HID or HIE'),
    'CYS': ('CYS', 'CYM'),
    'LYS': ('LYS', 'LYN'),
}

# Always protonated: the hydrogen adder has no deprotonated form for these.
ALWAYS_PROTONATED = ('TYR', 'ARG')
MAX_UNREPRESENTED_FRACTION = 0.01

# Histidine tautomer geometry. A partner atom "sits where an N's hydrogen or
# lone pair points" when it is within this distance of the N and within this
# angle of the N's outward direction (the same tolerant criteria OpenMM uses).
HBOND_MAX_DISTANCE_ANGSTROM = 3.5
HBOND_MAX_ANGLE_DEGREES = 50.0

# Heavy atoms that carry a hydrogen and can donate it. Any oxygen counts as an
# acceptor (backbone C=O, carboxylates, hydroxyls, amide C=O, cap C=O).
SIDE_CHAIN_DONORS = {
    'LYS': {'NZ'},
    'ARG': {'NE', 'NH1', 'NH2'},
    'ASN': {'ND2'},
    'GLN': {'NE2'},
    'TRP': {'NE1'},
}
HISTIDINE_SIDE_CHAIN = {'CB', 'CG', 'ND1', 'CD2', 'CE1', 'NE2'}


@dataclass
class Site:
    """One titratable side chain and its two possible states."""
    key: str                  # chain:residue, e.g. "A:329"
    residue_name: str         # e.g. "HIS"
    pka: float
    fraction_protonated: float
    protonated_state: str     # e.g. "HIP"
    deprotonated_state: str   # e.g. "HIE"


def fraction_protonated(pka, ph):
    """Henderson-Hasselbalch: fraction of a site carrying its titratable proton."""
    return 1.0 / (1.0 + 10.0 ** (ph - pka))


def residue_key(residue):
    """Same key format add_hydrogens_generic.py uses: chain:id plus insertion code."""
    return f'{residue.chain.id}:{residue.id}{residue.insertionCode.strip()}'


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# Reading and checking the input
# ---------------------------------------------------------------------------

def load_capped_structure(path):
    """Load a single-model PDB and confirm every chain runs ACE ... NME."""
    pdb = PDBFile(str(path))
    checks.require(pdb.getNumFrames() == 1, 'Input must contain exactly one model')
    for chain in pdb.topology.chains():
        residues = list(chain.residues())
        checks.require(len(residues) >= 3 and residues[0].name == 'ACE' and residues[-1].name == 'NME',
                       f'Chain {chain.id} is not ACE/NME capped; run cap_generic.py first')
    keys = [residue_key(r) for r in pdb.topology.residues()]
    checks.require(len(keys) == len(set(keys)), 'Residue keys (chain:id) are not unique')
    return pdb


def is_disulfide_bonded(cysteine, bonded_atoms):
    """True if this Cys sulfur is bonded to a sulfur in another residue."""
    sulfur = next(a for a in cysteine.atoms() if a.name == 'SG')
    return any(other.element == element.sulfur and other.residue != cysteine
               for other in bonded_atoms[sulfur])


# ---------------------------------------------------------------------------
# Histidine tautomer
# ---------------------------------------------------------------------------

def outward_direction(nitrogen, ring_neighbor_a, ring_neighbor_b):
    """Unit vector along which a ring N's hydrogen (or lone pair) points.

    It points away from both ring neighbors, along the bisector of the
    exterior angle.
    """
    away_from_a = (nitrogen - ring_neighbor_a) / np.linalg.norm(nitrogen - ring_neighbor_a)
    away_from_b = (nitrogen - ring_neighbor_b) / np.linalg.norm(nitrogen - ring_neighbor_b)
    direction = away_from_a + away_from_b
    return direction / np.linalg.norm(direction)


def partners_in_line(nitrogen, direction, partner_positions):
    """Count partner atoms close to the N and roughly along `direction`."""
    if len(partner_positions) == 0:
        return 0
    offsets = partner_positions - nitrogen
    distances = np.linalg.norm(offsets, axis=1)
    cosines = offsets @ direction / distances
    close = distances <= HBOND_MAX_DISTANCE_ANGSTROM
    aligned = cosines >= math.cos(math.radians(HBOND_MAX_ANGLE_DEGREES))
    return int(np.count_nonzero(close & aligned))


def choose_histidine_tautomer(histidine, acceptor_atoms, donor_atoms, xyz):
    """Pick HID (H on ND1) or HIE (H on NE2) from the local hydrogen-bond partners.

    Each ring nitrogen gets a score:
        +1 if an acceptor (any oxygen) sits where its H would point, because
           putting the H there makes a hydrogen bond;
        -1 if a donor (an N-H or similar) sits there, because that nitrogen
           should stay bare so its lone pair can accept that donor's H.
    The H goes on the higher-scoring nitrogen. Ties go to NE2 (HIE), the
    tautomer free histidine prefers about 4:1 in water.
    """
    atoms = {a.name: a for a in histidine.atoms()}
    position = {name: xyz[atom.index] for name, atom in atoms.items()}

    # Ignore this histidine's own side chain; its geometry is fixed.
    def outside_own_side_chain(atom_list):
        kept = [a for a in atom_list if not (a.residue == histidine and a.name in HISTIDINE_SIDE_CHAIN)]
        return xyz[[a.index for a in kept]] if kept else np.empty((0, 3))

    acceptors = outside_own_side_chain(acceptor_atoms)
    donors = outside_own_side_chain(donor_atoms)

    def score(nitrogen_name, neighbor_a, neighbor_b):
        nitrogen = position[nitrogen_name]
        direction = outward_direction(nitrogen, position[neighbor_a], position[neighbor_b])
        acceptor_count = partners_in_line(nitrogen, direction, acceptors)
        donor_count = partners_in_line(nitrogen, direction, donors)
        return int(acceptor_count > 0) - int(donor_count > 0), acceptor_count, donor_count

    nd1_score, nd1_acceptors, nd1_donors = score('ND1', 'CG', 'CE1')
    ne2_score, ne2_acceptors, ne2_donors = score('NE2', 'CE1', 'CD2')
    tautomer = 'HID' if nd1_score > ne2_score else 'HIE'
    evidence = dict(ND1=dict(score=nd1_score, acceptors=nd1_acceptors, donors=nd1_donors),
                    NE2=dict(score=ne2_score, acceptors=ne2_acceptors, donors=ne2_donors),
                    rule='H on the higher-scoring N; ties go to NE2 (HIE)')
    return tautomer, evidence


def hydrogen_bond_partners(topology):
    """Return (acceptor atoms, donor atoms) for the histidine tautomer scoring."""
    acceptors, donors = [], []
    for atom in topology.atoms():
        residue = atom.residue.name
        if atom.element == element.oxygen:
            acceptors.append(atom)
        elif atom.name == 'N' and residue not in ('PRO', 'ACE'):
            donors.append(atom)  # backbone amide N-H (Pro has none; ACE has no N)
        elif atom.name in SIDE_CHAIN_DONORS.get(residue, ()):
            donors.append(atom)
    return acceptors, donors


# ---------------------------------------------------------------------------
# Finding sites and drawing microstates
# ---------------------------------------------------------------------------

def find_sites(topology, xyz, ph):
    """List the sampled sites, plus notes on residues that are not sampled."""
    bonded_atoms = {atom: [] for atom in topology.atoms()}
    for a, b in topology.bonds():
        bonded_atoms[a].append(b)
        bonded_atoms[b].append(a)
    acceptors, donors = hydrogen_bond_partners(topology)

    sites, histidine_evidence, not_sampled = [], {}, []
    for residue in topology.residues():
        name, key = residue.name, residue_key(residue)
        if name not in MODEL_PKA:
            continue
        pka = MODEL_PKA[name]
        fraction = fraction_protonated(pka, ph)

        if name in ALWAYS_PROTONATED:
            not_sampled.append(dict(residue=key, name=name, pka=pka, fraction_protonated=fraction,
                                    state='protonated', reason='hydrogen adder has no deprotonated form'))
            continue
        if name == 'CYS' and is_disulfide_bonded(residue, bonded_atoms):
            not_sampled.append(dict(residue=key, name=name, state='CYX', reason='disulfide bond'))
            continue

        protonated, deprotonated = STATE_NAMES[name]
        if name == 'HIS':
            deprotonated, histidine_evidence[key] = choose_histidine_tautomer(residue, acceptors, donors, xyz)
        sites.append(Site(key, name, pka, fraction, protonated, deprotonated))
    return sites, histidine_evidence, not_sampled


def draw_microstates(sites, n_samples, seed):
    """Draw n_samples microstates. Each is a {site key: state name} dict."""
    rng = np.random.default_rng(seed)
    fractions = np.array([site.fraction_protonated for site in sites])
    microstates = []
    for _ in range(n_samples):
        protonated = rng.random(len(sites)) < fractions
        microstates.append({site.key: site.protonated_state if is_protonated else site.deprotonated_state
                            for site, is_protonated in zip(sites, protonated)})
    return microstates


def minority_sites(microstate, sites):
    """Sites drawn in their less likely state (the interesting part of a sample)."""
    minority = []
    for site in sites:
        majority = site.protonated_state if site.fraction_protonated >= 0.5 else site.deprotonated_state
        if microstate[site.key] != majority:
            minority.append(f'{site.key} {microstate[site.key]}')
    return minority


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--input', type=Path, required=True, help='ACE/NME-capped PDB (hydrogens are ignored)')
    parser.add_argument('--ph', type=float, required=True)
    parser.add_argument('--samples', type=int, default=10, help='number of microstates to draw (default 10)')
    parser.add_argument('--seed', type=int, default=2026, help='random seed (default 2026)')
    parser.add_argument('--output-dir', type=Path, required=True,
                        help='writes protonation_report.json and sample_NNN/variants.json here')
    args = parser.parse_args()
    checks.require(math.isfinite(args.ph) and 0 <= args.ph <= 14, 'pH must be between 0 and 14')
    checks.require(args.samples >= 1, '--samples must be at least 1')
    return args


def main():
    args = parse_args()
    source = args.input.resolve()
    output_dir = args.output_dir.resolve()
    input_hash = sha256(source)

    pdb = load_capped_structure(source)
    xyz = np.asarray(pdb.positions.value_in_unit(unit.angstrom))
    sites, histidine_evidence, not_sampled = find_sites(pdb.topology, xyz, args.ph)

    # Stop if a residue we cannot deprotonate has a real chance of being deprotonated.
    unrepresentable = [r for r in not_sampled
                       if r['name'] in ALWAYS_PROTONATED
                       and 1 - r['fraction_protonated'] > MAX_UNREPRESENTED_FRACTION]
    if unrepresentable:
        names = sorted({r['name'] for r in unrepresentable})
        print(f'NEEDS REVIEW: at pH {args.ph}, deprotonated {"/".join(names)} is more than '
              f'{MAX_UNREPRESENTED_FRACTION:.0%} likely, but the hydrogen adder cannot build it.')
        raise SystemExit(2)

    microstates = draw_microstates(sites, args.samples, args.seed)
    width = max(3, len(str(args.samples)))
    sample_names = [f'sample_{i:0{width}d}' for i in range(1, args.samples + 1)]

    report = dict(
        method='Independent-site Henderson-Hasselbalch sampling from model pKa values',
        input=str(source), input_sha256=input_hash, script_sha256=sha256(__file__),
        ph=args.ph, seed=args.seed, samples=args.samples,
        model_pka=MODEL_PKA,
        averaging='Each sample is a probability-weighted draw: average results with equal weight.',
        sites=[dict(residue=s.key, name=s.residue_name, pka=s.pka, fraction_protonated=s.fraction_protonated,
                    protonated_state=s.protonated_state, deprotonated_state=s.deprotonated_state)
               for s in sites],
        histidine_tautomers=histidine_evidence,
        not_sampled=not_sampled,
        expected_minority_sites_per_sample=sum(min(s.fraction_protonated, 1 - s.fraction_protonated) for s in sites),
        sample_list=[dict(name=name, variants=f'{name}/variants.json', minority_sites=minority_sites(state, sites))
                     for name, state in zip(sample_names, microstates)],
        distinct_microstates=len({json.dumps(state, sort_keys=True) for state in microstates}),
        limitations=['Model pKa values ignore the protein environment.',
                     'Sites are treated as independent; stacked fibril layers are not.',
                     'His tautomers come from one geometric rule on the input structure.',
                     'Do not rank samples by raw minimized energy: they have different numbers of protons.'],
    )

    outputs = {output_dir / 'protonation_report.json': json.dumps(report, indent=2) + '\n'}
    for name, state in zip(sample_names, microstates):
        (output_dir / name).mkdir(parents=True, exist_ok=True)
        outputs[output_dir / name / 'variants.json'] = json.dumps(state, indent=2) + '\n'
    checks.require(sha256(source) == input_hash, 'Input changed while it was being read')
    checks.write_outputs(outputs)  # refuses to overwrite existing files

    # Short summary for the pipeline log.
    counts = {}
    for site in sites:
        counts[site.residue_name] = counts.get(site.residue_name, 0) + 1
    print(f'pH {args.ph}: {len(sites)} sampled sites ' + ', '.join(f'{n} {name}' for name, n in sorted(counts.items())))
    for name in sorted(counts):
        site = next(s for s in sites if s.residue_name == name)
        print(f'  {name}: pKa {site.pka}, protonated fraction {site.fraction_protonated:.4f}')
    tautomers = [s.deprotonated_state for s in sites if s.residue_name == 'HIS']
    if tautomers:
        print(f'  Neutral His tautomers: {tautomers.count("HID")} HID, {tautomers.count("HIE")} HIE')
    print(f'Expected sites in their minority state per sample: {report["expected_minority_sites_per_sample"]:.2f}')
    for entry in report['sample_list']:
        drawn = ', '.join(entry['minority_sites']) or 'all sites in majority state'
        print(f'  {entry["name"]}: {drawn}')
    print(f'Report: {output_dir / "protonation_report.json"}')


if __name__ == '__main__':
    main()
