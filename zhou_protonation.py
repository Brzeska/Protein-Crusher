#!/usr/bin/env python3
"""Apply Zhou's histidine states with deterministic planar N-H geometry.

Run: /opt/miniconda3/bin/python zhou_protonation.py
Requires preparation_checks.py beside this script. Defaults resolve here.

Zhou et al. (2021), doi:10.1021/acschemneuro.1c00536: H329 HIP,
H330 HID, H362/H374 HIE. Their 5O3T assignment is transferred to 5O3L
as a modeling assumption. OpenMM reads/builds/writes the bonded topology.
All 50 N-bound histidine ring hydrogens are rebuilt, including previously
present ones: N-H=1.01 A (Amber ff14SB equilibrium value), outward bisector
projected into the least-squares heavy-atom ring plane. No random placement.

Other hydrogens, heavy atoms, and fragment ends are unchanged. No energy
minimization or full force-field parameterization is performed. This does not
normalize all other inherited protein hydrogens. Output is a PDB plus JSON
provenance and validation report. Existing output files are never overwritten.
"""
import argparse
from collections import Counter
import hashlib
import io
import json
from pathlib import Path
import platform

import numpy as np
import openmm
from openmm import unit
from openmm.app import PDBFile, Topology, element
import preparation_checks as checks

ASSIGNMENTS = {"329": "HIP", "330": "HID", "362": "HIE", "374": "HIE"}
PARENTS = {"HD1": ("ND1", "CG", "CE1"), "HE2": ("NE2", "CE1", "CD2")}


def key(a):
    r = a.residue
    return r.chain.id, r.id, r.name, a.name


def snapshot(topology, xyz):
    result = {key(a): np.array(xyz[a.index]) for a in topology.atoms()}
    checks.require(len(result) == topology.getNumAtoms(), "Duplicate atom identities")
    return result


def planar_hydrogen(atoms, xyz, name):
    ring = np.array([xyz[atoms[n].index] for n in checks.RING])
    normal = np.linalg.svd(ring - ring.mean(0))[2][-1]
    n, a, b = (xyz[atoms[s].index] for s in PARENTS[name])
    u, v = a - n, b - n
    direction = -(u / np.linalg.norm(u) + v / np.linalg.norm(v))
    direction -= np.dot(direction, normal) * normal
    checks.require(np.linalg.norm(direction) > 1e-8, "Degenerate histidine geometry")
    return n + 1.01 * direction / np.linalg.norm(direction)


def main():
    folder = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=folder / "ad_tau_5O3L_full_filament.pdb")
    parser.add_argument("--output", type=Path, default=folder / "ad_tau_5O3L_full_filament_zhou_protonated.pdb")
    args = parser.parse_args()
    source, output = args.input.resolve(), args.output.resolve()
    report_path = output.with_suffix('.json')
    checks.require(source not in (output, report_path), "Input must not be overwritten")
    for path in (output, report_path):
        checks.require(not path.exists(), f"Refusing to overwrite {path}")
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    pdb = PDBFile(str(source))
    old_xyz = np.array(pdb.positions.value_in_unit(unit.angstrom))
    before = snapshot(pdb.topology, old_xyz)
    chains = list(pdb.topology.chains())
    checks.require(len(chains) == 10 and {c.id for c in chains} == set('ABCDEFGHIJ'),
                   "Expected ten chains A-J")
    topology, positions, copied, rebuilt = Topology(), [], {}, set()
    states = Counter()
    for chain in chains:
        residues = list(chain.residues())
        checks.require([r.id for r in residues] == [str(i) for i in range(306, 379)] and
                       all(not r.insertionCode.strip() for r in residues),
                       f"Unexpected sequence numbering in chain {chain.id}")
        new_chain = topology.addChain(chain.id)
        for r in residues:
            target = r.id in ASSIGNMENTS
            checks.require((r.name == 'HIS') == target, f"Unexpected histidine identity: {r}")
            nr = topology.addResidue(r.name, new_chain, r.id, r.insertionCode)
            atoms = {a.name: a for a in r.atoms()}
            for a in r.atoms():
                if target and a.name in PARENTS:
                    continue
                copied[a] = topology.addAtom(a.name, a.element, nr)
                positions.append(old_xyz[a.index])
            if target:
                checks.require(set(checks.RING) <= atoms.keys(), f"Incomplete ring: {r}")
                states[ASSIGNMENTS[r.id]] += 1
                for name in sorted(checks.STATES[r.id]):
                    h = topology.addAtom(name, element.hydrogen, nr)
                    positions.append(planar_hydrogen(atoms, old_xyz, name))
                    topology.addBond(copied[atoms[PARENTS[name][0]]], h)
                    rebuilt.add(h.index)
    for a, b in pdb.topology.bonds():
        if a in copied and b in copied:
            topology.addBond(copied[a], copied[b])
    xyz = np.array(positions)
    checks.require(np.isfinite(xyz).all(), "Nonfinite coordinates")
    after = snapshot(topology, xyz)
    expected = {k for k in before if not (k[2] == 'HIS' and k[3] in PARENTS)}
    expected.update((c, r, 'HIS', h) for c in 'ABCDEFGHIJ' for r, hs in checks.STATES.items() for h in hs)
    checks.require(set(after) == expected, "Unexpected atom changes")
    preserved = set(before) - {k for k in before if k[2] == 'HIS' and k[3] in PARENTS}
    checks.require(all(np.array_equal(before[k], after[k]) for k in preserved),
                   "An atom outside the intended ring hydrogens moved")
    charges = checks.formal_charges(topology, capped=False)
    checks.histidine_geometry(topology, xyz)
    buffer = io.StringIO()
    for line in ["ZHOU STATES: 329 HIP; 330 HID; 362/374 HIE ON CHAINS A-J",
                 "ALL 50 N-BOUND HIS RING HYDROGENS REBUILT AT 1.01 A, PLANAR",
                 "OTHER ATOM COORDINATES PRESERVED; FRAGMENT ENDS UNCHANGED",
                 "NO FORCE-FIELD PARAMETERIZATION, CAPS, OR RELAXATION",
                 "SOURCE DOI 10.1021/acschemneuro.1c00536; SEE COMPANION JSON"]:
        buffer.write(f"REMARK 900 {line}\n")
    PDBFile.writeFile(topology, xyz * unit.angstrom, buffer, keepIds=True)
    contents = buffer.getvalue()
    saved = PDBFile(io.StringIO(contents))
    saved_xyz = np.array(saved.positions.value_in_unit(unit.angstrom))
    serialized = snapshot(saved.topology, saved_xyz)
    checks.require(set(serialized) == set(after), "PDB serialization changed atom identities")
    checks.require(all(np.allclose(serialized[k], after[k], rtol=0, atol=.000501) for k in after),
                   "Unexpected coordinate rounding")
    geometry = checks.histidine_geometry(saved.topology, saved_xyz)
    checks.formal_charges(saved.topology, capped=False)
    saved_rebuilt = {a.index for a in saved.topology.atoms() if a.residue.name == 'HIS' and a.name in PARENTS}
    contacts = checks.contact_screen(saved.topology, saved_xyz, saved_rebuilt)
    checks.require(hashlib.sha256(source.read_bytes()).hexdigest() == source_hash, "Input changed")
    report = dict(input=str(source), input_sha256=source_hash, output=str(output),
                  output_sha256=hashlib.sha256(contents.encode()).hexdigest(),
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  checks_sha256=hashlib.sha256(Path(checks.__file__).read_bytes()).hexdigest(),
                  openmm=openmm.__version__, numpy=np.__version__, python=platform.python_version(),
                  assignments=ASSIGNMENTS, ring_hydrogens_rebuilt=50,
                  new_hydrogens=len(set(after)-set(before)), removed_hydrogens=len(set(before)-set(after)),
                  existing_ring_hydrogens_repositioned=20, other_atoms_preserved=len(preserved),
                  histidine_geometry=geometry, formal_charge_check=charges,
                  contact_rule=checks.CONTACT_RULE, contact_scope="50 rebuilt histidine N-H atoms",
                  close_contacts=contacts, parameterized=False, minimized=False,
                  note="Other inherited hydrogen geometries are unchanged; this is not full hydrogen normalization.")
    checks.write_outputs({output: contents, report_path: json.dumps(report, indent=2)+'\n'})
    print(f"Saved: {output}\nReport: {report_path}")
    print(f"Verified 40 histidines: {dict(sorted(states.items()))}; all 50 ring N-H geometries checked.")
    print("30 H added, 20 removed, 20 retained ring H repositioned; other atoms unchanged.")
    print(f"Contact screen: {dict(Counter(r['category'] for r in contacts))} (see JSON).")
    print("Side-chain formal charge +6 per chain; ends incomplete. Capping and parameterization remain.")


if __name__ == '__main__':
    main()
