#!/usr/bin/env python3
"""Add neutral ACE/NME peptide caps to the Zhou-protonated ten-chain AD PHF.

Pipeline (run in the directory containing these scripts and the original PDB):
    /opt/miniconda3/bin/python zhou_protonation.py
    /opt/miniconda3/bin/python cap_tau.py

Requires preparation_checks.py beside this script.
Defaults resolve beside this script. Existing output files are never replaced.
Output: a new capped PDB and a .json provenance/validation report.

Each chain becomes ACE305--tau306:378--NME379. All original coordinates,
including histidine hydrogens, remain fixed. Missing amide H on Val306 is added.
This script intentionally accepts the specific uncapped input from this pipeline;
it is not a general repair tool for arbitrary proteins or charged termini.

Cap positions are modeled, not experimentally measured. The new peptide bonds
are trans. The N-cap phi defaults to -135 degrees, an extended starting geometry.
The C-terminal carbonyl plane determines placement of the NME nitrogen.
Methyl H atoms have tetrahedral geometry and a reproducible rotational phase.
Bond lengths and heavy-atom angles below are taken from OpenMM's bundled Amber
ff14SB parameters for initial construction only. No full force field, charges,
solvent, minimization, or dynamics are applied here. Close-contact warnings
are reported for review before subsequent relaxation.
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
import preparation_checks as checks
from openmm import unit
from openmm.app import PDBFile, Topology, element


# Angstroms and degrees. Amber ff14SB equilibrium values, used only for geometry.
GEOMETRY = dict(C_N=1.335, C_C=1.522, C_O=1.229, N_C=1.449,
                N_H=1.010, C_H=1.090, CA_C_N=116.6, C_N_CA=121.9,
                N_C_O=122.9, peptide_omega=180.0)
HISTIDINES = {"329": {"HD1", "HE2"}, "330": {"HD1"},
              "362": {"HE2"}, "374": {"HE2"}}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def normalized(vector):
    length = np.linalg.norm(vector)
    require(length > 1e-10, "Cannot construct geometry from coincident/collinear atoms.")
    return vector / length


def dihedral(a, b, c, d):
    axis = normalized(c - b)
    v = a - b - np.dot(a - b, axis) * axis
    w = d - c - np.dot(d - c, axis) * axis
    require(min(np.linalg.norm(v), np.linalg.norm(w)) > 1e-10,
            "Undefined dihedral from collinear atoms.")
    return float(np.degrees(np.arctan2(np.dot(np.cross(axis, v), w), np.dot(v, w))))


def place(a, b, c, length, angle, torsion):
    """Place d with distance c-d, angle b-c-d, and dihedral a-b-c-d."""
    axis = normalized(c - b)
    normal = normalized(np.cross(b - a, axis))
    in_plane = np.cross(normal, axis)
    theta, phi = np.radians([angle, torsion])
    return c + length * (-np.cos(theta) * axis + np.sin(theta) *
                         (np.cos(phi) * in_plane + np.sin(phi) * normal))


def methyl_hydrogens(center, parent, reference, phase):
    axis = normalized(parent - center)
    v = reference - center
    u = normalized(v - np.dot(v, axis) * axis)
    w = np.cross(axis, u)
    return [center + GEOMETRY["C_H"] *
            (-axis / 3 + np.sqrt(8) / 3 * (np.cos(t) * u + np.sin(t) * w))
            for t in np.radians(phase + np.array([0, 120, 240]))]


def amide_hydrogen(n, c, ca):
    return n - GEOMETRY["N_H"] * normalized(normalized(c - n) + normalized(ca - n))


def snapshot(topology, xyz):
    result = {}
    for a in topology.atoms():
        key = (a.residue.chain.id, a.residue.id, a.residue.name, a.name)
        require(key not in result, f"Duplicate atom: {key}")
        result[key] = np.array(xyz[a.index])
    return result


def main():
    folder = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path,
                        default=folder / "ad_tau_5O3L_full_filament_zhou_protonated.pdb")
    parser.add_argument("--output", type=Path,
                        default=folder / "ad_tau_5O3L_full_filament_zhou_capped.pdb")
    parser.add_argument("--n-cap-phi", type=float, default=-135.0)
    parser.add_argument("--methyl-phase", type=float, default=60.0)
    args = parser.parse_args()
    require(np.isfinite([args.n_cap_phi, args.methyl_phase]).all(), "Invalid geometry setting.")
    source, output = args.input.resolve(), args.output.resolve()
    report_path = output.with_suffix(".json")
    require(source not in (output, report_path), "Cannot overwrite the input.")
    for path in (output, report_path):
        require(not path.exists(), f"Refusing to overwrite {path}")
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    pdb = PDBFile(str(source))
    old_xyz = np.array(pdb.positions.value_in_unit(unit.angstrom))
    checks.histidine_geometry(pdb.topology, old_xyz)
    checks.formal_charges(pdb.topology, capped=False)
    before = snapshot(pdb.topology, old_xyz)
    chains = list(pdb.topology.chains())
    require(len(chains) == 10 and {c.id for c in chains} == set("ABCDEFGHIJ"),
            "Expected the full filament, chains A through J.")
    old_neighbors = {a: [] for a in pdb.topology.atoms()}
    for a, b in pdb.topology.bonds():
        old_neighbors[a].append(b)
        old_neighbors[b].append(a)

    topology, positions, copied, added, boundary_atoms = Topology(), [], {}, set(), []
    expected_new_bonds = []

    def add_atom(residue, name, symbol, position, new=True):
        atom = topology.addAtom(name, element.Element.getBySymbol(symbol), residue)
        positions.append(np.array(position))
        if new:
            added.add(atom.index)
        return atom

    def bond(a, b, length):
        topology.addBond(a, b)
        expected_new_bonds.append((a.index, b.index, length))

    for chain in chains:
        residues = list(chain.residues())
        require([r.id for r in residues] == [str(i) for i in range(306, 379)] and
                all(not r.insertionCode.strip() for r in residues),
                f"Chain {chain.id} is not the expected uncapped 306–378 fragment.")
        for r in residues:
            if r.id in HISTIDINES:
                hs = {a.name for a in r.atoms() if a.name in {"HD1", "HE2"}}
                require(r.name == "HIS" and hs == HISTIDINES[r.id],
                        f"Zhou assignment missing at {chain.id}:{r.id}; run step 1 first.")
        first = {a.name: a for a in residues[0].atoms()}
        last = {a.name: a for a in residues[-1].atoms()}
        require(residues[0].name == "VAL" and residues[-1].name == "PHE",
                "Expected Val306 and Phe378 endpoints.")
        require({"N", "CA", "C"} <= first.keys() and {"N", "CA", "C", "O"} <= last.keys(),
                "Missing terminal backbone atoms.")
        require(not any(a.element == element.hydrogen for a in old_neighbors[first["N"]])
                and "OXT" not in last,
                "Input has different terminal chemistry; refusing to remove atoms automatically.")
        n, ca, c = (old_xyz[first[name].index] for name in ("N", "CA", "C"))
        ac = place(c, ca, n, GEOMETRY["C_N"], GEOMETRY["C_N_CA"], args.n_cap_phi)
        am = place(ca, n, ac, GEOMETRY["C_C"], GEOMETRY["CA_C_N"], 180.0)
        ao = place(ca, n, ac, GEOMETRY["C_O"], GEOMETRY["N_C_O"], 0.0)
        new_chain = topology.addChain(chain.id)
        ace = topology.addResidue("ACE", new_chain, "305")
        ace_c = add_atom(ace, "C", "C", ac)
        ace_o = add_atom(ace, "O", "O", ao)
        ace_m = add_atom(ace, "CH3", "C", am)
        bond(ace_c, ace_o, GEOMETRY["C_O"])
        bond(ace_c, ace_m, GEOMETRY["C_C"])
        for i, h in enumerate(methyl_hydrogens(am, ac, ao, args.methyl_phase), 1):
            bond(ace_m, add_atom(ace, f"H{i}", "H", h), GEOMETRY["C_H"])
        for r in residues:
            new_residue = topology.addResidue(r.name, new_chain, r.id, r.insertionCode)
            for atom in r.atoms():
                require(atom.element is not None, f"Unknown element for {atom}")
                copied[atom] = add_atom(new_residue, atom.name, atom.element.symbol,
                                        old_xyz[atom.index], new=False)
            if r == residues[0]:
                h = add_atom(new_residue, "H", "H", amide_hydrogen(n, ac, ca))
                bond(copied[first["N"]], h, GEOMETRY["N_H"])
        bond(ace_c, copied[first["N"]], GEOMETRY["C_N"])

        ln, lca, lc, lo = (old_xyz[last[name].index] for name in ("N", "CA", "C", "O"))
        psi = dihedral(ln, lca, lc, lo) + 180.0
        mn = place(ln, lca, lc, GEOMETRY["C_N"], GEOMETRY["CA_C_N"], psi)
        mm = place(lca, lc, mn, GEOMETRY["N_C"], GEOMETRY["C_N_CA"], 180.0)
        nme = topology.addResidue("NME", new_chain, "379")
        nme_n = add_atom(nme, "N", "N", mn)
        # OpenMM's standard PDB naming uses C for the NME methyl carbon.
        nme_m = add_atom(nme, "C", "C", mm)
        nme_h = add_atom(nme, "H", "H", amide_hydrogen(mn, lc, mm))
        bond(copied[last["C"]], nme_n, GEOMETRY["C_N"])
        bond(nme_n, nme_m, GEOMETRY["N_C"])
        bond(nme_n, nme_h, GEOMETRY["N_H"])
        for i, h in enumerate(methyl_hydrogens(mm, mn, lc, args.methyl_phase), 1):
            bond(nme_m, add_atom(nme, f"H{i}", "H", h), GEOMETRY["C_H"])
        boundary_atoms.append((chain.id, ace_m.index, ace_c.index, copied[first["N"]].index,
                               copied[first["CA"]].index, copied[last["CA"]].index,
                               copied[last["C"]].index, nme_n.index, nme_m.index))
    for a, b in pdb.topology.bonds():
        topology.addBond(copied[a], copied[b])
    xyz = np.array(positions)
    require(np.isfinite(xyz).all(), "Nonfinite coordinates.")
    after = snapshot(topology, xyz)
    require(set(before) <= set(after) and
            all(np.array_equal(v, after[k]) for k, v in before.items()),
            "An original atom was lost or moved.")
    require(len(added) == 130, "Expected 20 caps plus ten new Val306 amide H atoms.")
    require(Counter(r.name for r in topology.residues())["ACE"] == 10 and
            Counter(r.name for r in topology.residues())["NME"] == 10, "Wrong cap counts.")
    for i, j, length in expected_new_bonds:
        require(abs(np.linalg.norm(xyz[i] - xyz[j]) - length) < 1e-8,
                "Constructed bond length differs from its specification.")
    for chain, *indices in boundary_atoms:
        for quartet in (indices[:4], indices[4:]):
            omega = dihedral(*(xyz[i] for i in quartet))
            require(abs(abs(omega) - 180) < 1e-8, f"Non-trans cap in chain {chain}.")
    charges = checks.formal_charges(topology, capped=True)
    checks.histidine_geometry(topology, xyz)
    buffer = io.StringIO()
    for line in ["NEUTRAL ACE305 AND NME379 CAPS; ZHOU HISTIDINES RETAINED",
                 "ALL ORIGINAL ATOM COORDINATES PRESERVED; 130 ATOMS ADDED",
                 "CAP GEOMETRY MODELED; NO ENERGY MINIMIZATION OR SOLVATION",
                 f"N-CAP PHI {args.n_cap_phi}; METHYL PHASE {args.methyl_phase}",
                 "SEE COMPANION JSON FOR PROVENANCE AND CLOSE-CONTACT WARNINGS"]:
        buffer.write(f"REMARK 900 {line}\n")
    PDBFile.writeFile(topology, xyz * unit.angstrom, buffer, keepIds=True)
    contents = buffer.getvalue()
    # Verify that PDB serialization preserves atoms and cap-to-protein bonds.
    reread = PDBFile(io.StringIO(contents))
    reread_xyz = np.array(reread.positions.value_in_unit(unit.angstrom))
    serialized = snapshot(reread.topology, reread_xyz)
    require(set(serialized) == set(after), "PDB serialization changed atom identities.")
    require(all(np.allclose(after[k], serialized[k], rtol=0, atol=0.000501) for k in after),
            "PDB coordinate differences exceed normal rounding.")
    def bond_keys(t):
        def key(a):
            return (a.residue.chain.id, a.residue.id, a.name)
        return {frozenset((key(a), key(b))) for a, b in t.bonds()}
    require(bond_keys(topology) == bond_keys(reread.topology),
            "Bond connectivity was not preserved by the PDB round trip.")
    geometry = checks.histidine_geometry(reread.topology, reread_xyz)
    checks.formal_charges(reread.topology, capped=True)
    changed_keys = set(after) - set(before)
    inspected = {a.index for a in reread.topology.atoms()
                 if (a.residue.chain.id, a.residue.id, a.residue.name, a.name) in changed_keys
                 or (a.residue.name == "HIS" and a.name in ("HD1", "HE2"))}
    contacts = checks.contact_screen(reread.topology, reread_xyz, inspected)
    report = dict(input=str(source), input_sha256=source_hash,
                  output=str(output), output_sha256=hashlib.sha256(contents.encode()).hexdigest(),
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  checks_sha256=hashlib.sha256(Path(checks.__file__).read_bytes()).hexdigest(),
                  python=platform.python_version(), openmm=openmm.__version__, numpy=np.__version__,
                  geometry_angstrom_degrees=GEOMETRY, n_cap_phi=args.n_cap_phi,
                  methyl_phase=args.methyl_phase, caps={"ACE": 10, "NME": 10},
                  added_atoms=130, original_atoms_preserved=len(before),
                  histidines="329 HIP, 330 HID, 362 HIE, 374 HIE on each of ten chains",
                  contact_rule=checks.CONTACT_RULE,
                  contact_scope="130 capping additions plus 50 corrected histidine N-H atoms",
                  histidine_geometry=geometry, formal_charge_check=charges,
                  close_contacts=contacts, minimized=False, parameterized=False,
                  note="Starting cap geometry only. Review close contacts before relaxation; no stability conclusion.")
    require(hashlib.sha256(source.read_bytes()).hexdigest() == source_hash,
            "Input changed during processing.")
    checks.write_outputs({output: contents, report_path: json.dumps(report, indent=2) + "\n"})
    print(f"Saved: {output}\nReport: {report_path}")
    print("Verified: 10 ACE caps, 10 NME caps; original atoms and Zhou states preserved.")
    print(f"Contact screen: {dict(Counter(r['category'] for r in contacts))} (see JSON).")
    print("Formal charge verified: +6 per capped chain, +60 total; partial charges not assigned.")
    print("This is a modeled starting structure, not a relaxed or parameterized system.")


if __name__ == "__main__":
    main()
