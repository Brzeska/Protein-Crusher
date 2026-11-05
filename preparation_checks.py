"""Shared validation/output utilities for the tau preparation scripts.

The contact screen is a geometric warning system, not a MolProbity clashscore
or an energy calculation. Formal charges are inferred from validated hydrogen
occupancy, not force-field partial charges. Coordinates here are in angstroms.
"""
import os
from pathlib import Path
import tempfile

import numpy as np
from openmm.app import element

STATES = {"329": {"HD1", "HE2"}, "330": {"HD1"},
          "362": {"HE2"}, "374": {"HE2"}}
RING = ("CG", "ND1", "CE1", "NE2", "CD2")
RADII = {"H": 1.20, "C": 1.70, "N": 1.55, "O": 1.52, "S": 1.80}
CONTACT_RULE = ("Bondi radii H1.20 C1.70 N1.55 O1.52 S1.80 A; overlap >0.40 A; "
                "exclude 1-2/1-3; label 1-4 pairs and geometrically plausible "
                "hydrogen bonds separately; screen only pairs involving specified atoms")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def hydrogen_variant_arguments(topology, requested):
    """Translate chemical CYM to OpenMM's no-S-H placement label CYX.

    CYX here describes hydrogen occupancy only. Explicit CYM template selection
    supplies thiolate parameters; it never creates a disulfide bond.
    https://docs.openmm.org/latest/api-python/generated/openmm.app.modeller.Modeller.html
    ff14SB supplies internal CYM, but not free-terminal NCYM/CCYM templates.
    """
    residues = list(topology.residues())
    require(len(requested) == len(residues), 'Wrong variant-list length')
    translated = list(requested)
    templates = {}
    bonded = {a: [] for a in topology.atoms()}
    for a, b in topology.bonds():
        bonded[a].append(b); bonded[b].append(a)
    for r, variant in zip(residues, requested):
        if variant != 'CYM':
            continue
        label = f'{r.chain.id}:{r.id}{r.insertionCode.strip()}'
        require(r.name == 'CYS', f'{label}: CYM requires a CYS residue')
        atoms = {a.name: a for a in r.atoms()}
        require(all(n in atoms for n in ('SG', 'CB', 'N', 'C')),
                f'{label}: incomplete cysteine heavy atoms')
        sg_heavy = [a for a in bonded[atoms['SG']] if a.element != element.hydrogen]
        require(sg_heavy == [atoms['CB']],
                f'{label}: CYM cannot have a disulfide or other sulfur crosslink')
        for name, partner in (('N', 'C'), ('C', 'N')):
            require(any(a.residue != r and a.name == partner for a in bonded[atoms[name]]),
                    f'{label}: CYM requires peptide-bonded or capped ends; free-terminal CYM is not supported')
        translated[r.index] = 'CYX'
        templates[r] = 'CYM'
    return translated, templates


def verify_cym(topology, system, requested, forcefield):
    """Check actual parameters and sulfur occupancy, including after PDB reload."""
    import openmm
    from openmm import unit
    residues = list(topology.residues())
    targets = [r for r, v in zip(residues, requested) if v == 'CYM']
    if not targets:
        return []
    require(len(requested) == len(residues), 'Wrong variant-list length')
    hydrogen_variant_arguments(topology, requested)  # Recheck covalent context.
    matched = forcefield.getMatchingTemplates(topology)
    forces = [f for f in system.getForces() if isinstance(f, openmm.NonbondedForce)]
    require(len(forces) == 1, 'Expected one NonbondedForce for CYM validation')
    adjacency = neighbors(topology)
    atoms = list(topology.atoms())
    result = []
    for r in targets:
        label = f'{r.chain.id}:{r.id}{r.insertionCode.strip()}'
        sg = next(a for a in r.atoms() if a.name == 'SG')
        require(not any(atoms[i].element == element.hydrogen for i in adjacency[sg.index]),
                f'{label}: CYM still has a sulfur hydrogen')
        require(matched[r.index].name == 'CYM', f'{label}: did not match CYM force-field template')
        charge = sum(forces[0].getParticleParameters(a.index)[0].value_in_unit(unit.elementary_charge)
                     for a in r.atoms())
        require(abs(charge + 1.0) < 1e-5, f'{label}: CYM charge is {charge}, expected -1')
        result.append(dict(residue=label, template='CYM', charge_e=charge, sulfur_hydrogens=0))
    return result


def neighbors(topology):
    result = {a.index: set() for a in topology.atoms()}
    for a, b in topology.bonds():
        result[a.index].add(b.index)
        result[b.index].add(a.index)
    return result


def histidine_geometry(topology, xyz):
    records = []
    bonded = neighbors(topology)
    for residue in topology.residues():
        if residue.name != "HIS":
            continue
        atoms = {a.name: a for a in residue.atoms()}
        require(residue.id in STATES, f"Unexpected histidine: {residue}")
        require(set(atoms) & {"HD1", "HE2"} == STATES[residue.id],
                f"Wrong histidine occupancy: {residue.chain.id}:{residue.id}")
        ring = np.array([xyz[atoms[n].index] for n in RING])
        normal = np.linalg.svd(ring - ring.mean(0))[2][-1]
        for h in sorted(STATES[residue.id]):
            n = "ND1" if h == "HD1" else "NE2"
            hi, ni = atoms[h].index, atoms[n].index
            require(bonded[hi] == {ni}, f"Incorrect bond for {residue}:{h}")
            v = xyz[hi] - xyz[ni]
            length = float(np.linalg.norm(v))
            tilt = float(np.degrees(np.arcsin(np.clip(abs(np.dot(v, normal)) / length, 0, 1))))
            require(abs(length - 1.01) <= 0.002 and tilt <= 0.2,
                    f"Bad histidine H geometry at {residue.chain.id}:{residue.id}:{h}: "
                    f"N-H {length:.4f} A, tilt {tilt:.3f} degrees. Rerun zhou_protonation.py.")
            records.append(dict(chain=residue.chain.id, residue=residue.id,
                                hydrogen=h, length_angstrom=length, tilt_degrees=tilt))
    require(len(records) == 50, "Expected 50 N-bound histidine ring hydrogens.")
    return records


def formal_charges(topology, capped):
    """Verify this construct's ionizable groups before assigning formal charge."""
    atoms = list(topology.atoms())
    bonded = neighbors(topology)
    result = {}
    for chain in topology.chains():
        charge, count, caps = 0, 0, []
        for r in chain.residues():
            aa = {a.name: a for a in r.atoms()}
            def hcount(name):
                require(name in aa, f"Missing atom {chain.id}:{r.id}:{name}")
                return sum(atoms[i].element == element.hydrogen for i in bonded[aa[name].index])
            def check(ok):
                require(ok, f"Unexpected protonation at {chain.id}:{r.id}:{r.name}")
            if r.name == "LYS":
                check(hcount("NZ") == 3); charge += 1
            elif r.name == "ARG":
                check((hcount("NE"), hcount("NH1"), hcount("NH2")) == (1, 2, 2)); charge += 1
            elif r.name == "ASP":
                check(hcount("OD1") + hcount("OD2") == 0); charge -= 1
            elif r.name == "GLU":
                check(hcount("OE1") + hcount("OE2") == 0); charge -= 1
            elif r.name == "CYS":
                check(hcount("SG") == 1 and len(bonded[aa["SG"].index]) == 2)
            elif r.name == "TYR":
                check(hcount("OH") == 1)
            elif r.name == "HIS":
                check(r.id in STATES)
                check(hcount("ND1") == int("HD1" in STATES[r.id]) and
                      hcount("NE2") == int("HE2" in STATES[r.id]))
                charge += int(r.id == "329"); count += 1
            elif r.name in ("ACE", "NME"):
                caps.append(r.name)
                if r.name == "ACE":
                    check(hcount("CH3") == 3 and len(bonded[aa["C"].index]) == 3)
                else:
                    check(hcount("N") == 1 and hcount("C") == 3 and
                          len(bonded[aa["N"].index]) == 3)
            if capped and r.id == "306":
                check(r.name == "VAL" and hcount("N") == 1 and len(bonded[aa["N"].index]) == 3)
        require(charge == 6 and count == 4, f"Unexpected charge or His count in {chain.id}")
        require(sorted(caps) == (["ACE", "NME"] if capped else []), "Unexpected cap composition")
        result[chain.id] = charge
    require(len(result) == 10, "Expected ten chains.")
    return dict(scope="whole capped chains" if capped else "side chains only; fragment ends incomplete",
                per_chain=result, summed_side_chain_charge=sum(result.values()),
                total_formal_charge=sum(result.values()) if capped else None,
                force_field_partial_charges_assigned=False)


def contact_screen(topology, xyz, indices):
    atoms = list(topology.atoms())
    bonded = neighbors(topology)
    radii = np.array([RADII[a.element.symbol] for a in atoms])
    def label(i):
        a = atoms[i]
        return f"{a.residue.chain.id}:{a.residue.name}{a.residue.id}:{a.name}"
    def acceptor(i):
        a = atoms[i]
        return (a.element == element.oxygen or
                (a.residue.name == "HIS" and a.name in ("ND1", "NE2") and
                 not any(atoms[j].element == element.hydrogen for j in bonded[i])))
    def hbond(i, j):
        # Label plausible H-bond overlaps; do not silently discard them.
        if atoms[i].element != element.hydrogen:
            i, j = j, i
        if atoms[i].element != element.hydrogen or not acceptor(j) or len(bonded[i]) != 1:
            return False
        donor = next(iter(bonded[i]))
        if atoms[donor].element not in (element.nitrogen, element.oxygen):
            return False
        ha, hd = xyz[j] - xyz[i], xyz[donor] - xyz[i]
        length = np.linalg.norm(ha)
        if not (1.4 <= length <= 2.6 and np.linalg.norm(xyz[donor] - xyz[j]) <= 3.5):
            return False
        angle = np.degrees(np.arccos(np.clip(np.dot(ha, hd) / length / np.linalg.norm(hd), -1, 1)))
        return angle >= 120
    records, seen = [], set()
    for i in sorted(indices):
        first = bonded[i]
        second = set().union(*(bonded[j] for j in first)) if first else set()
        excluded = {i} | first | second
        third = set().union(*(bonded[j] for j in second)) if second else set()
        distances = np.linalg.norm(xyz - xyz[i], axis=1)
        overlaps = radii[i] + radii - distances
        for j in np.flatnonzero(overlaps > 0.4):
            pair = tuple(sorted((i, int(j))))
            if j in excluded or pair in seen:
                continue
            seen.add(pair)
            category = "possible_hydrogen_bond" if hbond(i, j) else (
                "one_four_overlap" if j in third else "steric_overlap")
            records.append(dict(atoms=[label(i), label(j)], category=category,
                                distance_angstrom=float(distances[j]),
                                overlap_angstrom=float(overlaps[j])))
    return sorted(records, key=lambda x: -x["overlap_angstrom"])


def write_outputs(files):
    """Stage complete files, publish without overwriting, roll back on exceptions.

    Not a multi-file filesystem transaction: power loss/SIGKILL between publishes
    may require cleanup. Ordinary write/publish errors leave no owned final files.
    """
    staged, published = [], []
    try:
        for path, text in files.items():
            path = Path(path)
            require(not path.exists(), f"Refusing to overwrite {path}")
            fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
            temp = Path(name)
            staged.append((temp, path))
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(text)
                handle.flush()
                os.fsync(handle.fileno())
        for temp, path in staged:
            os.link(temp, path)  # atomic exclusive publication; fails if path exists
            published.append(path)
    except BaseException:
        for path in reversed(published):
            path.unlink(missing_ok=True)
        raise
    finally:
        for temp, _ in staged:
            temp.unlink(missing_ok=True)
