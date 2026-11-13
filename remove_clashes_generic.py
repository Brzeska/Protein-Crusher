#!/usr/bin/env python3
"""Conditional soft-core clash cleanup before ordinary hydrogen relaxation.

Only severe heavy-atom overlaps (> --severe-overlap) trigger repair. Inputs without
them are copied byte-for-byte without constructing an OpenMM Context.
Triggered inputs use bonded forces, soft-core LJ, CA, trans-omega and L-chirality restraints.
Exit 2 retains a candidate/report that failed convergence or geometry gates.
Protonation is inherited: rerun assignment/H addition if affected pKa matters.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import time

import numpy as np
import openmm
from openmm import unit
from openmm.app import PDBFile, element

import preparation_checks as checks
import clash_geometry as geometry
from relax_full_structure_generic import build_system, bond_lengths, Progress
from relax_hydrogens_generic import create_compute_context, forcefield_sources, kappa_nm, sha

DEFAULT_ALPHAS = (0.5, 0.2, 0.05, 0.01, 0.001)
DEFAULT_CHIRAL_K = 1000.
# Normal L centers have signed volumes of ~2.4-2.8 A^3 (5O3L: 2.42-2.78), so the
# chirality restraint is zero for them and only acts on flattening or inversion.
CHIRAL_FLOOR_ANGSTROM3 = 2.0
# After cleanup, only require each peptide to be on the trans side with margin
# from the 90 degree barrier; full relaxation can flatten the rest, and the final
# check reports any peptide still beyond the usual 30 degree limit.
CLEANUP_MAX_OMEGA_DEVIATION = 60.


def build_soft_system(original, topology, initial_nm, ca_k=1000., omega_k=100., chiral_k=DEFAULT_CHIRAL_K):
    """Clone, retaining only known bonded forces; never mutate the real system.

    Exact proposed softened LJ (including its attractive well). Exception pairs,
    including scaled 1-4 interactions, are all excluded as requested. GB and
    Coulomb are absent. Zero-LJ sites retain epsilon=0 and get a safe sigma.
    """
    checks.require(original.getNumConstraints() == 0, 'Cleanup requires unconstrained atoms')
    checks.require(original.getNumParticles() == topology.getNumAtoms(), 'Unexpected particles')
    checks.require(not any(original.isVirtualSite(i) for i in range(original.getNumParticles())), 'Virtual sites unsupported')
    system = openmm.XmlSerializer.deserialize(openmm.XmlSerializer.serialize(original))
    nonbonded = [f for f in original.getForces() if isinstance(f, openmm.NonbondedForce)]
    checks.require(len(nonbonded) == 1, 'Expected one NonbondedForce')
    nb = nonbonded[0]
    checks.require(nb.getNonbondedMethod() == openmm.NonbondedForce.NoCutoff, 'Only nonperiodic NoCutoff supported')
    checks.require(nb.getNumParticleParameterOffsets() == 0 and nb.getNumExceptionParameterOffsets() == 0,
                   'Nonbonded parameter offsets unsupported')
    bonded = (openmm.HarmonicBondForce, openmm.HarmonicAngleForce, openmm.PeriodicTorsionForce, openmm.RBTorsionForce)
    removed = (openmm.NonbondedForce, openmm.CustomGBForce, openmm.GBSAOBCForce, openmm.CMMotionRemover)
    for i in reversed(range(system.getNumForces())):
        force = system.getForce(i)
        checks.require(isinstance(force, bonded+removed), f'Unsupported force in cleanup: {type(force).__name__}')
        if isinstance(force, removed):
            system.removeForce(i)
    force = openmm.CustomNonbondedForce(
        '4*epsilon*(1/u^2-1/u); u=alpha+(r/sigma)^6; '
        'sigma=0.5*(sigma1+sigma2); epsilon=sqrt(epsilon1*epsilon2)')
    force.setName('Soft-core LJ (temporary)')
    force.addGlobalParameter('alpha', DEFAULT_ALPHAS[0])
    force.addPerParticleParameter('sigma')
    force.addPerParticleParameter('epsilon')
    force.setNonbondedMethod(openmm.CustomNonbondedForce.NoCutoff)
    for i in range(nb.getNumParticles()):
        _, sigma, epsilon = nb.getParticleParameters(i)
        s = sigma.value_in_unit(unit.nanometer)
        e = epsilon.value_in_unit(unit.kilojoule_per_mole)
        checks.require(np.isfinite([s, e]).all() and s >= 0 and e >= 0, 'Invalid LJ parameters')
        force.addParticle([max(s, 1e-6), e])
    for i in range(nb.getNumExceptions()):
        a, b, _, _, _ = nb.getExceptionParameters(i)
        force.addExclusion(a, b)
    system.addForce(force)
    ca = openmm.CustomExternalForce('0.5*ca_k*((x-x0)^2+(y-y0)^2+(z-z0)^2)')
    ca.setName('Temporary CA restraints')
    ca.addGlobalParameter('ca_k', ca_k)
    for name in ('x0', 'y0', 'z0'):
        ca.addPerParticleParameter(name)
    for atom in topology.atoms():
        if atom.name == 'CA' and atom.element == element.carbon:
            ca.addParticle(atom.index, list(initial_nm[atom.index]))
    system.addForce(ca)
    # Periodic and smooth at +/-pi; near trans this is 0.5*k*delta^2.
    omega = openmm.CustomTorsionForce('omega_k*(1+cos(theta))')
    omega.setName('Temporary trans peptide restraints')
    omega.addGlobalParameter('omega_k', omega_k)
    for ids in geometry.peptide_torsions(topology):
        omega.addTorsion(*ids, [])
    system.addForce(omega)
    # ff14SB has no chirality term; only angle bending resists inversion, and with
    # 1-4 and electrostatic terms removed a deep overlap can push a substituent
    # through the plane. v is the gate's own signed volume (A^3), so no dihedral
    # sign convention is involved. Zero above chiral_floor; quadratic below it.
    chiral = openmm.CustomCompoundBondForce(4,
        'chiral_k*min(0, v/chiral_floor-1)^2; '
        'v=1000*(ax*(by*cz-bz*cy)-ay*(bx*cz-bz*cx)+az*(bx*cy-by*cx)); '
        'ax=x2-x1; ay=y2-y1; az=z2-z1; bx=x3-x1; by=y3-y1; bz=z3-z1; cx=x4-x1; cy=y4-y1; cz=z4-z1')
    chiral.setName('Temporary L-chirality restraints')
    chiral.addGlobalParameter('chiral_k', chiral_k)
    chiral.addGlobalParameter('chiral_floor', CHIRAL_FLOOR_ANGSTROM3)
    for center, arms in geometry.stereocenters(topology):
        chiral.addBond([center, *arms], [])
    system.addForce(chiral)
    return system


def separate_coincident_atoms(xyz_nm, seed=2026, minimum_nm=1e-3):
    """Deterministically break r=0 symmetry only on the repair path.

    A finite pair energy alone does not define a radial force direction at zero.
    This is a small initial perturbation, not a clash-removal algorithm.
    """
    xyz = np.array(xyz_nm, dtype=float, copy=True)
    rng = np.random.default_rng(seed)
    moved = []
    for i in range(1, len(xyz)):
        for attempt in range(100):
            if np.all(np.linalg.norm(xyz[:i]-xyz[i], axis=1) >= minimum_nm):
                break
            direction = rng.normal(size=3)
            xyz[i] += 2*minimum_nm*direction/np.linalg.norm(direction)
            if i not in moved:
                moved.append(i)
        else:
            raise ValueError('Unable to separate coincident atoms')
    return xyz, moved


def atomkeys(topology):
    return [(a.residue.chain.index, a.residue.chain.id, a.residue.id,
             a.residue.insertionCode, a.residue.name, a.name) for a in topology.atoms()]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--platform', choices=('auto', 'CPU', 'CUDA', 'HIP', 'Metal'), default='auto')
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--max-iterations', type=int, default=2000, help='Per alpha round')
    p.add_argument('--tolerance', type=float, default=10., help='Target RMS force, kJ/mol/nm')
    p.add_argument('--save-every', type=int, default=5)
    p.add_argument('--alphas', type=float, nargs='+', default=DEFAULT_ALPHAS)
    p.add_argument('--ca-k', type=float, default=1000., help='kJ/mol/nm^2; energy is k*displacement^2/2')
    p.add_argument('--omega-k', type=float, default=100., help='kJ/mol, energy k*(1+cos(omega))')
    p.add_argument('--chiral-k', type=float, default=DEFAULT_CHIRAL_K,
                   help=f'kJ/mol; energy k*min(0, V/{CHIRAL_FLOOR_ANGSTROM3:g}-1)^2 on each gated stereocenter volume V (A^3)')
    p.add_argument('--severe-overlap', type=float, default=geometry.SEVERE_OVERLAP_ANGSTROM,
                   help='Angstrom; heavy-atom Bondi overlaps above this trigger cleanup and fail its gates')
    p.add_argument('--seed', type=int, default=2026)
    args = p.parse_args()
    checks.require(all(np.isfinite(v) and v > 0 for v in
                       [args.threads, args.max_iterations, args.tolerance, args.save_every, args.ca_k, args.omega_k, args.chiral_k, *args.alphas]),
                   'Settings must be finite and positive')
    checks.require(all(a > b for a, b in zip(args.alphas, args.alphas[1:])) and max(args.alphas) < 1,
                   'Alphas must decrease strictly and lie between zero and one')
    checks.require(np.isfinite(args.severe_overlap) and args.severe_overlap >= 0.4,
                   'Severe overlap must be finite and at least the 0.40 A screen threshold')
    source, output = args.input.resolve(), args.output.resolve()
    report_path, frames_path = output.with_suffix('.json'), output.with_name(output.stem+'_frames.pdb')
    checks.require(output.suffix.lower() == '.pdb', 'Output must end in .pdb')
    for path in (output, report_path, frames_path):
        checks.require(path != source and not path.exists(), f'Refusing to overwrite {path}')
    original_bytes = source.read_bytes()
    input_hash = hashlib.sha256(original_bytes).hexdigest()
    pdb = PDBFile(str(source))
    topology = pdb.topology
    atoms = list(topology.atoms())
    checks.require(pdb.getNumFrames() == 1 and atoms, 'Expected nonempty single-model PDB')
    checks.require(all(a.element is not None and a.element.symbol in checks.RADII for a in atoms), 'Unsupported elements')
    checks.require(any(a.element == element.hydrogen for a in atoms), 'Add hydrogens before cleanup')
    xyz = np.array(pdb.positions.value_in_unit(unit.nanometer))
    checks.require(np.isfinite(xyz).all(), 'Nonfinite input coordinates')
    contacts = geometry.heavy_contacts(topology, xyz*10)
    trigger = geometry.severe(contacts, args.severe_overlap)
    report = dict(input=str(source), input_sha256=input_hash, output=str(output),
                  script_sha256=sha(__file__), checks_sha256=sha(checks.__file__), geometry_sha256=sha(geometry.__file__),
                  openmm=openmm.__version__, numpy=np.__version__,
                  settings={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                  contact_rule=checks.CONTACT_RULE, severe_overlap_angstrom=args.severe_overlap,
                  chiral_floor_angstrom3=CHIRAL_FLOOR_ANGSTROM3, contacts_before=contacts,
                  triggered=bool(trigger), protonation_preserved=True)
    if not trigger:
        # Preserve the entire input file, including REMARK records and PDB rounding.
        report.update(status='skipped_no_severe_overlaps', output_sha256=input_hash,
                      note='No Context, minimization, perturbation, or extra geometry gate on clean inputs.')
        checks.require(sha(source) == input_hash, 'Input changed during screening')
        # PDB is ASCII; retain exact line endings with newline="" in write_outputs.
        checks.write_outputs({output: original_bytes.decode('ascii'), report_path: json.dumps(report, indent=2)+'\n'})
        print(f'Cleanup skipped: no heavy-atom overlaps above {args.severe_overlap:g} A. Input copied unchanged.')
        return
    print(f'Cleanup triggered by {len(trigger)} heavy-atom overlaps above {args.severe_overlap:g} A.', flush=True)
    fields = ['amber14/protein.ff14SB.xml', 'implicit/gbn2.xml']
    original, movable, _ = build_system(topology, fields, 1., 78.3, kappa_nm(.15, 298.15, 78.3))
    soft = build_soft_system(original, topology, xyz, args.ca_k, args.omega_k, args.chiral_k)
    report['geometry_before'] = geometry.geometry_report(topology, xyz*10, contacts, severe_overlap=args.severe_overlap)
    start_nm, moved = separate_coincident_atoms(xyz, args.seed)
    context, integrator, backend, properties, failures = create_compute_context(soft, args.threads, args.platform)
    context.setPositions(start_nm*unit.nanometer)
    reporter = Progress(topology, movable, start_nm, args.save_every)
    rounds = []
    started = time.monotonic()
    for alpha in args.alphas:
        context.setParameter('alpha', alpha)
        initial = context.getState(energy=True, forces=True)
        energy = float(initial.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole))
        checks.require(np.isfinite(energy) and np.isfinite(initial.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole/unit.nanometer)).all(), 'Nonfinite starting energy/forces')
        print(f'Soft-core alpha={alpha:g}', flush=True)
        calls = reporter.calls
        openmm.LocalEnergyMinimizer.minimize(context, args.tolerance, args.max_iterations, reporter)
        state = context.getState(positions=True, energy=True, forces=True)
        final_nm = np.array(state.getPositions(asNumpy=True).value_in_unit(unit.nanometer))
        forces = np.array(state.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole/unit.nanometer))
        end_energy = float(state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole))
        checks.require(np.isfinite(final_nm).all() and np.isfinite(forces).all() and np.isfinite(end_energy), 'Nonfinite cleanup result')
        rms = float(np.sqrt(np.mean(forces**2)))
        rounds.append(dict(alpha=alpha, iterations=reporter.calls-calls, initial_energy_kj_mol=energy,
                           final_energy_kj_mol=end_energy, rms_force_kj_mol_nm=rms,
                           converged=rms <= args.tolerance, energy_decreased=end_energy <= energy+1e-4))
        reporter.frames.append(final_nm.copy())
    # Serialize and gate the exact coordinates handed to the next script.
    buf = io.StringIO()
    buf.write('REMARK 900 TEMPORARY SOFT-CORE CLASH CLEANUP; SEE JSON FOR ACCEPTANCE\n')
    PDBFile.writeFile(topology, final_nm*unit.nanometer, buf, keepIds=True)
    text = buf.getvalue()
    reread = PDBFile(io.StringIO(text))
    bonds = lambda t: sorted(tuple(sorted((a.index, b.index))) for a, b in t.bonds())
    checks.require(atomkeys(topology) == atomkeys(reread.topology) and bonds(topology) == bonds(reread.topology), 'PDB roundtrip changed atoms or connectivity')
    written = np.array(reread.positions.value_in_unit(unit.nanometer))
    gate = geometry.geometry_report(reread.topology, written*10, severe_overlap=args.severe_overlap,
                                    max_omega_deviation=CLEANUP_MAX_OMEGA_DEVIATION)
    lengths = [r for r in bond_lengths(original, topology, written) if abs(r['deviation_angstrom']) > .10]
    accepted = bool(rounds[-1]['converged'] and all(r['energy_decreased'] for r in rounds) and gate['passed'] and not lengths)
    frames = io.StringIO()
    PDBFile.writeHeader(topology, frames)
    for number, frame in enumerate(reporter.frames, 1):
        PDBFile.writeModel(topology, frame*unit.nanometer, frames, modelIndex=number, keepIds=True)
    PDBFile.writeFooter(topology, frames)
    report.update(status='clash_cleanup_passed' if accepted else 'needs_review', converged=rounds[-1]['converged'],
                  output_sha256=hashlib.sha256(text.encode()).hexdigest(), forcefield_files=forcefield_sources(fields),
                  platform=backend, platform_properties=properties, gpu_initialization_failures=failures,
                  elapsed_seconds=time.monotonic()-started, rounds=rounds, geometry_after=gate,
                  bond_length_outliers=lengths, perturbed_atom_indices=moved,
                  maximum_ca_shift_angstrom=max((float(np.linalg.norm(written[a.index]-xyz[a.index])*10) for a in atoms if a.name == 'CA'), default=0.),
                  trajectory=str(frames_path), trajectory_sha256=hashlib.sha256(frames.getvalue().encode()).hexdigest(),
                  note='Temporary softened LJ includes attraction; all original nonbonded exceptions excluded. No Coulomb/GB. '
                       'Only coordinates are handed off; downstream scripts rebuild the original force field without restraints. '
                       'Protonation was sampled from model pKas and does not depend on geometry, but His tautomers were chosen '
                       'from the pre-cleanup geometry: reassess them on the cleaned structure if those sites matter.')
    checks.require(sha(source) == input_hash, 'Input changed during cleanup')
    checks.write_outputs({output: text, report_path: json.dumps(report, indent=2)+'\n', frames_path: frames.getvalue()})
    print(f'Cleanup: {report["status"]}. Candidate: {output}; report: {report_path}', flush=True)
    if not accepted:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
