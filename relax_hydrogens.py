#!/usr/bin/env python3
"""Hydrogen-only OpenMM minimization of the capped tau PHF in implicit solvent.

Requires preparation_checks.py beside this script. No structure or topology
changes are made: all heavy atoms are immobilized, and only H atoms can move.
Bond constraints are disabled, allowing inherited X-H lengths to be corrected.
Protonation states remain fixed; minimization cannot add/remove protons.

Project defaults: Amber ff14SB with GBn2 implicit solvent. These are selected
starting models, not a biologically validated project protocol. Override with:
    /opt/miniconda3/bin/python relax_hydrogens.py \
        --protein-forcefield amber14/protein.ff14SB.xml --implicit-model gbn2

Other documented starting defaults: salt ionic strength 0.15 mol/L, dielectric
78.3 outside / 1.0 inside, temperature 298.15 K for Debye screening only.
This is minimization, not finite-temperature dynamics. Adjust these defaults
when deciding the protocol; they do not reproduce a neuronal environment.

Outputs: PDB, JSON diagnostics/provenance, and a multi-model PDB of optimization
frames (not a physical-time trajectory). Names resolve beside this script.
No existing outputs are overwritten. Exit status 2 means the saved candidate
failed the convergence or geometry acceptance checks; consult the JSON.
No full-protein stability or aggregation conclusion follows from this step.
"""
import argparse
from collections import Counter
import hashlib
import io
import json
from pathlib import Path
import platform
import time
import xml.etree.ElementTree as ET

import numpy as np
import openmm
from openmm import unit
from openmm.app import ForceField, NoCutoff, PDBFile, element
import openmm.app as app
import preparation_checks as checks

PROTEIN_FIELDS = ('amber14/protein.ff14SB.xml', 'amber99sbildn.xml')
GB_MODELS = ('gbn2', 'gbn', 'obc2', 'obc1', 'hct')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def kappa_nm(ionic_strength, temperature, dielectric):
    # Debye-Huckel inverse length for monovalent electrolyte, ionic strength in M.
    # SI constants; the GB model remains an approximation to ionic screening.
    return float(np.sqrt(2 * 6.02214076e23 * (1.602176634e-19)**2 *
                         1000 * ionic_strength / (8.8541878128e-12 * dielectric *
                                                   1.380649e-23 * temperature)) * 1e-9)


def forcefield_sources(names):
    base = Path(app.__file__).parent / 'data'
    result = {}
    def visit(path):
        path = path.resolve()
        if str(path) in result:
            return
        result[str(path)] = sha(path)
        for include in ET.parse(path).getroot().findall('Include'):
            target = path.parent / include.attrib['file']
            visit(target if target.exists() else base / include.attrib['file'])
    for name in names:
        visit(base / name)
    return result


def build_system(topology, protein_field, implicit_model, solute, solvent, kappa):
    forcefield = ForceField(protein_field, f'implicit/{implicit_model}.xml')
    system = forcefield.createSystem(topology, nonbondedMethod=NoCutoff,
                                    constraints=None, rigidWater=False, removeCMMotion=False,
                                    soluteDielectric=solute, solventDielectric=solvent,
                                    implicitSolventKappa=kappa / unit.nanometer)
    checks.require(system.getNumConstraints() == 0, 'Unexpected bond constraints')
    atoms = list(topology.atoms())
    checks.require(system.getNumParticles() == len(atoms), 'Unexpected extra particles')
    movable = np.array([a.element == element.hydrogen for a in atoms])
    checks.require(movable.any() and (~movable).any(), 'Expected both hydrogens and heavy atoms')
    for i, is_h in enumerate(movable):
        if not is_h:
            system.setParticleMass(i, 0 * unit.dalton)
    nbf = [f for f in system.getForces() if isinstance(f, openmm.NonbondedForce)]
    checks.require(len(nbf) == 1, 'Expected one nonbonded force')
    per_chain = {}
    for a in atoms:
        q = nbf[0].getParticleParameters(a.index)[0].value_in_unit(unit.elementary_charge)
        per_chain[a.residue.chain.id] = per_chain.get(a.residue.chain.id, 0) + q
    checks.require(all(abs(q-6) < 1e-4 for q in per_chain.values()),
                   f'Force-field partial charge sums disagree with +6 per chain: {per_chain}')
    gb = [f for f in system.getForces() if isinstance(f, (openmm.CustomGBForce, openmm.GBSAOBCForce))]
    checks.require(len(gb) == 1, 'Implicit-solvent force not found')
    return system, movable, per_chain


def hydrogen_bond_lengths(system, topology, xyz_nm):
    atoms = list(topology.atoms())
    rows = []
    for force in system.getForces():
        if not isinstance(force, openmm.HarmonicBondForce):
            continue
        for i in range(force.getNumBonds()):
            a, b, length, spring = force.getBondParameters(i)
            if not (atoms[a].element == element.hydrogen or atoms[b].element == element.hydrogen):
                continue
            observed = float(np.linalg.norm(xyz_nm[a]-xyz_nm[b]) * 10)
            target = float(length.value_in_unit(unit.angstrom))
            def label(j):
                x = atoms[j]
                return f'{x.residue.chain.id}:{x.residue.name}{x.residue.id}:{x.name}'
            rows.append(dict(atoms=[label(a), label(b)], length_angstrom=observed,
                             forcefield_equilibrium_angstrom=target,
                             deviation_angstrom=observed-target))
    checks.require(len(rows) == sum(a.element == element.hydrogen for a in atoms),
                   'Not every hydrogen has exactly one parameterized covalent bond')
    return rows


class Progress(openmm.MinimizationReporter):
    def __init__(self, topology, movable, initial_nm, every):
        super().__init__()
        self.topology, self.movable, self.every = topology, movable, every
        self.trace, self.frames, self.calls = [], [np.array(initial_nm)], 0
        self.last_print = time.monotonic()

    def report(self, iteration, x, grad, args):
        self.calls += 1
        gradient = np.asarray(grad).reshape((-1, 3))[self.movable]
        rms = float(np.sqrt(np.mean(gradient**2)))
        self.trace.append(dict(iteration=int(iteration), energy_kj_mol=float(args['system energy']),
                               hydrogen_rms_force_kj_mol_nm=rms))
        if self.calls % self.every == 0:
            self.frames.append(np.asarray(x).reshape((-1, 3)).copy())
        if time.monotonic() - self.last_print >= 15:
            print(f'Iteration {iteration}: hydrogen RMS force {rms:.3g} kJ/mol/nm', flush=True)
            self.last_print = time.monotonic()
        return False


def main():
    folder = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--input', type=Path, default=folder/'ad_tau_5O3L_full_filament_zhou_capped.pdb')
    p.add_argument('--output', type=Path, default=folder/'ad_tau_5O3L_full_filament_hydrogen_relaxation.pdb')
    p.add_argument('--protein-forcefield', choices=PROTEIN_FIELDS, default='amber14/protein.ff14SB.xml',
                   help='Protein force field (default: Amber ff14SB)')
    p.add_argument('--implicit-model', choices=GB_MODELS, default='gbn2',
                   help='Implicit solvent model (default: GBn2)')
    p.add_argument('--salt-molar', type=float, default=0.15, help='Ionic strength in mol/L')
    p.add_argument('--temperature-k', type=float, default=298.15, help='For Debye screening only; no thermostat')
    p.add_argument('--solvent-dielectric', type=float, default=78.3)
    p.add_argument('--solute-dielectric', type=float, default=1.0)
    p.add_argument('--tolerance', type=float, default=1.0, help='Target hydrogen RMS force, kJ/mol/nm')
    p.add_argument('--max-iterations', type=int, default=2000)
    p.add_argument('--save-every', type=int, default=50, help='Optimization iterations between saved frames')
    p.add_argument('--threads', type=int, default=2)
    args = p.parse_args()
    positive = [args.temperature_k, args.solvent_dielectric, args.solute_dielectric, args.tolerance,
                args.max_iterations, args.save_every, args.threads]
    checks.require(all(np.isfinite(x) and x > 0 for x in positive) and
                   np.isfinite(args.salt_molar) and args.salt_molar >= 0, 'Invalid numeric setting')
    source, output = args.input.resolve(), args.output.resolve()
    checks.require(output.suffix.lower() == '.pdb', 'Output must have a .pdb extension')
    report_path, frames_path = output.with_suffix('.json'), output.with_name(output.stem+'_frames.pdb')
    for path in (output, report_path, frames_path):
        checks.require(path != source and not path.exists(), f'Refusing to overwrite {path}')
    input_hash = sha(source)
    pdb = PDBFile(str(source))
    topology = pdb.topology
    checks.formal_charges(topology, capped=True)
    initial_nm = np.array(pdb.positions.value_in_unit(unit.nanometer))
    checks.require(np.isfinite(initial_nm).all(), 'Nonfinite input coordinates')
    # Check the corrected starting geometry; after minimization the force field,
    # not a rigid 1.01 A rule, determines the final H positions.
    checks.histidine_geometry(topology, initial_nm * 10)
    kappa = kappa_nm(args.salt_molar, args.temperature_k, args.solvent_dielectric)
    system, movable, partial_charges = build_system(topology, args.protein_forcefield,
                                                   args.implicit_model, args.solute_dielectric,
                                                   args.solvent_dielectric, kappa)
    start_lengths = hydrogen_bond_lengths(system, topology, initial_nm)
    # Full hydrogen-contact screen includes inherited H atoms, not just caps/His.
    indices = set(np.flatnonzero(movable).tolist())
    before_contacts = checks.contact_screen(topology, initial_nm * 10, indices)
    integrator = openmm.VerletIntegrator(0.001 * unit.picoseconds)  # no MD steps taken
    cpu = openmm.Platform.getPlatformByName('CPU')
    properties = {'Threads': str(args.threads), 'DeterministicForces': 'true'}
    context = openmm.Context(system, integrator, cpu, properties)
    context.setPositions(initial_nm * unit.nanometer)
    start_energy = context.getState(energy=True).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    checks.require(np.isfinite(start_energy), 'Nonfinite initial energy')
    reporter = Progress(topology, movable, initial_nm, args.save_every)
    print(f'Relaxing {int(movable.sum())} hydrogens; {int((~movable).sum())} heavy atoms fixed.', flush=True)
    print(f'{args.protein_forcefield} + {args.implicit_model}; screening kappa={kappa:.4f} /nm.', flush=True)
    # Scale for OpenMM implementations that include zeroed fixed-atom gradients
    # in their RMS criterion. Independently test movable-atom forces afterward.
    internal_tolerance = args.tolerance * np.sqrt(movable.mean())
    start = time.monotonic()
    openmm.LocalEnergyMinimizer.minimize(context, internal_tolerance, args.max_iterations, reporter)
    elapsed = time.monotonic() - start
    state = context.getState(positions=True, forces=True, energy=True)
    final_nm = np.array(state.getPositions().value_in_unit(unit.nanometer))
    forces = np.array(state.getForces().value_in_unit(unit.kilojoule_per_mole/unit.nanometer))
    final_energy = float(state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole))
    checks.require(np.isfinite(final_nm).all() and np.isfinite(forces).all() and np.isfinite(final_energy),
                   'Nonfinite minimization result')
    heavy_shift = float(np.linalg.norm(final_nm[~movable]-initial_nm[~movable], axis=1).max()*10)
    checks.require(heavy_shift < 1e-8, f'Heavy atoms moved by {heavy_shift} A')
    rms = float(np.sqrt(np.mean(forces[movable]**2)))
    end_lengths = hydrogen_bond_lengths(system, topology, final_nm)
    length_outliers = [r for r in end_lengths if abs(r['deviation_angstrom']) > 0.10]
    energy_ok = final_energy <= start_energy + 1e-4
    converged = rms <= args.tolerance
    accepted = bool(converged and energy_ok and not length_outliers)
    status = 'hydrogen_minimization_passed' if accepted else 'needs_review'
    reporter.frames.append(final_nm.copy())
    after_contacts = checks.contact_screen(topology, final_nm*10, indices)
    buffer = io.StringIO()
    for line in [f'HYDROGEN-ONLY IMPLICIT SOLVENT MINIMIZATION: {status}',
                 'HEAVY ATOMS FIXED; PROTONATION AND BOND CONNECTIVITY UNCHANGED',
                 'NOT FULL-PROTEIN RELAXATION OR A STABILITY TEST; SEE JSON']:
        buffer.write(f'REMARK 900 {line}\n')
    PDBFile.writeFile(topology, final_nm*unit.nanometer, buffer, keepIds=True)
    contents = buffer.getvalue()
    reread = PDBFile(io.StringIO(contents))
    charges = checks.formal_charges(reread.topology, capped=True)
    def atomkeys(t):
        return [(a.residue.chain.id,a.residue.id,a.residue.name,a.name) for a in t.atoms()]
    checks.require(atomkeys(topology) == atomkeys(reread.topology), 'PDB changed atom identities')
    written_nm = np.array(reread.positions.value_in_unit(unit.nanometer))
    checks.require(np.allclose(written_nm[~movable], initial_nm[~movable], rtol=0, atol=1e-10),
                   'Saved heavy coordinates changed')
    checks.require(np.allclose(written_nm, final_nm, rtol=0, atol=.0000501), 'Unexpected PDB rounding')
    frames = io.StringIO()
    PDBFile.writeHeader(topology, frames)
    for number, frame in enumerate(reporter.frames, 1):
        PDBFile.writeModel(topology, frame*unit.nanometer, frames, modelIndex=number, keepIds=True)
    PDBFile.writeFooter(topology, frames)
    report = dict(status=status, converged=bool(converged), input=str(source), input_sha256=input_hash,
                  output=str(output), output_sha256=hashlib.sha256(contents.encode()).hexdigest(),
                  script_sha256=sha(__file__), checks_sha256=sha(checks.__file__),
                  forcefield_files=forcefield_sources([args.protein_forcefield, f'implicit/{args.implicit_model}.xml']),
                  openmm=openmm.__version__, numpy=np.__version__, python=platform.python_version(),
                  settings={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
                  platform='CPU', platform_properties=properties, elapsed_seconds=elapsed,
                  implicit_solvent_kappa_per_nm=kappa, constraints=0, heavy_atoms_fixed=int((~movable).sum()),
                  mobile_hydrogens=int(movable.sum()), heavy_atom_max_shift_angstrom=heavy_shift,
                  initial_energy_kj_mol=float(start_energy), final_energy_kj_mol=final_energy,
                  energy_decreased=bool(energy_ok), final_hydrogen_rms_force_kj_mol_nm=rms,
                  final_max_hydrogen_force_kj_mol_nm=float(np.linalg.norm(forces[movable],axis=1).max()),
                  formal_charge_check=charges, forcefield_charge_per_chain=partial_charges,
                  initial_hydrogen_bond_lengths=start_lengths, final_hydrogen_bond_lengths=end_lengths,
                  bond_length_outliers_over_0_10_angstrom=length_outliers,
                  contact_rule=checks.CONTACT_RULE, contact_scope='All pairs involving at least one hydrogen',
                  contacts_before=before_contacts, contacts_after=after_contacts,
                  contact_counts_before=dict(Counter(r['category'] for r in before_contacts)),
                  contact_counts_after=dict(Counter(r['category'] for r in after_contacts)),
                  optimization_trace=reporter.trace, trajectory=str(frames_path), frames=len(reporter.frames),
                  trajectory_sha256=hashlib.sha256(frames.getvalue().encode()).hexdigest(),
                  note='Optimization frames are not physical-time dynamics. Remaining contact warnings need review; no assembly stability inference.')
    checks.require(sha(source)==input_hash, 'Input changed during run')
    checks.write_outputs({output:contents, report_path:json.dumps(report,indent=2)+'\n', frames_path:frames.getvalue()})
    print(f'Saved candidate: {output}\nReport: {report_path}\nOptimization frames: {frames_path}')
    print(f'Status: {status}; hydrogen RMS force {rms:.5g} kJ/mol/nm; heavy atoms unchanged.')
    print(f'Contacts after: {report["contact_counts_after"]} (geometric screen, not an energy criterion).')
    if not accepted:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
