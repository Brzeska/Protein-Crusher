#!/usr/bin/env python3
"""Relax all atoms in a prepared protein PDB using implicit solvent.

Usage: python relax_full_structure_generic.py --input prepared.pdb
Requires OpenMM, numpy, and preparation_checks.py alongside this script.
Input must already contain complete hydrogens, deliberate protonation and
force-field-compatible termini. Does NOT add atoms, caps, or choose protonation.
No tau-specific chain count, sequence, residue numbering, or charge assumptions.
Only single-model PDBs are accepted. Ligands/nonstandard residues require
compatible parameter files supplied using repeated --forcefield options.
All atoms move; no position restraints. Reports aligned heavy-atom RMSD and
checks all harmonic bond lengths (0.10 A deviation flags review).
Default force fields: amber14/protein.ff14SB.xml and implicit/gbn2.xml.
No explicit solvent, constraints, or dynamics; default ionic strength 0.15 M,
dielectrics 1/78.3, temperature 298.15 K for screening only.
Tries CUDA, HIP, Metal, then CPU with a warning. OpenCL is not auto-selected
because it can expose CPU devices. --platform CPU forces CPU execution.
Saves a new PDB, JSON report, and optimization frames every five iterations.
Refuses overwrites; exit 2 means the saved result needs review. Can restart
from a previous relaxed PDB by selecting a different --output filename.
"""
import argparse
from collections import Counter
import hashlib
import io
import json
from pathlib import Path
import platform
import time
import warnings
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
        visit(Path(name) if Path(name).is_file() else base / name)
    return result


def build_system(topology, fields, solute, solvent, kappa):
    forcefield = ForceField(*fields)
    system = forcefield.createSystem(topology, nonbondedMethod=NoCutoff,
                                    constraints=None, rigidWater=False, removeCMMotion=False,
                                    soluteDielectric=solute, solventDielectric=solvent,
                                    implicitSolventKappa=kappa / unit.nanometer)
    checks.require(system.getNumConstraints() == 0, 'Unexpected bond constraints')
    atoms = list(topology.atoms())
    checks.require(system.getNumParticles() == len(atoms), 'Unexpected extra particles')
    movable = np.ones(len(atoms), dtype=bool)
    checks.require(all(system.getParticleMass(i).value_in_unit(unit.dalton)>0
                       for i in range(len(atoms))), 'All atoms must have positive mass')
    nbf = [f for f in system.getForces() if isinstance(f, openmm.NonbondedForce)]
    checks.require(len(nbf) == 1, 'Expected one nonbonded force')
    per_chain = {}
    for a in atoms:
        q = nbf[0].getParticleParameters(a.index)[0].value_in_unit(unit.elementary_charge)
        per_chain[f'{a.residue.chain.index}:{a.residue.chain.id}'] = per_chain.get(f'{a.residue.chain.index}:{a.residue.chain.id}', 0) + q
    gb = [f for f in system.getForces() if isinstance(f, (openmm.CustomGBForce, openmm.GBSAOBCForce))]
    checks.require(len(gb) == 1, 'Implicit-solvent force not found')
    return system, movable, per_chain


def bond_lengths(system, topology, xyz_nm):
    atoms = list(topology.atoms())
    rows = []
    for force in system.getForces():
        if not isinstance(force, openmm.HarmonicBondForce):
            continue
        for i in range(force.getNumBonds()):
            a, b, length, spring = force.getBondParameters(i)
            observed = float(np.linalg.norm(xyz_nm[a]-xyz_nm[b]) * 10)
            target = float(length.value_in_unit(unit.angstrom))
            def label(j):
                x = atoms[j]
                return f'{x.residue.chain.id}:{x.residue.name}{x.residue.id}:{x.name}'
            rows.append(dict(atoms=[label(a), label(b)], length_angstrom=observed,
                             forcefield_equilibrium_angstrom=target,
                             deviation_angstrom=observed-target))
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
                               all_atom_rms_force_kj_mol_nm=rms))
        if self.calls % self.every == 0:
            self.frames.append(np.asarray(x).reshape((-1, 3)).copy())
        if time.monotonic() - self.last_print >= 15:
            print(f'Iteration {iteration}: all-atom RMS force {rms:.3g} kJ/mol/nm', flush=True)
            self.last_print = time.monotonic()
        return False


def create_compute_context(system, threads, requested="auto"):
    """Try GPU-only backends; successful Context creation verifies availability.

    OpenCL is not auto-selected because it can expose CPU devices as well.
    Each attempt gets a fresh integrator, including after initialization failure.
    """
    available = {openmm.Platform.getPlatform(i).getName()
                 for i in range(openmm.Platform.getNumPlatforms())}
    failures = {}
    for name in (('CUDA', 'HIP', 'Metal') if requested == 'auto' else (() if requested == 'CPU' else (requested,))):
        if name not in available:
            continue
        backend = openmm.Platform.getPlatformByName(name)
        properties = {'Precision': 'mixed'} if 'Precision' in backend.getPropertyNames() else {}
        integrator = openmm.VerletIntegrator(0.001 * unit.picoseconds)
        try:
            context = openmm.Context(system, integrator, backend, properties)
        except Exception as exc:
            failures[name] = str(exc)
            print(f'{name} GPU initialization failed: {exc}', flush=True)
            continue
        actual = {key: backend.getPropertyValue(context, key)
                  for key in backend.getPropertyNames()}
        print(f'Using {name} GPU acceleration.', flush=True)
        return context, integrator, name, actual, failures
    if requested not in ('auto', 'CPU'):
        raise RuntimeError(f'Requested {requested} backend unavailable: {failures}')
    if requested == 'auto':
        warnings.warn(
        'No usable CUDA, HIP, or Metal GPU backend found. Falling back to CPU; '
        'this will generally be significantly slower than a supported discrete GPU.',
        RuntimeWarning, stacklevel=2)
    backend = openmm.Platform.getPlatformByName('CPU')
    properties = {'Threads': str(threads), 'DeterministicForces': 'true'}
    integrator = openmm.VerletIntegrator(0.001 * unit.picoseconds)
    context = openmm.Context(system, integrator, backend, properties)
    actual = {key: backend.getPropertyValue(context, key)
              for key in backend.getPropertyNames()}
    print(f'Using CPU with {threads} threads.', flush=True)
    return context, integrator, 'CPU', actual, failures

def main():
    folder = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output', type=Path, help='Default: INPUT_STEM_full_relaxation.pdb')
    p.add_argument('--platform', choices=('auto','CPU','CUDA','HIP','Metal'), default='auto')
    p.add_argument('--forcefield', action='append', help='XML file; repeat to replace the default protein and implicit XML pair')
    p.add_argument('--protein-forcefield', choices=PROTEIN_FIELDS, default='amber14/protein.ff14SB.xml',
                   help='Protein force field (default: Amber ff14SB)')
    p.add_argument('--implicit-model', choices=GB_MODELS, default='gbn2',
                   help='Implicit solvent model (default: GBn2)')
    p.add_argument('--salt-molar', type=float, default=0.15, help='Ionic strength in mol/L')
    p.add_argument('--temperature-k', type=float, default=298.15, help='For Debye screening only; no thermostat')
    p.add_argument('--solvent-dielectric', type=float, default=78.3)
    p.add_argument('--solute-dielectric', type=float, default=1.0)
    p.add_argument('--tolerance', type=float, default=10., help='Target all-atom RMS force, kJ/mol/nm')
    p.add_argument('--max-iterations', type=int, default=2000)
    p.add_argument('--save-every', type=int, default=5, help='Optimization iterations between saved frames')
    p.add_argument('--threads', type=int, default=2)
    args = p.parse_args()
    positive = [args.temperature_k, args.solvent_dielectric, args.solute_dielectric, args.tolerance,
                args.max_iterations, args.save_every, args.threads]
    checks.require(all(np.isfinite(x) and x > 0 for x in positive) and
                   np.isfinite(args.salt_molar) and args.salt_molar >= 0, 'Invalid numeric setting')
    source = args.input.resolve()
    output = (args.output or source.with_name(source.stem+'_full_relaxation.pdb')).resolve()
    checks.require(output.suffix.lower() == '.pdb', 'Output must have a .pdb extension')
    report_path, frames_path = output.with_suffix('.json'), output.with_name(output.stem+'_frames.pdb')
    for path in (output, report_path, frames_path):
        checks.require(path != source and not path.exists(), f'Refusing to overwrite {path}')
    input_hash = sha(source)
    pdb = PDBFile(str(source))
    topology = pdb.topology
    checks.require(pdb.getNumFrames() == 1, 'Input has multiple models; select one model first')
    atoms = list(topology.atoms())
    checks.require(any(a.element == element.hydrogen for a in atoms),
                   'No hydrogens found. Add hydrogens and choose protonation before relaxation.')
    checks.require(all(a.element is not None for a in atoms), 'Unknown atom elements')
    initial_nm = np.array(pdb.positions.value_in_unit(unit.nanometer))
    checks.require(np.isfinite(initial_nm).all(), 'Nonfinite input coordinates')
    kappa = kappa_nm(args.salt_molar, args.temperature_k, args.solvent_dielectric)
    fields = args.forcefield or [args.protein_forcefield, f'implicit/{args.implicit_model}.xml']
    try:
        system, movable, partial_charges = build_system(topology, fields, args.solute_dielectric,
                                                       args.solvent_dielectric, kappa)
    except ValueError as exc:
        raise ValueError('Input cannot be parameterized. Check missing atoms/hydrogens, termini, and residue templates. Original error: '+str(exc)) from exc
    start_lengths = bond_lengths(system, topology, initial_nm)
    # Full-system geometric contact screen, distinct from force-field energy.
    indices = set(np.flatnonzero(movable).tolist())
    contact_supported = all(a.element.symbol in checks.RADII for a in atoms)
    before_contacts = checks.contact_screen(topology, initial_nm * 10, indices) if contact_supported else []
    context, integrator, platform_name, properties, failures = create_compute_context(system, args.threads, args.platform)
    context.setPositions(initial_nm * unit.nanometer)
    start_energy = context.getState(energy=True).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    checks.require(np.isfinite(start_energy), 'Nonfinite initial energy')
    reporter = Progress(topology, movable, initial_nm, args.save_every)
    print(f'Relaxing all {int(movable.sum())} atoms; no positional restraints.', flush=True)
    print(f'Force fields: {fields}; screening kappa={kappa:.4f} /nm.', flush=True)
    # Independently verify final forces; OpenMM can return before convergence.
    internal_tolerance = args.tolerance
    start = time.monotonic()
    openmm.LocalEnergyMinimizer.minimize(context, internal_tolerance, args.max_iterations, reporter)
    elapsed = time.monotonic() - start
    state = context.getState(positions=True, forces=True, energy=True)
    final_nm = np.array(state.getPositions().value_in_unit(unit.nanometer))
    forces = np.array(state.getForces().value_in_unit(unit.kilojoule_per_mole/unit.nanometer))
    final_energy = float(state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole))
    checks.require(np.isfinite(final_nm).all() and np.isfinite(forces).all() and np.isfinite(final_energy),
                   'Nonfinite minimization result')
    heavy = np.array([a.element != element.hydrogen for a in atoms])
    checks.require(heavy.any(), 'No heavy atoms')
    heavy_shift = float(np.linalg.norm(final_nm[heavy]-initial_nm[heavy], axis=1).max()*10)
    x = initial_nm[heavy]-initial_nm[heavy].mean(axis=0)
    y = final_nm[heavy]-final_nm[heavy].mean(axis=0)
    u, _, vt = np.linalg.svd(y.T @ x)
    correction = np.eye(3)
    correction[2,2] = np.linalg.det(u @ vt)
    aligned_rmsd = float(np.sqrt(np.mean(np.sum((y @ (u @ correction @ vt)-x)**2,axis=1)))*10)
    rms = float(np.sqrt(np.mean(forces[movable]**2)))
    end_lengths = bond_lengths(system, topology, final_nm)
    length_outliers = [r for r in end_lengths if abs(r['deviation_angstrom']) > 0.10]
    energy_ok = final_energy <= start_energy + 1e-4
    converged = rms <= args.tolerance
    accepted = bool(converged and energy_ok and not length_outliers)
    status = 'full_atom_minimization_passed' if accepted else 'needs_review'
    stop_assessment = ('force_target_met' if converged else
                       'iteration_limit_reached' if reporter.calls >= args.max_iterations else
                       'returned_before_limit_without_force_convergence')
    reporter.frames.append(final_nm.copy())
    after_contacts = checks.contact_screen(topology, final_nm*10, indices) if contact_supported else []
    buffer = io.StringIO()
    for line in [f'FULL-ATOM IMPLICIT SOLVENT MINIMIZATION: {status}',
                 'ALL ATOMS MOBILE; PROTONATION AND BOND CONNECTIVITY UNCHANGED',
                 'MINIMIZATION DOES NOT ESTABLISH THERMODYNAMIC STABILITY; SEE JSON']:
        buffer.write(f'REMARK 900 {line}\n')
    PDBFile.writeFile(topology, final_nm*unit.nanometer, buffer, keepIds=True)
    contents = buffer.getvalue()
    reread = PDBFile(io.StringIO(contents))
    bonds = lambda t: sorted(tuple(sorted((a.index,b.index))) for a,b in t.bonds())
    checks.require(bonds(topology) == bonds(reread.topology), 'PDB roundtrip changed connectivity')
    def atomkeys(t):
        return [(a.residue.chain.index,a.residue.chain.id,a.residue.id,a.residue.insertionCode.strip(),a.residue.name,a.name) for a in t.atoms()]
    checks.require(atomkeys(topology) == atomkeys(reread.topology), 'PDB changed atom identities')
    written_nm = np.array(reread.positions.value_in_unit(unit.nanometer))
    checks.require(np.allclose(written_nm, final_nm, rtol=0, atol=.0000501), 'Unexpected PDB rounding')
    frames = io.StringIO()
    PDBFile.writeHeader(topology, frames)
    for number, frame in enumerate(reporter.frames, 1):
        PDBFile.writeModel(topology, frame*unit.nanometer, frames, modelIndex=number, keepIds=True)
    PDBFile.writeFooter(topology, frames)
    report = dict(status=status, stop_assessment=stop_assessment,
                  optimizer_exit_code_available=False, iterations_reported=reporter.calls,
                  converged=bool(converged), input=str(source), input_sha256=input_hash,
                  output=str(output), output_sha256=hashlib.sha256(contents.encode()).hexdigest(),
                  script_sha256=sha(__file__), checks_sha256=sha(checks.__file__),
                  forcefield_files=forcefield_sources(fields),
                  openmm=openmm.__version__, numpy=np.__version__, python=platform.python_version(),
                  settings={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
                  platform=platform_name, platform_properties=properties, gpu_initialization_failures=failures,
                  elapsed_seconds=elapsed,
                  implicit_solvent_kappa_per_nm=kappa, constraints=0, heavy_atoms_fixed=int((~movable).sum()),
                  mobile_atoms=int(movable.sum()), heavy_atom_max_shift_angstrom=heavy_shift,
                  aligned_heavy_atom_rmsd_angstrom=aligned_rmsd,
                  initial_energy_kj_mol=float(start_energy), final_energy_kj_mol=final_energy,
                  energy_decreased=bool(energy_ok), final_all_atom_rms_force_kj_mol_nm=rms,
                  final_max_atom_force_kj_mol_nm=float(np.linalg.norm(forces[movable],axis=1).max()),
                  forcefield_charge_per_chain=partial_charges, protonation_preserved=True,
                  contact_screen_performed=contact_supported,
                  initial_bond_lengths=start_lengths, final_bond_lengths=end_lengths,
                  bond_length_outliers_over_0_10_angstrom=length_outliers,
                  contact_rule=checks.CONTACT_RULE, contact_scope='All atom pairs',
                  contacts_before=before_contacts, contacts_after=after_contacts,
                  contact_counts_before=dict(Counter(r['category'] for r in before_contacts)),
                  contact_counts_after=dict(Counter(r['category'] for r in after_contacts)),
                  optimization_trace=reporter.trace, trajectory=str(frames_path), frames=len(reporter.frames),
                  trajectory_sha256=hashlib.sha256(frames.getvalue().encode()).hexdigest(),
                  note='Optimization frames are not physical-time dynamics. Remaining contact warnings need review; no assembly stability inference.')
    checks.require(sha(source)==input_hash, 'Input changed during run')
    checks.write_outputs({output:contents, report_path:json.dumps(report,indent=2)+'\n', frames_path:frames.getvalue()})
    print(f'Saved candidate: {output}\nReport: {report_path}\nOptimization frames: {frames_path}')
    print(f'Status: {status}; all-atom RMS force {rms:.5g} kJ/mol/nm; all atoms were mobile.')
    print(f'Stopping assessment: {stop_assessment}; internal optimizer exit code is not exposed.')
    print(f'Contacts after: {report["contact_counts_after"]} (geometric screen, not an energy criterion).')
    if not accepted:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
