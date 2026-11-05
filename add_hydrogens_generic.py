#!/usr/bin/env python3
"""Add hydrogens to a prepared, single-model protein PDB.

Example: python add_hydrogens_generic.py --input capped.pdb --ph 7.0
Optional --variants variants.json accepts {"A:8": "HIP", "A:14": "HIE"}.
CYM requests negatively charged cysteine (thiolate), with peptide-bonded/capped ends.
Keys are chain ID:residue ID with insertion code appended (e.g. A:14B).
Existing hydrogens are rejected unless --existing-hydrogens keep or replace
is specified. Keep retains existing H unless an explicit variant overrides it;
replace removes ALL input hydrogens before assigning states at the chosen pH.
No caps, heavy atoms, missing residues, or terminal oxygens are added.
Standard proteins are supported; arbitrary ligands require additional hydrogen
and force-field definitions beyond this script's built-in scope.
Uses ff14SB by default. OpenMM positions new H with this force field (vacuum
placement optimization); run the separate implicit-solvent H relaxation next.
Requires OpenMM, numpy, and preparation_checks.py beside this script.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import platform as pyplatform
import random
import warnings
import xml.etree.ElementTree as ET

import numpy as np
import openmm
from openmm import unit
from openmm.app import PDBFile, Modeller, ForceField, NoCutoff, element
import openmm.app as app
import preparation_checks as checks

VARIANTS = {
    'ASP': {'ASP', 'ASH'}, 'GLU': {'GLU', 'GLH'},
    'HIS': {'HID', 'HIE', 'HIP', 'HIN'},
    'CYS': {'CYS', 'CYX', 'CYM'}, 'LYS': {'LYS', 'LYN'},
}

# Absolute displacement guard: 0.0001 A, below PDB's 0.001 A resolution.
# Accept numerical roundoff only, then restore original coordinates exactly.
HEAVY_ROUNDOFF_LIMIT_NM = 1e-5


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def residue_key(r):
    return f'{r.chain.id}:{r.id}{r.insertionCode.strip()}'


def atom_key(a):
    r = a.residue
    return (r.chain.index, r.chain.id, r.index, r.id, r.insertionCode, r.name, a.name)


def heavy_snapshot(topology, positions):
    xyz = np.asarray(positions.value_in_unit(unit.nanometer))
    keys, coordinates = [], []
    for a in topology.atoms():
        if a.element != element.hydrogen:
            keys.append(atom_key(a)); coordinates.append(xyz[a.index])
    return keys, np.array(coordinates)


def heavy_bonds(topology):
    return sorted(tuple(sorted((atom_key(a), atom_key(b)))) for a,b in topology.bonds()
                  if a.element != element.hydrogen and b.element != element.hydrogen)


def restore_heavy_coordinates(modeller, original_keys, original_heavy):
    keys, xyz = heavy_snapshot(modeller.topology, modeller.positions)
    checks.require(keys == original_keys, 'Heavy-atom identities changed during hydrogen placement')
    positions = np.array(modeller.positions.value_in_unit(unit.nanometer), dtype=float, copy=True)
    checks.require(np.isfinite(positions).all(), 'Nonfinite generated coordinates')
    checks.require(xyz.shape == original_heavy.shape and np.isfinite(original_heavy).all(),
                   'Invalid original heavy-atom coordinates')
    displacement = np.linalg.norm(xyz-original_heavy, axis=1)
    maximum = float(displacement.max()) if len(displacement) else 0.0
    checks.require(maximum <= HEAVY_ROUNDOFF_LIMIT_NM,
                   f'Heavy atoms moved {maximum:.12g} nm during hydrogen placement; '
                   f'roundoff limit is {HEAVY_ROUNDOFF_LIMIT_NM:g} nm. Coordinates not restored.')
    heavy_indices = [a.index for a in modeller.topology.atoms() if a.element != element.hydrogen]
    positions[heavy_indices] = original_heavy
    modeller.positions = unit.Quantity(positions, unit.nanometer)
    restored_keys, restored = heavy_snapshot(modeller.topology, modeller.positions)
    checks.require(restored_keys == original_keys and np.array_equal(restored, original_heavy),
                   'Failed to restore original heavy-atom coordinates exactly')
    return dict(max_displacement_before_restoration_nm=maximum,
                roundoff_limit_nm=HEAVY_ROUNDOFF_LIMIT_NM,
                original_heavy_coordinates_restored=True)


def source_hashes(names):
    base = Path(app.__file__).parent/'data'
    result = {}
    def visit(path):
        path = path.resolve()
        if str(path) in result: return
        result[str(path)] = sha(path)
        for inc in ET.parse(path).getroot().findall('Include'):
            target = path.parent/inc.attrib['file']
            visit(target if target.exists() else base/inc.attrib['file'])
    for name in names:
        path = Path(name)
        visit(path if path.is_file() else base/name)
    return result


def load_overrides(path, residues):
    if path is None: return {}, [None]*len(residues)
    def unique_pairs(pairs):
        obj = {}
        for k,v in pairs:
            if k in obj: raise ValueError(f'Duplicate override key: {k}')
            obj[k] = v
        return obj
    data = json.loads(path.read_text(), object_pairs_hook=unique_pairs)
    if not isinstance(data,dict): raise ValueError('Variants JSON must be an object')
    indexed = {}
    for i,r in enumerate(residues): indexed.setdefault(residue_key(r), []).append(i)
    variants = [None]*len(residues)
    for key,value in data.items():
        matches = indexed.get(key,[])
        if len(matches)!=1: raise ValueError(f'Override {key}: expected one matching residue, found {len(matches)}')
        i = matches[0]; residue = residues[i]
        if not isinstance(value,str) or value not in VARIANTS.get(residue.name,set()):
            raise ValueError(f'Unsupported variant {value!r} for {key} ({residue.name})')
        variants[i] = value
    return data, variants


def choose_platform(name):
    available = {openmm.Platform.getPlatform(i).getName() for i in range(openmm.Platform.getNumPlatforms())}
    if name == 'auto':
        name = next((x for x in ('CUDA','HIP','Metal') if x in available), 'CPU')
        if name == 'CPU': warnings.warn('No CUDA/HIP/Metal backend registered; using CPU for hydrogen placement.')
    if name not in available: raise ValueError(f'{name} unavailable; registered platforms: {sorted(available)}')
    print(f'Hydrogen placement platform: {name}',flush=True)
    return openmm.Platform.getPlatformByName(name)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--ph',type=float,default=7.0)
    parser.add_argument('--variants',type=Path,help='JSON mapping chain:residue to protonation variant')
    parser.add_argument('--existing-hydrogens',choices=('error','keep','replace'),default='error')
    parser.add_argument('--forcefield',action='append',help='Protein XML; repeat for additional parameters. Default amber14/protein.ff14SB.xml')
    parser.add_argument('--platform',choices=('auto','CPU','CUDA','HIP','Metal','Reference'),default='auto')
    parser.add_argument('--seed',type=int,default=2026)
    args = parser.parse_args()
    checks.require(np.isfinite(args.ph) and 0<=args.ph<=14,'pH must be finite and between 0 and 14')
    source = args.input.resolve()
    output = (args.output or source.with_name(source.stem+'_hydrogenated.pdb')).resolve()
    report_path = output.with_suffix('.json')
    checks.require(output.suffix.lower()=='.pdb','Output must have .pdb extension')
    for path in (output,report_path):
        checks.require(path!=source and not path.exists(),f'Refusing to overwrite {path}')
    input_hash = sha(source)
    pdb = PDBFile(str(source))
    checks.require(pdb.getNumFrames()==1,'Select a single model before adding hydrogens')
    atoms = list(pdb.topology.atoms())
    checks.require(atoms and all(a.element is not None for a in atoms),'Empty input or unknown atom elements')
    checks.require(np.isfinite(pdb.positions.value_in_unit(unit.nanometer)).all(),'Nonfinite input coordinates')
    n_old = sum(a.element==element.hydrogen for a in atoms)
    if n_old and args.existing_hydrogens=='error':
        raise ValueError(f'Input has {n_old} hydrogens; explicitly choose --existing-hydrogens keep or replace')
    original_keys, original_heavy = heavy_snapshot(pdb.topology,pdb.positions)
    original_bonds = heavy_bonds(pdb.topology)
    modeller = Modeller(pdb.topology,pdb.positions)
    if args.existing_hydrogens=='replace':
        modeller.delete([a for a in modeller.topology.atoms() if a.element==element.hydrogen])
    residues = list(modeller.topology.residues())
    overrides, requested = load_overrides(args.variants,residues)
    fields = args.forcefield or ['amber14/protein.ff14SB.xml']
    ff_hashes = source_hashes(fields)
    ff = ForceField(*fields)
    random.seed(args.seed)
    backend = choose_platform(args.platform)
    print('Adding hydrogens; existing heavy atoms remain fixed.',flush=True)
    try:
        placement_variants, templates = checks.hydrogen_variant_arguments(modeller.topology, requested)
        selected = modeller.addHydrogens(ff,pH=args.ph,variants=placement_variants,platform=backend,
                                         residueTemplates=templates)
        system = ff.createSystem(modeller.topology, nonbondedMethod=NoCutoff,
                                 constraints=None,rigidWater=False,removeCMMotion=False)
    except Exception as exc:
        raise RuntimeError('Hydrogen addition/parameterization failed. Check missing heavy atoms, terminal OXT or caps, nonstandard residues, variant compatibility, and platform initialization. No output saved. OpenMM: '+str(exc)) from exc
    cym_validation = checks.verify_cym(modeller.topology, system, requested, ff)
    checks.require(heavy_bonds(modeller.topology)==original_bonds,'Heavy-atom connectivity changed')
    coordinate_restoration = restore_heavy_coordinates(modeller, original_keys, original_heavy)
    print(f'Heavy-atom numerical drift: {coordinate_restoration["max_displacement_before_restoration_nm"]:.6g} nm; '
          'original coordinates restored exactly.', flush=True)
    checks.require(np.isfinite(modeller.positions.value_in_unit(unit.nanometer)).all(),'Nonfinite generated coordinates')
    checks.require(system.getNumParticles()==modeller.topology.getNumAtoms(),'Unexpected force-field extra particles')
    adjacency = checks.neighbors(modeller.topology)
    final_atoms = list(modeller.topology.atoms())
    for a in final_atoms:
        if a.element==element.hydrogen:
            checks.require(len(adjacency[a.index])==1,f'Hydrogen {atom_key(a)} does not have exactly one bond')
    assignments=[]
    for r,variant in zip(modeller.topology.residues(),selected):
        names={a.name for a in r.atoms()}
        observed=None
        if r.name=='HIS':
            observed={(True,True):'HIP',(True,False):'HID',(False,True):'HIE',(False,False):'HIN'}[('HD1' in names,'HE2' in names)]
            if requested[r.index] is not None: checks.require(observed==requested[r.index],'Histidine override not honored')
        assignments.append(dict(chain_index=r.chain.index,chain_id=r.chain.id,residue_id=r.id,
                                insertion_code=r.insertionCode,residue_name=r.name,
                                requested_variant=requested[r.index],openmm_selected_variant=variant,
                                observed_histidine_variant=observed,
                                hydrogen_names=[a.name for a in r.atoms() if a.element==element.hydrogen]))
    nonbonded=[f for f in system.getForces() if isinstance(f,openmm.NonbondedForce)]
    checks.require(len(nonbonded)==1,'Expected one NonbondedForce')
    charges={}
    for a in final_atoms:
        key=f'{a.residue.chain.index}:{a.residue.chain.id}'
        charges[key]=charges.get(key,0)+nonbonded[0].getParticleParameters(a.index)[0].value_in_unit(unit.elementary_charge)
    buf=io.StringIO();buf.write('REMARK 900 HYDROGEN ADDITION; HEAVY ATOMS PRESERVED; SEE JSON FOR STATES\n')
    PDBFile.writeFile(modeller.topology,modeller.positions,buf,keepIds=True)
    text=buf.getvalue(); roundtrip=PDBFile(io.StringIO(text))
    rkeys,rxyz=heavy_snapshot(roundtrip.topology,roundtrip.positions)
    checks.require(rkeys==original_keys and np.allclose(rxyz,original_heavy,rtol=0,atol=1e-10),'Saved heavy atoms changed')
    checks.require([atom_key(a) for a in roundtrip.topology.atoms()]==[atom_key(a) for a in final_atoms],'PDB atom identities changed')
    bonds=lambda t: sorted(tuple(sorted((a.index,b.index))) for a,b in t.bonds())
    checks.require(bonds(roundtrip.topology)==bonds(modeller.topology),'PDB roundtrip changed bonds')
    roundtrip_system = ff.createSystem(roundtrip.topology, nonbondedMethod=NoCutoff,
                                       constraints=None, rigidWater=False, removeCMMotion=False)
    checks.verify_cym(roundtrip.topology, roundtrip_system, requested, ff)
    report=dict(input=str(source),input_sha256=input_hash,output=str(output),
                output_sha256=hashlib.sha256(text.encode()).hexdigest(),script_sha256=sha(__file__),
                helper_sha256=sha(checks.__file__),forcefield_files=ff_hashes,
                openmm=openmm.__version__,numpy=np.__version__,python=pyplatform.python_version(),
                settings={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
                cysteine_thiolate_validation=cym_validation,
                heavy_coordinate_restoration=coordinate_restoration,
                cysteine_placement_note='CYM uses the OpenMM CYX no-S-H placement label with the CYM force-field template; no disulfide is created.',
                platform=backend.getName(),overrides=overrides,residue_assignments=assignments,
                original_hydrogens=n_old,final_hydrogens=sum(a.element==element.hydrogen for a in final_atoms),
                heavy_atoms_preserved=True,forcefield_charge_per_chain=charges,
                note='pH-based heuristics, not a pKa calculation. Existing H in keep mode can override automatic pH choices. Placement uses protein force field without implicit solvent; run hydrogen relaxation next. Terminal chemistry is inherited, not capped or repaired. Seeds aid reproducibility but do not ensure bitwise agreement across hardware.')
    checks.require(sha(source)==input_hash,'Input changed during preparation')
    checks.write_outputs({output:text,report_path:json.dumps(report,indent=2)+'\n'})
    print(f'Saved: {output}\nReport: {report_path}\nHydrogens: {n_old} input, {report["final_hydrogens"]} output. Heavy atoms preserved.')


if __name__=='__main__':
    main()
