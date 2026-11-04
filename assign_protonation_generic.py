#!/usr/bin/env python3
"""Propose fixed protonation states for an ACE/NME-capped protein assembly.

python assign_protonation_generic.py --input protein_capped.pdb --ph 7.0
Requires propka==3.5.1, OpenMM, numpy, and preparation_checks.py.
Outputs *_variants.json (for add_hydrogens_generic.py), *_protonation.json
(audit report), and *_sensitivity.json (one-site alternative assignments).
No structure files are altered and no production simulations are run.

PROPKA analyzes the whole assembly. Independent-site Henderson-Hasselbalch
fractions are heuristics, not coupled microstate probabilities. A +/-1 pH-unit
ambiguity window is a screening rule, not a prediction confidence interval.
Neutral histidine tautomers use OpenMM's local H-bond heuristic at pH 7 on an
in-memory heavy-atom copy. Existing H therefore do not bias that selection.
Choices are independent per chain; symmetry is NOT imposed.

Version 1 deliberately requires standard protein chains with ACE/NME caps.
Missing atoms, chain breaks, alternate conformations, and unsupported predicted
states require review. Cysteine thiolate is NOT misrepresented as disulfide CYX.
If an assignment cannot be encoded by the current hydrogen adder, only the
report is written and exit status is 2. Ambiguous but supported states produce
provisional assignments, never a claim of experimentally established protonation.
"""
import argparse
import hashlib
import importlib.metadata
import io
import json
import logging
from pathlib import Path
import platform
import random
import contextlib

import numpy as np
import openmm
from openmm import unit
from openmm.app import PDBFile, Modeller, ForceField, NoCutoff, element
import openmm.app as app
import preparation_checks as checks

STANDARD=set('ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL'.split())
IONIZABLE={'ASP','GLU','HIS','LYS','CYS','TYR','ARG'}
FF='amber14/protein.ff14SB.xml'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def rkey(r):return f'{r.chain.id}:{r.id}{r.insertionCode.strip()}'
def gkey(g):return f'{g.atom.chain_id}:{g.atom.res_num}{g.atom.icode.strip()}'


def decision(name,pka,ph,neutral_his):
    """Return selected state and independent-site protonated fraction."""
    fraction=1/(1+10**float(np.clip(ph-pka,-300,300)))
    protonated=fraction>=0.5
    states={'ASP':('ASH','ASP'),'GLU':('GLH','GLU'),
            'HIS':('HIP',neutral_his),'LYS':('LYS','LYN'),
            'CYS':('CYS','CYM'),'TYR':('TYR','TYM'),'ARG':('ARG','neutral_ARG')}
    return states[name][0 if protonated else 1],fraction


def validate(pdb):
    checks.require(pdb.getNumFrames()==1,'Select one PDB model')
    chains=list(pdb.topology.chains());residues=list(pdb.topology.residues())
    checks.require(chains and len({c.id for c in chains})==len(chains),'Require nonempty structure and unique chain IDs')
    checks.require(all(c.id.strip() for c in chains),'Assign nonblank chain IDs first')
    checks.require(len({rkey(r) for r in residues})==len(residues),'Ambiguous residue IDs')
    xyz=np.asarray(pdb.positions.value_in_unit(unit.angstrom))
    checks.require(np.isfinite(xyz).all(),'Nonfinite coordinates')
    adjacency=checks.neighbors(pdb.topology)
    for c in chains:
        rs=list(c.residues())
        checks.require(len(rs)>=3 and rs[0].name=='ACE' and rs[-1].name=='NME',
                       f'Chain {c.id}: run cap_generic.py first; this version requires ACE/NME caps')
        checks.require(all(r.name in STANDARD for r in rs[1:-1]),'Only standard amino acids between caps are supported')
        for left,right in zip(rs,rs[1:]):
            aa={a.name:a for a in left.atoms()};bb={a.name:a for a in right.atoms()}
            checks.require('C' in aa and 'N' in bb,'Missing peptide-link atoms')
            dist=np.linalg.norm(xyz[aa['C'].index]-xyz[bb['N'].index])
            checks.require(bb['N'].index in adjacency[aa['C'].index] and 1<=dist<=1.8,
                           f'Broken peptide/cap link in chain {c.id} near {left.id}')
    return residues


def normalized_propka_input(modeller):
    buf=io.StringIO();PDBFile.writeFile(modeller.topology,modeller.positions,buf,keepIds=True)
    # PROPKA chooses the N-terminal residue from the first ATOM record.
    # Mark caps ATOM in this temporary representation so it sees ACE, which
    # has no N, instead of incorrectly charging the first amino-acid N.
    return '\n'.join('ATOM  '+line[6:] if line.startswith('HETATM') and line[17:20] in ('ACE','NME')
                     else line for line in buf.getvalue().splitlines())+'\n'


def verify_propka_caps(conf,topology):
    byid={(a.chain_id,str(a.res_num),a.icode.strip(),a.name):a for a in conf.atoms if a.element!='H'}
    for c in topology.chains():
        rs=list(c.residues())
        for left,right in ((rs[0],rs[1]),(rs[-2],rs[-1])):
            a=byid[(c.id,left.id,left.insertionCode.strip(),'C')]
            b=byid[(c.id,right.id,right.insertionCode.strip(),'N')]
            checks.require(b in a.bonded_atoms,'PROPKA lost a cap peptide bond')
    checks.require(not any(g.type in ('N+','C-') for g in conf.get_titratable_groups()),
                   'PROPKA introduced a free terminal group into the capped model')


def main():
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--ph',type=float,required=True)
    parser.add_argument('--ambiguity-window',type=float,default=1.0)
    parser.add_argument('--output-prefix',type=Path,help='Default: input path without extension')
    args=parser.parse_args()
    checks.require(np.isfinite(args.ph) and 0<=args.ph<=14,'pH must be between 0 and 14')
    checks.require(np.isfinite(args.ambiguity_window) and args.ambiguity_window>=0,'Invalid ambiguity window')
    try:import propka.run
    except ImportError:raise SystemExit('Install PROPKA first: python -m pip install "propka==3.5.1"')
    source=args.input.resolve();prefix=(args.output_prefix or source.with_suffix('')).resolve()
    variants_path=Path(str(prefix)+'_variants.json');report_path=Path(str(prefix)+'_protonation.json')
    sweep_path=Path(str(prefix)+'_sensitivity.json')
    for path in (variants_path,report_path,sweep_path):checks.require(not path.exists() and path!=source,f'Refusing to overwrite {path}')
    source_hash=sha(source);raw=source.read_text()
    checks.require(not any(l.startswith(('ATOM  ','HETATM')) and l[16].strip() for l in raw.splitlines()),
                   'Resolve alternate locations before protonation assessment')
    pdb=PDBFile(str(source));validate(pdb)
    model=Modeller(pdb.topology,pdb.positions)
    model.delete([a for a in model.topology.atoms() if a.element==element.hydrogen])
    residues=list(model.topology.residues())
    protein={rkey(r):r for r in residues if r.name in STANDARD}
    ff=ForceField(FF)
    # Obtain neutral His tautomers from an isolated temporary copy, not the
    # submitted H pattern. This also checks completeness and Amber compatibility.
    neutral_model=Modeller(model.topology,model.positions);random.seed(2026)
    print('Checking force-field compatibility and neutral histidine tautomers.',flush=True)
    neutral_variants=neutral_model.addHydrogens(ff,pH=7.0,platform=openmm.Platform.getPlatformByName('CPU'))
    neutral_his={rkey(r):neutral_variants[r.index] for r in residues if r.name=='HIS'}
    checks.require(all(v in ('HID','HIE') for v in neutral_his.values()),'Unexpected neutral histidine selection')
    propka_text=normalized_propka_input(model)
    capture=io.StringIO();handler=logging.StreamHandler(capture)
    logger=logging.getLogger('propka');logger.addHandler(handler)
    print('Predicting pKa values for the entire assembly.',flush=True)
    try:
        with contextlib.redirect_stdout(capture),contextlib.redirect_stderr(capture):
            molecule=propka.run.single('assembly.pdb',stream=io.StringIO(propka_text),
                                      optargs=['--pH',str(args.ph),'--display-coupled-residues'],write_pka=False)
    finally:logger.removeHandler(handler)
    checks.require(len(molecule.conformation_names)==1,'Expected a single PROPKA conformation')
    conf=molecule.conformations[molecule.conformation_names[0]]
    verify_propka_caps(conf,model.topology)
    groups={}
    for g in conf.get_titratable_groups():
        k=gkey(g)
        if k in protein and g.residue_type in IONIZABLE:
            checks.require(k not in groups,f'Multiple titratable groups for {k}')
            checks.require(np.isfinite(g.pka_value),f'Nonfinite pKa at {k}')
            groups[k]=g
    adjacency=checks.neighbors(model.topology);atoms=list(model.topology.atoms())
    overrides={};records=[];blocking=[]
    for k,r in protein.items():
        if r.name not in IONIZABLE:continue
        row=dict(residue=k,name=r.name)
        if r.name=='CYS':
            sg=next(a for a in r.atoms() if a.name=='SG')
            ss=any(atoms[i].element==element.sulfur and atoms[i].residue!=r for i in adjacency[sg.index])
            if ss:
                overrides[k]='CYX';row.update(selected='CYX',reason='Existing disulfide bond',ambiguous=False,alternatives=[])
                records.append(row);continue
        if k not in groups:
            blocking.append(f'Missing PROPKA side-chain prediction for {k} {r.name}')
            row.update(selected=None,reason='Missing prediction',ambiguous=True,alternatives=[]);records.append(row);continue
        g=groups[k];pka=float(g.pka_value)
        selected,fraction=decision(r.name,pka,args.ph,neutral_his.get(k))
        coupled=sorted({gkey(other) for other in g.non_covalently_coupled_groups+g.covalently_coupled_groups})
        near=abs(pka-args.ph)<=args.ambiguity_window
        pairs={'ASP':['ASH','ASP'],'GLU':['GLH','GLU'],'LYS':['LYS','LYN'],
               'CYS':['CYS','CYM'],'TYR':['TYR','TYM'],'ARG':['ARG','neutral_ARG'],
               'HIS':['HIP','HID','HIE']}
        alternatives=[v for v in pairs[r.name] if v!=selected] if near or coupled else []
        if r.name=='HIS' and selected!='HIP':
            alternatives=sorted(set(alternatives+['HIE' if selected=='HID' else 'HID']))
        supported=selected not in ('CYM','TYM','neutral_ARG')
        if not supported:blocking.append(f'{k} favors {selected}, unsupported by the current hydrogen-adder interface; expert review/parameterization needed')
        elif r.name not in ('TYR','ARG'):overrides[k]=selected
        row.update(pka=pka,model_pka=float(g.model_pka),selected=selected,
                   independent_site_protonated_fraction=fraction,
                   within_ambiguity_window=near,coupled_residues=coupled,
                   ambiguous=bool(near or coupled or alternatives),alternatives=alternatives,
                   neutral_histidine_heuristic=neutral_his.get(k),supported=supported,
                   reason='Majority state from predicted pKa; neutral His tautomer from OpenMM local H-bond heuristic')
        records.append(row)
    if not blocking:
        requested=[overrides.get(rkey(r)) for r in residues]
        candidate=Modeller(model.topology,model.positions);random.seed(2026)
        try:
            candidate.addHydrogens(ff,pH=args.ph,variants=requested,platform=openmm.Platform.getPlatformByName('CPU'))
            ff.createSystem(candidate.topology,nonbondedMethod=NoCutoff,constraints=None,rigidWater=False)
        except Exception as exc:blocking.append('Proposed assignment fails OpenMM compatibility check: '+str(exc))
    sweeps=[];unsupported_alternatives=[]
    for row in records:
        for alternative in row['alternatives']:
            if alternative in ('CYM','TYM','neutral_ARG') or row['name'] in ('ARG','TYR'):
                unsupported_alternatives.append(dict(residue=row['residue'],state=alternative));continue
            variant=overrides.copy();variant[row['residue']]=alternative
            sweeps.append(dict(changed_residue=row['residue'],alternative=alternative,variants=variant))
    report=dict(status='blocked' if blocking else 'provisional_assignments_ready',
                references=['https://propka.readthedocs.io/en/stable/quickstart.html',
                            'https://doi.org/10.1021/ct100578z',
                            'https://docs.openmm.org/latest/api-python/generated/openmm.app.modeller.Modeller.html'],
                input=str(source),input_sha256=source_hash,
                ph=args.ph,ambiguity_window=args.ambiguity_window,assembly_scope='all chains; no imposed symmetry',
                terminal_policy='ACE/NME required; cap ATOM record normalization for PROPKA only',
                propka_input_sha256=hashlib.sha256(propka_text.encode()).hexdigest(),
                propka=importlib.metadata.version('propka'),openmm=openmm.__version__,python=platform.python_version(),
                script_sha256=sha(__file__),helper_sha256=sha(checks.__file__),
                amber_xml_sha256=sha(Path(app.__file__).parent/'data'/FF),
                neutral_his_method='OpenMM pH 7 local hydrogen-bond heuristic on H-free in-memory assembly',
                residues=records,blocking_issues=blocking,propka_messages=capture.getvalue(),
                unsupported_sensitivity_states=unsupported_alternatives,
                variants_file=str(variants_path) if not blocking else None,
                limitations=['Not an experimental protonation map or a joint microstate inference.',
                             'Independent-site fractions are approximate, particularly for coupled sites.',
                             'The ambiguity window is a screening policy, not a calibrated confidence interval.',
                             'Neutral His tautomers are geometric guesses; both are listed for sensitivity testing.',
                             'Redox state is inherited from disulfide connectivity, not determined by pH.',
                             'Sensitivity cases change one site at a time; they are not an exhaustive permutation sweep.',
                             'Do not rank different proton counts by raw minimized energy.'])
    outputs={report_path:json.dumps(report,indent=2)+'\n'}
    if not blocking:
        outputs[variants_path]=json.dumps(overrides,indent=2)+'\n'
        outputs[sweep_path]=json.dumps(dict(ph=args.ph,baseline=overrides,cases=sweeps,
                                           note='One-site sensitivity proposals only; alternatives are not individually parameterization-tested. No simulations run.'),indent=2)+'\n'
    checks.require(sha(source)==source_hash,'Input changed during assessment')
    checks.write_outputs(outputs)
    print(f'Report: {report_path}\nStatus: {report["status"]}')
    if blocking:
        print('\n'.join(blocking));raise SystemExit(2)
    print(f'Overrides: {variants_path}\nSensitivity proposals: {sweep_path}')
    print(f'{sum(r["ambiguous"] for r in records)} sites flagged; review report before production use.')


if __name__=='__main__':main()
