#!/usr/bin/env python3
"""Add heavy atoms of neutral ACE/NME caps to each continuous protein chain.

Usage: python cap_generic.py --input protein.pdb
Then run add_hydrogens_generic.py on the capped output, followed by relaxation.
Works with standard amino acids, including N-terminal proline. Every chain is
capped at BOTH ends. Missing residues/internal chain breaks are rejected.
Input hydrogens require explicit --strip-hydrogens; this removes all H, so
reapply deliberate protonation choices during subsequent hydrogen addition.
Terminal OXT (if present) is removed to replace the free carboxyl terminus with
an amide; all other input heavy atoms and their coordinates are preserved.
Existing caps, nonprotein chains, and cyclic/externally linked ends are rejected.
Cap coordinates are a modeled starting geometry, not an optimized structure.
Requires numpy, OpenMM, and preparation_checks.py beside this script.
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

STANDARD=set('ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL'.split())
GEOMETRY=dict(C_N=1.335,C_C=1.522,C_O=1.229,N_C=1.449,
              CA_C_N=116.6,C_N_CA=121.9,N_C_O=122.9,peptide_omega=180.0)

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

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def key(a):
    r=a.residue
    return (r.chain.id,r.id,r.insertionCode.strip(),r.name,a.name)


def cap_ids(residues):
    used={r.id for r in residues}
    try:
        ids=[int(r.id) for r in residues]
    except ValueError:
        raise ValueError('PDB residue IDs must be integers')
    choices=[]
    for preferred in (min(ids)-1,max(ids)+1):
        if -999<=preferred<=9999 and str(preferred) not in used:
            chosen=str(preferred)
        else:
            chosen=next((str(i) for i in range(0,10000) if str(i) not in used),None)
        require(chosen is not None,'No unused PDB residue numbers for caps')
        used.add(chosen);choices.append(chosen)
    return choices


def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path)
    p.add_argument('--strip-hydrogens',action='store_true')
    p.add_argument('--n-cap-phi',type=float,default=-135.0,
                   help='Starting phi for non-Pro N termini; Pro uses its existing ring geometry')
    args=p.parse_args()
    require(np.isfinite(args.n_cap_phi),'Nonfinite N-cap phi')
    source=args.input.resolve()
    output=(args.output or source.with_name(source.stem+'_capped.pdb')).resolve()
    report_path=output.with_suffix('.json')
    require(output.suffix.lower()=='.pdb','Output must end in .pdb')
    for path in (output,report_path):require(path!=source and not path.exists(),f'Refusing to overwrite {path}')
    source_hash=sha(source)
    pdb=PDBFile(str(source));old=pdb.topology
    require(pdb.getNumFrames()==1,'Select one PDB model first')
    xyz=np.asarray(pdb.positions.value_in_unit(unit.angstrom))
    require(np.isfinite(xyz).all(),'Nonfinite coordinates')
    atoms=list(old.atoms());chains=list(old.chains())
    require(atoms and all(a.element is not None for a in atoms),'Empty input or unknown elements')
    require(len({c.id for c in chains})==len(chains),'Duplicate chain IDs; assign unique IDs first')
    require(len({key(a) for a in atoms})==len(atoms),'Duplicate atom identifiers')
    hydrogens={a for a in atoms if a.element==element.hydrogen}
    require(not hydrogens or args.strip_hydrogens,
            'Input contains hydrogens: explicitly use --strip-hydrogens, then reapply protonation during hydrogen addition')
    removed=set(hydrogens);plans=[]
    adjacency={a:set() for a in atoms}
    for a,b in old.bonds():adjacency[a].add(b);adjacency[b].add(a)
    for chain in chains:
        rs=list(chain.residues())
        require(all(r.name in STANDARD for r in rs),f'Chain {chain.id}: nonstandard residue or existing cap; standard uncapped proteins only')
        maps=[{a.name:a for a in r.atoms()} for r in rs]
        for r,aa in zip(rs,maps):
            require({'N','CA','C','O'}<=aa.keys(),f'Missing backbone atoms at {chain.id}:{r.id}')
            if r!=rs[-1]:require('OXT' not in aa,f'Internal OXT at {chain.id}:{r.id}; possible chain break')
        for left,right in zip(maps,maps[1:]):
            c,n=left['C'],right['N'];dist=float(np.linalg.norm(xyz[c.index]-xyz[n.index]))
            require(n in adjacency[c] and 1.0<=dist<=1.8,
                    f'Chain {chain.id}: missing/nonphysical peptide link near {c.residue.id} ({dist:.3f} A); split or repair first')
        first,last=maps[0],maps[-1]
        require(all(a.residue==rs[0] for a in adjacency[first['N']]),'N terminus already linked outside its residue')
        require(all(a.residue==rs[-1] for a in adjacency[last['C']]),'C terminus already linked outside its residue')
        if 'OXT' in last:
            oxt=last['OXT']
            require(oxt.element==element.oxygen and adjacency[oxt]-hydrogens=={last['C']},'Unexpected OXT bonding')
            removed.add(oxt)
        n,ca,c=(xyz[first[name].index] for name in ('N','CA','C'))
        if rs[0].name=='PRO':
            require('CD' in first,'N-terminal proline missing CD')
            cd=xyz[first['CD'].index]
            ac=n-GEOMETRY['C_N']*normalized(normalized(ca-n)+normalized(cd-n))
        else:
            ac=place(c,ca,n,GEOMETRY['C_N'],GEOMETRY['C_N_CA'],args.n_cap_phi)
        am=place(ca,n,ac,GEOMETRY['C_C'],GEOMETRY['CA_C_N'],180)
        ao=place(ca,n,ac,GEOMETRY['C_O'],GEOMETRY['N_C_O'],0)
        ln,lca,lc,lo=(xyz[last[name].index] for name in ('N','CA','C','O'))
        mn=place(ln,lca,lc,GEOMETRY['C_N'],GEOMETRY['CA_C_N'],dihedral(ln,lca,lc,lo)+180)
        mm=place(lca,lc,mn,GEOMETRY['N_C'],GEOMETRY['C_N_CA'],180)
        plans.append((chain,rs,first,last,cap_ids(rs),(ac,ao,am,mn,mm)))
    top=Topology();positions=[];copied={};added=[];newbonds=[];cap_records=[]
    def add(res,name,symbol,pos,is_new=False):
        a=top.addAtom(name,element.Element.getBySymbol(symbol),res);positions.append(pos)
        if is_new:added.append(a.index)
        return a
    def bond(a,b,length):
        top.addBond(a,b);newbonds.append((a.index,b.index,length))
    for chain,rs,first,last,ids,coords in plans:
        ac,ao,am,mn,mm=coords
        nc=top.addChain(chain.id);ace=top.addResidue('ACE',nc,ids[0])
        acec=add(ace,'C','C',ac,True);aceo=add(ace,'O','O',ao,True);acem=add(ace,'CH3','C',am,True)
        bond(acec,aceo,GEOMETRY['C_O']);bond(acec,acem,GEOMETRY['C_C'])
        for r in rs:
            nr=top.addResidue(r.name,nc,r.id,r.insertionCode)
            for a in r.atoms():
                if a not in removed:copied[a]=add(nr,a.name,a.element.symbol,xyz[a.index])
        nme=top.addResidue('NME',nc,ids[1])
        nmen=add(nme,'N','N',mn,True);nmec=add(nme,'C','C',mm,True)
        bond(acec,copied[first['N']],GEOMETRY['C_N'])
        bond(copied[last['C']],nmen,GEOMETRY['C_N']);bond(nmen,nmec,GEOMETRY['N_C'])
        omega_n=dihedral(am,ac,xyz[first['N'].index],xyz[first['CA'].index])
        omega_c=dihedral(xyz[last['CA'].index],xyz[last['C'].index],mn,mm)
        require(all(abs(abs(x)-180)<1e-7 for x in (omega_n,omega_c)),'Non-trans modeled cap')
        cap_records.append(dict(chain=chain.id,ace_id=ids[0],nme_id=ids[1],
                                first_residue=rs[0].name,last_residue=rs[-1].name,
                                n_cap_omega=omega_n,c_cap_omega=omega_c,
                                n_cap_method='Pro ring geometry' if rs[0].name=='PRO' else 'specified phi'))
    for a,b in old.bonds():
        if a in copied and b in copied:top.addBond(copied[a],copied[b])
    positions=np.asarray(positions)
    require(len(added)==5*len(chains),'Unexpected cap atom count')
    for a,b,length in newbonds:require(abs(np.linalg.norm(positions[a]-positions[b])-length)<1e-8,'Bad cap bond length')
    for a,b in copied.items():require(np.array_equal(xyz[a.index],positions[b.index]),'Original heavy atom moved')
    buf=io.StringIO();buf.write('REMARK 900 MODELED ACE/NME HEAVY-ATOM CAPS; ADD HYDROGENS BEFORE RELAXATION\n')
    PDBFile.writeFile(top,positions*unit.angstrom,buf,keepIds=True)
    contents=buf.getvalue();roundtrip=PDBFile(io.StringIO(contents))
    require([key(a) for a in top.atoms()]==[key(a) for a in roundtrip.topology.atoms()],'PDB changed atom identities')
    edges=lambda t:sorted(tuple(sorted((a.index,b.index))) for a,b in t.bonds())
    require(edges(top)==edges(roundtrip.topology),'PDB roundtrip changed connectivity')
    written=np.asarray(roundtrip.positions.value_in_unit(unit.angstrom))
    require(np.allclose(written,positions,rtol=0,atol=.000501),'Unexpected PDB rounding')
    for a,b in copied.items():require(np.allclose(xyz[a.index],written[b.index],rtol=0,atol=1e-9),'Saved original heavy atom moved')
    contacts=checks.contact_screen(roundtrip.topology,written,set(added))
    report=dict(input=str(source),input_sha256=source_hash,output=str(output),
                output_sha256=hashlib.sha256(contents.encode()).hexdigest(),script_sha256=sha(__file__),
                helper_sha256=sha(checks.__file__),openmm=openmm.__version__,numpy=np.__version__,python=platform.python_version(),
                settings=dict(n_cap_phi=args.n_cap_phi,strip_hydrogens=args.strip_hydrogens),
                geometry=GEOMETRY,caps=cap_records,added_heavy_atoms=len(added),
                removed_atoms=[key(a) for a in atoms if a in removed],preserved_heavy_atoms=len(copied),
                all_retained_heavy_coordinates_preserved=True,close_contacts=contacts,
                contact_counts=dict(Counter(x['category'] for x in contacts)),contact_rule=checks.CONTACT_RULE,
                parameterized=False,minimized=False,
                note='Heavy-atom caps only. OXT replaced by NME where present. Add hydrogens with deliberate protonation next; cap placement is not optimized. No stability inference.')
    require(sha(source)==source_hash,'Input changed during processing')
    checks.write_outputs({output:contents,report_path:json.dumps(report,indent=2)+'\n'})
    print(f'Saved: {output}\nReport: {report_path}')
    print(f'Added {len(chains)} ACE and {len(chains)} NME heavy-atom caps; {len(copied)} retained heavy atoms unchanged.')
    print(f'Removed {len(removed)} atoms (listed in report). Contacts: {report["contact_counts"]}')
    print('Next: add hydrogens, then relax. This structure is not yet parameterized.')


if __name__=='__main__':main()
