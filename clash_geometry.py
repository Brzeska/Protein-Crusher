"""Conservative geometry gates for standard L-proteins; coordinates in angstroms.

The 0.40 A Bondi screen also catches ordinary contacts (hydrogen-bonded N/O
pairs near 2.6 A, Calpha-H...O near 2.3 A), so only overlaps above
SEVERE_OVERLAP_ANGSTROM trigger cleanup or fail a gate; smaller ones are reported.
1-4 contacts are reported but not gated (they are excluded in the soft system).
Ring piercing is a geometric bond/surface screen, not a knot/link proof.
"""
import numpy as np
from openmm.app import element
import preparation_checks as checks

# 5O3L's worst heavy-atom overlap is 0.93 A and hydrogen relaxation handles it;
# every RibbonFold model checked has overlaps above 2.4 A. 1.0 A means C-C closer
# than 2.4 A or O-O closer than 2.04 A, shorter than any real nonbonded contact.
SEVERE_OVERLAP_ANGSTROM = 1.0

RINGS = {
    'HIS': [('CG', 'ND1', 'CE1', 'NE2', 'CD2')],
    'PHE': [('CG', 'CD1', 'CE1', 'CZ', 'CE2', 'CD2')],
    'TYR': [('CG', 'CD1', 'CE1', 'CZ', 'CE2', 'CD2')],
    'TRP': [('CG', 'CD1', 'NE1', 'CE2', 'CD2'),
            ('CD2', 'CE2', 'CZ2', 'CH2', 'CZ3', 'CE3')],
    'PRO': [('N', 'CA', 'CB', 'CG', 'CD')],
}


def label(atom):
    r = atom.residue
    return f'{r.chain.index}:{r.chain.id}:{r.name}{r.id}{r.insertionCode.strip()}:{atom.name}'


def severe(records, threshold=SEVERE_OVERLAP_ANGSTROM):
    """Steric-overlap records strictly deeper than threshold (angstrom)."""
    return [r for r in records if r['category'] == 'steric_overlap' and r['overlap_angstrom'] > threshold]


def heavy_contacts(topology, xyz):
    indices = {a.index for a in topology.atoms() if a.element != element.hydrogen}
    return checks.contact_screen(topology, xyz, indices, partner_indices=indices)


def peptide_torsions(topology):
    """Use actual C-N bonds, never adjacency in residue numbering; include caps."""
    rows = []
    for a, b in topology.bonds():
        if a.name == 'N' and b.name == 'C':
            a, b = b, a
        if a.name != 'C' or b.name != 'N' or a.residue == b.residue:
            continue
        left = {x.name: x for x in a.residue.atoms()}
        right = {x.name: x for x in b.residue.atoms()}
        first = 'CH3' if a.residue.name == 'ACE' else 'CA'
        last = 'C' if b.residue.name == 'NME' else 'CA'
        checks.require(first in left and last in right, 'Incomplete peptide/cap: '+label(a))
        rows.append((left[first].index, a.index, b.index, right[last].index))
    return rows


def stereocenters(topology):
    """(center, arms) atom indices gated for L-chirality: Calpha (not Gly/caps), Ile/Thr Cbeta.

    The ordered triple product of the arm-minus-center vectors is positive for
    standard L amino acids, including Cys (whose CIP label is R) and natural
    Ile/Thr at CB.
    """
    rows = []
    for residue in topology.residues():
        names = {a.name: a.index for a in residue.atoms()}
        centers = []
        if residue.name not in ('GLY', 'ACE', 'NME'):
            centers.append(('CA', ('N', 'C', 'CB')))
        if residue.name == 'ILE':
            centers.append(('CB', ('CA', 'CG1', 'CG2')))
        if residue.name == 'THR':
            centers.append(('CB', ('CA', 'OG1', 'CG2')))
        for center, arms in centers:
            checks.require(all(n in names for n in (center, *arms)), f'Incomplete stereocenter: {residue}')
            rows.append((names[center], tuple(names[n] for n in arms)))
    return rows


def dihedral(points):
    a, b, c, d = np.asarray(points)
    axis = c-b
    norm = np.linalg.norm(axis)
    if norm < 1e-10:
        return None
    axis /= norm
    u, v = a-b, d-c
    u -= np.dot(u, axis)*axis
    v -= np.dot(v, axis)*axis
    if min(np.linalg.norm(u), np.linalg.norm(v)) < 1e-10:
        return None
    return float(np.degrees(np.arctan2(np.dot(np.cross(axis, u), v), np.dot(u, v))))


def point_in_triangle(p, a, b, c, tolerance=1e-8):
    v, w, q = b-a, c-a, p-a
    vv, vw, ww = np.dot(v, v), np.dot(v, w), np.dot(w, w)
    det = vv*ww-vw*vw
    if det < 1e-16:
        return False
    s = (ww*np.dot(q, v)-vw*np.dot(q, w))/det
    t = (vv*np.dot(q, w)-vw*np.dot(q, v))/det
    return s >= -tolerance and t >= -tolerance and s+t <= 1+tolerance


def segment_triangle(p, q, a, b, c):
    normal = np.cross(b-a, c-a)
    norm = np.linalg.norm(normal)
    if norm < 1e-10:
        return False
    normal /= norm
    dp, dq = np.dot(p-a, normal), np.dot(q-a, normal)
    if abs(dp) < 1e-8 and abs(dq) < 1e-8:
        # Coplanar segments: endpoints or intersection with a triangle edge.
        if point_in_triangle(p, a, b, c) or point_in_triangle(q, a, b, c):
            return True
        for u, v in ((a, b), (b, c), (c, a)):
            matrix = np.column_stack((q-p, u-v))
            st, _, _, _ = np.linalg.lstsq(matrix, u-p, rcond=None)
            if np.linalg.norm(matrix@st-(u-p)) < 1e-8 and np.all(st >= 0) and np.all(st <= 1):
                return True
        return False
    if dp*dq > 0 or abs(dp-dq) < 1e-12:
        return False
    t = dp/(dp-dq)
    return 0 <= t <= 1 and point_in_triangle(p+t*(q-p), a, b, c)


def ring_piercings(topology, xyz):
    atoms = list(topology.atoms())
    adjacency = checks.neighbors(topology)
    bonds = np.array([(a.index, b.index) for a, b in topology.bonds()], dtype=int).reshape(-1, 2)
    found, malformed = [], []
    for residue in topology.residues():
        names = {a.name: a.index for a in residue.atoms()}
        for ring in RINGS.get(residue.name, []):
            if not all(n in names for n in ring):
                malformed.append(f'{residue}: incomplete ring'); continue
            ids = [names[n] for n in ring]
            if not all(ids[(j+1) % len(ids)] in adjacency[i] for j, i in enumerate(ids)):
                malformed.append(f'{residue}: missing ring bond'); continue
            vertices = xyz[ids]
            center = vertices.mean(axis=0)
            triangles = [(center, vertices[j], vertices[(j+1) % len(ids)]) for j in range(len(ids))]
            if any(np.linalg.norm(np.cross(b-a, c-a)) < 1e-8 for a, b, c in triangles):
                malformed.append(f'{residue}: degenerate ring surface'); continue
            # Vectorized bounding-box filter avoids Python ring x bond loops.
            lo, hi = vertices.min(0)-1e-8, vertices.max(0)+1e-8
            starts, ends = xyz[bonds[:, 0]], xyz[bonds[:, 1]]
            mask = np.all(np.maximum(starts, ends) >= lo, axis=1) & np.all(np.minimum(starts, ends) <= hi, axis=1)
            for i, j in bonds[mask]:
                if i in ids or j in ids:
                    continue
                if any(segment_triangle(xyz[i], xyz[j], a, b, c) for a, b, c in triangles):
                    found.append(dict(ring=[label(atoms[k]) for k in ids], bond=[label(atoms[i]), label(atoms[j])]))
    return found, malformed


def geometry_report(topology, xyz, contacts=None, include_hydrogens=False,
                    severe_overlap=SEVERE_OVERLAP_ANGSTROM, max_omega_deviation=30.):
    """Fail on omega more than max_omega_deviation from trans (30 for final structures).

    Peptides between 30 degrees and that limit are listed as twisted_trans_peptides:
    they are on the trans side of the ~90 degree barrier, so unrestrained
    minimization can flatten them without crossing it (it may not, if the
    surrounding structure holds the twist).
    """
    checks.require(np.isfinite(severe_overlap) and severe_overlap >= 0.4,
                   'Severe overlap threshold must be finite and at least the 0.40 A screen')
    checks.require(np.isfinite(max_omega_deviation) and 30 <= max_omega_deviation < 90,
                   'Omega limit must be at least 30 and below the 90 degree barrier')
    atoms = list(topology.atoms())
    xyz = np.asarray(xyz, dtype=float)
    checks.require(xyz.shape == (len(atoms), 3) and np.isfinite(xyz).all(), 'Invalid geometry coordinates')
    contacts = heavy_contacts(topology, xyz) if contacts is None else contacts
    peptide, omega, twisted, chirality = [], [], [], []
    for ids in peptide_torsions(topology):
        length = float(np.linalg.norm(xyz[ids[1]]-xyz[ids[2]]))
        if not 1.20 <= length <= 1.45:
            peptide.append(dict(atoms=[label(atoms[i]) for i in ids[1:3]], length_angstrom=length))
        angle = dihedral(xyz[list(ids)])
        if angle is None or 180-abs(angle) > 30:
            row = dict(atoms=[label(atoms[i]) for i in ids], omega_degrees=angle)
            (omega if angle is None or 180-abs(angle) > max_omega_deviation else twisted).append(row)
    for center, arms in stereocenters(topology):
        volume = float(np.linalg.det([xyz[i]-xyz[center] for i in arms]))
        if volume <= 0.5:
            chirality.append(dict(atom=label(atoms[center]), signed_volume_angstrom3=volume))
    piercings, malformed = ring_piercings(topology, xyz)
    all_contacts = checks.contact_screen(topology, xyz, set(range(len(atoms)))) if include_hydrogens else []
    hydrogen_rows = [r for r in all_contacts if r['category'] == 'steric_overlap' and r not in contacts]
    heavy_rows = [r for r in contacts if r['category'] == 'steric_overlap']
    mild = lambda rows: [r for r in rows if r['overlap_angstrom'] <= severe_overlap]
    # Only severe overlaps fail; ordinary close contacts are listed, not gated.
    failures = dict(hydrogen_steric_overlaps=severe(hydrogen_rows, severe_overlap),
                    heavy_atom_overlaps=severe(heavy_rows, severe_overlap),
                    peptide_bond_outliers=peptide, omega_outliers=omega, chirality_outliers=chirality,
                    ring_piercings=piercings, malformed_rings=malformed)
    return dict(passed=not any(failures.values()), **failures,
                heavy_atom_close_contacts=mild(heavy_rows), hydrogen_close_contacts=mild(hydrogen_rows),
                one_four_contacts=[r for r in contacts if r['category'] == 'one_four_overlap'],
                twisted_trans_peptides=twisted,
                thresholds=dict(screen_overlap_angstrom=0.4, severe_overlap_angstrom=severe_overlap,
                                peptide_bond_angstrom=[1.20, 1.45],
                                max_trans_omega_deviation_degrees=max_omega_deviation,
                                reported_omega_deviation_degrees=30, min_chiral_volume_angstrom3=0.5),
                scope='Standard L-proteins, trans peptides including caps; ring bond/surface intersections (not a topology proof).')
