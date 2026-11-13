"""Tiny CPU/Reference tests only: no protein minimization or GPU required."""
import hashlib
import io
import json
from pathlib import Path
import sys
import os
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import openmm
from openmm import unit
from openmm.app import Topology, PDBFile, element

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import clash_geometry as geometry
import preparation_checks as checks
import remove_clashes_generic as cleanup
import validate_clash_output as gate


def fixture(n=2):
    topology = Topology()
    chain = topology.addChain('A')
    for i in range(n):
        r = topology.addResidue('GLY', chain, str(i+1))
        topology.addAtom('CA', element.carbon, r)
    system = openmm.System()
    nb = openmm.NonbondedForce()
    for _ in range(n):
        system.addParticle(12.)
        nb.addParticle(.3, .3, .5)
    system.addForce(nb)
    system.addForce(openmm.HarmonicBondForce())
    system.addForce(openmm.HarmonicAngleForce())
    system.addForce(openmm.PeriodicTorsionForce())
    system.addForce(openmm.CustomGBForce())
    return topology, system


class SoftForceTests(unittest.TestCase):
    def test_clone_removes_real_nonbonded_and_gb_without_mutating_original(self):
        top, original = fixture(3)
        original.getForce(0).addException(0, 1, 0., .2, 0.)
        original.getForce(0).addException(1, 2, .01, .25, .2)
        before = openmm.XmlSerializer.serialize(original)
        soft = cleanup.build_soft_system(original, top, np.zeros((3, 3)))
        self.assertEqual(before, openmm.XmlSerializer.serialize(original))
        self.assertFalse(any(isinstance(f, (openmm.NonbondedForce, openmm.CustomGBForce)) for f in soft.getForces()))
        for force_type in (openmm.HarmonicBondForce, openmm.HarmonicAngleForce, openmm.PeriodicTorsionForce):
            self.assertTrue(any(isinstance(f, force_type) for f in soft.getForces()))
        nb = next(f for f in soft.getForces() if isinstance(f, openmm.CustomNonbondedForce))
        self.assertEqual([tuple(nb.getExclusionParticles(i)) for i in range(nb.getNumExclusions())], [(0, 1), (1, 2)])

    def test_expression_energy_force_and_alpha_updates(self):
        top, original = fixture()
        positions = np.array([[0., 0., 0.], [.2, 0., 0.]])
        system = cleanup.build_soft_system(original, top, positions)
        integrator = openmm.VerletIntegrator(.001)
        context = openmm.Context(system, integrator, openmm.Platform.getPlatformByName('Reference'))
        context.setPositions(positions)
        for alpha in cleanup.DEFAULT_ALPHAS:
            context.setParameter('alpha', alpha)
            state = context.getState(energy=True, forces=True)
            u = alpha+(.2/.3)**6
            expected = 4*.5*(1/u**2-1/u)
            self.assertAlmostEqual(state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole), expected, places=7)
            force = state.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole/unit.nanometer)
            self.assertTrue(np.isfinite(force).all())
            self.assertLess(force[0, 0], 0)
            self.assertAlmostEqual(force[0, 0], -force[1, 0])
        # Nonzero but tiny distances remain finite, unlike hard LJ.
        context.setPositions([[0, 0, 0], [1e-8, 0, 0]])
        state = context.getState(energy=True, forces=True)
        self.assertTrue(np.isfinite(state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)))
        self.assertTrue(np.isfinite(state.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole/unit.nanometer)).all())
        del context, integrator

    def test_trans_restraint_has_minimum_at_pi_and_ca_spring_uses_half_k(self):
        top = Topology(); chain = top.addChain('A')
        first, second = top.addResidue('GLY', chain, '1'), top.addResidue('GLY', chain, '2')
        atoms = []
        for r, names in ((first, ('CA', 'C')), (second, ('N', 'CA'))):
            atoms.extend(top.addAtom(n, element.carbon, r) for n in names)
        top.addBond(atoms[1], atoms[2])
        original = openmm.System(); nb = openmm.NonbondedForce()
        for _ in atoms:
            original.addParticle(12.); nb.addParticle(0., .3, 0.)
        original.addForce(nb)
        trans = np.array([[0., .1, 0], [0., 0, 0], [.13, 0, 0], [.13, -.1, 0]])
        system = cleanup.build_soft_system(original, top, trans)
        integrator = openmm.VerletIntegrator(.001)
        context = openmm.Context(system, integrator, openmm.Platform.getPlatformByName('Reference'))
        context.setPositions(trans)
        energy = lambda: context.getState(energy=True).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        self.assertAlmostEqual(energy(), 0.)
        moved = trans.copy(); moved[0, 1] += .01
        context.setPositions(moved)
        self.assertAlmostEqual(energy(), .5*1000*.01**2)
        context.setParameter('ca_k', 0)
        cis = trans.copy(); cis[-1, 1] *= -1
        context.setPositions(cis)
        self.assertAlmostEqual(energy(), 200.)
        del context, integrator

    def test_chirality_restraint_zero_for_l_and_resists_flattening_and_inversion(self):
        top = Topology(); r = top.addResidue('ALA', top.addChain('A'), '1')
        for name, el in (('N', element.nitrogen), ('CA', element.carbon), ('C', element.carbon), ('CB', element.carbon)):
            top.addAtom(name, el, r)
        self.assertEqual(geometry.stereocenters(top), [(1, (0, 2, 3))])
        original = openmm.System(); nb = openmm.NonbondedForce()
        for _ in range(4):
            original.addParticle(12.); nb.addParticle(0., .3, 0.)
        original.addForce(nb)
        # CA at the origin; N, C, CB along +x, +y, +z: signed volume +3.375 A^3 (L side).
        l_form = np.array([[.15, 0, 0], [0., 0, 0], [0., .15, 0], [0., 0, .15]])
        system = cleanup.build_soft_system(original, top, l_form)
        integrator = openmm.VerletIntegrator(.001)
        context = openmm.Context(system, integrator, openmm.Platform.getPlatformByName('Reference'))
        def evaluate(xyz):
            context.setPositions(xyz)
            s = context.getState(energy=True, forces=True)
            return (s.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole),
                    s.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole/unit.nanometer))
        k, floor = cleanup.DEFAULT_CHIRAL_K, cleanup.CHIRAL_FLOOR_ANGSTROM3
        self.assertAlmostEqual(evaluate(l_form)[0], 0.)
        flat = l_form.copy(); flat[3] = [.1, .1, 0.]
        self.assertAlmostEqual(evaluate(flat)[0], k, places=4)
        inverted = l_form.copy(); inverted[3, 2] *= -1
        energy, force = evaluate(inverted)
        self.assertAlmostEqual(energy, k*(-3.375/floor-1)**2, places=4)
        self.assertGreater(force[3, 2], 0)  # pushes CB back toward the L side
        del context, integrator

    def test_unknown_force_rejected(self):
        top, system = fixture()
        system.addForce(openmm.CustomExternalForce('x*x'))
        with self.assertRaisesRegex(ValueError, 'Unsupported force'):
            cleanup.build_soft_system(system, top, np.zeros((2, 3)))

    def test_zero_lj_site_is_safe(self):
        top, original = fixture()
        original.getForce(0).setParticleParameters(0, 0., 0., 0.)
        system = cleanup.build_soft_system(original, top, np.zeros((2, 3)))
        nb = next(f for f in system.getForces() if isinstance(f, openmm.CustomNonbondedForce))
        sigma, epsilon = nb.getParticleParameters(0)
        self.assertGreater(sigma, 0)
        self.assertEqual(epsilon, 0)

    def test_coincidences_separated_deterministically_without_moving_other_atoms(self):
        start = np.array([[0., 0., 0.], [0., 0., 0.], [1., 0., 0.]])
        result, moved = cleanup.separate_coincident_atoms(start)
        again, _ = cleanup.separate_coincident_atoms(start)
        np.testing.assert_array_equal(result, again)
        np.testing.assert_array_equal(result[[0, 2]], start[[0, 2]])
        self.assertEqual(moved, [1])
        self.assertGreater(np.linalg.norm(result[1]-result[0]), .001)
        np.testing.assert_array_equal(start[0], start[1])


class GeometryTests(unittest.TestCase):
    def test_hydrogen_contacts_do_not_trigger_heavy_cleanup(self):
        top, _ = fixture()
        r = list(top.residues())[-1]
        h = top.addAtom('H', element.hydrogen, r)
        top.addBond(list(top.atoms())[1], h)
        xyz = np.array([[0., 0., 0.], [4., 0., 0.], [.1, 0., 0.]])
        self.assertTrue(checks.contact_screen(top, xyz, {0, 1}))
        self.assertEqual(geometry.heavy_contacts(top, xyz), [])

    def test_one_two_one_three_excluded_one_four_labeled(self):
        top, _ = fixture(4)
        atoms = list(top.atoms())
        for a, b in zip(atoms, atoms[1:]): top.addBond(a, b)
        contacts = geometry.heavy_contacts(top, np.array([[i*.2, 0, 0] for i in range(4)]))
        self.assertEqual(len(contacts), 1)
        self.assertEqual(contacts[0]['category'], 'one_four_overlap')

    def test_nonbonded_heavy_overlap_is_flagged(self):
        top, _ = fixture()
        result = geometry.geometry_report(top, np.array([[0., 0., 0.], [1., 0., 0.]]))
        self.assertFalse(result['passed'])
        self.assertEqual(len(result['heavy_atom_overlaps']), 1)

    def test_mild_overlap_reported_but_not_gated(self):
        top, _ = fixture()
        # 2.8 A between carbons: a 0.6 A Bondi overlap, below the severe threshold.
        result = geometry.geometry_report(top, np.array([[0., 0., 0.], [2.8, 0., 0.]]))
        self.assertTrue(result['passed'])
        self.assertEqual(result['heavy_atom_overlaps'], [])
        self.assertEqual(len(result['heavy_atom_close_contacts']), 1)

    def test_trans_cis_and_degenerate_dihedral(self):
        self.assertAlmostEqual(abs(geometry.dihedral([[0., 1, 0], [0., 0, 0], [1., 0, 0], [1., -1, 0]])), 180)
        self.assertAlmostEqual(geometry.dihedral([[0., 1, 0], [0., 0, 0], [1., 0, 0], [1., 1, 0]]), 0)
        self.assertIsNone(geometry.dihedral(np.zeros((4, 3))))

    def test_actual_peptide_bonds_not_numbering_and_caps(self):
        top = Topology(); chain = top.addChain('A')
        ace = top.addResidue('ACE', chain, '20')
        gly = top.addResidue('GLY', chain, '77')
        nme = top.addResidue('NME', chain, '99')
        atoms = []
        for r, names in ((ace, ('CH3', 'C')), (gly, ('N', 'CA', 'C')), (nme, ('N', 'C'))):
            atoms.extend(top.addAtom(n, element.nitrogen if n == 'N' else element.carbon, r) for n in names)
        top.addBond(atoms[1], atoms[2]); top.addBond(atoms[4], atoms[5])
        self.assertEqual(geometry.peptide_torsions(top), [(0, 1, 2, 3), (3, 4, 5, 6)])

    def test_cleanup_gate_allows_twisted_trans_but_final_gate_does_not(self):
        top = Topology(); chain = top.addChain('A')
        first, second = top.addResidue('GLY', chain, '1'), top.addResidue('GLY', chain, '2')
        ca1 = top.addAtom('CA', element.carbon, first); c1 = top.addAtom('C', element.carbon, first)
        n2 = top.addAtom('N', element.nitrogen, second); ca2 = top.addAtom('CA', element.carbon, second)
        for a, b in ((ca1, c1), (c1, n2), (n2, ca2)): top.addBond(a, b)
        def place(omega):
            t = np.radians(omega)
            return np.array([[-.7, 1.3, 0.], [0., 0., 0.], [1.33, 0., 0.], [2.03, 1.3*np.cos(t), 1.3*np.sin(t)]])
        xyz = place(145.)
        self.assertAlmostEqual(geometry.dihedral(xyz), 145.)
        final = geometry.geometry_report(top, xyz)
        self.assertFalse(final['passed'])
        self.assertEqual(len(final['omega_outliers']), 1)
        after_cleanup = geometry.geometry_report(top, xyz, max_omega_deviation=cleanup.CLEANUP_MAX_OMEGA_DEVIATION)
        self.assertTrue(after_cleanup['passed'])
        self.assertEqual(len(after_cleanup['twisted_trans_peptides']), 1)
        near_barrier = geometry.geometry_report(top, place(100.), max_omega_deviation=cleanup.CLEANUP_MAX_OMEGA_DEVIATION)
        self.assertFalse(near_barrier['passed'])
        self.assertEqual(len(near_barrier['omega_outliers']), 1)

    def test_chirality_inversion_and_flattening(self):
        top = Topology(); r = top.addResidue('ALA', top.addChain('A'))
        for name in ('CA', 'N', 'C', 'CB'): top.addAtom(name, element.carbon, r)
        xyz = np.array([[0., 0, 0], [1., 0, 0], [0., 1, 0], [0., 0, 1]])
        self.assertEqual(geometry.geometry_report(top, xyz)['chirality_outliers'], [])
        xyz[3, 2] = -1
        self.assertEqual(len(geometry.geometry_report(top, xyz)['chirality_outliers']), 1)
        xyz[3, 2] = 0
        self.assertEqual(len(geometry.geometry_report(top, xyz)['chirality_outliers']), 1)

    def test_ring_piercing_coplanar_and_nonpiercing(self):
        top = Topology(); chain = top.addChain('A'); r = top.addResidue('PHE', chain)
        atoms = [top.addAtom(n, element.carbon, r) for n in geometry.RINGS['PHE'][0]]
        for a, b in zip(atoms, atoms[1:]+atoms[:1]): top.addBond(a, b)
        r2 = top.addResidue('GLY', chain)
        a, b = [top.addAtom(n, element.carbon, r2) for n in ('CA', 'C')]
        top.addBond(a, b)
        theta = np.arange(6)*np.pi/3
        xyz = np.vstack((np.column_stack((np.cos(theta), np.sin(theta), np.zeros(6))), [[0, 0, -1], [0, 0, 1]]))
        self.assertEqual(len(geometry.ring_piercings(top, xyz)[0]), 1)
        xyz[-2:] = [[-2, 0, 0], [2, 0, 0]]
        self.assertEqual(len(geometry.ring_piercings(top, xyz)[0]), 1)
        xyz[-2:] = [[3, 0, -1], [3, 0, 1]]
        self.assertEqual(geometry.ring_piercings(top, xyz), ([], []))


class CliTests(unittest.TestCase):
    def clean_pdb(self, path):
        top = Topology(); r = top.addResidue('GLY', top.addChain('A'), '1')
        n = top.addAtom('N', element.nitrogen, r)
        h = top.addAtom('H', element.hydrogen, r); top.addBond(n, h)
        buf = io.StringIO()
        PDBFile.writeFile(top, np.array([[0., 0., 0.], [.1, 0., 0.]])*unit.nanometer, buf)
        path.write_bytes(buf.getvalue().replace('\n', '\r\n').encode())

    def test_clean_skip_preserves_bytes_and_never_builds_system_or_context(self):
        with tempfile.TemporaryDirectory() as directory:
            source, output = Path(directory)/'input.pdb', Path(directory)/'output.pdb'
            self.clean_pdb(source)
            with patch.object(sys, 'argv', ['cleanup', '--input', str(source), '--output', str(output)]), \
                 patch.object(cleanup, 'build_system', side_effect=AssertionError('must skip')), \
                 patch.object(cleanup, 'create_compute_context', side_effect=AssertionError('must skip')):
                cleanup.main()
            self.assertEqual(source.read_bytes(), output.read_bytes())
            report = json.loads(output.with_suffix('.json').read_text())
            self.assertEqual(report['status'], 'skipped_no_severe_overlaps')
            self.assertFalse(report['triggered'])
            self.assertFalse(output.with_name('output_frames.pdb').exists())

    def test_mild_heavy_overlap_does_not_trigger(self):
        # Two CAs 2.8 A apart: a 0.6 A Bondi overlap, recorded but not severe.
        top = Topology(); chain = top.addChain('A')
        first = top.addResidue('GLY', chain, '1')
        a = top.addAtom('CA', element.carbon, first)
        h = top.addAtom('H', element.hydrogen, first); top.addBond(a, h)
        top.addAtom('CA', element.carbon, top.addResidue('GLY', chain, '2'))
        with tempfile.TemporaryDirectory() as directory:
            source, output = Path(directory)/'input.pdb', Path(directory)/'output.pdb'
            with source.open('w') as f:
                PDBFile.writeFile(top, np.array([[0., 0, 0], [-.1, 0, 0], [.28, 0, 0]])*unit.nanometer, f)
            with patch.object(sys, 'argv', ['cleanup', '--input', str(source), '--output', str(output)]), \
                 patch.object(cleanup, 'build_system', side_effect=AssertionError('must skip')), \
                 patch.object(cleanup, 'create_compute_context', side_effect=AssertionError('must skip')):
                cleanup.main()
            self.assertEqual(source.read_bytes(), output.read_bytes())
            report = json.loads(output.with_suffix('.json').read_text())
            self.assertEqual(report['status'], 'skipped_no_severe_overlaps')
            self.assertTrue(any(r['category'] == 'steric_overlap' for r in report['contacts_before']))

    def test_tiny_repair_runs_rounds_and_hands_off_unrestrained_system(self):
        # Only three atoms, with an injected toy force field; never a protein run.
        # CAs start 1.5 A apart (1.9 A overlap) so the severe trigger fires.
        top = Topology(); chain = top.addChain('A')
        first = top.addResidue('GLY', chain, '1')
        a = top.addAtom('CA', element.carbon, first)
        h = top.addAtom('H', element.hydrogen, first); top.addBond(a, h)
        second = top.addResidue('GLY', chain, '2')
        top.addAtom('CA', element.carbon, second)
        original = openmm.System(); nb = openmm.NonbondedForce()
        for mass, epsilon in ((12., .5), (1., 0.), (12., .5)):
            original.addParticle(mass); nb.addParticle(0., .3, epsilon)
        nb.addException(0, 1, 0., .3, 0.)
        original.addForce(nb)
        bond = openmm.HarmonicBondForce(); bond.addBond(0, 1, .1, 10000.)
        original.addForce(bond)
        snapshot = openmm.XmlSerializer.serialize(original)
        with tempfile.TemporaryDirectory() as directory:
            source, output = Path(directory)/'input.pdb', Path(directory)/'output.pdb'
            with source.open('w') as f:
                PDBFile.writeFile(top, np.array([[0., 0, 0], [-.1, 0, 0], [.15, 0, 0]])*unit.nanometer, f)
            with patch.object(sys, 'argv', ['cleanup', '--input', str(source), '--output', str(output),
                                          '--platform', 'CPU', '--threads', '1', '--max-iterations', '200', '--tolerance', '.1']), \
                 patch.object(cleanup, 'build_system', return_value=(original, np.ones(3, dtype=bool), {})):
                cleanup.main()
            report = json.loads(output.with_suffix('.json').read_text())
            self.assertEqual(report['status'], 'clash_cleanup_passed')
            self.assertTrue(report['triggered'])
            self.assertEqual(len(report['rounds']), len(cleanup.DEFAULT_ALPHAS))
            self.assertTrue(report['geometry_after']['passed'])
            self.assertTrue(report['converged'])
            self.assertEqual(snapshot, openmm.XmlSerializer.serialize(original))
            self.assertEqual(report['output_sha256'], hashlib.sha256(output.read_bytes()).hexdigest())

    def test_pipeline_skips_relaxation_for_samples_whose_cleanup_needs_review(self):
        # Shell orchestration only: every stage is a stub that records its name.
        # The sampling stub makes two sample folders; cleanup "needs review" in
        # both, so each sample stops before relaxation but both samples run.
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            (folder/'relax_pipeline.sh').write_text((root/'relax_pipeline.sh').read_text())
            names = ('cap_generic.py', 'sample_protonation.py', 'add_hydrogens_generic.py',
                     'remove_clashes_generic.py', 'clash_geometry.py', 'validate_clash_output.py',
                     'relax_hydrogens_generic.py', 'relax_full_structure_generic.py', 'preparation_checks.py')
            stub = (
                "from pathlib import Path\n"
                "import sys\n"
                "name = Path(__file__).name\n"
                "with (Path(__file__).parent/'called.txt').open('a') as f:\n"
                "    f.write(name + '\\n')\n"
                "if name == 'sample_protonation.py':\n"
                "    output_dir = Path(sys.argv[sys.argv.index('--output-dir') + 1])\n"
                "    for i in (1, 2):\n"
                "        (output_dir/f'sample_{i:03d}').mkdir()\n"
                "if name == 'remove_clashes_generic.py':\n"
                "    sys.exit(2)\n")
            for name in names: (folder/name).write_text(stub)
            source = folder/'input.pdb'; self.clean_pdb(source)
            env = dict(os.environ, PYTHON=sys.executable, RUNS_DIR=str(folder/'runs'),
                       PYTHONPATH=str(folder)+os.pathsep+os.environ.get('PYTHONPATH', ''))
            result = subprocess.run(['bash', str(folder/'relax_pipeline.sh'), str(source)],
                                    capture_output=True, text=True, env=env, timeout=20)
            self.assertEqual(result.returncode, 2, result.stdout+result.stderr)
            per_sample = ['add_hydrogens_generic.py', 'remove_clashes_generic.py']
            self.assertEqual((folder/'called.txt').read_text().splitlines(),
                             ['cap_generic.py', 'sample_protonation.py'] + per_sample + per_sample)
            self.assertIn('Samples: 0 passed, 2 need review, 0 failed (of 10).', result.stdout)
            self.assertNotIn('SUCCESS:', result.stdout)

    def test_final_gate_records_findings_without_failing_unless_strict(self):
        with tempfile.TemporaryDirectory() as directory:
            source, report = Path(directory)/'input.pdb', Path(directory)/'cleanup.json'
            self.clean_pdb(source)
            report.write_text(json.dumps(dict(status='clash_cleanup_passed', severe_overlap_angstrom=1.0, output=str(source),
                                             output_sha256=hashlib.sha256(source.read_bytes()).hexdigest())))
            output, strict = Path(directory)/'gate.json', Path(directory)/'gate_strict.json'
            base = ['gate', '--input', str(source), '--cleanup-report', str(report)]
            with patch.object(sys, 'argv', base+['--output', str(output)]), \
                 patch.object(geometry, 'geometry_report', return_value=dict(passed=False, ring_piercings=['bad'])):
                gate.main()
            recorded = json.loads(output.read_text())
            self.assertEqual(recorded['status'], 'final_geometry_findings')
            self.assertFalse(recorded['passed'])
            with patch.object(sys, 'argv', base+['--output', str(strict), '--strict']), \
                 patch.object(geometry, 'geometry_report', return_value=dict(passed=False, ring_piercings=['bad'])):
                with self.assertRaises(SystemExit) as exc: gate.main()
            self.assertEqual(exc.exception.code, 2)
            self.assertEqual(json.loads(strict.read_text())['status'], 'final_geometry_findings')

    def test_final_gate_clean_path_skips_extra_checks(self):
        with tempfile.TemporaryDirectory() as directory:
            source, report = Path(directory)/'input.pdb', Path(directory)/'cleanup.json'
            self.clean_pdb(source)
            report.write_text(json.dumps(dict(status='skipped_no_severe_overlaps', severe_overlap_angstrom=1.0, output=str(source),
                                             output_sha256=hashlib.sha256(source.read_bytes()).hexdigest())))
            output = Path(directory)/'gate.json'
            with patch.object(sys, 'argv', ['gate', '--input', str(source), '--cleanup-report', str(report), '--output', str(output)]), \
                 patch.object(geometry, 'geometry_report', side_effect=AssertionError('must skip')):
                gate.main()
            self.assertTrue(json.loads(output.read_text())['passed'])

    def test_final_gate_uses_cleanup_threshold(self):
        with tempfile.TemporaryDirectory() as directory:
            source, report = Path(directory)/'input.pdb', Path(directory)/'cleanup.json'
            self.clean_pdb(source)
            report.write_text(json.dumps(dict(output=str(source), status='clash_cleanup_passed', severe_overlap_angstrom=1.3,
                                             output_sha256=hashlib.sha256(source.read_bytes()).hexdigest())))
            output = Path(directory)/'gate.json'
            with patch.object(sys, 'argv', ['gate', '--input', str(source), '--cleanup-report', str(report), '--output', str(output)]), \
                 patch.object(geometry, 'geometry_report', return_value=dict(passed=True)) as mocked:
                gate.main()
            self.assertEqual(mocked.call_args.kwargs['severe_overlap'], 1.3)
            self.assertTrue(json.loads(output.read_text())['passed'])


if __name__ == '__main__':
    unittest.main()
