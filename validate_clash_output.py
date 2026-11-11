#!/usr/bin/env python3
"""Recheck repaired structures after full relaxation; clean-input path is unchanged.

Fails only on overlaps above the cleanup report's severe threshold; ordinary
close contacts (e.g. hydrogen bonds, Calpha-H...O) are listed but not gated.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from openmm import unit
from openmm.app import PDBFile
import clash_geometry as geometry
import preparation_checks as checks


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--cleanup-report', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True, help='New JSON gate report')
    args = p.parse_args()
    cleanup = json.loads(args.cleanup_report.read_text())
    checks.require(cleanup.get('status') in ('clash_cleanup_passed', 'skipped_no_severe_overlaps'),
                   'Cleanup did not pass')
    severe = cleanup.get('severe_overlap_angstrom')
    checks.require(isinstance(severe, (int, float)) and np.isfinite(severe) and severe >= 0.4,
                   'Cleanup report lacks a valid severe_overlap_angstrom')
    checks.require(hashlib.sha256(Path(cleanup['output']).read_bytes()).hexdigest() == cleanup['output_sha256'],
                   'Cleanup output hash mismatch')
    source = args.input.resolve()
    input_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    if cleanup['status'] == 'skipped_no_severe_overlaps':
        report = dict(status='skipped_clean_input', passed=True)
    else:
        pdb = PDBFile(str(source))
        checks.require(pdb.getNumFrames() == 1, 'Expected single-model final PDB')
        report = geometry.geometry_report(pdb.topology, np.array(pdb.positions.value_in_unit(unit.angstrom)),
                                          include_hydrogens=True, severe_overlap=severe)
        report['status'] = 'final_geometry_passed' if report['passed'] else 'needs_review'
    report.update(input=str(source), input_sha256=input_hash, cleanup_report=str(args.cleanup_report.resolve()),
                  geometry_sha256=hashlib.sha256(Path(geometry.__file__).read_bytes()).hexdigest())
    checks.require(hashlib.sha256(source.read_bytes()).hexdigest() == input_hash, 'Input changed during gate')
    checks.write_outputs({args.output: json.dumps(report, indent=2)+'\n'})
    print('Final cleanup geometry gate: '+report['status'])
    if not report['passed']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
