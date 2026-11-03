# Experimental AD tau reference structures (5O3L)

Prepared 2026-11-02 from the user's saved experimental reference, not a RibbonFold prediction.
PDB entry: https://www.rcsb.org/structure/5O3L
Structure: paired helical filament from Alzheimer's disease brain.

- `ad_tau_5O3L_full_filament.pdb`: the complete supplied file, preserved byte-for-byte. Two protofilaments, chains A/C/E/G/I and B/D/F/H/J; five layers each, ten protein chains total.
- `ad_tau_5O3L_single_protofilament.pdb`: chains A/C/E/G/I only, selected using the repeated stacking geometry. Five layers, 365 residues, 5680 atom records (2895 hydrogen records). Each chain spans tau residues 306–378. All retained atom records and coordinates are unchanged.

These are experimental-structure-derived reference inputs. The supplied file already includes hydrogen atoms; preserving those does not imply their positions were experimentally resolved. No atoms were added, and no relaxation, repacking, or coordinate transformation was performed. Original chain IDs and atom/residue numbering are retained. The extracted file's header identifies the subset; obsolete full-assembly remarks and the original MASTER count record are omitted.

Original location (moved to the full-filament filename):
`/Users/phillipbrandel/Desktop/RibbonFold Experiment/experimental_results/5O3L.pdb`

SHA-256 of original/full file: `2eabac8b92d32e98dbe181a652afb28d5d91c39604adf63ab516395d975bedd4`
SHA-256 of extracted protofilament: `eab00d0cec036886e9d111b2540ae10be84b24dd4c34b89268418eb31d0f49ce`

This is a finite five-layer single-protofilament reference. Its exposed ends and missing partner protofilament matter when setting up subsequent energy calculations; no energy-preparation choices have been made here.
