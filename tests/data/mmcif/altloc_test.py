from pathlib import Path

import pytest
from pydantic import TypeAdapter

from gmol.base.data.mmcif import (
    Assembly,
    ChemComp,
    load_mmcif_single,
    mmcif_assemblies,
)


@pytest.fixture(scope="module")
def ccd_with_sugars(
    test_data: Path, ccd_components: dict[str, ChemComp]
) -> dict[str, ChemComp]:
    extra = TypeAdapter(dict[str, ChemComp]).validate_json(
        (test_data / "ccd" / "components_7qe4.json").read_bytes()
    )
    return {**ccd_components, **extra}


@pytest.fixture(scope="module")
def assembly_7qe4(
    test_data: Path, ccd_with_sugars: dict[str, ChemComp]
) -> Assembly:
    data = load_mmcif_single(test_data / "mmcif" / "7qe4.cif")
    assemblies = mmcif_assemblies(data, ccd_with_sugars)
    assert len(assemblies) >= 1
    return assemblies[0]


@pytest.fixture(scope="module")
def assembly_7qgw(
    test_data: Path, ccd_components: dict[str, ChemComp]
) -> Assembly:
    data = load_mmcif_single(test_data / "mmcif" / "7qgw.cif")
    assemblies = mmcif_assemblies(data, ccd_components)
    assert len(assemblies) >= 1
    return assemblies[0]


def test_within_residue_altlocs_cleared(assembly_7qgw: Assembly):
    """Within residue altlocs should have label_alt_id cleared after altloc resolution."""
    assert len(assembly_7qgw.atoms) > 0
    for atom in assembly_7qgw.atoms:
        assert atom.label_alt_id is None, (
            f"Within residue altloc atom {atom.atom_id} in {atom.residue_id} should have",
            f"label_alt_id=None, but has {atom.label_alt_id}",
        )


def test_cross_chain_altlocs_preserved(assembly_7qe4: Assembly):
    """Cross chain altloc atoms should retain their label_alt_id since it is not resolved in gmol-base."""
    chain_d_alt_ids = {
        atom.label_alt_id
        for atom in assembly_7qe4.atoms
        if atom.chain_id == "D"
    }
    chain_f_alt_ids = {
        atom.label_alt_id
        for atom in assembly_7qe4.atoms
        if atom.chain_id == "F"
    }

    assert "D" in assembly_7qe4.chains, "Chain D A2G should exist"
    assert "F" in assembly_7qe4.chains, "Chain F NGA should exist"

    assert "A" in chain_d_alt_ids, (
        f"Chain D A2G should have label_alt_id='A', but got {chain_d_alt_ids}"
    )
    assert "B" in chain_f_alt_ids, (
        f"Chain F NGA should have label_alt_id='B', but got {chain_f_alt_ids}"
    )

    protein_atoms_with_altloc = [
        atom
        for atom in assembly_7qe4.atoms
        if atom.chain_id == "A" and atom.label_alt_id is not None
    ]
    assert len(protein_atoms_with_altloc) == 0, (
        "Protein chain A should have all label_alt_id=None after "
        f"within residue altloc resolution, but {len(protein_atoms_with_altloc)} "
        "atoms have non-None label_alt_id"
    )


def _parse_label_alt_ids(mmcif_str: str):
    """Parse (label_asym_id, label_alt_id) tuples from mmCIF _atom_site block."""
    atom_site_lines: list[str] = []
    in_atom_site = False

    for line in mmcif_str.splitlines():
        stripped = line.strip()
        if stripped.startswith("_atom_site."):
            in_atom_site = True
            continue
        if in_atom_site and stripped and not stripped.startswith(("_", "#")):
            atom_site_lines.append(stripped)
        elif in_atom_site and (stripped.startswith("#") or stripped == ""):
            if atom_site_lines:
                break

    return [
        (t[6], t[4]) for asl in atom_site_lines if len(t := asl.split()) > 6
    ]


def test_cross_chain_altloc_mmcif_output(assembly_7qe4: Assembly):
    """Cross chain altloc label_alt_id should be written to mmCIF output."""
    mmcif_str = assembly_7qe4.to_mmcif("7QE4")
    rows = _parse_label_alt_ids(mmcif_str)
    assert len(rows) > 0

    chain_d_alt_ids = {val for cid, val in rows if cid == "D"}
    chain_f_alt_ids = {val for cid, val in rows if cid == "F"}

    assert "A" in chain_d_alt_ids, (
        f"mmCIF chain D should have label_alt_id='A', got {chain_d_alt_ids}"
    )
    assert "B" in chain_f_alt_ids, (
        f"mmCIF chain F should have label_alt_id='B', got {chain_f_alt_ids}"
    )


def test_within_residue_altloc_mmcif_output(assembly_7qgw: Assembly):
    """Within residue altlocs should be written as '.' in mmCIF output."""
    mmcif_str = assembly_7qgw.to_mmcif("7qgw")
    rows = _parse_label_alt_ids(mmcif_str)
    assert len(rows) > 0

    alt_ids = {val for _, val in rows}
    assert alt_ids == {"."}, f"All label_alt_id should be '.', got {alt_ids}"


def test_cross_chain_altloc_pdb_output(assembly_7qe4: Assembly):
    """Cross chain altloc atoms should have altloc indicator in PDB."""
    pdb_str = assembly_7qe4.to_pdb()

    a2g_altlocs: set[str] = set()
    nga_altlocs: set[str] = set()
    for line in pdb_str.splitlines():
        if not (line.startswith("ATOM") or line.startswith("HETATM")):
            continue

        alt_id = line[16]
        res_name = line[17:20].strip()

        if res_name == "A2G":
            a2g_altlocs.add(alt_id)
        elif res_name == "NGA":
            nga_altlocs.add(alt_id)

    assert "A" in a2g_altlocs, (
        f"PDB A2G atoms should have altloc 'A', got {a2g_altlocs}"
    )
    assert "B" in nga_altlocs, (
        f"PDB NGA atoms should have altloc 'B', got {nga_altlocs}"
    )
