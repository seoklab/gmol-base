from pathlib import Path

from pydantic import TypeAdapter

from gmol.base.data.mmcif import (
    ChemComp,
    assembly_ligand_chains_mol2,
    load_components,
    load_mmcif_single,
    mmcif_assemblies,
    mmcif_ligand_chains_mol2,
)


def test_assembly_ligand_chains_mol2_branched_nag(test_data: Path):
    """Branched NAG chain produces valid Open Babel MOL2 with expected title."""
    ccd = load_components(test_data / "ccd" / "components_stdres.cif")
    ccd.update(
        TypeAdapter(dict[str, ChemComp]).validate_json(
            (test_data / "ccd" / "components_4mb4.json").read_bytes()
        )
    )
    data = load_mmcif_single(test_data / "mmcif" / "4mb4.cif")
    assembly = mmcif_assemblies(data, ccd)[0]

    items = assembly_ligand_chains_mol2(assembly)
    by_id = dict(items)
    assert "B" in by_id
    mol2_b = by_id["B"]
    assert "@<TRIPOS>MOLECULE" in mol2_b
    assert "@<TRIPOS>ATOM" in mol2_b
    assert "@<TRIPOS>BOND" in mol2_b
    title_line = mol2_b.splitlines()[1]
    assert title_line.endswith("_B")


def test_mmcif_ligand_chains_mol2_filtered(test_data: Path):
    """End-to-end mmCIF path returns one MOL2 per surviving ligand chain."""
    ccd = load_components(test_data / "ccd" / "components_stdres.cif")
    ccd.update(
        TypeAdapter(dict[str, ChemComp]).validate_json(
            (test_data / "ccd" / "components_4mb4.json").read_bytes()
        )
    )
    path = test_data / "mmcif" / "4mb4.cif"
    items = mmcif_ligand_chains_mol2(path, ccd, is_test=True)
    assert len(items) >= 1
    for chain_id, mol2 in items:
        assert chain_id
        assert mol2.startswith("@<TRIPOS>MOLECULE")
