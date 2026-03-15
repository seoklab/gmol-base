from pathlib import Path

import pytest
from pydantic import TypeAdapter

from gmol.base.data.mmcif import (
    Assembly,
    ChemComp,
    MolType,
    load_components,
    load_mmcif_single,
    mmcif_assemblies,
)


@pytest.fixture(scope="module")
def ccd_with_ligands(test_data: Path) -> dict[str, ChemComp]:
    ccd = load_components(test_data / "ccd" / "components_stdres.cif")
    extra = TypeAdapter(dict[str, ChemComp]).validate_json(
        (test_data / "ccd" / "components_5m7m.json").read_bytes()
    )
    ccd.update(extra)
    return ccd


@pytest.fixture(scope="module")
def assembly_with_ligands(
    test_data: Path, ccd_with_ligands: dict[str, ChemComp]
) -> Assembly:
    data = load_mmcif_single(test_data / "mmcif" / "5m7m.cif")
    assemblies = mmcif_assemblies(data, ccd_with_ligands)
    assert len(assemblies) == 1
    return assemblies[0]


def test_ligand_label_seq_id_is_dot(assembly_with_ligands: Assembly):
    """Non-polymer (ligand) chains should have '.' for label_seq_id in mmCIF output."""
    mmcif_str = assembly_with_ligands.to_mmcif("test")

    # Parse atom_site lines from the mmCIF output
    in_atom_site = False
    field_names: list[str] = []
    label_seq_id_col = -1
    label_asym_id_col = -1

    ligand_chain_ids = {
        cid
        for cid, chain in assembly_with_ligands.chains.items()
        if chain.type == MolType.Ligand
    }
    polymer_chain_ids = {
        cid
        for cid, chain in assembly_with_ligands.chains.items()
        if chain.type.is_polymer
    }

    assert len(ligand_chain_ids) > 0, "Test requires ligand chains"
    assert len(polymer_chain_ids) > 0, "Test requires polymer chains"

    ligand_label_seq_ids: set[str] = set()
    polymer_label_seq_ids: set[str] = set()

    for line in mmcif_str.splitlines():
        line = line.strip()

        if line == "loop_":
            in_atom_site = False
            field_names = []
            continue

        if line.startswith("_atom_site."):
            field_names.append(line.split(".")[1])
            if field_names[-1] == "label_seq_id":
                label_seq_id_col = len(field_names) - 1
            if field_names[-1] == "label_asym_id":
                label_asym_id_col = len(field_names) - 1
            in_atom_site = True
            continue

        if (
            in_atom_site
            and label_seq_id_col >= 0
            and line
            and not line.startswith("_")
            and not line.startswith("#")
        ):
            tokens = line.split()
            if len(tokens) >= len(field_names):
                asym_id = tokens[label_asym_id_col]
                seq_id = tokens[label_seq_id_col]

                if asym_id in ligand_chain_ids:
                    ligand_label_seq_ids.add(seq_id)
                elif asym_id in polymer_chain_ids:
                    polymer_label_seq_ids.add(seq_id)

        if line.startswith("#") and in_atom_site:
            in_atom_site = False

    # Ligand label_seq_id must all be "."
    assert ligand_label_seq_ids == {"."}, (
        f"Ligand label_seq_id should be '.', got {ligand_label_seq_ids}"
    )

    # Polymer label_seq_id must be numeric (not ".")
    assert "." not in polymer_label_seq_ids, (
        "Polymer label_seq_id should not contain '.'"
    )
    assert len(polymer_label_seq_ids) > 0
