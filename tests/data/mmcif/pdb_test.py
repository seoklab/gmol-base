import nuri
import pytest

from gmol.base.data.mmcif import Assembly, ResidueId


def _atom_lines(pdb_str: str) -> list[str]:
    return [
        line
        for line in pdb_str.splitlines()
        if line.startswith("ATOM  ") or line.startswith("HETATM")
    ]


def test_to_pdb_roundtrips(sample_assembly: Assembly):
    result = sample_assembly.to_pdb()

    mols = list(nuri.readstring("pdb", result, sanitize=False))
    assert len(mols) == 1

    mol = mols[0]
    assert len(mol) == len(sample_assembly.atoms)


def test_to_pdb_writes_group_pdb_from_atoms(sample_assembly: Assembly):
    assembly = sample_assembly.model_copy(deep=True)
    residue_id = assembly.atoms[0].residue_id
    residue_atoms = [
        atom for atom in assembly.atoms if atom.residue_id == residue_id
    ]

    for atom in assembly.atoms:
        if atom.residue_id == residue_id:
            atom.group_PDB = "HETATM"

    result = assembly.to_pdb()
    atom_lines = _atom_lines(result)

    atoms_sorted = sorted(
        assembly.atoms,
        key=lambda atom: (atom.residue_id, atom.atom_idx),
    )
    actual = [line[:6] for line in atom_lines]
    expected = [
        "ATOM  " if atom.group_PDB == "ATOM" else "HETATM"
        for atom in atoms_sorted
    ]

    assert actual == expected
    assert actual[: len(residue_atoms)] == ["HETATM"] * len(residue_atoms)


def test_to_pdb_writes_b_factor(sample_assembly: Assembly):
    result = sample_assembly.to_pdb()
    atom_lines = _atom_lines(result)
    assert len(atom_lines) == len(sample_assembly.atoms)

    atoms_sorted = sorted(
        sample_assembly.atoms,
        key=lambda atom: (atom.residue_id, atom.atom_idx),
    )
    expected = [f"{atom.b_factor:.2f}" for atom in atoms_sorted]
    actual = [line[60:66].strip() for line in atom_lines]

    assert actual == expected


def test_to_pdb_writes_insertion_code(sample_assembly: Assembly):
    assembly = sample_assembly.model_copy(deep=True)
    residue_id = assembly.atoms[-1].residue_id
    residue_id_with_ins = ResidueId(
        residue_id.chain_id,
        residue_id.seq_id,
        "A",
    )

    for atom in assembly.atoms:
        if atom.residue_id == residue_id:
            atom.residue_id = residue_id_with_ins

    result = assembly.to_pdb()
    atom_lines = _atom_lines(result)

    assert atom_lines[-1][26] == "A"

    ter_line = next(
        line for line in result.splitlines() if line.startswith("TER")
    )
    assert ter_line[26] == "A"


def test_to_pdb_rejects_unknown_group_pdb(sample_assembly: Assembly):
    assembly = sample_assembly.model_copy(deep=True)
    assembly.atoms[0].group_PDB = "ANISOU"

    with pytest.raises(ValueError, match="Unsupported PDB atom record type"):
        assembly.to_pdb()


def test_to_pdb_raises_when_b_factor_exceeds_field_width(
    sample_assembly: Assembly,
):
    assembly = sample_assembly.model_copy(deep=True)
    assembly.atoms[0].b_factor = 1000.0

    with pytest.raises(AssertionError):
        assembly.to_pdb()
