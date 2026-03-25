import nuri
import pytest

from gmol.base.data.mmcif import Assembly


def test_to_pdb(sample_assembly: Assembly):
    result = sample_assembly.to_pdb()

    mols = list(nuri.readstring("pdb", result, sanitize=False))
    assert len(mols) == 1

    mol = mols[0]
    assert len(mol) == len(sample_assembly.atoms)


def test_to_pdb_writes_b_factor(sample_assembly: Assembly):
    result = sample_assembly.to_pdb()
    atom_lines = [
        line
        for line in result.splitlines()
        if line.startswith("ATOM  ") or line.startswith("HETATM")
    ]
    assert len(atom_lines) == len(sample_assembly.atoms)

    atoms_sorted = sorted(
        sample_assembly.atoms,
        key=lambda atom: (atom.residue_id, atom.atom_idx),
    )
    expected = [f"{atom.b_factor:.2f}" for atom in atoms_sorted]
    actual = [line[60:66].strip() for line in atom_lines]

    assert actual == expected


def test_to_pdb_raises_when_b_factor_exceeds_field_width(
    sample_assembly: Assembly,
):
    assembly = sample_assembly.model_copy(deep=True)
    assembly.atoms[0].b_factor = 1000.0

    with pytest.raises(AssertionError):
        assembly.to_pdb()
