import nuri

from gmol.base.data.mmcif import Assembly


def test_to_pdb(sample_assembly: Assembly):
    result = sample_assembly.to_pdb()

    mols = list(nuri.readstring("pdb", result, sanitize=False))
    assert len(mols) == 1

    mol = mols[0]
    assert len(mol) == len(sample_assembly.atoms)
