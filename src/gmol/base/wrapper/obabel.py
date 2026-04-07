"""Open Babel helpers for small-molecule file formats."""

from openbabel import pybel


def mol_block_to_mol2(mol_block: str, *, title: str | None = None) -> str:
    """Convert an MDL mol block string to Tripos MOL2 via Open Babel."""
    ob_mol = pybel.readstring("mol", mol_block)
    if title is not None:
        ob_mol.title = title
    return ob_mol.write("mol2")
