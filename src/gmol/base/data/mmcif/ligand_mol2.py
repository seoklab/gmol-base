"""Ligand chains as MOL2 (Open Babel), from mmCIF / assembly."""

from __future__ import annotations

from pathlib import Path

from rdkit import Chem
from rdkit.Geometry import Point3D

from gmol.base.wrapper.obabel import mol_block_to_mol2
from .assembly import Assembly, MolType, mmcif_assemblies
from .filter import filter_mmcif
from .input import build_ref_ligand_atom_coords
from .parse import ChemComp, load_mmcif_single
from .smiles import mol_from_chem_comp, reference_from_mmcif


def assembly_ligand_chains_mol2(assembly: Assembly) -> list[tuple[str, str]]:
    """One MOL2 string per ligand chain (``label_asym_id`` / ``chain_id``).

    Branched or multi-residue ligand chains use the same graph construction as
    :func:`gmol.base.data.mmcif.input.process_ligand_chain`; coordinates come
    from the assembly, bond orders and typing from RDKit then Open Babel.
    """
    entry = assembly.metadata.entry_id
    out: list[tuple[str, str]] = []

    for chain in assembly.chains_of_type(MolType.Ligand):
        ccs = [
            (residue.residue_id, residue.chem_comp)
            for residue in assembly.residues_of_chain(chain)
        ]
        ref = reference_from_mmcif(ccs, chain.branches)
        mol = mol_from_chem_comp(ref.atoms, ref.bonds)
        atom_ids = [a.GetProp("atom_id") for a in mol.GetAtoms()]
        residues = [assembly.residues[rid] for rid, _ in ccs]
        coords, _ = build_ref_ligand_atom_coords(assembly, residues, atom_ids)

        conf = Chem.Conformer(mol.GetNumAtoms())
        for i in range(mol.GetNumAtoms()):
            c = coords[i]
            conf.SetAtomPosition(
                i, Point3D(float(c[0]), float(c[1]), float(c[2]))
            )
        mol.RemoveAllConformers()
        mol.AddConformer(conf)

        mol_block = Chem.MolToMolBlock(mol, confId=0)
        title = f"{entry}_{chain.chain_id}"
        mol2 = mol_block_to_mol2(mol_block, title=title)
        out.append((chain.chain_id, mol2))

    return out


def mmcif_ligand_chains_mol2(
    mmcif_path: str | Path,
    chem_comp_dict: dict[str, ChemComp],
    *,
    is_test: bool = False,
) -> list[tuple[str, str]]:
    """Load mmCIF, apply :func:`filter_mmcif`, return ligand-chain MOL2 strings.

    If the structure fails the mmcif pre-filter, returns an empty list (same
    idea as :func:`gmol.base.data.mmcif.input.build_input_from_mmcif`).
    """
    path = Path(mmcif_path)
    metadata = load_mmcif_single(path)
    assemblies = mmcif_assemblies(metadata, chem_comp_dict)
    filtered = filter_mmcif(assemblies[0], chem_comp_dict, is_test=is_test)
    if filtered is None:
        return []
    return assembly_ligand_chains_mol2(filtered)
