import datetime as dt
from pathlib import Path

import numpy as np
import pytest
from pydantic import TypeAdapter

from gmol.base.data.mmcif import (
    ChemComp,
    filter_mmcif,
    load_mmcif_single,
    mmcif_assemblies,
)
from gmol.base.data.mmcif.input import build_input


@pytest.fixture(scope="session")
def ccd_min(test_data: Path) -> dict[str, ChemComp]:
    return TypeAdapter(dict[str, ChemComp]).validate_json(
        test_data.joinpath("ccd", "components_min.json").read_bytes()
    )


def _build_input_for_pdb(
    pdb_id: str,
    test_data: Path,
    ccd: dict[str, ChemComp],
):
    mmcif = test_data / "mmcif" / f"{pdb_id}.cif"
    data = load_mmcif_single(mmcif)

    assemblies = mmcif_assemblies(data, ccd)
    assert len(assemblies) == 1

    filtered = filter_mmcif(
        assemblies[0], ccd, cutoff_date=dt.date(9999, 12, 31)
    )
    assert filtered is not None

    return build_input(filtered, ccd, split_modified=False)


def test_atom_site_bfactor_is_parsed(test_data: Path):
    mmcif = test_data / "mmcif" / "1ubq.cif"
    data = load_mmcif_single(mmcif)

    assert len(data.atom_site) > 0

    bvals = [a.b_iso_or_equiv for a in data.atom_site[:50]]
    assert any(v is not None for v in bvals), "B-factor should be present"

    non_null = [v for v in bvals if v is not None]
    assert all(float(v) >= 0.0 for v in non_null)


def test_polymer_chain_bfactor_tensor(
    test_data: Path, ccd_min: dict[str, ChemComp]
):
    input_data = _build_input_for_pdb("4hf7", test_data, ccd_min)
    assert len(input_data.polymers) > 0

    poly = input_data.polymers[0]
    assert poly.atom_coords.shape[:2] == poly.atom_b_factors.shape

    # If a coordinate exists for an atom type, B-factor should also be set
    # (at least for some atoms; missing values are NaN)
    present = ~np.isnan(poly.atom_coords[..., 0])
    b_present = ~np.isnan(poly.atom_b_factors)
    assert int(np.sum(present & b_present)) > 0


def test_ligand_chain_bfactor_vector(
    test_data: Path, ccd_min: dict[str, ChemComp]
):
    input_data = _build_input_for_pdb("7qgw", test_data, ccd_min)
    assert len(input_data.ligands) > 0

    lig = next(
        (l for l in input_data.ligands if l.atom_b_factors is not None), None
    )
    assert lig is not None

    b_factors = lig.atom_b_factors
    assert b_factors is not None

    assert b_factors.shape == (len(lig.atom_ids),)
    assert int(np.sum(~np.isnan(b_factors))) > 0
