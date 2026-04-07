"""Test parsing of Boltz output mmCIF file."""

from pathlib import Path
from typing import Any, TypeAlias

import numpy as np
import pytest
from nuri.fmt import cif_ddl2_frame_as_dict, read_cif

from gmol.base.data.mmcif import ChemComp, load_mmcif_single, mmcif_assemblies

DataDict: TypeAlias = dict[str, list[dict[str, Any]]]


@pytest.fixture(scope="session")
def data_dict(test_data: Path):
    cif_path = test_data / "model_mmcif" / "boltz.cif"
    cif_block = next(read_cif(cif_path)).data
    return cif_ddl2_frame_as_dict(cif_block)


def test_boltz_roundtrip(
    test_data: Path,
    tmp_path: Path,
    ccd_components: dict[str, ChemComp],
):
    mmcif = load_mmcif_single(test_data / "model_mmcif" / "boltz.cif")
    asm = mmcif_assemblies(mmcif, ccd_components)[0]

    cif = asm.to_mmcif("boltz")
    out = tmp_path / "boltz_roundtrip.cif"
    out.write_text(cif)

    reloaded = load_mmcif_single(out)
    assert len(reloaded.entity_poly) == 1
    assert len(reloaded.entity_poly_seq) == 1

    assert reloaded.entity_poly_seq[1] == mmcif.entity_poly_seq[1]

    assert reloaded.entity_poly[1].type == "polypeptide(L)"
    assert reloaded.entity_poly[1].pdbx_strand_id == ["A"]
    assert len(reloaded.entity_poly_seq[1]) == 169

    assert np.allclose(
        reloaded.atom_site[42].cartn, mmcif.atom_site[42].cartn, atol=1e-3
    )
    assert np.allclose(
        reloaded.atom_site[-1].occupancy,
        mmcif.atom_site[-1].occupancy,
        atol=1e-3,
    )
