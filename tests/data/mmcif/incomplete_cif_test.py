import datetime as dt
from pathlib import Path

import pytest

from gmol.base.data.mmcif import (
    Assembly,
    ChemComp,
    load_mmcif_single,
    mmcif_assemblies,
)
from gmol.base.data.mmcif.parse import Mmcif


@pytest.fixture(scope="module")
def incomplete_mmcif(test_data: Path) -> Mmcif:
    return load_mmcif_single(
        test_data / "mmcif" / "metadata_incomplete_8aq1.cif"
    )


@pytest.fixture(scope="module")
def incomplete_assembly(
    incomplete_mmcif: Mmcif, ccd_components: dict[str, ChemComp]
) -> Assembly:
    assemblies = mmcif_assemblies(incomplete_mmcif, ccd_components)
    assert len(assemblies) == 1
    return assemblies[0]


def test_load_incomplete_mmcif(incomplete_mmcif: Mmcif):
    assert incomplete_mmcif.entry_id == "metadata_incomplete_8AQ1"
    assert incomplete_mmcif.exptl_method == ""
    assert incomplete_mmcif.pdbx_keywords == ""
    assert incomplete_mmcif.revision_date == dt.date(1970, 1, 1)
    assert incomplete_mmcif.resolution == 999.9
    assert len(incomplete_mmcif.entity) == 2
    assert incomplete_mmcif.entity[1].pdbx_description is None
    assert incomplete_mmcif.pdbx_struct_assembly == []
    assert incomplete_mmcif.pdbx_struct_oper_list == {}


def test_incomplete_atom_site_auth_comp_id(incomplete_mmcif: Mmcif):
    for atom in incomplete_mmcif.atom_site:
        assert atom.auth_comp_id == atom.label_comp_id


def test_incomplete_nonpoly_scheme(incomplete_mmcif: Mmcif):
    assert "B" in incomplete_mmcif.pdbx_nonpoly_scheme
    schemes = incomplete_mmcif.pdbx_nonpoly_scheme["B"]
    assert len(schemes) == 1
    assert schemes[0].mon_id == "NJC"
    assert schemes[0].seq_id == schemes[0].pdb_seq_num


def test_incomplete_struct_conn(incomplete_mmcif: Mmcif):
    assert len(incomplete_mmcif.struct_conn) == 1
    conn = incomplete_mmcif.struct_conn[0]
    assert conn.conn_type_id == "covale"
    assert conn.pdbx_leaving_atom_flag == 0
    assert conn.pdbx_dist_value is None
    assert conn.ptnr1.label_comp_id == "CYS"
    assert conn.ptnr2.label_comp_id == "NJC"
    assert conn.ptnr1.auth_comp_id == "CYS"
    assert conn.ptnr2.auth_comp_id == "NJC"


def test_incomplete_assembly_structure(incomplete_assembly: Assembly):
    assert "A" in incomplete_assembly.chains
    assert incomplete_assembly.chains["A"].type.is_polymer
    assert len(incomplete_assembly.atoms) == 1856
    assert len(incomplete_assembly.residues) == 236


def test_incomplete_assembly_metadata(incomplete_assembly: Assembly):
    m = incomplete_assembly.metadata
    assert m.entry_id == "metadata_incomplete_8AQ1"
    assert m.exptl_method == ""
    assert m.revision_date == dt.date(1970, 1, 1)


def test_incomplete_assembly_to_mmcif(incomplete_assembly: Assembly):
    mmcif_str = incomplete_assembly.to_mmcif("metadata_incomplete_8aq1")
    assert "None" not in mmcif_str
