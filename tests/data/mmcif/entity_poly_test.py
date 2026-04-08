from pathlib import Path

import pytest

from gmol.base.data.mmcif import (
    ChemComp,
    EntityPolySeq,
    load_mmcif_single,
    load_one_assembly,
    mmcif_assemblies,
)


def test_parse_entity_poly_single_entity(test_data: Path):
    mmcif = load_mmcif_single(test_data / "mmcif" / "1ubq.cif")

    assert len(mmcif.entity_poly) == 1
    assert 1 in mmcif.entity_poly

    ep = mmcif.entity_poly[1]
    assert ep.entity_id == 1
    assert ep.type == "polypeptide(L)"
    assert ep.nstd_linkage is False
    assert ep.nstd_monomer is False
    assert ep.pdbx_strand_id == ["A"]
    assert ep.pdbx_seq_one_letter_code is not None
    assert len(ep.pdbx_seq_one_letter_code) == 76


def test_parse_entity_poly_seq_single_entity(test_data: Path):
    mmcif = load_mmcif_single(test_data / "mmcif" / "1ubq.cif")

    assert len(mmcif.entity_poly_seq) == 1
    assert 1 in mmcif.entity_poly_seq

    seq = mmcif.entity_poly_seq[1]
    assert len(seq) == 76
    assert seq[42] == EntityPolySeq(
        entity_id=1, num=43, mon_id="LEU", hetero=False
    )
    assert seq[-1] == EntityPolySeq(
        entity_id=1, num=76, mon_id="GLY", hetero=False
    )


def test_parse_entity_poly_multi_entity(test_data: Path):
    mmcif = load_mmcif_single(test_data / "mmcif" / "7rtb.cif")

    assert len(mmcif.entity_poly) == 6
    for eid in range(1, 7):
        assert eid in mmcif.entity_poly

    assert mmcif.entity_poly[1].pdbx_strand_id == ["A"]
    assert mmcif.entity_poly[2].pdbx_strand_id == ["B"]


def test_parse_entity_poly_multi_strand(test_data: Path):
    mmcif = load_mmcif_single(test_data / "mmcif" / "7qgw.cif")

    ep = mmcif.entity_poly[1]
    assert ep.pdbx_strand_id == ["A", "B"]


@pytest.mark.parametrize("pdb_id", ["1ubq", "7rtb", "7qgw"])
def test_entity_poly_roundtrip(
    test_data: Path,
    tmp_path: Path,
    ccd_components: dict[str, ChemComp],
    pdb_id: str,
):
    src = test_data / "mmcif" / f"{pdb_id}.cif"
    original = load_mmcif_single(src)
    asm = mmcif_assemblies(original, ccd_components)[0]

    cif = asm.to_mmcif(pdb_id)
    out = tmp_path / f"{pdb_id}_roundtrip.cif"
    out.write_text(cif)

    reloaded = load_mmcif_single(out)

    for eid, ep in original.entity_poly.items():
        if eid not in reloaded.entity_poly:
            continue
        rp = reloaded.entity_poly[eid]
        assert rp.type == ep.type

    for eid, seq in original.entity_poly_seq.items():
        if eid not in reloaded.entity_poly_seq:
            continue
        rseq = reloaded.entity_poly_seq[eid]
        assert len(rseq) == len(seq)
        for orig, rel in zip(seq, rseq):
            assert rel.num == orig.num
            assert rel.mon_id == orig.mon_id


@pytest.mark.parametrize("pdb_id", ["7rtb", "7qgw", "6akd"])
def test_entity_poly_in_roundtrip_cif(
    test_data: Path,
    tmp_path: Path,
    pdb_id: str,
):
    src = test_data / "mmcif" / f"{pdb_id}.cif"
    asm = load_one_assembly(src)

    cif = asm.to_mmcif(pdb_id)
    out = tmp_path / f"{pdb_id}_roundtrip.cif"
    out.write_text(cif)

    reloaded = load_one_assembly(out)
    cif2 = reloaded.to_mmcif(pdb_id)

    assert cif.splitlines() == cif2.splitlines()


@pytest.mark.parametrize(
    ("pdb_id", "nstd_monomer_flag"),
    [("4hf7", True), ("3m8z", True), ("13sy", False), ("7rtb", False)],
)
def test_nstd_parsing(
    test_data: Path,
    tmp_path: Path,
    ccd_components: dict[str, ChemComp],
    pdb_id: str,
    nstd_monomer_flag: bool,
):
    mmcif = load_mmcif_single(test_data / "mmcif" / f"{pdb_id}.cif")
    assert mmcif.entity_poly[1].nstd_monomer is nstd_monomer_flag
    assert mmcif.entity_poly[1].nstd_linkage is False
    asm = mmcif_assemblies(mmcif, ccd_components)[0]
    cif = asm.to_mmcif(pdb_id)
    out = tmp_path / f"{pdb_id}_roundtrip.cif"
    out.write_text(cif)
    reloaded = load_mmcif_single(out)
    assert reloaded.entity_poly[1].nstd_monomer is nstd_monomer_flag
    assert reloaded.entity_poly[1].nstd_linkage is False
