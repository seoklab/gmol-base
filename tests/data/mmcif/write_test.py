from collections.abc import Sequence
from pathlib import Path

import pytest
from nuri.fmt import cif_ddl2_frame_as_dict, read_cif

from gmol.base.data.mmcif import load_mmcif_single, mmcif_assemblies
from gmol.base.data.mmcif.write import mmcif_write_block


def _parse_block(
    tmp_path,
    fields: Sequence[str],
    data: Sequence[Sequence[str | int]],
):
    cif = f"data_test{mmcif_write_block('test', fields, data)}\n#\n"
    path = tmp_path / "write.cif"
    path.write_text(cif)

    block = next(read_cif(path))
    parsed = cif_ddl2_frame_as_dict(block.data)
    return parsed["test"]


def _parse_loop_rows(cif_text: str, block_name: str) -> list[dict[str, str]]:
    in_block = False
    field_names: list[str] = []
    rows: list[dict[str, str]] = []

    for line in cif_text.splitlines():
        line = line.strip()

        if line == "loop_":
            in_block = False
            field_names = []
            continue

        if line.startswith(f"_{block_name}."):
            in_block = True
            field_names.append(line.split(".", 1)[1])
            continue

        if (
            in_block
            and field_names
            and line
            and not line.startswith("_")
            and not line.startswith("#")
        ):
            tokens = line.split()
            if len(tokens) >= len(field_names):
                rows.append(dict(zip(field_names, tokens)))

        if line.startswith("#") and in_block:
            in_block = False

    return rows


@pytest.mark.parametrize(
    "value",
    [
        "data_block",
        "save_a",
        "loop_",
        "stop_",
        "global_",
        "#comment",
        "$frame",
        "[bracket",
        "]bracket",
        "_tag",
        ";text",
    ],
)
def test_write_block_quotes_reserved_starts(tmp_path, value: str):
    parsed = _parse_block(tmp_path, ["v"], [(value,)])
    assert parsed == [{"v": value}]


@pytest.mark.parametrize("value", [".", "?"])
def test_write_block_keeps_special_unknown_tokens_as_nulls(
    tmp_path, value: str
):
    parsed = _parse_block(tmp_path, ["v"], [(value,)])
    assert parsed == [{"v": None}]


def test_write_block_quotes_empty_string(tmp_path):
    parsed = _parse_block(tmp_path, ["v"], [("",)])
    assert parsed == [{"v": ""}]


@pytest.mark.parametrize("value", ["'start", '"start'])
def test_write_block_quotes_values_starting_with_quote(tmp_path, value: str):
    parsed = _parse_block(tmp_path, ["v"], [(value,)])
    assert parsed == [{"v": value}]


def test_write_block_chooses_safe_quote_for_apostrophe(tmp_path):
    value = "ab' cd"
    parsed = _parse_block(tmp_path, ["v"], [(value,)])
    assert parsed == [{"v": value}]


def test_write_block_uses_text_field_for_both_quotes(tmp_path):
    value = "non-polymer syn \"'5'-O-(N-(L-ALANYL)-SULFAMOYL)ADENOSINE\""
    parsed = _parse_block(tmp_path, ["v"], [(value,)])
    assert parsed == [{"v": value}]


def test_write_block_serializes_multiline_values(tmp_path):
    parsed = _parse_block(
        tmp_path,
        ["v1", "v2"],
        [("line1\nline2", "x"), ("plain", "y")],
    )
    assert parsed == [
        {"v1": "line1\nline2", "v2": "x"},
        {"v1": "plain", "v2": "y"},
    ]


def test_write_block_rejects_unencodable_multiline_values():
    with pytest.raises(ValueError, match="starts with ';'"):
        mmcif_write_block("test", ["v"], [("line1\n;line2",)])


def test_write_block_rejects_row_length_mismatch():
    with pytest.raises(ValueError, match="Row length does not match"):
        mmcif_write_block("test", ["v1", "v2"], [("only_one",)])


@pytest.mark.parametrize("value", ["loop_foo", "stop_foo", "global_foo"])
def test_write_block_allows_non_reserved_keyword_prefixes(value: str):
    block = mmcif_write_block("test", ["v"], [(value,)])
    assert f"\n{value}" in block
    assert f"\n'{value}'" not in block


@pytest.mark.parametrize("pdb_id", ["7rtb", "7qgw", "6akd"])
def test_write_block_roundtrip_real_entity_descriptions(
    test_data: Path, tmp_path, pdb_id: str
):
    mmcif = test_data / "mmcif" / f"{pdb_id}.cif"
    metadata = load_mmcif_single(mmcif)

    descriptions = [
        entity.pdbx_description for entity in metadata.entity.values()
    ]
    parsed = _parse_block(
        tmp_path,
        ["pdbx_description"],
        [(desc,) for desc in descriptions],
    )
    assert [row["pdbx_description"] for row in parsed] == descriptions


def test_write_block_quotes_leading_bracket_description_from_13sy(
    test_data: Path,
):
    mmcif = test_data / "mmcif" / "13sy.cif"

    metadata = load_mmcif_single(mmcif)
    target = next(
        desc
        for desc in (
            entity.pdbx_description for entity in metadata.entity.values()
        )
        if desc.startswith("[")
    )

    block = mmcif_write_block("entity", ["pdbx_description"], [(target,)])
    assert block.splitlines()[-1] == f"'{target}'"


def test_write_block_quotes_leading_single_quote_description_from_6akd(
    test_data: Path,
):
    mmcif = test_data / "mmcif" / "6akd.cif"
    metadata = load_mmcif_single(mmcif)
    target = next(
        desc
        for desc in (
            entity.pdbx_description for entity in metadata.entity.values()
        )
        if desc.startswith("'")
    )

    block = mmcif_write_block("entity", ["pdbx_description"], [(target,)])
    assert block.splitlines()[-1] == f'"{target}"'


@pytest.mark.parametrize("pdb_id", ["7rtb", "7qgw", "6akd"])
def test_assembly_to_mmcif_roundtrip_cif_parsing(
    test_data: Path,
    tmp_path,
    ccd_components,
    pdb_id: str,
):
    src = test_data / "mmcif" / f"{pdb_id}.cif"
    mmcif = load_mmcif_single(src)
    assemblies = mmcif_assemblies(mmcif, ccd_components)
    assert assemblies

    cif = assemblies[0].to_mmcif(pdb_id)
    out = tmp_path / f"{pdb_id}_roundtrip.cif"
    out.write_text(cif)

    block = next(read_cif(out))
    parsed = cif_ddl2_frame_as_dict(block.data)
    assert "atom_site" in parsed
    assert len(parsed["atom_site"]) > 0


def test_assembly_operation_preserves_chain_auth_asym_id_mapping(
    test_data: Path,
    ccd_components,
):
    mmcif = load_mmcif_single(test_data / "mmcif" / "4hf7.cif")
    assembly = mmcif_assemblies(mmcif, ccd_components)[0]

    auth_by_label = {
        chain.chain_id: chain.auth_asym_id
        for chain in assembly.chains.values()
        if chain.chain_id in {"A_1", "A_2"}
    }

    assert auth_by_label == {"A_1": "A", "A_2": "A"}
