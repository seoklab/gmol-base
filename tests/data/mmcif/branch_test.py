from pathlib import Path

from pydantic import TypeAdapter

from gmol.base.data.mmcif import (
    ChemComp,
    load_mmcif_single,
    mmcif_assemblies,
)


def test_load_heterogeneous_half_occupancy_branch(
    test_data: Path,
    ccd_components: dict[str, ChemComp],
):
    data = load_mmcif_single(test_data / "mmcif" / "4mb4.cif")

    ccd = ccd_components.copy()
    ccd.update(
        TypeAdapter(dict[str, ChemComp]).validate_json(
            (test_data / "ccd" / "components_4mb4.json").read_bytes()
        )
    )

    assemblies = mmcif_assemblies(data, ccd)
    asm = assemblies[0]

    chain_b = asm.chains["B"]

    # NDG discarded by altloc selection logic. Should not overwrite NAG
    assert len(chain_b.branches) == 3
    for branch in chain_b.branches:
        assert branch.ptnr1.comp_id == "NAG"
        assert branch.ptnr2.comp_id == "NAG"

    mmcif_out = asm.to_mmcif("4MB4")
    in_branch_scheme = False
    branch_scheme_lines: list[str] = []

    for line in mmcif_out.splitlines():
        stripped = line.strip()
        if stripped.startswith("_pdbx_branch_scheme."):
            in_branch_scheme = True
            continue
        if (
            in_branch_scheme
            and stripped
            and not stripped.startswith(("_", "#"))
        ):
            branch_scheme_lines.append(stripped)
        elif in_branch_scheme and (stripped.startswith("#") or stripped == ""):
            if branch_scheme_lines:
                break

    assert len(branch_scheme_lines) == 4
    for bsl in branch_scheme_lines:
        tokens = bsl.split()
        mon_id = tokens[3]
        assert mon_id == "NAG"
