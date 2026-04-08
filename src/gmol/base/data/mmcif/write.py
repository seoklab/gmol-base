"""This module is written to follow the syntax rules in the reference below.

Reference:
    CIF 1.1 syntax (general):
    https://www.iucr.org/resources/cif/spec/version1.1/cifsyntax#general
"""

from collections.abc import Sequence


def mmcif_bond_order(order: int) -> str:
    return {1: "sing", 2: "doub", 3: "trip", 4: "quad"}[order]


def mmcif_bool(pred: bool | None, lower: bool = False) -> str:
    if pred is None:
        return "?"
    val = "Y" if pred else "N"
    if lower:
        val = val.lower()
    return val


_RESERVED_PREFIXES = ("data_", "save_")
_RESERVED_EXACT = frozenset({"loop_", "stop_", "global_"})
_RESERVED_START_CHARS = frozenset("_#$[];'\"")


def _mmcif_text_field(v: str) -> str:
    normalized = v.replace("\r\n", "\n").replace("\r", "\n")
    if "\n;" in normalized:
        raise ValueError(
            "Cannot encode a CIF text field containing a line that starts with ';'"
        )
    return f";{normalized}\n;"


def _needs_quoted(v: str) -> bool:
    if not v:
        return True

    lower = v.lower()

    return (
        any(c.isspace() for c in v)
        or v[0] in _RESERVED_START_CHARS
        or lower in _RESERVED_EXACT
        or lower.startswith(_RESERVED_PREFIXES)
    )


def _mmcif_quote(v: str) -> str:
    if "\n" in v or "\r" in v:
        return _mmcif_text_field(v)

    if "'" not in v:
        return f"'{v}'"
    if '"' not in v:
        return f'"{v}"'
    return _mmcif_text_field(v)


def _mmcif_escape(v: str) -> str:
    if _needs_quoted(v):
        return _mmcif_quote(v)
    return v


def _assert_join(row: Sequence[str | int], num_fields: int) -> str:
    if len(row) != num_fields:
        raise ValueError("Row length does not match field count")

    tokens = [_mmcif_escape(str(c)) for c in row]
    if any("\n" in tok for tok in tokens):
        return "\n".join(tokens)
    return " ".join(tokens)


def mmcif_write_block(
    block: str,
    fields: Sequence[str],
    data: Sequence[Sequence[str | int]],
) -> str:
    if not fields:
        raise ValueError("Fields must not be empty")

    if not data:
        return ""

    content = """
#
loop_
""" + "\n".join(f"_{block}.{field}" for field in fields)

    content += "\n" + "\n".join(_assert_join(row, len(fields)) for row in data)

    return content
