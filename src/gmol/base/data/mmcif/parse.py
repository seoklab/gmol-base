"""
cif_parse() and related subroutines are based on the BioPython library.
Here follows the full license text from BioPython:


Copyright (C) 2002, Thomas Hamelryck (thamelry@binf.ku.dk)

Copyright (c) 1999-2024, The Biopython Contributors
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

# mypy: disallow-any-explicit=false

import datetime as dt
import logging
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from nuri.fmt import cif_ddl2_frame_as_dict, read_cif
from pydantic import (
    AliasChoices,
    AliasPath,
    Field,
    field_validator,
    model_validator,
)
from tqdm import tqdm

from gmol.base.types import LooseModel
from .write import mmcif_bond_order, mmcif_write_block

__all__ = ["ChemComp", "Mmcif", "load_components", "load_mmcif_single"]

_logger = logging.getLogger(__name__)


class Entity(LooseModel):
    id: int
    type: str
    pdbx_description: str


class ChemCompAtom(LooseModel):
    atom_id: str
    type_symbol: str
    charge: float = 0.0
    pdbx_aromatic_flag: bool = False
    pdbx_leaving_atom_flag: bool = False
    pdbx_stereo_config: str | None = None

    @field_validator("type_symbol", mode="after")
    @staticmethod
    def _correct_case(v: str) -> str:
        return v.capitalize()

    @field_validator("pdbx_stereo_config", mode="before")
    @staticmethod
    def _coerce_stereo_config(v: str | None) -> str | None:
        return None if v == "N" else v

    @field_validator("charge", mode="before")
    @staticmethod
    def _coerce_charge(v: str | None) -> str:
        return v or "0"


class ChemCompBond(LooseModel):
    atom_id_1: str
    atom_id_2: str
    value_order: int
    pdbx_aromatic_flag: bool = False
    pdbx_stereo_config: str | None = None

    @field_validator("value_order", mode="before")
    @staticmethod
    def _coerce_order(v: str | int) -> int:
        if isinstance(v, int):
            return v
        return {"sing": 1, "doub": 2, "trip": 3, "quad": 4}[v.lower()]

    @field_validator("pdbx_stereo_config", mode="before")
    @staticmethod
    def _coerce_stereo_config(v: str | None) -> str | None:
        return None if v == "N" else v


class ChemComp(LooseModel):
    id: str
    name: str
    type: str
    formula: str | None
    formula_weight: float | None
    mon_nstd_flag: bool | None = None

    atoms: list[ChemCompAtom]
    bonds: list[ChemCompBond]


class Scheme(LooseModel):
    asym_id: str
    entity_id: int
    mon_id: str

    seq_id: int = Field(
        validation_alias=AliasChoices(
            "seq_id",  # poly_seq_scheme
            "ndb_seq_num",  # nonpoly_scheme
            "num",  # branch_scheme
        )
    )

    pdb_seq_num: int | None  # this is auth_seq_id (!!!)
    pdb_ins_code: str | None = None

    def __eq__(self, other):
        if not isinstance(other, Scheme):
            return False

        return self.asym_id == other.asym_id and self.seq_id == other.seq_id

    def __lt__(self, other):
        if not isinstance(other, Scheme):
            return NotImplemented

        return (self.asym_id, self.seq_id) < (other.asym_id, other.seq_id)

    def __hash__(self):
        return hash((self.asym_id, self.seq_id))


class AtomSite(LooseModel):
    id: int
    type_symbol: str
    group_PDB: str

    label_atom_id: str
    label_alt_id: str | None
    label_comp_id: str
    label_asym_id: str
    label_seq_id: int | None

    auth_seq_id: int
    auth_comp_id: str
    auth_asym_id: str

    pdbx_PDB_ins_code: str | None
    pdbx_PDB_model_num: int

    cartn: NDArray[np.float64]
    occupancy: float

    @property
    def is_hydrogen(self):
        return self.type_symbol == "H"

    @model_validator(mode="before")
    @staticmethod
    def _gather_cartn(v: dict[str, Any]):
        v["cartn"] = np.array(
            [v["Cartn_x"], v["Cartn_y"], v["Cartn_z"]], dtype=np.float64
        )
        return v


class BioAssemblyGen(LooseModel):
    asym_id_list: list[str]
    operations: list[list[str]]

    @field_validator("asym_id_list", mode="before")
    @staticmethod
    def _parse_asym_ids(v: str):
        return v.split(",")

    @model_validator(mode="before")
    @staticmethod
    def _parse_operations(v: dict[str, str]):
        op_exprs = v["oper_expression"]

        if "(" not in op_exprs:
            v["operations"] = [  # type: ignore[assignment]
                _parse_expr(op_exprs)
            ]
            return v

        v["operations"] = [  # type: ignore[assignment]
            _parse_expr(ops)
            for token in op_exprs.split("(")
            if (ops := token.strip(")"))
        ]

        return v


def _parse_single_range(op: str) -> Iterable[str]:
    if "-" not in op:
        return [op]

    start, end = map(int, op.split("-"))
    return map(str, range(start, end + 1))


def _parse_expr(ops: str) -> list[str]:
    return [
        elem for expr in ops.split(",") for elem in _parse_single_range(expr)
    ]


class BioAssembly(LooseModel):
    id: int
    details: str
    oligomeric_details: str
    oligomeric_count: int

    assembly_gens: list[BioAssemblyGen]

    # XXX: 1hya has missing oligomeric_* fields. Assume monomeric for now.

    @field_validator("oligomeric_details", mode="before")
    @staticmethod
    def _parse_ol_detail(v: str | None):
        if v is None:
            return "monomeric (!MISSING!)"
        return v

    @field_validator("oligomeric_count", mode="before")
    @staticmethod
    def _parse_ol_cnt(v: str | None):
        if v is None:
            return 1
        return v


class SymOp(LooseModel):
    type: str
    name: str | None
    symmetry_operation: str | None

    matrix: NDArray[np.float64]
    vector: NDArray[np.float64]

    @model_validator(mode="before")
    @staticmethod
    def _gather_matrix_vector(v: dict[str, Any]):
        v["matrix"] = np.array(
            [
                [v[f"matrix[{i}][{j}]"] for j in range(1, 4)]
                for i in range(1, 4)
            ],
            dtype=np.float64,
        )
        v["vector"] = np.array(
            [v[f"vector[{i}]"] for i in range(1, 4)], dtype=np.float64
        )
        return v


class StructConnPartner(LooseModel):
    label_atom_id: str
    label_comp_id: str
    label_asym_id: str
    label_seq_id: int | None

    auth_seq_id: int
    auth_comp_id: str
    auth_asym_id: str

    pdbx_PDB_ins_code: str | None

    symmetry: str


class StructConn(LooseModel):
    id: str
    conn_type_id: str

    pdbx_leaving_atom_flag: int = Field(ge=0, le=2)
    pdbx_dist_value: float | None

    ptnr1: StructConnPartner
    ptnr2: StructConnPartner

    @model_validator(mode="before")
    @staticmethod
    def _gather_ptnrs(d: dict[str, Any]):
        for ptnr in ("ptnr1", "ptnr2"):
            d[ptnr] = {
                k.replace(f"{ptnr}_", ""): v
                for k, v in d.items()
                if f"{ptnr}_" in k
            }
        return d

    @field_validator("pdbx_leaving_atom_flag", mode="before")
    @staticmethod
    def _coerce_leaving_atom(v):
        return {"one": 1, "both": 2}.get(v, 0)


class StructConnType(LooseModel):
    criteria: str | None
    reference: str | None


class BranchLinkPartner(LooseModel):
    entity_branch_list_num: int
    comp_id: str
    atom_id: str
    leaving_atom_id: str


class BranchLink(LooseModel):
    value_order: int

    ptnr1: BranchLinkPartner
    ptnr2: BranchLinkPartner

    @model_validator(mode="before")
    @staticmethod
    def _gather_ptnrs(d: dict[str, Any]):
        for idx in range(1, 3):
            key = f"ptnr{idx}"
            suffix = f"_{idx}"
            d[key] = {
                k[: -len(suffix)]: v
                for k, v in d.items()
                if k.endswith(suffix)
            }
        return d

    @field_validator("value_order", mode="before")
    @staticmethod
    def _coerce_order(v: str) -> int:
        return {"sing": 1, "doub": 2, "trip": 3, "quad": 4}[v.lower()]


def _join_chem_comp(
    v: dict[str, Any],
) -> dict[str, dict[str, list[dict[str, str | None]]]]:
    chem_comps: dict[str, dict[str, list[dict[str, str | None]]]] = {
        cc["id"]: cc for cc in v.get("chem_comp", [])
    }
    for cc in chem_comps.values():
        cc["atoms"] = []
        cc["bonds"] = []

    for atom in v.get("chem_comp_atom", []):
        chem_comps[atom["comp_id"]]["atoms"].append(atom)

    for bond in v.get("chem_comp_bond", []):
        chem_comps[bond["comp_id"]]["bonds"].append(bond)

    return chem_comps


def _filter_assembly_gens(
    gens: list[BioAssemblyGen],
    kept_asym_ids: set[str],
) -> list[BioAssemblyGen]:
    filtered = []
    for g in gens:
        kept_ids = [aid for aid in g.asym_id_list if aid in kept_asym_ids]
        if not kept_ids:
            continue
        if len(g.operations) == 1:
            oper_expr = ",".join(g.operations[0])
        else:
            oper_expr = "".join(
                "(" + ",".join(ops) + ")" for ops in g.operations
            )
        filtered.append(
            BioAssemblyGen.model_validate(
                {
                    "asym_id_list": ",".join(kept_ids),
                    "oper_expression": oper_expr,
                }
            )
        )
    return filtered


class Mmcif(LooseModel):
    entry_id: str = Field(validation_alias=AliasPath("entry", 0, "id"))
    exptl_method: str = Field(validation_alias=AliasPath("exptl", 0, "method"))
    pdbx_keywords: str = Field(
        validation_alias=AliasPath("struct_keywords", 0, "pdbx_keywords")
    )

    revision_date: dt.date = Field(
        validation_alias=AliasPath(
            "pdbx_audit_revision_history", "revision_date"
        )
    )

    resolution: float = Field(default=999.9)

    entity: dict[int, Entity] = Field(default_factory=dict)

    pdbx_poly_seq_scheme: dict[str, list[Scheme]] = Field(default_factory=dict)
    pdbx_branch_scheme: dict[str, list[Scheme]] = Field(default_factory=dict)
    pdbx_nonpoly_scheme: dict[str, list[Scheme]] = Field(default_factory=dict)

    atom_site: list[AtomSite]

    pdbx_struct_assembly: list[BioAssembly]
    pdbx_struct_oper_list: dict[str, SymOp]
    struct_asym: dict[str, int]

    struct_conn: list[StructConn] = Field(default_factory=list)
    struct_conn_type: dict[str, StructConnType] = Field(default_factory=dict)

    pdbx_entity_branch_link: dict[int, list[BranchLink]] = Field(
        default_factory=dict
    )

    @field_validator("pdbx_keywords", mode="before")
    @staticmethod
    def _coerce_pdbx_kwds(v: Any) -> Any:
        if v is None:
            return ""
        return v

    @model_validator(mode="before")
    @staticmethod
    def _find_oldest(v: dict[str, list[dict[str, Any]]]):
        min_rev = min(
            (rev for rev in v.get("pdbx_audit_revision_history", [])),
            key=lambda r: int(r["ordinal"]),
        )
        v["pdbx_audit_revision_history"] = min_rev  # type: ignore[assignment]
        return v

    @model_validator(mode="before")
    @staticmethod
    def _join_bioassembly(v: dict[str, list[dict[str, Any]]]):
        bas = {ba["id"]: ba for ba in v.get("pdbx_struct_assembly", [])}
        for ba in bas.values():
            ba["assembly_gens"] = []

        for bag in v.get("pdbx_struct_assembly_gen", []):
            bas[bag["assembly_id"]]["assembly_gens"].append(bag)

        return v

    @model_validator(mode="before")
    @staticmethod
    def _find_resolution(v: dict[str, list[dict[str, Any]]]):
        for k1, k2 in (
            ("refine", "ls_d_res_high"),
            ("em_3d_reconstruction", "resolution"),
            ("reflns", "d_resolution_high"),
        ):
            try:
                res = v.get(k1, [])[0][k2]
            except (KeyError, IndexError):
                continue

            if res is not None:
                v["resolution"] = res
                break

        return v

    @model_validator(mode="before")
    @staticmethod
    def _join_branch_link(v: dict[str, list[dict[str, Any]]]):
        branch_links = defaultdict(list)
        for bl in v.get("pdbx_entity_branch_link", []):
            branch_links[bl["entity_id"]].append(bl)

        v["pdbx_entity_branch_link"] = branch_links  # type: ignore[assignment]

        return v

    @field_validator(
        "entity",
        "pdbx_struct_oper_list",
        "struct_conn_type",
        mode="before",
    )
    @staticmethod
    def _list_as_dict(v: list[dict[str, Any]]):
        return {d["id"]: d for d in v}

    @field_validator(
        "pdbx_poly_seq_scheme",
        "pdbx_branch_scheme",
        "pdbx_nonpoly_scheme",
        mode="before",
    )
    @staticmethod
    def _xform_schemes(v: list[dict[str, Any]]):
        ret = defaultdict(list)
        for d in v:
            ret[d["asym_id"]].append(d)
        return ret

    @field_validator("struct_asym", mode="before")
    @staticmethod
    def _xform_struct_asym(v: list[dict[str, Any]]):
        ret = {}
        for d in v:
            ret[d["id"]] = d["entity_id"]
        return ret

    def filter_chains(self, chain_ids: list[str]) -> "Mmcif":
        selected = np.array(
            [site.label_asym_id in chain_ids for site in self.atom_site],
            dtype=np.bool_,
        )
        return self.filter(selected)

    def filter(self, selected: NDArray[np.bool_]) -> "Mmcif":
        if selected.shape != (len(self.atom_site),):
            raise ValueError(
                f"selected must have shape (len(atom_site),), got {selected.shape}"
            )
        new_sites = []
        for i, a in enumerate(self.atom_site):
            if not selected[i]:
                continue

            raw = {
                "id": i + 1,
                "type_symbol": a.type_symbol,
                "group_PDB": a.group_PDB,
                "label_atom_id": a.label_atom_id,
                "label_alt_id": a.label_alt_id,
                "label_comp_id": a.label_comp_id,
                "label_asym_id": a.label_asym_id,
                "label_seq_id": a.label_seq_id,
                "auth_seq_id": a.auth_seq_id,
                "auth_comp_id": a.auth_comp_id,
                "auth_asym_id": a.auth_asym_id,
                "pdbx_PDB_ins_code": a.pdbx_PDB_ins_code,
                "pdbx_PDB_model_num": a.pdbx_PDB_model_num,
                "Cartn_x": float(a.cartn[0]),
                "Cartn_y": float(a.cartn[1]),
                "Cartn_z": float(a.cartn[2]),
                "occupancy": a.occupancy,
            }
            new_sites.append(AtomSite.model_validate(raw))
        kept_asym_ids = set(s.label_asym_id for s in new_sites)
        kept_atom_keys = set(
            (s.label_asym_id, s.label_seq_id, s.label_atom_id)
            for s in new_sites
        )

        new_struct_asym = {
            aid: eid
            for aid, eid in self.struct_asym.items()
            if aid in kept_asym_ids
        }
        kept_entity_ids = frozenset(new_struct_asym.values())
        new_entity = {
            eid: ent
            for eid, ent in self.entity.items()
            if eid in kept_entity_ids
        }

        new_poly = {
            aid: self.pdbx_poly_seq_scheme[aid]
            for aid in self.pdbx_poly_seq_scheme
            if aid in kept_asym_ids
        }
        new_branch = {
            aid: self.pdbx_branch_scheme[aid]
            for aid in self.pdbx_branch_scheme
            if aid in kept_asym_ids
        }
        new_nonpoly = {
            aid: self.pdbx_nonpoly_scheme[aid]
            for aid in self.pdbx_nonpoly_scheme
            if aid in kept_asym_ids
        }

        def conn_partner_in_kept(ptnr: StructConnPartner) -> bool:
            return (
                ptnr.label_asym_id in kept_asym_ids
                and (ptnr.label_asym_id, ptnr.label_seq_id, ptnr.label_atom_id)
                in kept_atom_keys
            )

        new_conn = [
            c
            for c in self.struct_conn
            if conn_partner_in_kept(c.ptnr1) and conn_partner_in_kept(c.ptnr2)
        ]
        kept_conn_type_ids = frozenset(c.conn_type_id for c in new_conn)
        new_conn_type = {
            cid: ct
            for cid, ct in self.struct_conn_type.items()
            if cid in kept_conn_type_ids
        }

        new_assemblies = []
        for ba in self.pdbx_struct_assembly:
            new_gens = _filter_assembly_gens(ba.assembly_gens, kept_asym_ids)
            if not new_gens:
                continue
            new_assemblies.append(
                BioAssembly(
                    id=ba.id,
                    details=ba.details,
                    oligomeric_details=ba.oligomeric_details,
                    oligomeric_count=ba.oligomeric_count,
                    assembly_gens=new_gens,
                )
            )

        new_branch_link = {
            eid: links
            for eid, links in self.pdbx_entity_branch_link.items()
            if eid in kept_entity_ids
        }

        return Mmcif.model_construct(
            entry_id=self.entry_id,
            exptl_method=self.exptl_method,
            pdbx_keywords=self.pdbx_keywords,
            revision_date=self.revision_date,
            resolution=self.resolution,
            entity=new_entity,
            pdbx_poly_seq_scheme=new_poly,
            pdbx_branch_scheme=new_branch,
            pdbx_nonpoly_scheme=new_nonpoly,
            atom_site=new_sites,
            pdbx_struct_assembly=new_assemblies,
            pdbx_struct_oper_list=self.pdbx_struct_oper_list,
            struct_asym=new_struct_asym,
            struct_conn=new_conn,
            struct_conn_type=new_conn_type,
            pdbx_entity_branch_link=new_branch_link,
        )

    def to_mmcif(self) -> str:
        parts = [f"data_{self.entry_id}"]

        parts.append(
            mmcif_write_block(
                "entry",
                ["id"],
                [(self.entry_id,)],
            )
        )
        parts.append(
            mmcif_write_block(
                "exptl",
                ["method"],
                [(self.exptl_method,)],
            )
        )
        parts.append(
            mmcif_write_block(
                "struct_keywords",
                ["pdbx_keywords"],
                [(self.pdbx_keywords,)],
            )
        )
        parts.append(
            mmcif_write_block(
                "pdbx_audit_revision_history",
                ["ordinal", "revision_date"],
                [(1, self.revision_date.isoformat())],
            )
        )

        parts.append(
            mmcif_write_block(
                "refine",
                ["ls_d_res_high"],
                [(f"{self.resolution:.2f}",)],
            )
        )

        if self.entity:
            parts.append(
                mmcif_write_block(
                    "entity",
                    ["id", "type", "pdbx_description"],
                    [
                        (e.id, e.type, e.pdbx_description)
                        for e in self.entity.values()
                    ],
                )
            )

        if self.struct_asym:
            parts.append(
                mmcif_write_block(
                    "struct_asym",
                    ["id", "entity_id"],
                    [(aid, eid) for aid, eid in self.struct_asym.items()],
                )
            )

        def _site_row(s: AtomSite):
            return (
                s.id,
                s.type_symbol,
                s.group_PDB,
                s.label_atom_id,
                s.label_alt_id or ".",
                s.label_comp_id,
                s.label_asym_id,
                s.label_seq_id if s.label_seq_id is not None else ".",
                s.auth_seq_id,
                s.auth_comp_id,
                s.auth_asym_id,
                s.pdbx_PDB_ins_code or ".",
                s.pdbx_PDB_model_num,
                f"{s.cartn[0]:.3f}",
                f"{s.cartn[1]:.3f}",
                f"{s.cartn[2]:.3f}",
                f"{s.occupancy:.2f}",
            )

        parts.append(
            mmcif_write_block(
                "atom_site",
                [
                    "id",
                    "type_symbol",
                    "group_PDB",
                    "label_atom_id",
                    "label_alt_id",
                    "label_comp_id",
                    "label_asym_id",
                    "label_seq_id",
                    "auth_seq_id",
                    "auth_comp_id",
                    "auth_asym_id",
                    "pdbx_PDB_ins_code",
                    "pdbx_PDB_model_num",
                    "Cartn_x",
                    "Cartn_y",
                    "Cartn_z",
                    "occupancy",
                ],
                [_site_row(s) for s in self.atom_site],
            )
        )

        def _scheme_row(s: Scheme):
            return (
                s.asym_id,
                s.entity_id,
                s.mon_id,
                s.seq_id,
                s.pdb_seq_num if s.pdb_seq_num is not None else ".",
                s.pdb_ins_code or ".",
            )

        def _scheme_fields(seq_key: str) -> list[str]:
            return [
                "asym_id",
                "entity_id",
                "mon_id",
                seq_key,
                "pdb_seq_num",
                "pdb_ins_code",
            ]

        if self.pdbx_poly_seq_scheme:
            rows = [
                _scheme_row(s)
                for aid in self.pdbx_poly_seq_scheme
                for s in self.pdbx_poly_seq_scheme[aid]
            ]
            parts.append(
                mmcif_write_block(
                    "pdbx_poly_seq_scheme",
                    _scheme_fields("seq_id"),
                    rows,
                )
            )
        if self.pdbx_branch_scheme:
            rows = [
                _scheme_row(s)
                for aid in self.pdbx_branch_scheme
                for s in self.pdbx_branch_scheme[aid]
            ]
            parts.append(
                mmcif_write_block(
                    "pdbx_branch_scheme",
                    _scheme_fields("num"),
                    rows,
                )
            )
        if self.pdbx_nonpoly_scheme:
            rows = [
                _scheme_row(s)
                for aid in self.pdbx_nonpoly_scheme
                for s in self.pdbx_nonpoly_scheme[aid]
            ]
            parts.append(
                mmcif_write_block(
                    "pdbx_nonpoly_scheme",
                    _scheme_fields("ndb_seq_num"),
                    rows,
                )
            )

        if self.pdbx_struct_assembly:
            parts.append(
                mmcif_write_block(
                    "pdbx_struct_assembly",
                    [
                        "id",
                        "details",
                        "oligomeric_details",
                        "oligomeric_count",
                    ],
                    [
                        (
                            ba.id,
                            ba.details,
                            ba.oligomeric_details,
                            ba.oligomeric_count,
                        )
                        for ba in self.pdbx_struct_assembly
                    ],
                )
            )
            gen_rows = []
            for ba in self.pdbx_struct_assembly:
                for g in ba.assembly_gens:
                    if len(g.operations) == 1:
                        oper_expr = ",".join(g.operations[0])
                    else:
                        oper_expr = "".join(
                            "(" + ",".join(ops) + ")" for ops in g.operations
                        )
                    gen_rows.append(
                        (ba.id, ",".join(g.asym_id_list), oper_expr)
                    )
            if gen_rows:
                parts.append(
                    mmcif_write_block(
                        "pdbx_struct_assembly_gen",
                        ["assembly_id", "asym_id_list", "oper_expression"],
                        gen_rows,
                    )
                )

        if self.pdbx_struct_oper_list:
            op_rows = []
            for op_id, op in self.pdbx_struct_oper_list.items():
                row = [
                    op_id,
                    op.type,
                    op.name or ".",
                    op.symmetry_operation or ".",
                ]
                for i in range(1, 4):
                    for j in range(1, 4):
                        row.append(f"{op.matrix[i - 1, j - 1]:.5f}")
                for i in range(1, 4):
                    row.append(f"{op.vector[i - 1]:.5f}")
                op_rows.append(tuple(row))
            parts.append(
                mmcif_write_block(
                    "pdbx_struct_oper_list",
                    [
                        "id",
                        "type",
                        "name",
                        "symmetry_operation",
                        "matrix[1][1]",
                        "matrix[1][2]",
                        "matrix[1][3]",
                        "matrix[2][1]",
                        "matrix[2][2]",
                        "matrix[2][3]",
                        "matrix[3][1]",
                        "matrix[3][2]",
                        "matrix[3][3]",
                        "vector[1]",
                        "vector[2]",
                        "vector[3]",
                    ],
                    op_rows,
                )
            )

        if self.struct_conn:
            leaving_str = {0: "none", 1: "one", 2: "both"}
            conn_rows: list[tuple[str | int, ...]] = []
            for c in self.struct_conn:
                p1, p2 = c.ptnr1, c.ptnr2
                dist_str = (
                    f"{c.pdbx_dist_value:.3f}"
                    if c.pdbx_dist_value is not None
                    else "."
                )
                conn_row: tuple[str | int, ...] = (
                    c.id,
                    c.conn_type_id,
                    leaving_str.get(c.pdbx_leaving_atom_flag, "none"),
                    dist_str,
                    p1.label_atom_id,
                    p1.label_comp_id,
                    p1.label_asym_id,
                    p1.label_seq_id if p1.label_seq_id is not None else ".",
                    p1.auth_seq_id,
                    p1.auth_comp_id,
                    p1.auth_asym_id,
                    p1.pdbx_PDB_ins_code or ".",
                    p1.symmetry,
                    p2.label_atom_id,
                    p2.label_comp_id,
                    p2.label_asym_id,
                    p2.label_seq_id if p2.label_seq_id is not None else ".",
                    p2.auth_seq_id,
                    p2.auth_comp_id,
                    p2.auth_asym_id,
                    p2.pdbx_PDB_ins_code or ".",
                    p2.symmetry,
                )
                conn_rows.append(conn_row)
            parts.append(
                mmcif_write_block(
                    "struct_conn",
                    [
                        "id",
                        "conn_type_id",
                        "pdbx_leaving_atom_flag",
                        "pdbx_dist_value",
                        "ptnr1_label_atom_id",
                        "ptnr1_label_comp_id",
                        "ptnr1_label_asym_id",
                        "ptnr1_label_seq_id",
                        "ptnr1_auth_seq_id",
                        "ptnr1_auth_comp_id",
                        "ptnr1_auth_asym_id",
                        "pdbx_ptnr1_PDB_ins_code",
                        "ptnr1_symmetry",
                        "ptnr2_label_atom_id",
                        "ptnr2_label_comp_id",
                        "ptnr2_label_asym_id",
                        "ptnr2_label_seq_id",
                        "ptnr2_auth_seq_id",
                        "ptnr2_auth_comp_id",
                        "ptnr2_auth_asym_id",
                        "pdbx_ptnr2_PDB_ins_code",
                        "ptnr2_symmetry",
                    ],
                    conn_rows,
                )
            )

        if self.struct_conn_type:
            parts.append(
                mmcif_write_block(
                    "struct_conn_type",
                    ["id", "criteria", "reference"],
                    [
                        (
                            cid,
                            ct.criteria or ".",
                            ct.reference or ".",
                        )
                        for cid, ct in self.struct_conn_type.items()
                    ],
                )
            )

        if self.pdbx_entity_branch_link:
            bl_rows = []
            for entity_id, links in self.pdbx_entity_branch_link.items():
                for bl in links:
                    bl_rows.append(
                        (
                            entity_id,
                            bl.ptnr1.entity_branch_list_num,
                            bl.ptnr1.comp_id,
                            bl.ptnr1.atom_id,
                            bl.ptnr1.leaving_atom_id,
                            bl.ptnr2.entity_branch_list_num,
                            bl.ptnr2.comp_id,
                            bl.ptnr2.atom_id,
                            bl.ptnr2.leaving_atom_id,
                            mmcif_bond_order(bl.value_order),
                        )
                    )
            parts.append(
                mmcif_write_block(
                    "pdbx_entity_branch_link",
                    [
                        "entity_id",
                        "entity_branch_list_num_1",
                        "comp_id_1",
                        "atom_id_1",
                        "leaving_atom_id_1",
                        "entity_branch_list_num_2",
                        "comp_id_2",
                        "atom_id_2",
                        "leaving_atom_id_2",
                        "value_order",
                    ],
                    bl_rows,
                )
            )

        return "\n".join(parts)


def load_mmcif_single(file: Path):
    data = next(read_cif(file)).data
    mmcif = Mmcif.model_validate(cif_ddl2_frame_as_dict(data))
    return mmcif


def load_components(file: Path, max_count: int = 0):
    components: dict[str, ChemComp] = {}

    for i, block in tqdm(enumerate(read_cif(file))):
        data = block.data
        name = block.name

        try:
            mmcif_dict = cif_ddl2_frame_as_dict(data)
            chem_comp = _join_chem_comp(mmcif_dict)

            assert len(chem_comp) == 1
            components[name] = ChemComp.model_validate(chem_comp[name])
        except Exception:
            _logger.exception("Failed to load component %s", name)

        if max_count and i >= max_count - 1:
            break

    return components
