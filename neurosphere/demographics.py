"""IXI demographics from IXI.xls (age, sex) plus site from the file name (Phase 3).

Missing or inconsistent entries are null + reason, never imputed:
  * subject not in IXI.xls                        -> age, sex null
  * all rows of the subject lack AGE              -> age null
  * duplicated rows disagree on AGE (> 0.01 y)    -> age null;  disagree on SEX -> sex null
"""
from __future__ import annotations

import datetime as dt
from collections import defaultdict
from pathlib import Path

import xlrd

from .nulls import set_null

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_XLS = ROOT / "data" / "raw" / "IXI.xls"


def _rows(xls_path: Path):
    book = xlrd.open_workbook(str(xls_path))
    sh = book.sheet_by_name("Table")
    head = sh.row_values(0)
    for i in range(1, sh.nrows):
        r = dict(zip(head, sh.row_values(i)))
        yield {"id": int(r["IXI_ID"]), "sex": None if r["SEX_ID (1=m, 2=f)"] in ("", None) else int(r["SEX_ID (1=m, 2=f)"]),
               "age": None if r["AGE"] in ("", None) else float(r["AGE"])}


def load_demographics(subject_ids, xls_path: Path = DEFAULT_XLS) -> dict[str, dict]:
    by_id = defaultdict(list)
    for r in _rows(xls_path):
        by_id[r["id"]].append(r)
    out = {}
    for sid in subject_ids:
        num = int(sid[3:])
        d: dict = {}
        rows = by_id.get(num, [])
        if not rows:
            set_null(d, "age", "subject not listed in IXI.xls")
            set_null(d, "sex", "subject not listed in IXI.xls")
            out[sid] = d
            continue
        ages = sorted({r["age"] for r in rows if r["age"] is not None})
        sexes = sorted({r["sex"] for r in rows if r["sex"] is not None})
        if not ages:
            set_null(d, "age", "AGE missing in IXI.xls")
        elif max(ages) - min(ages) > 0.01:
            set_null(d, "age", f"duplicated IXI.xls rows disagree on AGE ({', '.join(f'{a:.2f}' for a in ages)})")
        else:
            d["age"] = round(ages[0], 4)
        if not sexes:
            set_null(d, "sex", "SEX missing in IXI.xls")
        elif len(sexes) > 1:
            set_null(d, "sex", "duplicated IXI.xls rows disagree on SEX")
        else:
            d["sex"] = {1: "M", 2: "F"}.get(sexes[0])
            if d["sex"] is None:
                set_null(d, "sex", f"unknown SEX code {sexes[0]}")
        out[sid] = d
    return out
