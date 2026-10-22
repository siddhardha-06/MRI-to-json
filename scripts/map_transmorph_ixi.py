r"""TransMorph subject_k <-> IXI ID mapping, step 1: demographic matching (Phase 2).

Inputs (data/raw/, not committed):
  IXI_processed.csv  "Demographic information" linked from github.com/junyuchen245/Preprocessed_IXI_Dataset;
                     one row per subject_k with the IXI.xls columns but the ID replaced by k.
  IXI.xls            official IXI demographics (brain-development.org).

A CSV row is matched to IXI.xls rows on every shared field (sex, height, weight, ethnicity,
marital status, occupation, qualification, DOB, study date, age to 1e-6 years). A subject_k is
assigned only if all its CSV rows point to exactly one IXI ID that also has a raw T1 in ixi-t1/.
Everything else is left unassigned with a reason; step 2 (image matching) verifies a sample and
resolves the rest. Writes work/mapping/demographic_match.csv and a summary JSON.
"""
import csv
import datetime as dt
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import xlrd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from neurosphere.provenance import provenance, sha256_file  # noqa: E402

RAW = ROOT / "data" / "raw"
OUT = ROOT / "work" / "mapping"
FIELDS = ["SEX", "HEIGHT", "WEIGHT", "ETHNIC_ID", "MARITAL_ID", "OCCUPATION_ID", "QUALIFICATION_ID", "DOB", "STUDY_DATE", "AGE"]


def _num(v):
    if v in ("", None):
        return None
    return round(float(v), 6)


def _date_csv(s):
    s = (s or "").split(" ")[0]
    if not s:
        return None
    m, d, y = (int(x) for x in s.split("/"))
    return dt.date(y, m, d)


def _date_xls(v, datemode):
    if v in ("", None):
        return None
    if isinstance(v, str):
        return dt.date.fromisoformat(v)
    return xlrd.xldate_as_datetime(v, datemode).date()


def load_xls():
    book = xlrd.open_workbook(str(RAW / "IXI.xls"))
    sh = book.sheet_by_name("Table")
    head = sh.row_values(0)
    rows = []
    for i in range(1, sh.nrows):
        r = dict(zip(head, sh.row_values(i)))
        rows.append({"IXI_ID": int(r["IXI_ID"]), "SEX": _num(r["SEX_ID (1=m, 2=f)"]), "HEIGHT": _num(r["HEIGHT"]),
                     "WEIGHT": _num(r["WEIGHT"]), "ETHNIC_ID": _num(r["ETHNIC_ID"]), "MARITAL_ID": _num(r["MARITAL_ID"]),
                     "OCCUPATION_ID": _num(r["OCCUPATION_ID"]), "QUALIFICATION_ID": _num(r["QUALIFICATION_ID"]),
                     "DOB": _date_xls(r["DOB"], book.datemode), "STUDY_DATE": _date_xls(r["STUDY_DATE"], book.datemode),
                     "AGE": _num(r["AGE"])})
    return rows


def load_csv():
    rows = []
    with open(RAW / "IXI_processed.csv", newline="") as fh:
        for r in csv.DictReader(fh):
            rows.append({"k": int(r["Subject_ID"]), "SEX": _num(r["SEX_ID (1=m, 2=f)"]), "HEIGHT": _num(r["HEIGHT"]),
                         "WEIGHT": _num(r["WEIGHT"]), "ETHNIC_ID": _num(r["ETHNIC_ID"]), "MARITAL_ID": _num(r["MARITAL_ID"]),
                         "OCCUPATION_ID": _num(r["OCCUPATION_ID"]), "QUALIFICATION_ID": _num(r["QUALIFICATION_ID"]),
                         "DOB": _date_csv(r["DOB"]), "STUDY_DATE": _date_csv(r["STUDY_DATE"]), "AGE": _num(r["AGE"])})
    return rows


def main():
    xls, rows = load_xls(), load_csv()
    raw_ids = {int(m.group(1)) for d in (ROOT / "ixi-t1").iterdir() if (m := re.match(r"IXI(\d{3})-", d.name))}
    pkl = {int(re.search(r"subject_(\d+)", p.name).group(1)): p.parent.name for p in (ROOT / "IXI_data").glob("*/subject_*.pkl")}
    index = defaultdict(set)
    for x in xls:
        index[tuple(x[f] for f in FIELDS)].add(x["IXI_ID"])
    cand = defaultdict(list)
    for r in rows:
        cand[r["k"]].append(index.get(tuple(r[f] for f in FIELDS), set()))
    out = []
    for k in sorted(pkl):
        sets = cand.get(k)
        rec = {"k": k, "split": pkl[k], "ixi_id": "", "method": "", "reason": ""}
        if not sets:
            rec["reason"] = "subject_k has no row in IXI_processed.csv"
        else:
            ids = set.intersection(*sets) if all(sets) else set()
            ids &= raw_ids
            if len(ids) == 1:
                rec.update(ixi_id=f"IXI{next(iter(ids)):03d}", method="demographics_exact")
            elif not ids:
                rec["reason"] = "no IXI.xls row with identical demographics and a raw T1"
            else:
                rec["reason"] = "ambiguous: " + ",".join(f"IXI{i:03d}" for i in sorted(ids))
        out.append(rec)
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "demographic_match.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
    assigned = [r for r in out if r["ixi_id"]]
    dup = [i for i in {r["ixi_id"] for r in assigned} if sum(r["ixi_id"] == i for r in assigned) > 1]
    ks = [r["k"] for r in assigned]
    ids_in_k_order = [int(r["ixi_id"][3:]) for r in assigned]
    monotone = all(a < b for a, b in zip(ids_in_k_order, ids_in_k_order[1:]))
    summ = {"n_pkl": len(pkl), "assigned": len(assigned), "unassigned": len(out) - len(assigned),
            "ixi_ids_assigned_twice": sorted(dup), "ixi_id_increases_with_k": monotone,
            "unassigned_reasons": {r["reason"]: sum(x["reason"] == r["reason"] for x in out) for r in out if not r["ixi_id"]},
            "raw_t1_subjects": len(raw_ids),
            "inputs": {"IXI.xls": sha256_file(RAW / "IXI.xls"), "IXI_processed.csv": sha256_file(RAW / "IXI_processed.csv")},
            "provenance": provenance()}
    (OUT / "demographic_match.json").write_text(json.dumps(summ, indent=2, default=str))
    print(json.dumps({k: v for k, v in summ.items() if k != "provenance"}, indent=2, default=str))


if __name__ == "__main__":
    main()
