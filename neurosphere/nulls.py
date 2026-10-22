"""The null + reason convention.

A value that cannot be computed is stored as None, and its dict must also hold a sibling key
"<key>_reason" with a non-empty string. Lists never contain None.
"""
from __future__ import annotations

from typing import Any


def set_null(d: dict, key: str, reason: str) -> dict:
    if not reason or not isinstance(reason, str):
        raise ValueError("a null value needs a non-empty reason")
    d[key] = None
    d[f"{key}_reason"] = reason
    return d


def unexplained_nulls(obj: Any, path: str = "$") -> list[str]:
    """JSON paths of None values that lack a '<key>_reason' sibling (or sit inside a list)."""
    bad = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            p = f"{path}.{k}"
            if v is None:
                r = obj.get(f"{k}_reason")
                if not (isinstance(r, str) and r.strip()):
                    bad.append(p)
            else:
                bad.extend(unexplained_nulls(v, p))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            if v is None:
                bad.append(f"{path}[{i}]")
            else:
                bad.extend(unexplained_nulls(v, f"{path}[{i}]"))
    return bad


def assert_explained(obj: Any) -> None:
    bad = unexplained_nulls(obj)
    if bad:
        raise ValueError(f"null values without a reason: {bad[:10]}")
