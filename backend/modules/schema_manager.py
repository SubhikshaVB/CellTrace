"""Schema / result management"""
from __future__ import annotations


def not_implemented() -> dict:
    return {"ok": False, "status": "deferred", "module": "schema_manager"}
