"""Evaluation helpers"""
from __future__ import annotations


def not_implemented() -> dict:
    return {"ok": False, "status": "deferred", "module": "evaluation"}
