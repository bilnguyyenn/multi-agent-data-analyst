"""Read-only checks for a no-cost local-model run; never starts/stops other services."""

from __future__ import annotations

import httpx


def local_preflight(
    base_url: str = "http://localhost:11434", minimum_available_gib: float = 2.5
) -> dict:
    import psutil

    available = psutil.virtual_memory().available / 2**30
    result = {
        "available_gib": round(available, 3),
        "minimum_available_gib": minimum_available_gib,
        "cost_ceiling_usd": 0,
        "allowed": False,
        "reasons": [],
    }
    try:
        loaded = httpx.get(base_url + "/api/ps", timeout=3).raise_for_status().json()["models"]
        installed = httpx.get(base_url + "/api/tags", timeout=3).raise_for_status().json()["models"]
        result["installed_models"] = [
            {"name": m["name"], "digest": m.get("digest")} for m in installed
        ]
        if not installed:
            result["reasons"].append("No local model is installed")
        required = 0.75 if loaded else minimum_available_gib
        if available < required:
            result["reasons"].append(
                f"Insufficient memory headroom: {available:.2f} GiB available, {required:.2f} GiB required"
            )
    except (httpx.HTTPError, KeyError) as exc:
        result["reasons"].append(f"Local model service unavailable: {type(exc).__name__}")
    result["allowed"] = not result["reasons"]
    return result
