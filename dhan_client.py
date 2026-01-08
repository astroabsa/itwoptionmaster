import os
import datetime as dt
from typing import Any, Dict, List, Optional

import requests

API_BASE = "https://api.dhan.co/v2"

# These IDs are examples. Verify with Dhan instrument/master if needed.
NIFTY = {"name": "NIFTY", "UnderlyingScrip": 13, "UnderlyingSeg": "IDX_I"}
SENSEX = {"name": "SENSEX", "UnderlyingScrip": 51, "UnderlyingSeg": "IDX_I"}  # replace if different in your setup


class DhanAPIError(RuntimeError):
    pass


def _headers() -> Dict[str, str]:
    client_id = os.getenv("DHAN_CLIENT_ID")
    access_token = os.getenv("DHAN_ACCESS_TOKEN")
    if not client_id or not access_token:
        raise DhanAPIError("Missing DHAN_CLIENT_ID / DHAN_ACCESS_TOKEN in environment.")
    return {
        "client-id": client_id,
        "access-token": access_token,
        "Content-Type": "application/json",
        "Accept": "application/json",
    }


def _post(path: str, payload: Dict[str, Any], timeout: int = 15) -> Dict[str, Any]:
    url = f"{API_BASE}{path}"
    r = requests.post(url, json=payload, headers=_headers(), timeout=timeout)
    if not r.ok:
        raise DhanAPIError(f"HTTP {r.status_code}: {r.text[:500]}")
    return r.json()


def expiry_list(underlying_scrip: int, underlying_seg: str) -> List[str]:
    resp = _post("/optionchain/expirylist", {
        "UnderlyingScrip": underlying_scrip,
        "UnderlyingSeg": underlying_seg,
    })
    expiries = resp.get("data", [])
    if not expiries:
        raise DhanAPIError(f"Unexpected expiry list response: {resp}")
    return sorted(expiries)


def nearest_expiry(expiries: List[str], as_of: Optional[dt.date] = None) -> str:
    as_of = as_of or dt.date.today()
    parsed = []
    for e in expiries:
        try:
            parsed.append((dt.date.fromisoformat(e), e))
        except ValueError:
            pass
    if not parsed:
        raise DhanAPIError("No valid expiries returned.")
    future = [(d, e) for d, e in parsed if d >= as_of]
    return min(future, key=lambda x: x[0])[1] if future else max(parsed, key=lambda x: x[0])[1]


def option_chain(underlying_scrip: int, underlying_seg: str, expiry: str) -> Dict[str, Any]:
    return _post("/optionchain", {
        "UnderlyingScrip": underlying_scrip,
        "UnderlyingSeg": underlying_seg,
        "Expiry": expiry,
    })


def fetch_chain_for_symbol(symbol: Dict[str, Any]) -> Dict[str, Any]:
    expiries = expiry_list(symbol["UnderlyingScrip"], symbol["UnderlyingSeg"])
    exp = nearest_expiry(expiries)
    oc = option_chain(symbol["UnderlyingScrip"], symbol["UnderlyingSeg"], exp)
    oc["_meta"] = {"symbol": symbol["name"], "expiry": exp}
    return oc
