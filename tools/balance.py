"""Print the nanoGPT account balance (key from ~/.nanogpt-api-key)."""
import os
import pathlib

import httpx

KEY = pathlib.Path(os.path.expanduser("~/.nanogpt-api-key")).read_text().strip()
for url, kw in (
    ("https://nano-gpt.com/api/check-balance", {"method": "POST", "headers": {"x-api-key": KEY}}),
    ("https://nano-gpt.com/api/v1/balance", {"method": "GET", "headers": {"Authorization": f"Bearer {KEY}"}}),
):
    try:
        r = httpx.request(kw["method"], url, headers=kw["headers"], timeout=30)
        print(url, r.status_code, r.text[:300])
    except Exception as e:  # noqa: BLE001
        print(url, "error", e)
