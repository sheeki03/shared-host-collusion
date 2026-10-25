"""Print only the nanoGPT USD balance, without the key or unrelated account fields."""
from pathlib import Path

import httpx


def main() -> None:
    key = (Path.home() / ".nanogpt-api-key").read_text().strip()
    attempts = (
        ("POST", "https://nano-gpt.com/api/check-balance", {"x-api-key": key}),
        ("GET", "https://nano-gpt.com/api/v1/balance", {"Authorization": f"Bearer {key}"}),
    )
    for method, url, headers in attempts:
        try:
            response = httpx.request(method, url, headers=headers, timeout=30)
            response.raise_for_status()
            data = response.json()
            balance = data.get("usd_spendable_balance") or data.get("usd_balance")
            if balance is not None:
                print(f"usd_balance={balance}")
                return
        except (httpx.HTTPError, ValueError, AttributeError):
            continue
    raise SystemExit("nanoGPT balance unavailable")


if __name__ == "__main__":
    main()
