"""Firecrawl search helper (key from ~/.firecrawl-api-key). Read-only.

Usage:
  fc_search.py papers "<query>" [k]        # Research Index: GET /v2/search/research/papers
  fc_search.py research "<query>" [limit]  # web search restricted to academic sites (arXiv, PubMed, ...)
  fc_search.py dev "<query>" [limit]       # developer index: public repo READMEs, issues, merged PRs, docs
  fc_search.py web "<query>" [limit]       # general web search
  fc_search.py scrape "<url>"              # fetch one page as markdown (first 6000 chars)
"""
import json
import os
import pathlib
import sys

import httpx

KEY = pathlib.Path(os.path.expanduser("~/.firecrawl-api-key")).read_text().strip()
BASE = "https://api.firecrawl.dev/v2"
H = {"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"}


def show(items: list, fields: tuple) -> None:
    for it in items:
        print(" | ".join(str(it.get(f, ""))[:300] for f in fields if it.get(f)))
        print("-")


def main() -> None:
    mode, query = sys.argv[1], sys.argv[2]
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    if mode == "papers":
        r = httpx.get(f"{BASE}/search/research/papers", params={"query": query, "k": n}, headers=H, timeout=90)
        r.raise_for_status()
        data = r.json()
        items = data.get("results") or data.get("data") or []
        if not items:
            print(json.dumps(data)[:3000])
        for it in items:
            pid = it.get("primaryId", "")
            url = f"https://arxiv.org/abs/{pid.split(':', 1)[1]}" if pid.startswith("arxiv:") else pid
            print(f"{it.get('title', '')}\n  {url}\n  {it.get('abstract', '')[:400]}\n-")
    elif mode == "scrape":
        r = httpx.post(f"{BASE}/scrape", json={"url": query, "formats": ["markdown"]}, headers=H, timeout=120)
        r.raise_for_status()
        print((r.json().get("data") or {}).get("markdown", "")[:6000])
    else:
        body = {"query": query, "limit": n}
        if mode == "research":
            body["categories"] = ["research"]
        elif mode == "dev":
            body["categories"] = ["developer"]
        r = httpx.post(f"{BASE}/search", json=body, headers=H, timeout=90)
        r.raise_for_status()
        data = r.json().get("data") or {}
        items = data.get("web") or data.get("results") or (data if isinstance(data, list) else [])
        show(items, ("title", "url", "description", "highlights"))


if __name__ == "__main__":
    main()
