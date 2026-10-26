# D26: audited raise of the confirmatory L2-coder ledger cap (owner: "no cap needed"); run from the repo root.
from harness.ledger import BudgetLedger
led = BudgetLedger("runs/analysis/coding/ledger.jsonl", 50.0, create=False)
led.raise_cap(150.0, reason="D26: owner said no cap is needed; refused calls settle at est_max (D25) and would "
                            "exhaust the 50 USD admission cap before the final coding pass")
print("coder cap raised; spent", round(led.spent, 4), "flags", led.flags)
