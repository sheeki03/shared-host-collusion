# D27: audited raise of the S6 judge ledger cap for a census of the evaluation set; run from the repo root.
from harness.ledger import BudgetLedger
led = BudgetLedger("runs/analysis/judges/ledger.jsonl", 25.0, create=False)
led.raise_cap(80.0, reason="D27: judge budget 60 USD (census of the evaluation set; the 25 USD partial run was "
                           "already a census of 546 games); owner said no cap is needed")
print("judge cap raised; spent", round(led.spent, 4), "flags", led.flags)
