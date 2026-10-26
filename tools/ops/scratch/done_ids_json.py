import sys, json
from inspect_ai.log import list_eval_logs, read_eval_log
ids = []
for d in sys.argv[1:]:
    for info in list_eval_logs(d):
        log = read_eval_log(info)
        ids += [s.id for s in (log.samples or []) if not s.error and (s.metadata or {}).get("result")]
print(json.dumps(ids))
