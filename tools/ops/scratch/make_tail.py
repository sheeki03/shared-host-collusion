# Shard for instabox-ext-5: the unstarted GPT tails of E1a (instabox3-ext-2, idx 48-71) and E1b (vps1-ext-3, idx 52-71),
# each in reverse shard order, interleaved. D12 first-finished rule resolves any copy the original runner also completes.
import json
def tail(path, start):
    rows = [l.split() for l in open(path).read().splitlines()[1:]]
    return [g for i, g, st in rows if int(i) >= start and st == "-"][::-1]
a, b = tail("state-instabox3-ext-2.txt", 48), tail("state-vps1-ext-3.txt", 52)
ids = [x for pair in zip(a, b + [None] * (len(a) - len(b))) for x in pair if x]
assert len(ids) == len(set(ids)) == len(a) + len(b)
json.dump({"host": "instabox-ext-5", "game_ids": ids, "rounds": 40 * len(ids)}, open("instabox-ext-5.json", "w"), indent=1)
print(len(a), len(b), len(ids), ids[:3])
