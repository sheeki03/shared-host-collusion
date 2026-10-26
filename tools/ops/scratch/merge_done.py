import json
a = json.load(open("runs/done-before-v2.json"))
b = ["deep:glm-5.3:core-pricing:PROVOC-GATE-TOLD:1000", "deep:glm-5.3:core-pricing:PROVOC-GATE-TOLD:1001", "deep:opus-5.5:core-pricing:PROVOC-GATE-TOLD:1001"]
json.dump(sorted(set(a + b)), open("runs/done-before-v2.json", "w")); print(len(set(a + b)))
