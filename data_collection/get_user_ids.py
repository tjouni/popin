import json

d = json.load(open("hashtag.json"))
d = sorted(d, key=lambda x: x.get("likesCount", 0), reverse=True)[::20]


setit = {x["ownerId"] for x in d if "ownerId" in x if "username" not in x}

for setti in setit:
    print(setti)

with open("sorted_subset.json", "w") as f:
    json.dump(d, f)
