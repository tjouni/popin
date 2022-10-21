import json

d = json.load(open("sorted_subset.json"))
usernames = {str(x["pk"]): x["username"] for x in json.load(open("usernames.json"))}
new = []
print(len(d))

for image in d:
    try:
        if "username" not in image:
            image["username"] = usernames[image["ownerId"]]
            new.append(image["username"])
        new.append(image["username"])
    except Exception as e:
        pass

for u in {username for username in new}:
    print(u)

with open("data_with_usernames.json", "w") as f:
    json.dump(d, f)
