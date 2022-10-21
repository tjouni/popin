import json

d = json.load(open("data_with_usernames.json"))
profiles = {x.get("username", ""): x for x in json.load(open("userdata.json"))}
new = []

for image in d:
    try:
        profile = profiles[image["username"]]
        image["userFollowersCount"] = profile["followersCount"]
        image["userFollowsCount"] = profile["followsCount"]
        image["userHasChannel"] = profile["hasChannel"]
        image["userIsBusinessAccount"] = profile["isBusinessAccount"]
        image["userPostsCount"] = profile["postsCount"]
    except Exception as e:
        print(image)
        print(e)
        pass

print(len(d))
with open("final_data.json", "w") as f:
    json.dump(d, f)
