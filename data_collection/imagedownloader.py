import json


import requests
import shutil
import time

from random import randrange


d = json.load(open("final_data.json"))


images_done = 0

d = sorted(d, key=lambda x: x.get("likesCount", 0), reverse=True)


d = [x for x in d if "2022-10-07" not in x.get("timestamp", "2022-10-07")]

puppies = [x for x in d if "puppy" in x["hashtags"] and "kitten" not in x["hashtags"]]
kittens = [x for x in d if "kitten" in x["hashtags"] and "puppy" not in x["hashtags"]]

print(len(kittens))


for image in kittens:
    try:
        url = image["displayUrl"]
        response = requests.get(url, stream=True)
    except:
        print("skipped an image")
        continue
    with open(f"{image['id']}.jpg", "wb") as out_file:
        shutil.copyfileobj(response.raw, out_file)
    del response
    images_done += 1
    print("kitten" + str(images_done), flush=True)
    time.sleep(randrange(4))

for image in puppies:
    try:
        url = image["displayUrl"]
        response = requests.get(url, stream=True)
    except:
        print("skipped an image")
        continue
    with open(f"{image['id']}.jpg", "wb") as out_file:
        shutil.copyfileobj(response.raw, out_file)
    del response
    images_done += 1
    print("kitten" + str(images_done), flush=True)
    time.sleep(randrange(4))
