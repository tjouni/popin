## Data collection for kissat koiria äpplykäätiö

* Scrape for hashtags with https://console.apify.com/actors/reGe1ST3OBgYZSsZJ
* Save file as hashtag.json
* Run ```python get_user_ids.py > userids.txt```
* Scrape for usernames with https://console.apify.com/actors/lUpCGVpsNY1vFlxUj using userids.txt as input
* Save file as usernames.json
* Run ```python add_usernames.py > usernames.txt```
* Scrape for user data with https://console.apify.com/actors/dSCLg0C3YEZ83HzYX using usernames.txt as input
* Save file as userinfo.json
* Run ```python add_profile_data.py```
* Run ```python imagedownloader.py``` to download the images
