import json
import csv
from pprint import pprint

tweetTexts = []
with open('randomTweets.json') as f:
    data = json.load(f)
    for tweet in data:
    	text = tweet['text'].encode("utf-8").replace('\n', ' ')
    	if len(text) <= 140:
    		tweetTexts.append([text])

with open('csvTweets.csv', 'w') as csvFile:
    writer = csv.writer(csvFile)
    writer.writerows(tweetTexts)
csvFile.close()
