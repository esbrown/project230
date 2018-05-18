import json
from pprint import pprint

with open('randomTweets.json') as f:
    data = json.load(f)
    for tweet in data:
    	print 'user: ' + tweet['text']

#pprint(data)