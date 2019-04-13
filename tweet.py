#######################################################################################
#The tweet.py file is in charge of reading live tweets and dumping it to a file
#######################################################################################


import twython
import numpy as np
from time import sleep, time
import json
import codecs
import tweepy
import csv


CONSUMER_KEY = "ask amir"
CONSUMER_SECRET = "ask amir"
OAUTH_TOKEN = "ask amir"
OAUTH_TOKEN_SECRET = "ask amir"


#######################################################################################
#this is the main
#create streaming instance if the connection is lost ctahc the exception and try again
#######################################################################################

def getAndDumpTweetsOfUser(politician):
    global screen_name
    try :

        #clean screen name

        screen_name = politician[2]
        if screen_name.startswith('@'):
            screen_name =screen_name[1:]
        if screen_name == '?' or screen_name == '':
            return

        # get tweets

        status_cursor = tweepy.Cursor(api.user_timeline, screen_name=screen_name, count=200, tweet_mode='extended')

        with open("data//tweets_" + screen_name + ".txt", 'w',encoding='utf-8') as file:

            # write header
            for item in politician:
                file.write(item + ",")
            file.write('\n')

            # dump tweets
            for status in status_cursor.items():
                json.dump(status._json,file, ensure_ascii = False)
                file.write('\n')
                print(status)

    except Exception as e:
        sleep(1)
        print('error - ')
        print(e)
        with open("data//tweets_" + screen_name + "_error.txt", 'w', encoding='utf-8') as file:
            file.write("\n")



def getPoliticians(file):
    politicians = [[]]  # an empty list
    with open(file, 'r', encoding='utf-8') as csvFile:
        reader = csv.reader(csvFile)
        for row in reader:
            politicians.append(row)
    csvFile.close()
    return politicians[2:]


######################################################################################
#           Main
######################################################################################

# Init Tweeter API
auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
auth.set_access_token(OAUTH_TOKEN, OAUTH_TOKEN_SECRET)
api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True, compression=True)


politicians = getPoliticians('politicians.csv')
for politican in politicians:
    getAndDumpTweetsOfUser(politican)
