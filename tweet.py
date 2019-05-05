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
import os


CONSUMER_KEY = "7ltX8F0vUn30aHWytulVoot5S"
CONSUMER_SECRET = "cmTfSV81rwp6jWBoWOAHCyWoa95AYg22OYfSMrcuoWW8a4LF2d"
OAUTH_TOKEN = "90821027-4ET1pv1QdYS97jE9kXe2hz8zvlMclZcHgXqW4cg84"
OAUTH_TOKEN_SECRET = "qYu4urGMBywUNrOKnVCX1sTRozP7HlIQn3owKwx6g0a1a"


# CONSUMER_KEY = "ask amir"
# CONSUMER_SECRET = "ask amir"
# OAUTH_TOKEN = "ask amir"
# OAUTH_TOKEN_SECRET = "ask amir"

def getScreenName(politician):
    screen_name = politician[2]
    if screen_name.startswith('@'):
        screen_name = screen_name[1:]
    if screen_name == '?' or screen_name == '':
        screen_name =  ''
    return screen_name

#######################################################################################
#this is the main
#create streaming instance if the connection is lost ctahc the exception and try again
#######################################################################################

def getAndDumpTweetsOfUser(politician,firstScreenNames, secondScreenNames):
    global screen_name
    try :

        #clean screen name

        screen_name = getScreenName(politician)
        if screen_name == '':
            return


        with open("data//tweets_" + screen_name + ".txt", 'w',encoding='utf-8') as file:

            # write header
            for item in politician:
                file.write(item + ",")
            file.write('\n')

            # write friends
            for friend in firstScreenNames :
                result = api.show_friendship(source_screen_name=screen_name, target_screen_name= friend)
                if result[0].following :
                    file.write(friend + ",")
                    print(friend)
            for friend in secondScreenNames :
                result = api.show_friendship(source_screen_name=screen_name, target_screen_name= friend)
                if result[0].following:
                    file.write(friend + ",")
                    print(friend)
            file.write('\n')

            api.lookup_friendships(screen_names=['olswang','sdfsdf'])
            # first = True
            # for friend in tweepy.Cursor(api.friends, screen_name='Ayelet__Shaked').items():
            #     # Process the friend here
            #     if first == False : file.write(',')
            #     first = False
            #     file.write(friend.screen_name)
            # file.write('\n')

            # get tweets
            status_cursor = tweepy.Cursor(api.user_timeline, screen_name=screen_name, count=200, tweet_mode='extended')

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
            file.write("error\n")



def getPoliticians(file):
    politicians = [[]]  # an empty list
    with open(file, 'r', encoding='utf-8') as csvFile:
        reader = csv.reader(csvFile)
        for row in reader:
            politicians.append(row)
    csvFile.close()
    return politicians[2:]


def readAllFilesInDir(path):

    files = []
    # r=root, d=directories, f = files
    for r, d, f in os.walk(path):
        for file in f:
            if '.txt' in file:
                files.append(file)
    return files

def writeTextFile(fileName,path):
    with open(path+ fileName,encoding='utf-8') as inFile:
        line = inFile.readline()
        line = inFile.readline()
        line = inFile.readline()
        with open(path + 'text//' + fileName, 'w', encoding='utf-8') as outFile:
            while line:
                d = json.loads(line, encoding='utf-8')
                if "retweeted_status" in d:
                    outFile.write('מאוד חיובי' + "\n")
                else:
                    str = d['full_text'].replace('\n',". ")
                    outFile.write(str + '\n')
                line = inFile.readline()
    outFile.close()
    inFile.close()



######################################################################################
#           Main
######################################################################################

#tempMain
files = readAllFilesInDir('data//')
for f in files :
    writeTextFile(f,'data//')
exit()

# Init Tweeter API
auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
auth.set_access_token(OAUTH_TOKEN, OAUTH_TOKEN_SECRET)
api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True, compression=True)


politicians = getPoliticians('politicians.csv')

firstScreenNames = []
secondScreenNames = []

for idx, politican in enumerate(politicians):
    name = getScreenName(politican)
    if name != '':
        if idx < 90 :
            firstScreenNames.append(name)
        else:
            secondScreenNames.append(name)

for politican in politicians:
    getAndDumpTweetsOfUser(politican,firstScreenNames,secondScreenNames)
