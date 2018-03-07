import tweepy
import matplotlib.pyplot as plt
from textblob import TextBlob


#function to return percentage
def percent(part,whole):
    return 100* float(part)/float(whole)


#twitter credentials needed to get access for tweets 
consumer_key = #consumer key from https://apps.twitter.com
consumer_secret = #consumer secret from https://apps.twitter.com

access_token = #access_token from https://apps.twitter.com
access_token_secret = #access_token_secret from https://apps.twitter.com

#for authentication pass consumer key and secret for OauthHandler of tweepy module
auth = tweepy.OAuthHandler(consumer_key,consumer_secret)
#set access toke and secret
auth.set_access_token(access_token,access_token_secret)

#get search word and no of tweets to analyze
searchTerm = input('Enter word to search:')
noOftweets = int(input('Enter number of tweets to analyze:'))

#authnticat twitter with aforementioned keys
api = tweepy.API(auth)

#get tweets by passing search word and number of tweets to analyze
public_tweets = tweepy.Cursor(api.search,q=searchTerm).items(noOftweets)

#four variables to store results
positive = 0
negative = 0
neutral = 0
polarity = 0

#for each tweet extract the text and get polarity of it
for tweet in public_tweets:
    #print(tweet.text)
    analysis = TextBlob(tweet.text)
    polarity += analysis.sentiment.polarity
#based on polarity increment respective varible values
    if(analysis.sentiment.polarity == 0):
        neutral += 1
    elif(analysis.sentiment.polarity < 0):
        negative += 1
    elif(analysis.sentiment.polarity > 0):
        positive += 1

#get percentage of each category tweets from analyzes samples
positive  = percent(positive,noOftweets)
negative  = percent(negative,noOftweets)
neutral  = percent(neutral,noOftweets)
polarity  = percent(polarity,noOftweets)

#format for 2 decimal place
positive = format(positive,'.2f')
negative = format(negative,'.2f')
neutral = format(neutral,'.2f')

#print analyzed summary based on polarity
if(polarity == 0):
    print("Neutral")
elif(polarity<0.00):
    print("Negative")
elif(polarity>0.00):
    print("Positive")



#plotting on matplotlib pyplot
labels = ['Positive['+str(positive)+'%]','Neutral['+str(neutral)+'%]','Negative['+str(negative)+'%]','Polarity['+str(polarity)+'%]']
sizes = [positive,neutral,negative,polarity]
colors = ['yellowgreen','gold','red']
patches, texts = plt.pie(sizes,colors=colors,startangle=90.0)
plt.legend(patches,labels,loc="best")
plt.title('How people are reacting for '+searchTerm+'by analyzing' +str(noOftweets)+'tweets')
plt.axis('equal')
plt.tight_layout()
plt.show()
