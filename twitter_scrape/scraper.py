import tweepy
from tweepy.streaming import StreamListener, Stream
from http.client import IncompleteRead
import traceback
import codecs
import pandas as pd
import time
import sys

#### input your credentials here
consumer_key = 'Q8NYmNp0IH5sw7GRkmOQ'
consumer_secret = 'mZoaRAXIA6k4aLbjRHAC9pw49hv33WlTdzQvkTWZkI'
access_token = '483153462-roeo2ntKhmNNOJt54KjYMzJgJmJJvgnW5Xe5cLmm'
access_token_secret = 'TrmnImRZEfQ1AprXyYghylg2kHLz4X3eh3H7nO40Y'


def limit_handled(cursor):
    while True:
        try:
            yield cursor.next()
        except tweepy.RateLimitError:
            print('sleeping')
            time.sleep(15 * 60)


def getTimeString():
    t = time.localtime()
    return '%d-%d-%d-%d-%d-%d' % (t.tm_year, t.tm_mon, t.tm_mday, t.tm_hour, t.tm_min, t.tm_sec)


def get_api():
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    api = tweepy.API(auth, wait_on_rate_limit=True)
    return api


def scrape_hashtags_cursor(query, outfn, label):
    '''
    cursor raise RuntimeError: generator raised StopIteration
    '''

    ### get auth
    api = get_api()

    scraped_ids = set()
    # Use csv Writer
    with codecs.open(outfn, 'w', encoding='utf8') as outf:
        outf.write('{}\n'.format('\t'.join(['Tweet index', 'Label', 'Tweet text'])))
        idx = 0
        while True:
            try:
                # for tweet in limit_handled(tweepy.Cursor(api.search,
                #                                          q=query,
                #                                          tweet_mode="extended",
                #                                          since='2017-11-25',
                #                                          lang="en",
                #                                          ).items()):
                for tweet in tweepy.Cursor(api.search,
                                           q=query,
                                           tweet_mode="extended",
                                           since='2017-11-25',
                                           lang="en",
                                           count=400
                                           ).items():
                    tweet_id = tweet.id_str
                    if tweet_id in scraped_ids:
                        continue
                    scraped_ids.add(tweet_id)
                    print(tweet.created_at)
                    text = tweet.full_text.replace('\n', ' ').replace('\r', ' ')
                    outf.write('{}\n'.format('\t'.join([str(idx), label, text])))
                    idx += 1
                    if idx % 6000 == 0:
                        print('get number of tweets:', len(scraped_ids))
                        time.sleep(900)
                    if idx > 50000:
                        break
            except StopIteration as err:
                print(err)
                # time.sleep(1800)
                break


class MyStreamListener(StreamListener):
    def on_status(self, status):
        print(status.created_at, status.text)

    def on_error(self, status_code):
        if status_code == 420:
            return False


class KeywordStreamer:
    '''
    Streams in tweets matching a set of terms/from a set of user streams.
    '''

    def __init__(self, kw_list):

        self.kws = kw_list
        self.api = get_api()
        self.stream = Stream(auth=self.api.auth, listener=MyStreamListener())

    def streamTweets(self):
        goForIt = True
        while goForIt:
            try:
                goForIt = False
                self.stream.filter(track=self.kws)
            except IncompleteRead as ex:
                try:
                    tb = traceback.format_exc()
                except:
                    tb = ''
                print("%s: %s\n%s\n\n" % (getTimeString(),
                                          str(sys.exc_info()) + "\nStill ticking!", tb))
                time.sleep(15 * 60)
                goForIt = True
            except:
                try:
                    tb = traceback.format_exc()
                except:
                    tb = ''
                print('%s: %s\n%s\n\n' % (getTimeString(),
                                          str(sys.exc_info()), tb))
                time.sleep(15 * 60)
                goForIt = True


if __name__ == '__main__':
    positive_hashtags = ['#positive', '#love', '#motivation', '#positivevibes', '#happy',
                         '#amazing', '#positivity', '#inspiration', '#happiness', '#success']
    positive_query = "{} -filter:retweets".format(' OR '.join(positive_hashtags))  # :)
    print(positive_query)
    negative_hashtags = ['#sad', '#negativespace', '#negative', '#depressed', '#depression',
                         '#hate', '#frustrated', '#pain', '#struggle', '#failure']
    negative_query = "{} -filter:retweets".format(' OR '.join(negative_hashtags))  # :(
    print(negative_query)

    ### use cursor
    # label 0 for positive, 1 for negative
    scrape_hashtags_cursor(positive_query, 'twitter_scrape/positive_text.txt', label='0')
    print('done positive')
    time.sleep(900)
    scrape_hashtags_cursor(negative_query, 'twitter_scrape/negative_text.txt', label='1')
    print('done negative')

    ### use stream
    # streamer = KeywordStreamer(negative_hashtags)
    # streamer.streamTweets()
