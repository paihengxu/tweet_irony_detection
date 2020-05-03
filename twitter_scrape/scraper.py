import tweepy
import codecs
import pandas as pd
import time

def scrape_hashtags(query, outfn, label):
    #### input your credentials here
    consumer_key = 'Q8NYmNp0IH5sw7GRkmOQ'
    consumer_secret = 'mZoaRAXIA6k4aLbjRHAC9pw49hv33WlTdzQvkTWZkI'
    access_token = '483153462-roeo2ntKhmNNOJt54KjYMzJgJmJJvgnW5Xe5cLmm'
    access_token_secret = 'TrmnImRZEfQ1AprXyYghylg2kHLz4X3eh3H7nO40Y'

    ### get auth
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    api = tweepy.API(auth, wait_on_rate_limit=True)

    scraped_ids = set()
    #Use csv Writer
    with codecs.open(outfn, 'w', encoding='utf8') as outf:
        outf.write('{}\n'.format('\t'.join(['Tweet index', 'Label', 'Tweet text'])))
        idx = 0
        p = 1
        while True:
            try:
                for tweet in tweepy.Cursor(api.search,
                                           q=query,
                                           tweet_mode="extended",
                                           since='2017-11-25',
                                           count=400,
                                           lang="en",
                                           ).items():
                    tweet_id = tweet.id_str
                    if tweet_id in scraped_ids:
                        continue
                    scraped_ids.add(tweet_id)
                    print(tweet.created_at)
                    text = tweet.full_text.replace('\n', ' ').replace('\r', ' ')
                    outf.write('{}\n'.format('\t'.join([str(idx), label, text])))
                    idx += 1
                    if idx > 400000:
                        break
            except Exception as err:
                print(err)
                break
            print('get number of tweets:', len(scraped_ids))
            p += 1
            time.sleep(900)
            # csvWriter.writerow([tweet.created_at, tweet.full_text.encode('utf-8')])

if __name__ == '__main__':
    positive_hashtags = ['#positive', '#love', '#motivation', '#positivevibes', '#happy',
                         '#amazing', '#positivity', '#inspiration', '#happiness', '#success']
    positive_query = "{} -filter:retweets".format(' OR '.join(positive_hashtags)) # :)
    print(positive_query)
    negative_hashtags = ['#sad', '#negativespace', '#negative', '#depressed', '#depression',
                         '#hate', '#frustrated', '#pain', '#struggle', '#failure']
    negative_query = "{} -filter:retweets".format(' OR '.join(negative_hashtags))  # :(
    print(negative_query)
    # label 0 for positive, 1 for negative
    scrape_hashtags(positive_query, 'twitter_scrape/positive_text.txt', label='0')
    print('done positive')
    scrape_hashtags(negative_query, 'twitter_scrape/negative_text.txt', label='1')
    print('done negative')