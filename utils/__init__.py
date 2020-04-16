from typing import Dict, List, NamedTuple
from nltk.tokenize import word_tokenize

class Tweet(NamedTuple):
    tweet_id: int
    tweet_label:int
    tweet_text:str
    # emojis:List[str]

    def tweet_words(self):
        return word_tokenize(self.tweet_text)

    def __repr__(self):
        return (f"tweet_id: {self.tweet_id}\n" +
                f"tweet_label: {self.tweet_label}\n" +
                f"tweet_text: {self.tweet_text}")


def read_non_emoji_train_tweets(fp):
    tweets=[]
    with open(fp,encoding='utf8') as f:
        for line in f:
            line=line.strip()
            if not line.lower().startswith("tweet"):
                split=line.split("\t")
                id=int(split[0])
                label=int(split[1])
                text=split[2]


                tweets.append(Tweet(id,label,text))
    return tweets

def read_non_emoji_test_tweets(fp):
    tweets=[]
    with open(fp,encoding='utf8') as f:
        for line in f:
            line=line.strip()
            if not line.lower().startswith("tweet"):
                split=line.split("\t")
                id=int(split[0])
                label=None
                text=split[1]
                tweets.append(Tweet(id,label,text))
    return tweets
