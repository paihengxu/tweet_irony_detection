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


def read_non_emoji_tweets(fp,type):
    '''
    :param fp: file path
    :param type: either train or test data
    :return: List of Tweet
    '''
    tweets=[]
    with open(fp,encoding='utf8') as f:
        for line in f:
            line=line.strip()
            if not line.lower().startswith("tweet"):
                split=line.split("\t")
                id=int(split[0])
                if type.lower()=="train":
                    label=int(split[1])
                    text=split[2]
                else:
                    label = None
                    text = split[1]
                tweets.append(Tweet(id,label,text))
    return tweets

def get_label(fp):
    '''
    :param fp: Takes gold test data set
    :return: labels dict {tweet_id: label}
    '''
    labels={}
    with open(fp,encoding='utf8') as f:
        for line in f:
            line=line.strip()
            if not line.lower().startswith("tweet"):
                split=line.split("\t")
                id=int(split[0])
                label=int(split[1])
                labels[id]=label
    return labels


def print_data_stats(train_A,train_B,labels_A,labels_B):
    pass