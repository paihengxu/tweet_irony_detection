import re

from typing import Dict, List, NamedTuple
from collections import Counter
from nltk.tokenize import word_tokenize

class Tweet(NamedTuple):
    tweet_id: int
    tweet_label:int
    tweet_text:str

    def tweet_words(self):
        return word_tokenize(self.tweet_text)

    def __repr__(self):
        return (f"tweet_id: {self.tweet_id}\n" +
                f"tweet_label: {self.tweet_label}\n" +
                f"tweet_text: {self.tweet_text}")


def pre_process_tweet_url(text):
    '''
    1. Remove URLs r"http[s]*://"
    :param text: takes raw tweet_text
    :return: processed_text:
    '''
    processed_text=text
    if re.match(r"^.*http[s]*://",text):
        processed_text = re.sub(r"http[s]*://[\w,\.,\/]+", "*URL*", text);
        # print(f'pre-process: {text}')
        # print(f'post-process:{processed_text} \n')
    return processed_text

def read_non_emoji_tweets(fp,type,pre_process_url):
    '''
    TODO: some data not in utf8 eg: 355's FOLLOW
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
                if pre_process_url:
                    text=pre_process_tweet_url(text)
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


def print_class_stats(train_A,train_B,labels_A,labels_B):
    labels_train_A=[tweet.tweet_label for tweet in train_A]
    labels_train_B = [tweet.tweet_label for tweet in train_B]
    labels_test_A=labels_A.values()
    labels_test_B = labels_B.values()
    print('For Task A training ',Counter(labels_train_A))
    print('For Task B training ',Counter(labels_train_B))
    print('For Task A testing ',Counter(labels_test_A))
    print('For Task B testing ',Counter(labels_test_B))
