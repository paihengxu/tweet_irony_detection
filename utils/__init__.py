import re

from typing import Dict, List, NamedTuple
from collections import Counter
from nltk.tokenize import TweetTokenizer
from nltk.stem.snowball import SnowballStemmer

#  Replace repeated character sequences of length 3 or greater with sequences of length 3.
tknzr = TweetTokenizer(strip_handles=True, reduce_len=True)
stemmer = SnowballStemmer('english')
# TODO: Check on USR in test id - 147,148,160,230,319,336.398

class Tweet(NamedTuple):
    tweet_id: int
    tweet_label:int
    tweet_text:str

    def tweet_words(self):
        return tknzr.tokenize(self.tweet_text)

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
    processed_text = re.sub(r"http[s]*://[\w,\.,\/]+", "*URL*", text)
    # print(f'pre-process: {text}')
    # print(f'post-process:{processed_text} \n')
    return processed_text


def pre_process_usrname(text):
    #TODO: *USR* would be tokenized into three tokens
    processed_text = re.sub(r"@\w+", "*USR*", text)
    # print(f'{count}:pre-process: {text}')
    # print(f'{count}:post-process:{processed_text} \n')
    return processed_text


def read_non_emoji_tweets(fp,type,pre_process_url,pre_process_usr):
    '''
    TODO: some data not in utf8 eg: 355's FOLLOW, so issues when print
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
                if pre_process_usr:
                    text=pre_process_usrname(text)
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


def do_stemming(text):
    words=tknzr.tokenize(text)
    stemmed=[stemmer.stem(word) for word in words]
    return stemmed

def read_vocabulary(fn):
    """
    Read file to a dict with value as index
    """
    result = {}
    idx = 0
    with open(fn, 'r') as inf:
        for line in inf:
            k = line.strip().split('\t')
            if len(k) == 2:
                result[(k[0], k[1])] = idx
            else:
                result[k[0]] = idx
            idx += 1
    return result

def write_tokens_to_txt(corpus, fn):
    with open(fn, 'w') as outf:
        for data in corpus:
            tokens = data.tweet_words()
            lower_tokens = [t.lower() for t in tokens]
            outf.write("{}\n".format(' '.join(lower_tokens)))