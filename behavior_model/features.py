import os
import numpy as np
from collections import defaultdict

from utils import read_non_emoji_tweets
from utils import get_label
from utils import print_class_stats
from utils import read_vocabulary_with_occurrence, write_tokens_to_txt

### DEFINE CONST HERE

def twitter_sentiment_score(corpus, n):
    """
    input: whole dataset, n indicating n-gram
    output: two dictionaries, key: tweet_id, value: 1-dimensional binary numpy array
    Done: Out of Vocabulary (OOV) words
    """
    if not os.path.exists('behavior_model/positive_{}_vocab.txt'.format(n)) or not os.path.exists('behavior_model/negative_{}_vocab.txt'.format(n)):
        os.system('python -m behavior_model.sent_vocab')

    # key: word, value: index
    pos_vocab = read_vocabulary_with_occurrence('behavior_model/positive_{}_vocab.txt'.format(n), n)
    neg_vocab = read_vocabulary_with_occurrence('behavior_model/negative_{}_vocab.txt'.format(n), n)

    senti_features = defaultdict(float)
    for data in corpus:
        tokens = data.tweet_words()
        lower_tokens = [t.lower() for t in tokens]
        _id = data.tweet_id
        for idx, ele in enumerate(lower_tokens):
            # check idx for n-grams
            if idx >= len(lower_tokens) - (n-1):
                continue
            # set default value as 0, bc (-0.1, 0.1) is filtered out
            n_gram = [lower_tokens[w_idx] for w_idx in range(idx, idx+n)]
            assert len(n_gram) == n
            pos_score = pos_vocab.get(tuple(n_gram), 0)
            neg_score = neg_vocab.get(tuple(n_gram), 0)

            if pos_score + neg_score == 0:
                continue

            senti_score = (pos_score - neg_score)/(pos_score + neg_score)

            if -0.1 < senti_score < 0.1:
                continue

            senti_features[_id] += senti_score

    return senti_features

if __name__ == '__main__':
    ### read in corpus
    fp_train_A = 'train/SemEval2018-T3-train-taskA.txt'
    pre_process_url = True  # Set to remove URLs
    pre_process_usr = True
    train_A = read_non_emoji_tweets(fp_train_A, "train", pre_process_url, pre_process_usr)

    ### test sentiment score
    senti_score_bigram = twitter_sentiment_score(train_A, 2)
    senti_score_trigram = twitter_sentiment_score(train_A, 3)

    for i in range(1, 20):
        assert i==train_A[i-1].tweet_id
        print(senti_score_bigram[i])
        print(senti_score_trigram[i])
        print(train_A[i-1].tweet_text)
