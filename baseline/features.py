import pickle
import re
import os
import sys
import json
import numpy as np
from nltk.util import ngrams
from collections import Counter
import nltk
nltk.download('averaged_perceptron_tagger')

from utils import read_non_emoji_tweets
from utils import get_label
from utils import print_class_stats
from utils import read_vocabulary, write_tokens_to_txt

# TODO: tokenize takes time, and we are now doing it for every feature that need tokenization, optimize it later

### Part of Speech Implementation

def part_of_speech(data):
    '''
    Part of speech main function
    # TODO: TweeboParser has fewer class than nltk pos_tagger
    '''
    feature_dict = {}
    try:
        for tweet in data:

            # tokenize and tag tweet
            tokenized = nltk.word_tokenize(tweet.tweet_text)
            tagged = nltk.pos_tag(tokenized)

            # count the absolute number of each tag
            tweet_tag_counter = Counter()
            for _, tag in tagged:
                tweet_tag_counter[tag] += 1

            # compute/convert the absolute count of each tag
            tweet_tag_cnt = list(tweet_tag_counter.items())

            # compute the ratio of each tag
            tweet_tag_ratio = [(tag, tweet_tag_counter[tag] / len(tweet_tag_counter)) for tag in tweet_tag_counter]

            # compute lexical density: the number of unique tokens divided by the total number of words.
            tweet_lexical_density = len(set(tokenized))/len(tokenized)

            feature_dict[tweet.tweet_id] = {
                'tweet_tag_cnt' : tweet_tag_cnt,
                'tweet_tag_ratio' : tweet_tag_ratio,
                'tweet_lexical_density' : tweet_lexical_density
            }

        # print(feature_dict)

    except Exception as e:
        print(str(e))

### Pronunciation Implementation

def syllable_count(word):
    '''
    syllable count helper
    '''
    word = word.lower()
    count = 0
    vowels = "aeiouy"
    if word[0] in vowels:
        count += 1
    for index in range(1, len(word)):
        if word[index] in vowels and word[index - 1] not in vowels:
            count += 1
    if word.endswith("e"):
        count -= 1
    if count == 0:
        count += 1
    return count

def pronunciations(data):
    '''
    Pronunciation main function
    '''
    feature_dict = {}
    try:
        for tweet in data:

            # number of words with only alphabetic characters but no vowels
            tweet_no_vowel_cnt = 0

            #number of words with more than three syllables
            tweet_three_more_syllables_cnt = 0

            # tokenize and tag tweet
            tokenized = nltk.word_tokenize(tweet.tweet_text)

            for token in tokenized:
                if token.isalpha() and not any(letter in 'aeiou' for letter in token):
                    tweet_no_vowel_cnt += 1
                if syllable_count(token) > 3:
                    tweet_three_more_syllables_cnt += 1

            feature_dict[tweet.tweet_id] = {
                'tweet_no_vowel_cnt' : tweet_no_vowel_cnt,
                'tweet_three_more_syllables_cnt' : tweet_three_more_syllables_cnt
            }

        # print(feature_dict)

    except Exception as e:
        print(str(e))

### Capitalization Implementation

def capitalization(data):
    '''
    Capitalization main function
    '''
    feature_dict = {}
    try:
        for tweet in data:

            # number of words with initial cap
            tweet_initial_cap_cnt = 0

            # number of words with all cap
            tweet_all_cap_cnt = 0

            # number of tags with initial or all cap
            tweet_tag_cap_cnt = 0

            # tokenize tweet
            tokenized = nltk.word_tokenize(tweet.tweet_text)

            for token in tokenized:
                if token.istitle():
                    tweet_initial_cap_cnt += 1
                if token.isupper():
                    tweet_all_cap_cnt += 1

            tagged = nltk.pos_tag(tokenized)
            for _, tag in tagged:
                if tag.istitle() or tag.isupper():
                    tweet_tag_cap_cnt += 1

            feature_dict[tweet.tweet_id] = {
                'tweet_initial_cap_cnt' : tweet_initial_cap_cnt,
                'tweet_all_cap_cnt' : tweet_all_cap_cnt,
                'tweet_tag_cap_cnt' : tweet_tag_cap_cnt
            }

    except Exception as e:
        print(str(e))


### Unigram and Bigram features

def construct_vocabulary(corpus, min_freq=3):
    """
    construct vocabulary file first before extract ngrams
    TODO: discuss about min_freq
    """
    unigram_freq = Counter([])
    bigram_freq = Counter([])
    for tweet in corpus:
        tokens = tweet.tweet_words()
        lower_tokens = [t.lower() for t in tokens]
        # add start/end of sentence token
        bigram = ngrams(['<s>'] + lower_tokens + ['</s>'], 2)

        unigram_freq += Counter(lower_tokens)
        bigram_freq += Counter(bigram)

    # write to txt file
    def write_counter_to_file(freq, out_fn, min_freq=min_freq):
        with open(out_fn, 'w') as outf:
            for ele, count in freq.items():
                if count >= min_freq:
                    if type(ele) == str:
                        # write unigram
                        outf.write('{}\n'.format(ele))
                    else:
                        # write bigram
                        outf.write('{}\n'.format('\t'.join(ele)))

    write_counter_to_file(unigram_freq, 'baseline/unigram_vocab.txt')
    write_counter_to_file(bigram_freq, 'baseline/bigram_vocab.txt')

    return unigram_freq, bigram_freq


def extract_ngrams(corpus):
    """
    input: whole corpus
    output: 2 dicts for unigram and bigram features as arrays
    TODO: OOV words later
    """
    if not os.path.exists('baseline/unigram_vocab.txt') or not os.path.exists('baseline/bigram_vocab.txt'):
        construct_vocabulary(corpus)

    # key: word, value: index
    unigram_vocab = read_vocabulary('baseline/unigram_vocab.txt')
    bigram_vocab = read_vocabulary('baseline/bigram_vocab.txt')

    unigram_dict = {}
    bigram_dict = {}
    for data in corpus:
        tokens = data.tweet_words()
        lower_tokens = [t.lower() for t in tokens]
        _id = data.tweet_id
        unigram_dict[_id] = np.zeros(len(unigram_vocab))
        bigram_dict[_id] = np.zeros(len(bigram_vocab))
        for idx, ele in enumerate(lower_tokens):
            # unigram
            if ele in unigram_vocab:
                unigram_dict[_id][unigram_vocab[ele]] = 1.

            if idx == len(lower_tokens) - 1:
                continue

            # bigram
            if (ele, lower_tokens[idx+1]) in bigram_vocab:
                bigram_dict[_id][bigram_vocab[(ele, lower_tokens[idx+1])]] = 1.

    return unigram_dict, bigram_dict


def brown_cluster_ngrams(corpus, num_cluster=1000):
    # TODO: may need to adjust the num_cluster since we have smaller dataset/vocabulary size
    # TODO: check for spacy implementation later
    cluster_fn = 'baseline/brown_cluster_{}.txt'.format(num_cluster)
    if not os.path.exists(cluster_fn):
        print('brown cluster file not exist, run the repo first')
        write_tokens_to_txt(corpus, 'baseline/corpus_A.txt')
        sys.exit(1)

    cluster_vocab = {}
    idx = 0
    int2idx = {}
    with open(cluster_fn, 'r') as inf:
        for line in inf:
            # TODO: set min freq?
            cluster_bit, word, freq = line.strip().split('\t')
            cluster_int = int(cluster_bit, 2)
            if cluster_int not in int2idx:
                int2idx[cluster_int] = idx
                idx += 1
            cluster_vocab[word] = int2idx[cluster_int]

    unigram_dict = {}
    bigram_dict = {}
    for data in corpus:
        tokens = data.tweet_words()
        lower_tokens = [t.lower() for t in tokens]
        _id = data.tweet_id
        unigram_dict[_id] = np.zeros(len(cluster_vocab))
        bigram_dict[_id] = np.zeros(len(cluster_vocab))
        for idx, ele in enumerate(lower_tokens):
            # unigram
            if ele in cluster_vocab:
                unigram_dict[_id][cluster_vocab[ele]] = 1.
                bigram_dict[_id][cluster_vocab[ele]] = 1.

            if idx == len(lower_tokens) - 1:
                continue

            # bigram
            if lower_tokens[idx + 1] in cluster_vocab:
                bigram_dict[_id][cluster_vocab[lower_tokens[idx + 1]]] = 1.

    return unigram_dict, bigram_dict


def dependency():
    pass

if __name__ == '__main__':
    # File paths from project level
    # fp_train_A = 'tweet_irony_detection/train/SemEval2018-T3-train-taskA.txt'
    fp_train_A = 'train/SemEval2018-T3-train-taskA.txt'
    fp_train_B = 'train/SemEval2018-T3-train-taskB.txt'
    fp_test_A = 'test_TaskA/SemEval2018-T3_input_test_taskA.txt'
    fp_test_B = 'test_TaskB/SemEval2018-T3_input_test_taskB.txt'
    fp_labels_A= 'goldtest_TaskA/SemEval2018-T3_gold_test_taskA_emoji.txt'
    fp_labels_B='goldtest_TaskB/SemEval2018-T3_gold_test_taskB_emoji.txt'

    # Training data for task A and B , test data & correct labels for both tasks
    pre_process_url=True # Set to remove URLs
    pre_process_usr=True
    train_A = read_non_emoji_tweets(fp_train_A,"train",pre_process_url,pre_process_usr)
    train_B = read_non_emoji_tweets(fp_train_B,"train",pre_process_url,pre_process_usr)

    test_A= read_non_emoji_tweets(fp_test_A,"test",pre_process_url,pre_process_usr)
    test_B = read_non_emoji_tweets(fp_test_B, "test", pre_process_url,pre_process_usr)
    labels_A=get_label(fp_labels_A)
    labels_B = get_label(fp_labels_B)
    # Print class stats
    print_class_stats(train_A,train_B,labels_A,labels_B)


    # unit test for features
    # TODO: differentiate vocab for A,B and emoji task
    # unigram_feature, bigram_feature = extract_ngrams(train_A)
    # print(unigram_feature[2][:20], bigram_feature[2][:20])

    unigram_brown_feature, bigram_brown_feature = brown_cluster_ngrams(train_A)
    print(unigram_brown_feature[2][:20], bigram_brown_feature[2][:20])

    # unit test for part of speech
    # part_of_speech(train_A)

    # unit test for prounciation
    # pronunciations(train_A)

    # unit test for capitalization
    # capitalization(train_A)
