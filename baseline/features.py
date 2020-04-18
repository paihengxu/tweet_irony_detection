import pickle
import re

import os
import sys

from utils import read_non_emoji_tweets
from utils import get_label
from utils import print_class_stats

from collections import Counter
import nltk
nltk.download('averaged_perceptron_tagger')

### Part of Speech Implementation

def part_of_speech(data):
    '''
    Part of speech main function
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

def extract_ngrams():
    pass

def brown_cluster_ngrams():
    pass

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
    # extract_ngrams()

    # unit test for part of speech
    # part_of_speech(train_A)

    # unit test for prounciation
    # pronunciations(train_A)

    # unit test for capitalization
    # capitalization(train_A)
