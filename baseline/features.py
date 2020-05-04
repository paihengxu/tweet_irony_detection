import pickle
import re
import os
import sys
import json
import codecs
import numpy as np
import pandas as pd
from scipy import stats
from nltk.util import ngrams
from collections import Counter
from pycorenlp import StanfordCoreNLP
from sentistrength import PySentiStr

import nltk
from nltk.data import load
nltk.download('averaged_perceptron_tagger')
nltk.download("punkt")
nltk.download('tagsets')
from utils import read_non_emoji_tweets
from utils import get_label
from utils import print_class_stats
from utils import read_vocabulary, write_tokens_to_txt,write_dict_to_json,read_dict_from_json


# Define constant here
MIN_FREQ = 3
NUM_CLUSTER = 1000

### Part of Speech Implementation

def part_of_speech(data):
    '''
    Part of speech main function
    TODO: CMUTweetTagger from ARK-tweet-nlp has fewer class than nltk pos_tagger
    NOTE: I checked and TweeboParser requires running a server in the background
    are we sure we wan't this?
    Paiheng: I think it's the one mentioned in the paper. But let's skip it for now.

    # Parameters
    data : (Tweet : namedTuple) list

    # Return
    dict : 
        tweet_id : int -> {
            'tweet_tag_cnt' -> nparray : [45 x 1 vector in which each index represent the count of each tag in tagset],
            'tweet_tag_ratio' -> nparray : [45 x 1 vector in which each index represent the ratio of each tag in tagset],
            'tweet_lexical_density' -> float : tweet lexical density
        }
    
    '''

    # create a tagset dictionary of tag : str -> index : int so we can create np array
    # each tag is mapped to an index of our 45 x 1 dimension vector
    tagset = load('help/tagsets/upenn_tagset.pickle').keys()
    tagset_tag_to_index = dict(zip(tagset, [i for i in range(len(tagset))]))

    feature_dict = {}
    try:
        for tweet in data:

            # tokenize and tag tweet
            tokenized = tweet.tweet_words()
            tagged = nltk.pos_tag(tokenized)

            # count the absolute number of each tag
            tweet_tag_counter = Counter()
            for _, tag in tagged:
                tweet_tag_counter[tag] += 1

            # build np array for tweet tag counts
            tweet_tag_cnt = list(tweet_tag_counter.items())
            tweet_tag_cnt_vec = np.zeros((45,))
            for tag, cnt in tweet_tag_cnt:
                if tag in tagset:
                    tweet_tag_cnt_vec[tagset_tag_to_index[tag]] = cnt


            # build np array for tweet tag ratio
            tweet_tag_ratio = [(tag, tweet_tag_counter[tag] / len(tweet_tag_counter)) for tag in tweet_tag_counter]
            tweet_tag_ratio_vec = np.zeros((45,))
            for tag, ratio  in tweet_tag_ratio:
                if tag in tagset:
                    tweet_tag_ratio_vec[tagset_tag_to_index[tag]] = ratio

            # compute lexical density: the number of unique tokens divided by the total number of words.
            tweet_lexical_density = len(set(tokenized))/len(tokenized)

            feature_dict[tweet.tweet_id] = {
                'tweet_tag_cnt' : tweet_tag_cnt_vec,
                'tweet_tag_ratio' : tweet_tag_ratio_vec,
                'tweet_lexical_density' : tweet_lexical_density
            }

        # print(feature_dict)
        return feature_dict

    except Exception as e:
        print("In POS exceptions")
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

    # Parameters
    data : (Tweet : namedTuple) list

    # Return
    dict : 
        tweet_id : int -> {
            'tweet_no_vowel_cnt' -> int : number of tokens with no vowels,
            'tweet_three_more_syllables_cnt' -> int : number of tokens with more than 3 syllables
        }
    '''
    feature_dict = {}
    try:
        for tweet in data:

            # number of words with only alphabetic characters but no vowels
            tweet_no_vowel_cnt = 0

            #number of words with more than three syllables
            tweet_three_more_syllables_cnt = 0

            # tokenize and tag tweet
            tokenized = tweet.tweet_words()

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
        return feature_dict

    except Exception as e:
        print(str(e))

### Capitalization Implementation

def capitalization(data):
    '''
    Capitalization main function

    # Parameters
    data : (Tweet : namedTuple) list

    # Return
    dict : 
        tweet_id : int -> {
            'tweet_initial_cap_cnt' -> int : number of tokens with initial capitalization,
            'tweet_all_cap_cnt' -> int : number of tokens that are all cap,
            'tweet_tag_cap_cnt' -> int : number of tokens with tags that begin with capital letter
        }
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
            tokenized = tweet.tweet_words()

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
        # print(feature_dict)
        return feature_dict

    except Exception as e:
        print(str(e))


### Unigram and Bigram features

def construct_vocabulary(corpus, min_freq=MIN_FREQ):
    """
    construct vocabulary file first before extract ngrams
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
        with codecs.open(out_fn, 'w',encoding='utf8') as outf:
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
    input: whole dataset
    output: two dictionaries, key: tweet_id, value: 1-dimensional binary numpy array
    Done: Out of Vocabulary (OOV) words
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
        # +1 for OOV
        unigram_dict[_id] = np.zeros(len(unigram_vocab) + 1).tolist()
        bigram_dict[_id] = np.zeros(len(bigram_vocab) + 1).tolist()
        for idx, ele in enumerate(lower_tokens):
            # unigram
            unigram_dict[_id][unigram_vocab.get(ele, len(unigram_vocab))] = 1.

            if idx == len(lower_tokens) - 1:
                continue

            # bigram
            bigram_dict[_id][bigram_vocab.get((ele, lower_tokens[idx+1]), len(bigram_vocab))] = 1.

    return unigram_dict, bigram_dict


### brown cluster features

def read_brown_cluster(fn, min_freq):
    cluster_vocab = {}
    idx = 0
    int2idx = {}
    with codecs.open(fn, 'r',encoding='utf8') as inf:
        for line in inf:
            # set min freq
            cluster_bit, word, freq = line.strip().split('\t')
            if int(freq) < min_freq:
                continue
            cluster_int = int(cluster_bit, 2)
            if cluster_int not in int2idx:
                int2idx[cluster_int] = idx
                idx += 1
            cluster_vocab[word] = int2idx[cluster_int]
    return cluster_vocab


def brown_cluster_ngrams(corpus, num_cluster=NUM_CLUSTER, min_freq=MIN_FREQ):
    """
    input: whole dataset
    output: 2 dictionaries, key: tweet_id, value: 1-dimensional binary numpy array
    """
    # TODO: may need to adjust the num_cluster since we have smaller dataset/vocabulary size
    # TODO: check for spacy implementation later
    cluster_fn = 'baseline/brown_cluster_{}.txt'.format(num_cluster)
    if not os.path.exists(cluster_fn):
        print('brown cluster file not exist, run the repo first')
        write_tokens_to_txt(corpus, 'baseline/corpus_A.txt')
        sys.exit(1)

    cluster_vocab = read_brown_cluster(cluster_fn, min_freq=min_freq)

    unigram_dict = {}
    bigram_dict = {}
    for data in corpus:
        tokens = data.tweet_words()
        lower_tokens = [t.lower() for t in tokens]
        _id = data.tweet_id
        unigram_dict[_id] = np.zeros(len(cluster_vocab) + 1)
        bigram_dict[_id] = np.zeros(len(cluster_vocab) + 1)
        for idx, ele in enumerate(lower_tokens):
            # unigram
            unigram_dict[_id][cluster_vocab.get(ele, len(cluster_vocab))] = 1.
            bigram_dict[_id][cluster_vocab.get(ele, len(cluster_vocab))] = 1.

            if idx == len(lower_tokens) - 1:
                continue

            # bigram
            bigram_dict[_id][cluster_vocab.get(lower_tokens[idx + 1], len(cluster_vocab))] = 1.

    return unigram_dict, bigram_dict


def dependency(corpus, num_cluster=NUM_CLUSTER, min_freq=MIN_FREQ):
    """
    input: whole dataset
    output: two dictionaries, key: tweet_id, value: 1-dimensional binary numpy array
    """
    sorted_corpus = sorted(corpus, key=lambda tweet: tweet.tweet_id)

    depend_fn = 'baseline/dependency_A.txt.predict'
    if not os.path.exists(depend_fn):
        print('dependency parser file not exist, run the repo first')
        sys.exit(1)

    # reading vocab
    cluster_fn = 'baseline/brown_cluster_{}.txt'.format(num_cluster)
    if not os.path.exists(cluster_fn):
        print('brown cluster file not exist, run the repo first')
        sys.exit(1)

    if not os.path.exists('baseline/unigram_vocab.txt'):
        construct_vocabulary(corpus)

    cluster_vocab = read_brown_cluster(cluster_fn, min_freq)
    unigram_vocab = read_vocabulary('baseline/unigram_vocab.txt')

    # dict to store the features
    word_dict = {}
    cluster_dict = {}

    # NOTE: this requires tweets are sorted from 1 to n when passing to the txt file
    idx = 1
    with open(depend_fn, 'r') as inf:
        word_tmp = np.zeros((len(unigram_vocab) + 1, len(unigram_vocab) + 1))
        cluster_tmp = np.zeros((len(cluster_vocab) + 1, len(cluster_vocab) + 1))
        valid_arc = {}
        tweet_word_dict = {}
        tweet_tokens = []
        for line in inf:
            if line.strip():
                word_idx, word, _,  tag, tag, _, arc_idx, _ = line.split('\t')
                tweet_word_dict[word_idx] = word
                tweet_tokens.append(word)
                try:
                    int_arc_idx = int(arc_idx)
                except TypeError as err:
                    int_arc_idx = -1
                if int_arc_idx > 0:
                    valid_arc[word_idx] = arc_idx
            else:
                # tweets are separated by a empty line

                # There might be some exceptions due to space
                # lower_tweet_words = [t.lower() for t in sorted_corpus[idx-1].tweet_words()]
                # assert tweet_tokens == lower_tweet_words, \
                #     ' '.join(tweet_tokens) + '\n' + ' '.join(lower_tweet_words)
                # when encounter a empty line, summary and store last chunks, init. for next chunk
                # summary
                for k, v in valid_arc.items():
                    dim1_word, dim2_word = tweet_word_dict[k], tweet_word_dict[v]
                    word_tmp[unigram_vocab.get(dim1_word, len(unigram_vocab)), unigram_vocab.get(dim2_word, len(unigram_vocab))] = 1.
                    cluster_tmp[cluster_vocab.get(dim1_word, len(cluster_vocab)), cluster_vocab.get(dim2_word, len(cluster_vocab))] = 1.

                # flatten
                # TODO: sparse representation needed
                word_dict[idx] = word_tmp.flatten()
                cluster_dict[idx] = cluster_tmp.flatten()

                # init for next chunk
                # plus 1 for OOV
                valid_arc = {}
                tweet_word_dict = {}
                tweet_tokens = []
                word_tmp = np.zeros((len(unigram_vocab) + 1, len(unigram_vocab) + 1))
                cluster_tmp = np.zeros((len(cluster_vocab) + 1, len(cluster_vocab) + 1))
                idx += 1
                if idx % int(len(corpus)*0.1) == 0:
                    print(idx / int(len(corpus)*0.1))

    return word_dict, cluster_dict


#sentiment features

def tweet_whole_sentiment(data):
    '''
    input: whole corpus
    output: 1 dicts for tweet_whole_sentiment, 
            keys: tweet_id, values: sentimentValues (1--Positive,2--Neutral,3--Negative
    '''
    try:
        nlp_wrapper = StanfordCoreNLP('http://localhost:5000')
        feature_dict={}
        for tweet in data:
            tokenized= tweet.tweet_words()
            new_words= [word for word in tokenized if word.isalnum()]
            if not new_words:
                feature_dict[tweet.tweet_id] = 2
            text=" ".join(new_words)
            annotate=nlp_wrapper.annotate(text,properties={
                'annotators': 'sentiment',
                'outputFormat': 'json',
                'timeout': 10000,})
            for sentence in annotate["sentences"]:
                feature_dict[tweet.tweet_id]=sentence["sentimentValue"]
        # print(feature_dict)
        return feature_dict
    except Exception as e:
        print("In whole sentiment exception")
        print(str(e))
    
    
def tweet_word_sentiment(data):
    '''
    input: whole corpus
    output: 1 dicts for tweet_word_sentiment, 
            keys: tweet_id, values: dict (keys={"max","min","distance"})
                                    max--highest sentiment score among all words
                                    min--lowest sentiment score among all words
                                    distance-- difference between highest score and lowest score
    '''
    feature_dict={}
#     try:
    senti = PySentiStr()
    senti.setSentiStrengthPath('./SentiStrength.jar')
    senti.setSentiStrengthLanguageFolderPath('./SentiStrengthData/')

    for tweet in data:
        tokenized= tweet.tweet_words()
        new_words= [word for word in tokenized if word.isalnum()]
        if not new_words:
            feature_dict[tweet.tweet_id]={"max":0,"min":0,"distance":0}
            continue
        result = senti.getSentiment(new_words)
        max_,min_=result[0],result[0]
        for score in result:
            max_=max(max_,score)
            min_=min(min_,score)
        feature_dict[tweet.tweet_id]={"max":max_,"min":min_,"distance":max_-min_}
    return feature_dict

#     except Exception as e:
#         print("In word sentiment exception")
#         print(str(e))
    
    
def intensifier(data):
    '''
    input: whole corpus
    output: 1 dict for intensifier feature, 
            keys: tweet_id, values: 1,0 for containing an intensifier or not
    '''
    
    
    file=open("intensifier.txt")
    intense=set([x.rstrip("\n") for x in file.readlines()])
    feature_dict={}
    for tweet in data:
            tokenized= tweet.tweet_words()
            for word in tokenized:
                if word in intense:
                    feature_dict[tweet.tweet_id]=1
                    break
            if not tweet.tweet_id in feature_dict:
                feature_dict[tweet.tweet_id]=0      
    return feature_dict


def emoji_senti_eval(data):
    '''
    input: takes tweet's emojis.
    each emoji's negative, neutral,positive scores are extracted
    we average for all emojis in tweet these values
    :return: dict with keys: tweet_id, values: [all_emojis_avg_negative_prob, all_emojis_avg_neutral_prob, \
                       all_emojis_avg_positive_prob]
    '''
    emoji_senti_file = 'emoji_senti_data/Emoji_Sentiment_Data_v1.0.csv'
    df = pd.read_csv(emoji_senti_file)
    emoji_corpus={}
    for name, total, neg, neut,pos in zip(df['Unicode name'],df['Occurrences'], \
                                          df['Negative'],df['Neutral'],df['Positive']):
        emoji_corpus[name.lower()]=[neg/total,neut/total,pos/total]

    feature_dict={}
    for tweet in data:
        all_emojis=tweet.tweet_emojis
        count=len(all_emojis)
        values=[0,0,0]
        if count>0:
            for e in all_emojis:
                score=emoji_corpus.get(e,None)
                if score is not None:
                    score=np.array(score)
                    values+=score
            values=[x/count for x in values]
        feature_dict[tweet.tweet_id]=values
    return feature_dict




   



def get_features(data,generate,data_name):
    # unit test for ngrams
    if generate:
        unigram_feature, bigram_feature = extract_ngrams(data)
        file=data_name+"_unigram_feature.json"
        write_dict_to_json(unigram_feature,file)
        file = data_name + "_bigram_feature.json"
        write_dict_to_json(bigram_feature,file)
    else:
        file = data_name + "_unigram_feature.json"
        unigram_feature=read_dict_from_json(file)
        file = data_name + "_bigram_feature.json"
        bigram_feature=read_dict_from_json(file)


    print("1. Ngrams generated")
    print(f'Size of unigram={len(unigram_feature)} x {len(unigram_feature[1])}')
    print(f'Size of bigram={len(bigram_feature)} x {len(bigram_feature[1])}')


    # unit test for part of speech
    pos_dict=part_of_speech(data)
    print("2. POS Tagging done")
    print(f'Len of pos ={len(pos_dict)} x 3')



    # unit test for prounciation
    pronounce_dict= pronunciations(data)
    print("3. Pronounciation done")
    print(len(pronounce_dict))


    # unit test for capitalization
    caps=capitalization(data)
    print("4. CAPS done")
    print(len(caps))

    sent_senti=tweet_whole_sentiment(data)
    print("5.Sentence Sentiment done")
    print(len(sent_senti))

    word_senti=tweet_word_sentiment(data)
    print("6. Words sentiment done")
    print(len(word_senti))

    unigram_brown_feature, bigram_brown_feature = brown_cluster_ngrams(data)
    print("7.After Brown")
    print(f'Size of brown unigram={len(unigram_brown_feature)} x {len(unigram_brown_feature[1])}')
    print(f'Size of brown bigram={len(bigram_brown_feature)} x {len(bigram_brown_feature[1])}')

    emoji_senti=emoji_senti_eval(data)
    print("8. After emoji senti eval")
    print(len(emoji_senti))

    
    
    
    Vectors=[]
    for t in data:
        vec=[]
        vec.extend(unigram_feature.get(t.tweet_id))
        vec.extend(bigram_feature.get(t.tweet_id))

        vec.extend(pos_dict[t.tweet_id]['tweet_tag_cnt'])
        vec.extend(pos_dict[t.tweet_id]['tweet_tag_ratio'])
        vec.append(pos_dict[t.tweet_id]['tweet_lexical_density'])

        vec.append(pronounce_dict[t.tweet_id]['tweet_no_vowel_cnt'])
        vec.append(pronounce_dict[t.tweet_id]['tweet_three_more_syllables_cnt'])

        vec.append(caps[t.tweet_id]['tweet_initial_cap_cnt'])
        vec.append(caps[t.tweet_id]['tweet_all_cap_cnt'])
        vec.append(caps[t.tweet_id]['tweet_tag_cap_cnt'])

        vec.append(sent_senti[t.tweet_id])

        vec.append(word_senti[t.tweet_id]['max'])
        vec.append(word_senti[t.tweet_id]['min'])
        vec.append(word_senti[t.tweet_id]['distance'])


        vec.extend(unigram_brown_feature[t.tweet_id])
        vec.extend(bigram_brown_feature[t.tweet_id])

        vec.extend(emoji_senti[t.tweet_id])

        Vectors.append(vec)


    print(len(Vectors),len(Vectors[0]))
    return Vectors


def save_features(vectors,fn):
    #input vectors: feature vectors, fn:filename
    df=pd.DataFrame(vectors)
    df.to_csv(fn)
    return


def read_features(fp):
    print("In Read features from CSV")
    #input vectors: feature vectors, fn:filename
    df = pd.read_csv(fp, sep=',',dtype=np.float64)
    data=df.values.tolist()
    # print(len(data))
    # print(len(data[0]))
    # print(data[0])
    return data


def featurize(generate):
    # File paths from project level
    # fp_train_A = 'tweet_irony_detection/train/SemEval2018-T3-train-taskA.txt'
    fp_train_A = 'train/SemEval2018-T3-train-taskA.txt'
    fp_train_B = 'train/SemEval2018-T3-train-taskB.txt'
    fp_test_A = 'test_TaskA/SemEval2018-T3_input_test_taskA.txt'
    fp_test_B = 'test_TaskB/SemEval2018-T3_input_test_taskB.txt'
    fp_labels_A = 'goldtest_TaskA/SemEval2018-T3_gold_test_taskA_emoji.txt'
    fp_labels_B = 'goldtest_TaskB/SemEval2018-T3_gold_test_taskB_emoji.txt'


    # Training data for task A and B , test data & correct labels for both tasks
    pre_process_url = True  # Set to remove URLs
    pre_process_usr = True
    train_A = read_non_emoji_tweets(fp_train_A, "train", pre_process_url, pre_process_usr)
    train_B = read_non_emoji_tweets(fp_train_B, "train", pre_process_url, pre_process_usr)
    tr_labels_A = [t.tweet_label for t in train_A]
    tr_label_B = [t.tweet_label for t in train_B]

    test_A = read_non_emoji_tweets(fp_test_A, "test", pre_process_url, pre_process_usr)
    test_B = read_non_emoji_tweets(fp_test_B, "test", pre_process_url, pre_process_usr)
    gold_A = get_label(fp_labels_A)
    tst_labels_A = [v for k,v in gold_A.items()]
    gold_B = get_label(fp_labels_B)
    tst_labels_B = [v for k,v in gold_B.items()]

    # Print class stats

    print_class_stats(train_A, train_B, gold_A, gold_B)


    # Read features from files
    # if not generate:
    #     feats_tr_A = read_features("feats_tr_A.csv")
    #     feats_tst_A= read_features("feats_tst_A.csv")
    #     feats_tr_B= read_features("feats_tr_B.csv")
    #     feats_tst_B=read_features("feats_tst_B.csv")
    #     return feats_tr_A, feats_tst_A, feats_tr_B, feats_tst_B, tr_labels_A, tr_label_B, tst_labels_A, tst_labels_B

    # Generate features
    feats_tr_A = get_features(train_A,generate,'train_A')
    feats_tst_A = get_features(test_A,generate,'test_A')
    feats_tr_B=get_features(train_B,generate,'train_B') # Same as A's features
    feats_tst_B=get_features(test_B,generate,'test_B') # Same as A's features


    # save_features(feats_tr_A,"feats_tr_A.csv")
    # save_features(feats_tst_A,"feats_tst_A.csv")
    # save_features(feats_tr_B,"feats_tr_B.csv")
    # save_features(feats_tst_B,"feats_tst_B.csv")
    return feats_tr_A,feats_tst_A,feats_tr_B,feats_tst_B,tr_labels_A,tr_label_B,tst_labels_A,tst_labels_B


if __name__ == '__main__':
    generate=True
    featurize(generate)

