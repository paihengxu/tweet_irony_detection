# importing all necessary modules 
from nltk.tokenize import sent_tokenize, word_tokenize 
import warnings 
  
warnings.filterwarnings(action = 'ignore')

import numpy as np

from utils import read_non_emoji_tweets
from utils import write_dict_to_json
  
import gensim 
from gensim.models import Word2Vec


def create_corpus(data):
    corpus = []
    for tweet in data:
        tokenized = tweet.tweet_words()
        corpus.append(tokenized)
    return corpus


def cbow_embedding(data):
    '''
    data : (Tweet : namedTuple) list

    # Return
    dict : 
        tweet_id : int -> nparray : [300 x 1]
    
    '''
    feature_dict = {}
    try:
        corpus = create_corpus(data)
        
        # Create Skip Gram model 
        model = gensim.models.Word2Vec(corpus, min_count = 1, size = 100, window = 5) 
        
        for tweet in data:

            # tokenize and tag tweet
            tokenized = tweet.tweet_words()
            
            embedding_min = model[tokenized].min(axis=0)
            embedding_max = model[tokenized].max(axis=0)
            embedding_mean = model[tokenized].mean(axis=0)
            
            feature_dict[tweet.tweet_id] = np.concatenate([embedding_min, embedding_max, embedding_mean])
            
        return feature_dict

    except Exception as e:
        print("In CBOW Exceptions")
        print(str(e))


if __name__ == '__main__':
    
    fp_train_A = 'train/SemEval2018-T3-train-taskA.txt'
    fp_train_B = 'train/SemEval2018-T3-train-taskB.txt'
    fp_test_A = 'test_TaskA/SemEval2018-T3_input_test_taskA.txt'
    fp_test_B = 'test_TaskB/SemEval2018-T3_input_test_taskB.txt'
    
    pre_process_url = True  # Set to remove URLs
    pre_process_usr = True
    train_A = read_non_emoji_tweets(fp_train_A, "train", pre_process_url, pre_process_usr)
    train_B = read_non_emoji_tweets(fp_train_B, "train", pre_process_url, pre_process_usr)
    
    test_A = read_non_emoji_tweets(fp_test_A, "test", pre_process_url, pre_process_usr)
    test_B = read_non_emoji_tweets(fp_test_B, "test", pre_process_url, pre_process_usr)
    
    train_A_cbow = cbow_embedding(train_A)
    write_dict_to_json(train_A_cbow, 'train_A_cbow.json.gz')
    
    train_B_cbow = cbow_embedding(train_B)
    write_dict_to_json(train_B_cbow, 'train_B_cbow.json.gz')
    
    test_A_cbow = cbow_embedding(test_A)
    write_dict_to_json(test_A_cbow, 'test_A_cbow.json.gz')
    
    test_B_cbow = cbow_embedding(test_B)
    write_dict_to_json(test_B_cbow, 'test_B_cbow.json.gz')