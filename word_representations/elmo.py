import warnings
warnings.filterwarnings('ignore')

import numpy as np

from utils import *

# word embedding libraries
from allennlp.commands.elmo import ElmoEmbedder
import sister


def elmo_embedding(data):
    '''
    Get elmo word embeddings and convert into feature
    # Parameters
    data : (Tweet : namedTuple) list

    # Return
    dict : 
        tweet_id : int -> {
            'context_independent_layer' : [context_independent_layer_min, context_independent_layer_max, context_independent_layer_mean] (3072 x 1),
            'LSTM_layer1' : [LSTM_layer1_min, LSTM_layer1_max, LSTM_layer1_mean] (3072 x 1),
            'LSTM_layer2' : [LSTM_layer2_min, LSTM_layer2_max, LSTM_layer2_mean] (3072 x 1)
        }
    
    '''
    feature_dict = {}
    try:
        print('Creating ELMO Embedder...')
        
        elmo = ElmoEmbedder(
            options_file='https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway_5.5B/elmo_2x4096_512_2048cnn_2xhighway_5.5B_options.json', 
            weight_file='https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway_5.5B/elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5'
            )
        
        print('Embedding Tweets...')
        
        N = len(data)
        
        for i in range(3):
            tweet = data[i]

            # tokenize and tag tweet
            tokenized = tweet.tweet_words()
            vectors = elmo.embed_sentence(tokenized)
            
            assert(len(vectors) == 3)
            assert(len(vectors[0]) == len(tokenized))
            
            context_independent_layer = vectors[0]
            LSTM_layer1 = vectors[1]
            LSTM_layer2 = vectors[2]
            
            # One simple technique that seems to work reasonably well for short texts (e.g., a sentence or a tweet) is to compute 
            # the vector for each word in the document, and then aggregate them using the coordinate-wise mean, min, or max.
            # 
            # Reference:
            # 
            # Representation learning for very short texts using weighted word embedding aggregation. Cedric De Boom, Steven Van Canneyt, 
            # Thomas Demeester, Bart Dhoedt. Pattern Recognition Letters; arxiv:1607.00570. abstract, pdf. See especially Tables 1 and 2.
            
            context_independent_layer_min = context_independent_layer.min(axis=0)
            context_independent_layer_max = context_independent_layer.max(axis=0)
            context_independent_layer_mean = context_independent_layer.mean(axis=0)
            
            LSTM_layer1_min = LSTM_layer1.min(axis=0)
            LSTM_layer1_max = LSTM_layer1.max(axis=0)
            LSTM_layer1_mean = LSTM_layer1.mean(axis=0)
            
            LSTM_layer2_min = LSTM_layer2.min(axis=0)
            LSTM_layer2_max = LSTM_layer2.max(axis=0)
            LSTM_layer2_mean = LSTM_layer2.mean(axis=0)

            print('{}/{} tweets embedded'.format(i + 1, N))

            feature_dict[tweet.tweet_id] = {
                'context_independent_layer' : np.concatenate([context_independent_layer_min, context_independent_layer_max, context_independent_layer_mean]),
                'LSTM_layer1' : np.concatenate([LSTM_layer1_min, LSTM_layer1_max, LSTM_layer1_mean]),
                'LSTM_layer2' : np.concatenate([LSTM_layer2_min, LSTM_layer2_max, LSTM_layer2_mean])
            }
            
        return feature_dict

    except Exception as e:
        print("In ELMO exceptions")
        print(str(e))


# def fast_text_embedding(data):
#     '''
#     data : (Tweet : namedTuple) list

#     # Return
#     dict : 
#         tweet_id : int -> nparray : [300 x 1]
    
#     '''
#     feature_dict = {}
#     try:
        
#         for tweet in data:

#             # tokenize and tag tweet
#             tokenized = tweet.tweet_words()
#             embedder = sister.MeanEmbedding(lang="en")
#             vector = embedder(' '.join(tokenized))
#             feature_dict[tweet.tweet_id] = vector
            
#         return feature_dict

#     except Exception as e:
#         print("In Fast Text Exceptions")
#         print(str(e))
        
def vectorize(feture_dict):
    vecs=[]
    for f in feture_dict:
        vec=[]
        vec.extend(f.get('context_independent_layer'))
        vec.extend(f.get('LSTM_layer1'))
        vec.extend(f.get('LSTM_layer2'))
        vecs.append(vec)
    return vecs


def get_elmo_features(generate):
    fp_train_A = 'train/SemEval2018-T3-train-taskA.txt'
    fp_train_B = 'train/SemEval2018-T3-train-taskB.txt'
    fp_test_A = 'test_TaskA/SemEval2018-T3_input_test_taskA.txt'
    fp_test_B = 'test_TaskB/SemEval2018-T3_input_test_taskB.txt'
    fp_labels_A = 'goldtest_TaskA/SemEval2018-T3_gold_test_taskA_emoji.txt'
    fp_labels_B = 'goldtest_TaskB/SemEval2018-T3_gold_test_taskB_emoji.txt'

    pre_process_url = True  # Set to remove URLs
    pre_process_usr = True
    train_A = read_non_emoji_tweets(fp_train_A, "train", pre_process_url, pre_process_usr)
    train_B = read_non_emoji_tweets(fp_train_B, "train", pre_process_url, pre_process_usr)
    tr_labels_A = [t.tweet_label for t in train_A]
    tr_label_B = [t.tweet_label for t in train_B]

    test_A = read_non_emoji_tweets(fp_test_A, "test", pre_process_url, pre_process_usr)
    gold_A = get_label(fp_labels_A)
    tst_labels_A = [v for k, v in gold_A.items()]
    test_B = read_non_emoji_tweets(fp_test_B, "test", pre_process_url, pre_process_usr)
    gold_B = get_label(fp_labels_B)
    tst_labels_B = [v for k, v in gold_B.items()]



    if generate:

        train_A_elmo = elmo_embedding(train_A)
        write_dict_to_json(train_A_elmo, 'train_A_elmo.json')

        train_B_elmo = elmo_embedding(train_B)
        write_dict_to_json(train_B_elmo, 'train_B_elmo.json')

        test_A_elmo = elmo_embedding(test_A)
        write_dict_to_json(test_A_elmo, 'test_A_elmo.json')

        test_B_elmo = elmo_embedding(test_B)
        write_dict_to_json(test_B_elmo, 'test_B_elmo.json')


    else:
        train_A_elmo =read_dict_from_json('train_A_elmo.json')
        train_B_elmo =read_dict_from_json('train_B_elmo.json')
        test_A_elmo =read_dict_from_json('test_A_elmo.json')
        test_B_elmo =read_dict_from_json('test_B_elmo.json')

    feats_tr_A=vectorize(train_A_elmo)
    feats_tr_B=vectorize(train_B_elmo)
    feats_tst_A=vectorize(test_A_elmo)
    feats_tst_B=vectorize(test_B_elmo)

    return feats_tr_A, feats_tst_A, feats_tr_B, feats_tst_B, tr_labels_A, tr_label_B, tst_labels_A, tst_labels_B



if __name__ == '__main__':
    generate=False;
    get_elmo_features(generate)
    

    
    