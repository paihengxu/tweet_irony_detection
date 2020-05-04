import os
import numpy as np
import pandas as pd
# import re
# from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
# from sklearn.model_selection import train_test_split
import torch
# from keras.preprocessing.sequence import pad_sequences
# from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
# from tqdm import tqdm
from pytorch_transformers import *
from utils import read_non_emoji_tweets
from utils import write_dict_to_json, read_dict_from_json

# TODO: fine-tuning BERT on Twitter data we have

def data_processing(fn):
    df = pd.read_csv(fn, delimiter='\t')
    # test_df = pd.read_csv(os.path.join(directory_path, 'testData.tsv'), delimiter='\t')

    # preprocess
    # lm_df = pd.concat([train_df[['Tweet text']], test_df[['Tweet text']]])
    df['text'] = df['Tweet text'].str.lower()
    sentences = df.text.values
    sentences = ["[CLS] " + sentence + " [SEP]" for sentence in sentences]
    # labels = df.Label.values

    return sentences


def get_bert_embeddings(corpus):
    '''
    input: file name for the dataset
    '''

    # init pretrained model
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    bert_embed_features = {}
    for tweet in corpus:
        sent = tweet.tweet_text
        input_ids = torch.tensor(tokenizer.encode(sent, add_special_tokens=True)).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids)
        # NOTE: vector for last layer [cls] token. Since it's used for classification task
        bert_embed_features[tweet.tweet_id] = outputs[0][0][0].detach().numpy().tolist()
        # print('0', outputs[0][0][0].shape)
        # print('1', outputs[1].shape)
    return bert_embed_features


if __name__ == '__main__':
    ### generate feature files for all dataset
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


    train_A_feature = get_bert_embeddings(train_A)
    write_dict_to_json(train_A_feature, fn='train_A_bert.json.gz')

    train_B_feature = get_bert_embeddings(train_B)
    write_dict_to_json(train_B_feature, fn='train_B_bert.json.gz')

    test_A_feature = get_bert_embeddings(test_A)
    write_dict_to_json(test_A_feature, fn='test_A_bert.json.gz')

    test_B_feature = get_bert_embeddings(test_B)
    write_dict_to_json(test_B_feature, fn='test_B_bert.json.gz')

    ### test the readability
    featurse = read_dict_from_json(fn='train_A_bert.json.gz')
    print(len(featurse["1"]))

