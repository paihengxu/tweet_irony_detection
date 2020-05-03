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

# TODO: fine-tuning BERT on Twitter data we have

def data_processing(fn):
    df = pd.read_csv(fn, delimiter='\t')
    # test_df = pd.read_csv(os.path.join(directory_path, 'testData.tsv'), delimiter='\t')

    # preprocess
    # lm_df = pd.concat([train_df[['Tweet text']], test_df[['Tweet text']]])
    df['text'] = df['Tweet text'].str.lower()
    sentences = df.text.values
    sentences = ["[CLS] " + sentence + " [SEP]" for sentence in sentences]
    labels = df.Label.values

    return sentences, labels


def get_bert_embeddings(fn):
    sentences, labels = data_processing(fn)

    # init pretrained model
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    bert_embed_features = {}
    for idx, sent in enumerate(sentences):
        input_ids = torch.tensor(tokenizer.encode(sent)).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids)
        # NOTE: vector for last layer [cls] token. Since it's used for classification task
        bert_embed_features[idx+1] = outputs[0][0][0].detach().numpy()
        # print('0', outputs[0][0][0].shape)
        # print('1', outputs[1].shape)
    return bert_embed_features


if __name__ == '__main__':
    feature = get_bert_embeddings(fn='train/SemEval2018-T3-train-taskA.txt')
    print(type(feature[1]))
    print(len(feature[1]))

