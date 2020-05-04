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
from utils import *
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import classification_report, confusion_matrix

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

def vectorize(feature_dict):
    vecs=[]
    for k in list(feature_dict.keys()):
        vec=[]
        f=feature_dict.get(k)
        vec.extend(f)
        vecs.append(vec)
    return vecs

def get_bert_feature(generate):
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
        train_A_feature = get_bert_embeddings(train_A)
        write_dict_to_json(train_A_feature, fn='train_A_bert.json.gz')

        train_B_feature = get_bert_embeddings(train_B)
        write_dict_to_json(train_B_feature, fn='train_B_bert.json.gz')

        test_A_feature = get_bert_embeddings(test_A)
        write_dict_to_json(test_A_feature, fn='test_A_bert.json.gz')

        test_B_feature = get_bert_embeddings(test_B)
        write_dict_to_json(test_B_feature, fn='test_B_bert.json.gz')

    ### test the readability
    else:
        train_A_feature = read_dict_from_json(fn='train_A_bert.json.gz')
        train_B_feature =read_dict_from_json(fn='train_B_bert.json.gz')
        test_A_feature =read_dict_from_json(fn='test_A_bert.json.gz')
        test_B_feature =read_dict_from_json(fn='test_B_bert.json.gz')

    feats_tr_A = vectorize(train_A_feature)
    feats_tr_B = vectorize(train_B_feature)
    feats_tst_A = vectorize(test_A_feature)
    feats_tst_B = vectorize(test_B_feature)

    print(len(feats_tr_A), len(feats_tr_A[1]))
    print(len(feats_tst_A), len(feats_tst_A[1]))
    print(len(feats_tr_B), len(feats_tr_B[1]))
    print(len(feats_tst_B), len(feats_tst_B[1]))

    return feats_tr_A, feats_tst_A, feats_tr_B, feats_tst_B, tr_labels_A, tr_label_B, tst_labels_A, tst_labels_B


def fit_test_model(train, train_label, test, test_label, model):
    model.fit(train, train_label)
    # Predict
    # p_pred = model.predict_proba(feats_tst_A)
    # Metrics
    y_pred = model.predict(test)
    score_ = model.score(test, test_label)
    conf_m = confusion_matrix(test_label, y_pred)
    report = classification_report(test_label, y_pred,output_dict=True)

    # print('score_:', score_, end='\n\n')
    # print('conf_m:', conf_m, sep='\n', end='\n\n')
    # print('report:', report, sep='\n')



    print(f"Accuracy={report['accuracy']:.4},Precision={report['weighted avg']['precision']:.4}," \
          f"Recall={report['weighted avg']['recall']:.4},f1-score={report['weighted avg']['f1-score']:.4}")

if __name__ == '__main__':
    ### generate feature files for all dataset
    generate=False
    feats_tr_A, feats_tst_A, feats_tr_B, feats_tst_B, tr_labels_A, tr_labels_B, tst_labels_A, tst_labels_B=get_bert_feature(generate)

    # task A
    print("==============TASK A======================")
    model = LogisticRegression(solver='liblinear', penalty='l2', random_state=0)
    fit_test_model(train=feats_tr_A, train_label=tr_labels_A, test=feats_tst_A, test_label=tst_labels_A,
                   model=model)

    # task B
    print("==============TASK B======================")
    model2 = LogisticRegression(solver='liblinear', penalty='l2', random_state=0)
    fit_test_model(train=feats_tr_B, train_label=tr_labels_B, test=feats_tst_B, test_label=tst_labels_B,
                   model=model2)

