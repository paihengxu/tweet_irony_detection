from utils import *

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix
from utils import read_dict_from_json
import os


def vectorize(ids,data_name,feature_list,path):
    vecs=[]
    feature_dicts={}
    for feature in feature_list:
        feature_dicts[feature]=read_dict_from_json(path+data_name+feature+'.json')
    for id in ids:
        vec=[]
        for feature in feature_list:
            vec.extend(feature_dicts[feature][str(id)])
                        
        vecs.append(vec)
    return vecs


def get_hybrid_features(path):
    fp_train_A = 'train/SemEval2018-T3-train-taskA.txt'
    fp_train_B = 'train/SemEval2018-T3-train-taskB.txt'
    fp_test_A = 'test_TaskA/SemEval2018-T3_input_test_taskA.txt'
    fp_test_B = 'test_TaskB/SemEval2018-T3_input_test_taskB.txt'
    fp_labels_A = 'goldtest_TaskA/SemEval2018-T3_gold_test_taskA_emoji.txt'
    fp_labels_B = 'goldtest_TaskB/SemEval2018-T3_gold_test_taskB_emoji.txt'
    
    
    data_names=['train_A','test_A','train_B','test_B']
    feature_list=['_unigram_feature','_unigram_feature', '_part_of_speech','_pronounciation','_capitalization','_tweet_whole_sentiment',
                  '_word_sentiment','_unigram_brown_feature','_bigram_brown_feature','_emoji_sentiment']
    
    pre_process_url = True  # Set to remove URLs
    pre_process_usr = True
    train_A = read_non_emoji_tweets(fp_train_A, "train", pre_process_url, pre_process_usr)
    train_B = read_non_emoji_tweets(fp_train_B, "train", pre_process_url, pre_process_usr)
    tr_A_ids=[t.tweet_id for t in train_A]
    tr_B_ids = [t.tweet_id for t in train_B]
    tr_labels_A = [t.tweet_label for t in train_A]
    tr_label_B = [t.tweet_label for t in train_B]
    
    test_A = read_non_emoji_tweets(fp_test_A, "test", pre_process_url, pre_process_usr)
    gold_A = get_label(fp_labels_A)
    tst_labels_A = [v for k, v in gold_A.items()]
    test_B = read_non_emoji_tweets(fp_test_B, "test", pre_process_url, pre_process_usr)
    gold_B = get_label(fp_labels_B)
    tst_labels_B = [v for k, v in gold_B.items()]
    tst_A_ids = [t.tweet_id for t in test_A]
    tst_B_ids = [t.tweet_id for t in test_B]
    
    feats_tr_A=vectorize(tr_A_ids,'train_A',feature_list,path)
    feats_tr_B = vectorize(tr_B_ids,'train_B',feature_list,path)
    feats_tst_A = vectorize(tst_A_ids,'test_A',feature_list,path)
    feats_tst_B = vectorize(tst_B_ids,'test_B',feature_list,path)
        
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
    report = classification_report(test_label, y_pred)

    print('score_:', score_, end='\n\n')
    print('conf_m:', conf_m, sep='\n', end='\n\n')
    print('report:', report, sep='\n')

if __name__ == '__main__':
    path='./obtained_features/baselines/'
    feats_tr_A, feats_tst_A, feats_tr_B, feats_tst_B, tr_labels_A, tr_labels_B, tst_labels_A, tst_labels_B =get_hybrid_features(path)
    
    model1 = LogisticRegression(solver='liblinear', penalty='l2', random_state=0)
    model2 = GradientBoostingClassifier(n_estimators=500)
    
    # task A
    print("==============TASK A======================")
    fit_test_model(train=feats_tr_A, train_label=tr_labels_A, test=feats_tst_A, test_label=tst_labels_A,
                   model=model1)
    fit_test_model(train=feats_tr_A, train_label=tr_labels_A, test=feats_tst_A, test_label=tst_labels_A,
                   model=model2)

    # task B
    print("==============TASK B======================")
    model2 = LogisticRegression(solver='liblinear', penalty='l2', random_state=0)
    fit_test_model(train=feats_tr_B, train_label=tr_labels_B, test=feats_tst_B, test_label=tst_labels_B,
                   model=model1)
    fit_test_model(train=feats_tr_B, train_label=tr_labels_B, test=feats_tst_B, test_label=tst_labels_B,
                   model=model2)