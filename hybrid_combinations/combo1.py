from utils import *

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix



def vectorize(ids,bert_dict,elmo_dict):
    vecs=[]
    for id in ids:
        vec=[]
        # Append bert
        f = bert_dict.get(str(id))
        vec.extend(f)

        # Append elmo
        el = elmo_dict.get(str(id))
        vec.extend(el.get('context_independent_layer'))
        vec.extend(el.get('LSTM_layer1'))
        vec.extend(el.get('LSTM_layer2'))

        vecs.append(vec)
    return vecs



def get_hybrid_features():

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


    #dict[tweet_id]= list
    train_A_bert = read_dict_from_json(fn='train_A_bert.json.gz')
    train_B_bert = read_dict_from_json(fn='train_B_bert.json.gz')
    test_A_bert = read_dict_from_json(fn='test_A_bert.json.gz')
    test_B_bert = read_dict_from_json(fn='test_B_bert.json.gz')

    '''
    elmo_feature_dict[tweet.tweet_id] = {
                'context_independent_layer' : np.concatenate([context_independent_layer_min, context_independent_layer_max, context_independent_layer_mean]),
                'LSTM_layer1' : np.concatenate([LSTM_layer1_min, LSTM_layer1_max, LSTM_layer1_mean]),
                'LSTM_layer2' : np.concatenate([LSTM_layer2_min, LSTM_layer2_max, LSTM_layer2_mean])
            }
    '''

    train_A_elmo = read_dict_from_json('train_A_elmo.json.gz')
    train_B_elmo = read_dict_from_json('train_B_elmo.json.gz')
    test_A_elmo = read_dict_from_json('test_A_elmo.json.gz')
    test_B_elmo = read_dict_from_json('test_B_elmo.json.gz')


    # train_A_ = read_dict_from_json('train_A_.json.gz')
    # train_B_ = read_dict_from_json('train_B_.json.gz')
    # test_A_ = read_dict_from_json('test_A_.json.gz')
    # test_B_ = read_dict_from_json('test_B_.json.gz')



    feats_tr_A=vectorize(tr_A_ids,train_A_bert,train_A_elmo)
    feats_tr_B = vectorize(tr_B_ids,train_B_bert,train_B_elmo)
    feats_tst_A = vectorize(tst_A_ids,test_A_bert,test_A_elmo)
    feats_tst_B = vectorize(tst_B_ids,test_B_bert,test_B_elmo)

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
    # print('report:', str(report), sep='\n')

    print(f"Accuracy={report['accuracy']:.4},Precision={report['weighted avg']['precision']:.4}," \
          f"Recall={report['weighted avg']['recall']:.4},f1-score={report['weighted avg']['f1-score']:.4}")

if __name__ == '__main__':
    feats_tr_A, feats_tst_A, feats_tr_B, feats_tst_B, tr_labels_A, tr_labels_B, tst_labels_A, tst_labels_B =get_hybrid_features()

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