from utils import *

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV


# def vectorize(ids,data_name,feature_list,path):
def vectorize(ids,feat_dict,web):
    vecs=[]
    for id in ids:
        vec=[]
        # Append bert, skipgram,cbow
        if web in ['bert','cbow','skipgram']:
            vec.extend(feat_dict.get(str(id)))


        # Append elmo
        if web=='elmo':
            el = feat_dict.get(str(id))
            vec.extend(el.get('context_independent_layer'))
            vec.extend(el.get('LSTM_layer1'))
            vec.extend(el.get('LSTM_layer2'))



        vecs.append(vec)
    return vecs



def get_hybrid_features(web):

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

    train_A_web = None
    train_B_web = None
    test_A_web = None
    test_B_web = None

    if web=='bert':
        #dict[tweet_id]= list
        train_A_web = read_dict_from_json(fn='train_A_bert.json.gz')
        train_B_web = read_dict_from_json(fn='train_B_bert.json.gz')
        test_A_web = read_dict_from_json(fn='test_A_bert.json.gz')
        test_B_web = read_dict_from_json(fn='test_B_bert.json.gz')

    if web=='elmo':
        train_A_web = read_dict_from_json('train_A_elmo.json.gz')
        train_B_web = read_dict_from_json('train_B_elmo.json.gz')
        test_A_web = read_dict_from_json('test_A_elmo.json.gz')
        test_B_web = read_dict_from_json('test_B_elmo.json.gz')


    if web=='skipgram':
        train_A_web = read_dict_from_json('train_A_skipgram.json.gz')
        train_B_web = read_dict_from_json('train_B_skipgram.json.gz')
        test_A_web = read_dict_from_json('test_A_skipgram.json.gz')
        test_B_web = read_dict_from_json('test_B_skipgram.json.gz')


    if web=='cbow':
        train_A_web = read_dict_from_json('train_A_cbow.json.gz')
        train_B_web = read_dict_from_json('train_B_cbow.json.gz')
        test_A_web = read_dict_from_json('test_A_cbow.json.gz')
        test_B_web = read_dict_from_json('test_B_cbow.json.gz')



    feats_tr_A = vectorize(tr_A_ids, train_A_web,web)
    feats_tr_B = vectorize(tr_B_ids, train_B_web,web)
    feats_tst_A = vectorize(tst_A_ids,test_A_web,web)
    feats_tst_B = vectorize(tst_B_ids,test_B_web,web)




    # print(len(feats_tr_A), len(feats_tr_A[1]))
    # print(len(feats_tst_A), len(feats_tst_A[1]))
    # print(len(feats_tr_B), len(feats_tr_B[1]))
    # print(len(feats_tst_B), len(feats_tst_B[1]))

    return feats_tr_A, feats_tst_A, feats_tr_B, feats_tst_B, tr_labels_A, tr_label_B, tst_labels_A, tst_labels_B

if __name__ == '__main__':

    print(f"Embedding, Task, Accuracy, Precision, Recall,F1-Score", sep='\t |')
    webs = ['bert', 'elmo', 'skipgram', 'cbow']

    for web in webs:
        feats_tr_A, feats_tst_A, feats_tr_B, feats_tst_B, tr_labels_A, tr_labels_B, tst_labels_A, tst_labels_B = get_hybrid_features(web)

        task="Task A"

        parameter_space = {'C' : [0.001,0.01,0.1,1,10,100]}


        clf = GridSearchCV(LogisticRegression(solver='liblinear', penalty='l2',random_state=0), parameter_space)
        clf.fit(feats_tr_A,tr_labels_A)

        # Best paramete set
        # print('Best parameters found:\n', clf.best_params_)
        '''
        # All results
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
        '''
        y_pred=clf.predict(feats_tst_A)
        report = classification_report(tst_labels_A, y_pred, output_dict=True,zero_division=0)

        print(f"{web},{task},{report['accuracy']:.4},{report['weighted avg']['precision']:.4},"
              f"{report['weighted avg']['recall']:.4},{report['weighted avg']['f1-score']:.4}",
                 sep='\t |')

        task = "Task B"

        parameter_space = {'C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]}

        clf = GridSearchCV(LogisticRegression(solver='liblinear', penalty='l2',random_state=0), parameter_space)
        clf.fit(feats_tr_B, tr_labels_B)

        # Best paramete set
        # print('Best parameters found:\n', clf.best_params_)

        '''
        # All results
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
        '''
        y_pred = clf.predict(feats_tst_B)
        report = classification_report(tst_labels_B, y_pred, output_dict=True, zero_division=0)

        print(f"{web},{task},{report['accuracy']:.4},{report['weighted avg']['precision']:.4},"
              f"{report['weighted avg']['recall']:.4},{report['weighted avg']['f1-score']:.4}",
              sep='\t |')

