from .features import *
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import classification_report, confusion_matrix

# TODO: could use cross-validation to tune hyper-parameters later

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





def main():
    generate=True
    feats_tr_A, feats_tst_A, feats_tr_B, feats_tst_B, tr_labels_A, tr_labels_B, tst_labels_A, tst_labels_B = featurize(generate)


    # majority guess as another baseline
    major_model = DummyClassifier(strategy="most_frequent")
    fit_test_model(train=feats_tr_A, train_label=tr_labels_A, test=feats_tst_A, test_label=tst_labels_A,
                   model=major_model)

    # task A
    model = LogisticRegression(solver='liblinear', penalty='l2', random_state=0)
    fit_test_model(train=feats_tr_A, train_label=tr_labels_A, test=feats_tst_A, test_label=tst_labels_A,
                   model=model)

    # task B
    model2 = LogisticRegression(solver='liblinear', penalty='l2', random_state=0)
    fit_test_model(train=feats_tr_B, train_label=tr_labels_B, test=feats_tst_B, test_label=tst_labels_B,
                   model=model2)

if __name__ == '__main__':
    main()