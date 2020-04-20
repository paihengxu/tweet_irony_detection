from .features import *
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix


def main():
    feats_tr_A, feats_tst_A, feats_tr_B, feats_tst_B, tr_labels_A, tr_label_B, tst_labels_A, tst_labels_B = featurize()

    # Classifier
    model = LogisticRegression(solver='liblinear', penalty='l2', random_state=0)
    #  Fit
    model.fit(feats_tr_A, tr_labels_A)
    # Predict
    # p_pred = model.predict_proba(feats_tst_A)
    # Metrics
    y_pred = model.predict(feats_tst_A)
    score_ = model.score(feats_tst_A, tst_labels_A)
    conf_m = confusion_matrix(tst_labels_A, y_pred)
    report = classification_report(tst_labels_A, y_pred)

    print('score_:', score_, end='\n\n')
    print('conf_m:', conf_m, sep='\n', end='\n\n')
    print('report:', report, sep='\n')

    model2 = LogisticRegression(solver='liblinear', penalty='l2', random_state=0)
    model2.fit(feats_tr_B, tr_label_B)
    # Predict
    # p_pred = model.predict_proba(feats_tst_A)
    # Metrics
    y_pred = model2.predict(feats_tst_B)
    score_ = model2.score(feats_tst_B, tst_labels_B)
    conf_m = confusion_matrix(tst_labels_B, y_pred)
    report = classification_report(tst_labels_B, y_pred)

    print('score_:', score_, end='\n\n')
    print('conf_m:', conf_m, sep='\n', end='\n\n')
    print('report:', report, sep='\n')

if __name__ == '__main__':
    main()