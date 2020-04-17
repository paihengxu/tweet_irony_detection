import pickle
import re

from utils import read_non_emoji_tweets
from utils import get_label
from utils import print_class_stats


def extract_ngrams():
    pass

def brown_cluster_ngrams():
    pass

def dependency():
    pass

if __name__ == '__main__':
    # File paths from project level
    # fp_train_A = 'tweet_irony_detection/train/SemEval2018-T3-train-taskA.txt'
    fp_train_A = 'train/SemEval2018-T3-train-taskA.txt'
    fp_train_B = 'train/SemEval2018-T3-train-taskB.txt'
    fp_test_A = 'test_TaskA/SemEval2018-T3_input_test_taskA.txt'
    fp_test_B = 'test_TaskB/SemEval2018-T3_input_test_taskB.txt'
    fp_labels_A= 'goldtest_TaskA/SemEval2018-T3_gold_test_taskA_emoji.txt'
    fp_labels_B='goldtest_TaskB/SemEval2018-T3_gold_test_taskB_emoji.txt'

    # Training data for task A and B , test data & correct labels for both tasks
    pre_process_url=True # Set to remove URLs
    pre_process_usr=True
    train_A = read_non_emoji_tweets(fp_train_A,"train",pre_process_url,pre_process_usr)
    train_B = read_non_emoji_tweets(fp_train_B,"train",pre_process_url,pre_process_usr)

    test_A= read_non_emoji_tweets(fp_test_A,"test",pre_process_url,pre_process_usr)
    test_B = read_non_emoji_tweets(fp_test_B, "test", pre_process_url,pre_process_usr)
    labels_A=get_label(fp_labels_A)
    labels_B = get_label(fp_labels_B)
    # Print class stats
    print_class_stats(train_A,train_B,labels_A,labels_B)


    # unit test for features
    # extract_ngrams()