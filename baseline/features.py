import pickle

from ..utils import read_non_emoji_train_tweets
from ..utils import read_non_emoji_test_tweets




def extract_ngrams():
    pass

def brown_cluster_ngrams():
    pass

def dependency():
    pass

if __name__ == '__main__':
    fp_train_A = 'tweet_irony_detection/train/SemEval2018-T3-train-taskA.txt'
    fp_test_A='tweet_irony_detection/test_TaskA/SemEval2018-T3_input_test_taskA.txt'
    fp_train_B = 'tweet_irony_detection/train/SemEval2018-T3-train-taskB.txt'
    # fp_test_B = 'tweet_irony_detection/test_TaskB/SemEval2018-T3_input_test_taskB.txt'

    train_A=read_non_emoji_train_tweets(fp_train_A)
    test_A=read_non_emoji_test_tweets(fp_test_A)
    train_B = read_non_emoji_train_tweets(fp_train_B)
    # test_B = read_non_emoji_test_tweets(fp_test_B)

    print(len(train_A),len(train_B),len(test_A),len(test_A))


    # unit test for features
    # extract_ngrams()