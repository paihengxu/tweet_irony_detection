import pickle

from ..utils import read_non_emoji_tweets





def extract_ngrams():
    pass

def brown_cluster_ngrams():
    pass

def dependency():
    pass

if __name__ == '__main__':
    fp_train_A = 'tweet_irony_detection/train/SemEval2018-T3-train-taskA.txt'
    fp_train_B = 'tweet_irony_detection/train/SemEval2018-T3-train-taskB.txt'
    fp_test = 'tweet_irony_detection/test_TaskA/SemEval2018-T3_input_test_taskA.txt'

    train_A=read_non_emoji_tweets(fp_train_A,"train")
    train_B = read_non_emoji_tweets(fp_train_B,"train")
    test= read_non_emoji_tweets(fp_test,"test")

    print(len(train_A),len(train_B),len(test_A),len(test_A))


    # unit test for features
    # extract_ngrams()