from .utils import read_tweets

def extract_ngrams():
    pass

def brown_cluster_ngrams():
    pass

def dependency():
    pass

if __name__ == '__main__':
    fn = 'train/SemEval2018-T3-train-taskA_emoji.txt'
    read_tweets()

    # unit test for features
    extract_ngrams()