"""
This script is to convert tweet from train/scraped to format for fine tuning BERT
"""

import codecs
from utils import read_non_emoji_tweets
from utils.const import  *

def bert_fine_tune_output(fn, corpus_list):
    '''
    output to similar output in wiki dataset
    '''
    with codecs.open(fn, 'w', encoding='utf8') as outf:
        for corpus in corpus_list:
            for t in corpus:
                outf.write('\n = = {} = =\n\n'.format(t.tweet_id))
                outf.write('{}\n'.format(t.tweet_text))


if __name__ == '__main__':
    pre_process_url = True  # Set to remove URLs
    pre_process_usr = True
    train_A = read_non_emoji_tweets(fp_train_A, "train", pre_process_url, pre_process_usr)
    train_B = read_non_emoji_tweets(fp_train_B, "train", pre_process_url, pre_process_usr)

    test_A = read_non_emoji_tweets(fp_test_A, "test", pre_process_url, pre_process_usr)
    test_B = read_non_emoji_tweets(fp_test_B, "test", pre_process_url, pre_process_usr)

    pos = read_non_emoji_tweets('twitter_scrape/positive_text.txt', 'train', pre_process_url, pre_process_usr)
    neg = read_non_emoji_tweets('twitter_scrape/negative_text.txt', 'train', pre_process_url, pre_process_usr)
    corpus_list = [train_A, test_A, pos, neg]

    bert_fine_tune_output('combo_corpus.txt', corpus_list)