import codecs
import itertools
from nltk.util import ngrams
from collections import Counter

from utils import read_non_emoji_tweets

### CONST
MIN_FREQ = 5

def construct_sentiment_vocabulary(corpus, sent, n, min_freq=MIN_FREQ):
    """
    construct vocabulary file first before extract ngrams
    sent: pos or neg
    n: value of n in the n-gram
    """
    ngram_freq = Counter([])
    for tweet in corpus:
        tokens = tweet.tweet_words()
        lower_tokens = [t.lower() for t in tokens]
        # add start/end of sentence token
        ngram = ngrams(['<s>'] + lower_tokens + ['</s>'], n)

        ngram_freq += Counter(ngram)

    # write to txt file
    def write_counter_to_file_with_occurrence(freq, out_fn, min_freq=min_freq):
        with codecs.open(out_fn, 'w',encoding='utf8') as outf:
            for ele, count in freq.items():
                if count >= min_freq:
                    outf.write('{}\t{}\n'.format('\t'.join(ele), count))

    write_counter_to_file_with_occurrence(ngram_freq, 'behavior_model/{}_{}_vocab.txt'.format(sent, str(n)))

    return ngram_freq


if __name__ == '__main__':
    n_gram_values = [2, 3]
    sent_values = ['positive', 'negative']
    permutations = [
        n_gram_values,
        sent_values
    ]
    for n, sent in itertools.product(*permutations):
        print(n, sent)
        corpus = read_non_emoji_tweets('twitter_scrape/{}_text.txt'.format(sent), 'train', True, True)
        construct_sentiment_vocabulary(corpus, sent, n)