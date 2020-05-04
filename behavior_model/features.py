import os
import numpy as np
import pandas as pd
from collections import defaultdict
from scipy import stats

from utils import read_non_emoji_tweets
from utils import get_label
from utils import print_class_stats
from utils import read_vocabulary_with_occurrence, write_tokens_to_txt
from utils import write_dict_to_json, read_dict_from_json


### DEFINE CONST HERE

def twitter_sentiment_score(corpus, n):
    """
    input: whole dataset, n indicating n-gram
    output: two dictionaries, key: tweet_id, value: 1-dimensional numpy array
    """
    if not os.path.exists('behavior_model/positive_{}_vocab.txt'.format(n)) or not os.path.exists(
        'behavior_model/negative_{}_vocab.txt'.format(n)):
        os.system('python -m behavior_model.sent_vocab')

    # key: word, value: index
    pos_vocab = read_vocabulary_with_occurrence('behavior_model/positive_{}_vocab.txt'.format(n), n)
    neg_vocab = read_vocabulary_with_occurrence('behavior_model/negative_{}_vocab.txt'.format(n), n)

    senti_features = defaultdict(float)
    num_senti_features = defaultdict(int)
    for data in corpus:
        tokens = data.tweet_words()
        lower_tokens = [t.lower() for t in tokens]
        _id = data.tweet_id
        for idx, ele in enumerate(lower_tokens):
            # check idx for n-grams
            if idx >= len(lower_tokens) - (n - 1):
                continue
            # set default value as 0, bc (-0.1, 0.1) is filtered out
            n_gram = [lower_tokens[w_idx] for w_idx in range(idx, idx + n)]
            assert len(n_gram) == n
            pos_score = pos_vocab.get(tuple(n_gram), 0)
            neg_score = neg_vocab.get(tuple(n_gram), 0)

            if pos_score + neg_score == 0:
                continue

            senti_score = (pos_score - neg_score) / (pos_score + neg_score)

            if -0.1 < senti_score < 0.1:
                continue

            senti_features[_id] += senti_score
            num_senti_features[_id] += 1

        # assign 0 to empty score tweets
        if _id not in senti_features:
            senti_features[_id] = 0.
            num_senti_features[_id] = 0

    list_senti_features = {}
    for k, v in senti_features.items():
        list_senti_features[k] = [v, num_senti_features[k]]

    return list_senti_features



def word_affect(data):
    '''
    input: whole corpus
    output: 1 dicts for affect of word, 
            keys: tweet_id, values: dict (keys={"vmax","vmin","vdistance","amax","amin","adistance","dmax","dmin","ddistance"})
    '''
    feature_dict={}
    try:
        df=pd.read_csv('BRM-emot-submit.csv',index_col=0)
        Words=set(df['Word'].values.tolist())
        keys=['V.Mean.Sum','A.Mean.Sum','D.Mean.Sum']
        for tweet in data:
            tokenized= tweet.tweet_words()
            new_words= [word for word in tokenized if word in Words]
            if not new_words:
#                 feature_dict[tweet.tweet_id]={
#                 "vmax":0,"vmin":0,"vdistance":0,
#                  "amax":0,"amin":0,"adistance":0,
#                 "dmax":0,"dmin":0,"ddistance":0}
                feature_dict[tweet.tweet_id]=[0]*9
                continue
            vmax_,vmin_ = -1,10
            amax_,amin_ = -1,10
            dmax_,dmin_ = -1,10
            for word in new_words:
                df_selected = df[df['Word']==word]
#                 print(df_selected)
#                 return
                
                idx=df_selected.index[0]
                vmax_=max(vmax_,df_selected['V.Mean.Sum'][idx])
                vmin_=min(vmin_,df_selected['V.Mean.Sum'][idx])
                amax_=max(amax_,df_selected['A.Mean.Sum'][idx])
                amin_=min(amin_,df_selected['A.Mean.Sum'][idx])
                dmax_=max(dmax_,df_selected['D.Mean.Sum'][idx])
                dmin_=min(dmin_,df_selected['D.Mean.Sum'][idx])
            
            vmax_=0 if vmax_==-1 else vmax_
            vmin_=0 if vmin_==10 else vmin_
            amax_=0 if amax_==-1 else amax_
            amin_=0 if amin_==10 else amin_
            dmax_=0 if dmax_==-1 else dmax_
            dmin_=0 if dmin_==10 else dmin_
#             feature_dict[tweet.tweet_id]={
#                 "vmax": vmax_,"vmin": vmin_,"vdistance":vmax_-vmin_,
#                  "amax":amax_,"amin":amin_,"adistance":amax_-amin_,
#                 "dmax":dmax_,"dmin":dmin_,"ddistance":dmax_-dmin_}
            feature_dict[tweet.tweet_id]=[vmax_,vmin_,vmax_-vmin_,amax_,amin_,amax_-amin_,dmax_,dmin_,dmax_-dmin_]
        return feature_dict

    except Exception as e:
        print("In word affect")
        print(str(e))
    
def readability(data):
    '''
    input: whole corpus
    output: 1 dicts for readability, 
            keys: tweet_id, values: dict (keys={"mean","median","mode","sigma","min","max"})
    '''
    feature_dict={}
    try:
        for tweet in data:
            tokenized= tweet.tweet_words()
            new_words= [word for word in tokenized]
            l=[]
            for word in new_words:
                length=len(word)
                if length<20:
                    l.append(length)
            if not l:
#                 feature_dict[tweet.tweet_id]={"mean":0, "median":0,"mode":0,"sigma":0,"min":0,"max":0}
                feature_dict[tweet.tweet_id]=[0]*6
            else:
                arr_l=np.array(l)
                #feature_dict[tweet.tweet_id]={"mean":np.mean(arr_l), "median":np.median(arr_l),"mode":stats.mode(arr_l)[0],"sigma":np.std(arr_l),"min":arr_l.min(),"max":arr_l.max()}
                feature_dict[tweet.tweet_id]=[np.mean(arr_l),np.median(arr_l),float(stats.mode(arr_l)[0][0]),np.std(arr_l),float(arr_l.min()),float(arr_l.max())]
        return feature_dict

    except Exception as e:
        print("In readability")
        print(str(e))
    


def prosodic(data):
    '''
    input: whole corpus
    output: 1 dicts for prosodic variations, 
            keys: tweet_id, values: dict (keys={"repeat","total_character","ratio"})
    '''
    feature_dict={}
    try:
        for tweet in data:
            tokenized= tweet.tweet_words()
            new_words= [word for word in tokenized]
            total_character=0
            curr_charactor=None
            curr_repeat=0
            distinct_character=0
            visited_character=set()
            
            presence_of_repeat=False
            
            for word in new_words:
                for c in word:
                    total_character+=1
                    visited_character.add(c)
                    if not curr_charactor==c:
                        curr_charactor=c
                        curr_repeat=1
                    else:
                        curr_repeat+=1
                    if curr_repeat>=3:
                        presence_of_repeat=True
            ratio=len(visited_character)/total_character if total_character else 0
            feature_dict[tweet.tweet_id]=[presence_of_repeat*1.,total_character,ratio]
            
                        
        return feature_dict

    except Exception as e:
        print("In readability")
        print(str(e))


if __name__ == '__main__':
    fp_train_A = 'train/SemEval2018-T3-train-taskA.txt'
    fp_train_B = 'train/SemEval2018-T3-train-taskB.txt'
    fp_test_A = 'test_TaskA/SemEval2018-T3_input_test_taskA.txt'
    fp_test_B = 'test_TaskB/SemEval2018-T3_input_test_taskB.txt'
    fp_labels_A = 'goldtest_TaskA/SemEval2018-T3_gold_test_taskA_emoji.txt'
    fp_labels_B = 'goldtest_TaskB/SemEval2018-T3_gold_test_taskB_emoji.txt'

    ### read in corpus
    pre_process_url = True  # Set to remove URLs
    pre_process_usr = True
    train_A = read_non_emoji_tweets(fp_train_A, "train", pre_process_url, pre_process_usr)
    train_B = read_non_emoji_tweets(fp_train_B, "train", pre_process_url, pre_process_usr)
    test_A = read_non_emoji_tweets(fp_test_A, "test", pre_process_url, pre_process_usr)
    test_B = read_non_emoji_tweets(fp_test_B, "test", pre_process_url, pre_process_usr)

    name_2_dataset = {
        'train_A': train_A,
        'train_B': train_B,
        'test_A': test_A,
        'test_B': test_B
    }

    ### test sentiment score
    for name, dataset in name_2_dataset.items():
        feature_2 = twitter_sentiment_score(dataset, 2)
        feature_3 = twitter_sentiment_score(dataset, 3)        
        # word_aff=word_affect(dataset)
        # read=readability(dataset)
        # pros=prosodic(dataset)
                

        write_dict_to_json(feature_2, fn='{dataset}_{feature_name}.json.gz'.format(dataset=name,
                                                                                   feature_name='senti_bigram'))

        write_dict_to_json(feature_3, fn='{dataset}_{feature_name}.json.gz'.format(dataset=name,
                                                                                   feature_name='senti_trigram'))

#         write_dict_to_json(word_aff, fn='.\features\dataset\{dataset}_{feature_name}.json.gz'.format(dataset=name,
#                                                                                    feature_name='word_affect'))
#         write_dict_to_json(read, fn='{dataset}_{feature_name}.json.gz'.format(dataset=name,
#                                                                                    feature_name='readability'))
#         write_dict_to_json(pros, fn='{dataset}_{feature_name}.json.gz'.format(dataset=name,
#                                                                                    feature_name='prosodic'))
