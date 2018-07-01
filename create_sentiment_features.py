# -*- coding: utf-8 -*-
"""
Created on Sat Jun 30 17:24:45 2018

@author: Aditya
"""

import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import numpy as np
import random
import pickle
from collections import Counter


import warnings
warnings.filterwarnings("ignore")

lemmatizer = WordNetLemmatizer()
hm_lines = 10000000

def create_lexicon(pos, neg):
    lexicon = []
    for fi in [pos,neg]:
        with open(fi,'r') as f:
            contents = f.readlines()
            for l in contents[:hm_lines]:
                all_words = word_tokenize(l)
                lexicon += list(all_words)
    lexicon = [lemmatizer.lemmatize(i) for i in lexicon]
    stop_words = set(stopwords.words("english"))
    cleaned_text = filter(lambda x: x not in stop_words, lexicon)
    w_counts = Counter(cleaned_text)
    #print(w_counts.most_common(20))
    l2 = []
    for w in w_counts:
        if 1000 > w_counts[w] >30 and len(w) > 1:
            l2.append(w)
    #print(l2)
    print(l2)
    print(len(l2))
    return l2

def sample_handling(sample, lexicon, classification):
    featureset = []
    with open(sample,'r') as f:
        contents = f.readlines()
        for l in contents[:hm_lines]:
            current_words = word_tokenize(l.lower())
            current_words = [lemmatizer.lemmatize(i) for i in current_words]
            features = np.zeros(len(lexicon))
            for word in current_words:
                if word.lower() in lexicon:
                    index_value = lexicon.index(word.lower())
                    features[index_value] += 1
            features = list(features)
            featureset.append([features,classification])
    return featureset

def create_feature_sets_and_labels(pos, neg, test_size=0.1):
    lexicon = create_lexicon(pos,neg)
    features = []
    features += sample_handling('spam.txt',lexicon,[1,0])
    features += sample_handling('ham.txt',lexicon,[0,1])
    random.shuffle(features)
    
    features = np.array(features)
    
    testing_size = int(test_size*len(features))
    
    train_x = list(features[:,0][:-testing_size])
    train_y = list(features[:,1][:-testing_size])
    
    test_x = list(features[:,0][-testing_size:])
    test_y = list(features[:,1][-testing_size:])
    
    return train_x,train_y,test_x,test_y

if __name__ == '__main__':
    train_x,train_y,test_x,test_y = create_feature_sets_and_labels('spam.txt','ham.txt')
    with open('sentiment_set.pickle','wb') as f:
        pickle.dump([train_x,train_y,test_x,test_y], f)
            
            