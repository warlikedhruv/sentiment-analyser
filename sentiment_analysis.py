import nltk
import random
from nltk.classify.scikitlearn import SklearnClassifier
import pickle

from nltk.tokenize import word_tokenize
import re
from nltk.corpus import stopwords
import os



all_words = []
documents = []
stop_words = list(set(stopwords.words('english')))

#  j is adject, r is adverb, and v is verb
#allowed_word_types = ["J","R","V"]
allowed_word_types = ["J"]


documents_1 = open('static/documents.pickle', "rb")
documents = pickle.load(documents_1)
documents_1.close()

documents_2 = open('static/all_words.pickle', "rb")
all_words = pickle.load(documents_2)
documents_2.close()



all_words = nltk.FreqDist(all_words)


word_features = list(all_words.keys())[:5000]

def find_features(document):
    words = word_tokenize(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)
    return features


featuresets = []

documents_3 = open('static/featuresets.pickle', "rb")
featuresets = pickle.load(documents_3)
documents_3.close()


# Shuffling the documents
random.shuffle(featuresets)
training_set = featuresets[:20000]
testing_set = featuresets[20000:]


open_file = open('static/naive-model.pickle', "rb")
naive_classifier = pickle.load(open_file)
open_file.close()





feature_list = [f[0] for f in testing_set]

ensemble_preds = [naive_classifier.classify(features) for features in feature_list]

def sentiment(text):

    feats = find_features(text)
    print(naive_classifier.classify(feats))
    str = naive_classifier.classify(feats)
    return str






