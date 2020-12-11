import pandas as pd
import os
import pickle
import io
import demoji
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords,brown,words
import re
import string
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
import nltk
import random
import numpy as np
import tensorflow as tf
import fasttext as ft
from keras.models import *
from keras.layers import *
from keras.callbacks import *
import matplotlib.pyplot as plt
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier as rfc
from sklearn.svm import SVC
import re
from nltk.util import ngrams
import demoji

demoji.download_codes()

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('brown')
nltk.download('words')


users = dict()
labels = ['positive','negative']
path = input("enter the path to the tweet dataset")
for lab in labels:
    users[lab]=[]
    temp_path=os.path.join(path,lab,'tweet')
    for files in os.listdir(temp_path):
        print(files,' opened ')
        dt = pd.read_json(os.path.join(temp_path,files),lines=True)
        users[lab].append(dt['text'])
        print(dt['text'])
usess=dict()
for i in uses:
    tab = []
    c=0
    for j in uses[i]:
        print(c)
        print(j[0])
        tab.append(j[0])
        c=c+1
    usess[i]=tab
usesss=dict()
for i in usess:
    usesss[i]=[]
    for k in usess[i]:
        s=''
        h=demoji.findall(k)
        for j in range(len(k)):
            if k[j] in h :
                s=s+h[k[j]]
            else:
                s=s+k[j]
        usesss[i].append(s)
usesssss={'tweet':[],'class':[],'tokens':[],'vectors':[]}
for i in usess:
    usesss[i]=[]
    for k in usess[i]:
        s=''
        h=demoji.findall(k)
        for j in range(len(k)):
            if k[j] in h :
                s=s+h[k[j]]
            else:
                s=s+k[j]
        usesssss['tweet'].append(s)
        usesssss['class'].append(labels.index(i))
print(len(usesss['negative']))

usesss['positive']=usesss['positive'][0:5385]

print(len(usesss['positive']))



stop_words = set(stopwords.words('english')) 

for tweet in usesssss['tweet'] :

    temp = ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",tweet).split())
    temp = (word_tokenize(temp))
    temp = [word for word in temp if word not in stop_words and (word.encode('utf-8').isalpha()) ]
    temp = [porter.stem(word) for word in temp]
    usesssss['tokens'].append(temp)

us_file = open("uses.pickle","ab")

pickle.dump(usesssss,us_file)





