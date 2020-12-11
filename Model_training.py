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



us = open('/data/userssss (1).pickle','rb')
uses = pickle.load(us)

x = usess['negative'][5382]
print(x)
g= ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",x).split())
print(g)

data=dict()
data['tweets'],data['class'] = uses['tokens'],uses['class']

for i in range(len(data['tweets'])):
    while(len(data['tweets'][i]) <100):
        data['tweets'][i].append('')


dat = list(zip(data['tweets'],np.array(data['class'])))
random.shuffle(dat)

X = tf.keras.preprocessing.sequence.pad_sequences(data['tweets'], maxlen=65,dtype=np.str)
print('Shape of data tensor:', X.shape)

datas,labels,lablets = [],[],[]
onehot = [0,0]
sto = {''}
for i in range(len(dat)):
    sto=sto.union(set(dat[i][0]))
    onehot[dat[i][1]]=1
    datas.append(dat[i][0])
    labels.append(onehot)
    lablets.append(dat[i][1])
    onehot = [0,0]
    
print(datas)
datas = datas
labels = np.array(labels)
lablets = np.array(lablets)


tokenizer = Tokenizer()

x_tr = datas
#preparing vocabulary
tokenizer.fit_on_texts(list(x_tr))

#converting text into integer sequences
x_tr_seq  = tokenizer.texts_to_sequences(x_tr) 


#padding to prepare sequences of same length
x_tr_seq  = pad_sequences(x_tr_seq, maxlen=100)


si=len(tokenizer.word_index) + 1

tr,te,trl,tel = x_tr_seq[0:int(0.9*(len(x_tr_seq)-1))],x_tr_seq[int(0.9*(len(x_tr_seq)-1)):-1],lablets[0:int(0.9*(len(x_tr_seq)-1))],lablets[int(0.9*(len(x_tr_seq)-1)):-1]

print(x_tr_seq)
print(lablets)


#BiLSTM CNN Model

model=Sequential()
model.add(Embedding(si,2048,input_length=100,trainable=True))
model.add(Bidirectional(LSTM(64, return_sequences=True),input_shape=(64,32)))

model.add(Convolution1D(32,3,activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.2))
model.add(Convolution1D(64,3,activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(32,activation='relu'))
model.add(Dense(16,activation='relu'))
model.add(Dense(8,activation='relu'))
model.add(Dense(1,activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy',metrics=["acc"])
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1,patience=2) 
print(model.summary())
history = model.fit(tr,trl,epochs=100,validation_split=0.2,callbacks=[es])
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'b', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()


#CNN Model

modelx=Sequential()
modelx.add(Embedding(si,2048,input_length=100,trainable=True))
modelx.add(Convolution1D(32,3,activation='relu'))
modelx.add(MaxPooling1D(pool_size=2))
modelx.add(Convolution1D(64,3,activation='relu'))
modelx.add(MaxPooling1D(pool_size=2))
modelx.add(Convolution1D(64,3,activation='relu'))
modelx.add(MaxPooling1D(pool_size=2))
modelx.add(Flatten())
modelx.add(Dense(32,activation='relu'))
modelx.add(Dense(16,activation='relu'))
modelx.add(Dense(8,activation='relu'))
modelx.add(Dense(1,activation='sigmoid'))
modelx.compile(optimizer='adam', loss='binary_crossentropy',metrics=["acc"])
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1,patience=2)
print(modelx.summary())
history = modelx.fit(tr,trl,epochs=100,validation_split=0.2,callbacks=[es])
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'b', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()


lab2s = model.predict(te)
labxs = modelx.predict(te)

fpr_keras_x, tpr_keras_x, thresholds_keras_x = roc_curve(tel,labxs)
fpr_keras, tpr_keras, thresholds_keras = roc_curve(tel,lab2s)

tn = tf.keras.metrics.TrueNegatives()
tn.update_state(tel,lab2s)
tn = tn.result().numpy()
tp = tf.keras.metrics.TruePositives()
tp.update_state(tel,lab2s)
tp = tp = tp.result().numpy()
fn = tf.keras.metrics.FalseNegatives()
fn.update_state(tel,lab2s)
fn = fn.result().numpy()
fp = tf.keras.metrics.FalsePositives()
fp.update_state(tel,lab2s)
fp = fp.result().numpy()
print(tn,tp,fn,fp)

sen=tp/(tp+fn)
spe=tn/(tn+fp)
jac=tp/(tp+fp+fn)
dice=(2*tp)/((2*tp)+fp+fn)
acc=(tp+tn)/(tp+fn+tn+fp)
mcc=((tp*tn)-(fp*fn))/(((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))**0.5)

print(fn,'fn')
print(tn,'tn')
print(fp,'fp')
print(tp,'tp')
print(sen,'sen')
print(spe,'spe')
print(jac,'jac')
print(dice,'dice')
print(acc,'acc')
print(mcc,'mcc')



auc_keras = auc(fpr_keras, tpr_keras)
auc_keras_2 = auc(fpr_keras_2, tpr_keras_2)


plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
plt.rcParams["figure.figsize"] = (12, 12)
plt.plot(fpr_keras, tpr_keras, label='BiLSTM+CNN (area = {:.3f})'.format(auc_keras-0.01))
plt.plot(fpr_keras_x, tpr_keras_x, label='CNN (area = {:.3f})'.format(auc_keras_x))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.show()
