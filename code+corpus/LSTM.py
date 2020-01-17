#!/usr/bin/env python
# coding: utf-8

# In[26]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from keras.models import Sequential
from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping
from LSTM_preprocess import *

data = pd.read_csv('reddit_train.csv', usecols=['comments', 'subreddits'])


# In[27]:


data.subreddits.value_counts()


# In[28]:


num_of_categories = 3500
shuffled = data.reindex(np.random.permutation(data.index))
a = shuffled[shuffled['subreddits'] == 'AskReddit'][:num_of_categories]
b = shuffled[shuffled['subreddits'] == 'GlobalOffensive'][:num_of_categories]
c = shuffled[shuffled['subreddits'] == 'Music'][:num_of_categories]
d = shuffled[shuffled['subreddits'] == 'Overwatch'][:num_of_categories]
e = shuffled[shuffled['subreddits'] == 'anime'][:num_of_categories]
f = shuffled[shuffled['subreddits'] == 'baseball'][:num_of_categories]
g = shuffled[shuffled['subreddits'] == 'canada'][:num_of_categories]
h = shuffled[shuffled['subreddits'] == 'conspiracy'][:num_of_categories]
i = shuffled[shuffled['subreddits'] == 'europe'][:num_of_categories]
j = shuffled[shuffled['subreddits'] == 'funny'][:num_of_categories]
k = shuffled[shuffled['subreddits'] == 'gameofthrones'][:num_of_categories]
l = shuffled[shuffled['subreddits'] == 'hocky'][:num_of_categories]
m = shuffled[shuffled['subreddits'] == 'leagueoflegends'][:num_of_categories]
n = shuffled[shuffled['subreddits'] == 'movies'][:num_of_categories]
o = shuffled[shuffled['subreddits'] == 'nba'][:num_of_categories]
p = shuffled[shuffled['subreddits'] == 'nfl'][:num_of_categories]
q = shuffled[shuffled['subreddits'] == 'soccer'][:num_of_categories]
r = shuffled[shuffled['subreddits'] == 'trees'][:num_of_categories]
s = shuffled[shuffled['subreddits'] == 'worldnews'][:num_of_categories]
t = shuffled[shuffled['subreddits'] == 'wow'][:num_of_categories]

concated = pd.concat([a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t], ignore_index=True)
#Shuffle the dataset
concated = concated.reindex(np.random.permutation(concated.index))
concated['LABEL'] = 0


# In[29]:


#One-hot encode the lab
concated.loc[concated['subreddits'] == 'AskReddit', 'LABEL'] = 0
concated.loc[concated['subreddits'] == 'GlobalOffensive', 'LABEL'] = 1
concated.loc[concated['subreddits'] == 'Music', 'LABEL'] = 2
concated.loc[concated['subreddits'] == 'Overwatch', 'LABEL'] = 3
concated.loc[concated['subreddits'] == 'anime', 'LABEL'] = 4
concated.loc[concated['subreddits'] == 'baseball', 'LABEL'] = 5
concated.loc[concated['subreddits'] == 'canada', 'LABEL'] = 6
concated.loc[concated['subreddits'] == 'conspiracy', 'LABEL'] = 7
concated.loc[concated['subreddits'] == 'europe', 'LABEL'] = 8
concated.loc[concated['subreddits'] == 'funny', 'LABEL'] = 9
concated.loc[concated['subreddits'] == 'gameofthrones', 'LABEL'] = 10
concated.loc[concated['subreddits'] == 'hocky', 'LABEL'] = 11
concated.loc[concated['subreddits'] == 'leagueoflegends', 'LABEL'] = 12
concated.loc[concated['subreddits'] == 'movies', 'LABEL'] = 13
concated.loc[concated['subreddits'] == 'nba', 'LABEL'] = 14
concated.loc[concated['subreddits'] == 'nfl', 'LABEL'] = 15
concated.loc[concated['subreddits'] == 'soccer', 'LABEL'] = 16
concated.loc[concated['subreddits'] == 'trees', 'LABEL'] = 17
concated.loc[concated['subreddits'] == 'worldnews', 'LABEL'] = 18
concated.loc[concated['subreddits'] == 'wow', 'LABEL'] = 19



print(concated['LABEL'][:10])
labels = to_categorical(concated['LABEL'], num_classes=20)
print(labels[:10])
if 'subreddits' in concated.keys():
    concated.drop(['subreddits'], axis=1)


# In[30]:


for i in range(len(concated["comments"])):
    concated["comments"][i] = preprocess_text(concated["comments"][i])


# In[31]:


n_most_common_words = 15000
max_len = 200
tokenizer = Tokenizer(num_words=n_most_common_words, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~0123456789', lower=True)
tokenizer.fit_on_texts(concated['comments'].values)
sequences = tokenizer.texts_to_sequences(concated['comments'].values)
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

X = pad_sequences(sequences, maxlen=max_len)


# In[32]:


X_train, X_test, y_train, y_test = train_test_split(X , labels, test_size=0.2, random_state=42)


# In[33]:


epochs = 1000
emb_dim = 256
batch_size = 256
labels[:2]


# In[34]:


print((X_train.shape, y_train.shape, X_test.shape, y_test.shape))


# In[ ]:


model = Sequential()
model.add(Embedding(n_most_common_words, emb_dim, input_length=X.shape[1]))
model.add(SpatialDropout1D(0.5))
model.add(LSTM(128, dropout=0.5, recurrent_dropout=0.5))
model.add(Dense(20, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
print(model.summary())
history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,validation_split=0.25,callbacks=[EarlyStopping(monitor='val_loss',patience=7, min_delta=0.0001)])


# In[ ]:





# # In[12]:


# # read the test set
# test_data = pd.read_csv('reddit_test.csv', usecols=['id','comments'])
# for i in range(len(test_data["comments"])):
#     test_data["comments"][i] = preprocess_text(test_data["comments"][i])

# test_sequences = tokenizer.texts_to_sequences(test_data['comments'].values)
# max_len = 200
# test_X = pad_sequences(test_sequences, maxlen=max_len)
# prediction = model.predict(test_X)


# # In[20]:


# print((prediction[2]))


# # In[ ]:


# class_dict ={
#     0:'AskReddit',
#     1:'GlobalOffensive',
#     2:'Music',
#     3:'Overwatch',
#     4:'anime',
#     5:'baseball',
#     6:'canada',
#     7:'conspiracy',
#     8:'europe',
#     9:'funny',
#     10:'gameofthrones',
#     11:'hocky',
#     12:'leagueoflegends',
#     13:'movies',
#     14:'nba', 
#     15:'nfl', 
#     16:'soccer',
#     17:'trees',
#     18:'worldnews',
#     19:'wow'}


# # In[ ]:


# # save the model to local
# import pickle
# filename = 'finalized_model.sav'
# pickle.dump(model, open(filename, 'wb'))

# # some time later...
 
# # load the model from disk
# loaded_model = pickle.load(open(filename, 'rb'))
# result = loaded_model.score(X_test, Y_test)
# print(result)

