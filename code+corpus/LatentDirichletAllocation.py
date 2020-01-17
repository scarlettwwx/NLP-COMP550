#!/usr/bin/env python
# coding: utf-8
import pandas as pd
import os
import re
import nltk
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
from nltk.corpus import stopwords
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.metrics import f1_score
from sklearn import linear_model
from sklearn import metrics
from sklearn.model_selection import KFold
from sklearn.metrics import fbeta_score
import pyLDAvis.gensim
from sklearn.model_selection import train_test_split
from gensim import corpora, models
os.chdir('..')

# Read data into papers
comments = pd.read_csv('/Users/xuwenwen/OneDrive - McGill University/RedditCommentClassification/reddit_train.csv')

# Print head
comments.head()


# Next, let’s work to transform the textual data in a format that will serve as an input for training LDA model. We start by converting the documents into a simple vector representation (Bag of Words BOW). Next, we will convert a list of titles into lists of vectors, all with length equal to the vocabulary.
#
# We’ll then plot the ten most frequent words based on the outcome of this operation (the list of document vectors). As a check, these words should also occur in the word cloud.

def preprocess_text(text):
    text = text.lower().split()
    stops = set(stopwords.words("english")).union(set(stopwords.words("french")))
    text = [w for w in text if not w in stops and len(w) >= 3]
    text = " ".join(text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"^https?:\/\/.*[\r\n]*","", text)
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ',text)
    text = text.split()

    # lemmatization
    lemma = nltk.wordnet.WordNetLemmatizer()
    text = [lemma.lemmatize(word) for word in text]
    text_final = []

    # clean all non-English words, numbers, and other weirdos, stopwords
    for x in text:
        #x = spell.correction(x)
        if x.isalpha() and len(x)<20 and len(x) > 1 and x not in stops: #and is_english_word(x):
#             if len(x) >10:
#                 text_final.append(PorterStemmer().stem(x))
#             else:
            text_final.append(x)

    text = " ".join(text_final)
    return text


def bigrams(words, bi_min=15, tri_min=10):
    bigram = gensim.models.Phrases(words, min_count = bi_min)
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    return bigram_mod


stop_words = stopwords.words('english')

def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))

def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]



def get_corpus(df):
    df = df.map(lambda x: preprocess_text(x))
    words = list(sent_to_words(df))
    bigram = bigrams(words)
    bigram = [bigram[review] for review in words]
    id2word = gensim.corpora.Dictionary(bigram)
    id2word.filter_extremes(no_below=20, no_above=0.35)
    id2word.compactify()
    corpus = [id2word.doc2bow(text) for text in bigram]
    return corpus, id2word, bigram



train_corpus, train_id2word, bigram_train = get_corpus(comments['comments_processed'])

print(train_id2word[1])

print(len(train_id2word))

lda_train4 = gensim.models.ldamulticore.LdaMulticore(
                           corpus=train_corpus,
                           num_topics=20,
                           id2word=train_id2word,
                           chunksize=100,
                           workers=7, # Num. Processing Cores - 1
                           passes=50,
                           eval_every = 1,
                           per_word_topics=True)

lda_train4.print_topics(20,num_words=10)


# Now that we have an LDA model, we need to run all the reviews through it using 'get document topics'. A list comprehension on that output (2nd line in loop) will give the probability distribution of the topics for a specific review, and that's our feature vector.


train_vecs = []
for i in range(len(comments['comments_processed'])):
    top_topics = lda_train4.get_document_topics(train_corpus[i], minimum_probability=0.0)
    topic_vec = [top_topics[i][1] for i in range(20)]
    train_vecs.append(topic_vec)


train_vecs[2]

X = np.array(train_vecs)
y = np.array(comments['subreddits'])
print(X[2])
print(y[2])


train1_X,test1_X, train1_y, test1_y = train_test_split(X,y,test_size = 0.2, random_state = 10)

clf_nb2 = MultinomialNB();
clf_nb2.fit(train1_X, train1_y);
print("Multinomial Naive Bayes Score: {}".format(clf_nb2.score(test1_X, test1_y)))
y_predict = clf_nb2.predict(test1_X)
confusion_matrix(test1_y, y_predict)

processed_comments = comments['comments_processed'].map(preprocess_text)

words = list(sent_to_words(processed_comments))
dictionary = gensim.corpora.Dictionary(words)

count = 0
for k, v in dictionary.iteritems():
    print(k, v)
    count += 1
    if count > 10:
        break

bow_corpus = [dictionary.doc2bow(doc) for doc in words]


bow_corpus[0]
for i in range(len(bow_corpus[0])):
    print("Word {} (\"{}\") appears {} time.".format(bow_corpus[0][i][0],
                                                     dictionary[bow_corpus[0][i][0]],
                                                     bow_corpus[0][i][1]))

lda_modelling = gensim.models.LdaMulticore(corpus=bow_corpus,
                                           num_topics=20, id2word=dictionary,chunksize=400,passes=50,
                                           per_word_topics=True)


for idx, topic in lda_modelling.print_topics(-1):
    print('Topic: {} \nWords: {}'.format(idx, topic))


pyLDAvis.enable_notebook()
vis = pyLDAvis.gensim.prepare(lda_modelling, bow_corpus, dictionary=dictionary)
vis


# Compute Coherence Score
coherence_model_lda = CoherenceModel(model=lda_modelling, texts=words, dictionary=dictionary, coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence()
print('\nCoherence Score: ', coherence_lda)


train_vecs2 = []
for i in range(len(comments['comments_processed'])):
    top_topics = lda_modelling.get_document_topics(bow_corpus[i], minimum_probability=0.0)
    topic_vec = [top_topics[i][1] for i in range(20)]
    train_vecs2.append(topic_vec)
train_vecs2[2]
X = np.array(train_vecs2)
y = np.array(comments['subreddits'])


train1_X,test1_X, train1_y, test1_y = train_test_split(X,y,test_size = 0.2, random_state = 40)


# Compute Coherence Score
coherence_model_lda2 = CoherenceModel(model=lda_modelling2, texts=words, dictionary=dictionary, coherence='c_v')
coherence_lda2 = coherence_model_lda2.get_coherence()
print('\nCoherence Score: ', coherence_lda2)




tfidf = models.TfidfModel(bow_corpus)

corpus_tfidf = tfidf[bow_corpus]

lda_model_tfidf = gensim.models.LdaMulticore(corpus_tfidf, num_topics=20, id2word=dictionary, passes=2)


for idx, topic in lda_model_tfidf.print_topics(-1):
    print('Topic: {} Word: {}'.format(idx, topic))

processed_comments[:10]


print(lda_model_tfidf)

pyLDAvis.enable_notebook()
vis = pyLDAvis.gensim.prepare(lda_model_tfidf, bow_corpus, dictionary=dictionary)
vis
