#!/usr/bin/env python
# coding: utf-8

# In[4]:


get_ipython().system('pip install gensim')


# In[5]:


import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import gensim


# In[6]:


import warnings
warnings.filterwarnings(action='ignore')


# In[7]:


path = "../desktop/DT/movies/"


# In[8]:


import os
print(os.listdir(path))


# In[11]:


movie = pd.read_csv(path + 'ratings.csv', low_memory=False)
movie.head(2)


# In[14]:


movie = movie.sort_values(by='timestamp', ascending=True).reset_index(drop=True)
movie.head()


# In[15]:


# 영화의 Metadata를 불러와 movieID에 맞는 TITLE을 구해줌

meta = pd.read_csv(path + 'movies_metadata.csv', low_memory=False)
meta.head(2)


# In[16]:


meta.columns


# In[22]:


meta = meta.rename(columns={'id' :'movieId'})
movie['movieId'] = movie['movieId'].astype(str)
meta['movieId'] = meta['movieId'].astype(str)

movie=pd.merge(movie, meta[['movieId', 'original_title']], how='left', on='movieId')


# In[23]:


movie = movie[movie['original_title'].notnull()].reset_index(drop=True)


# In[24]:


agg = movie.groupby(['userId'])['original_title'].agg({'unique'})
agg.head()


# In[25]:


movie['original_title'].unique()


# In[26]:


# Word2Vec 적용


# In[27]:


# int 형식은 Word2vec에서 학습이 안되어 String으로 변경해줌
sentence = []

for user_sentence in agg['unique'].values:
    sentence.append(list(map(str, user_sentence)))


# In[28]:


# Word2vec의 학습진행
from gensim.models import Word2Vec
embedding_model = Word2Vec(sentence, size=20, window =5, min_count=1, workers=4, iter=200, sg=1)


# In[31]:


embedding_model.wv.most_similar(positive=['Spider-Man 2'], topn=10)


# In[32]:


# Doc2Vec 적용


# In[33]:


from gensim.models import doc2vec


# In[35]:


meta = pd.read_csv(path + 'movies_metadata.csv', low_memory=False)
meta = meta[meta['original_title'].notnull()].reset_index(drop=True)
meta = meta[meta['overview'].notnull()].reset_index(drop=True)


# In[55]:


from nltk.corpus import stopwords
from tqdm.notebook import tqdm
from nltk.tokenize import word_tokenize, sent_tokenize
import re
stop_words = set(stopwords.words('english'))

overview = []

for words in tqdm(meta['overview']):
    word_tokens = word_tokenize(words)
    sentence = re.sub('[^A-Za-z0-9]+', ' ', str(word_tokens))
    sentence = sentence.strip()
    
    sentence_tokens = word_tokenize(sentence)
    result = ''
    
    for token in sentence_tokens:
        if token not in stop_words:
            result += ' ' + token
    result = result.strip().lower()
    overview.append(result)


# In[53]:


import nltk
nltk.download('punkt')


# In[56]:


meta['pre_overview'] = overview


# In[57]:


doc_vectorizer = doc2vec.Doc2Vec(
    dm = 9, # PV-dbow / defalut 1
    dbow_words=1, # w2v simultaneous with DBOW d2b / defalust 0
    window = 10, #distance between the predicted word and context words
    size= 100, # vector size
    alpha=0.025, # learning-rate
    seed= 1234,
    min_count=5, #ignore with frq lower
    min_alpha=0.024, # min learning-rate
    workers=4, #multi cpu
    hs = 1, # hierar chical softmax / default 0
    negative =10 #negative sampling /default 5
)


# In[58]:


from collections import namedtuple

agg = meta[['id', 'original_title', 'pre_overview']]
TaggedDocument = namedtuple('TaggedDocument', 'words tags')
tagged_train_docs = [TaggedDocument((c), [d]) for d, c in agg[['original_title', 'pre_overview']].values]


# In[59]:


doc_vectorizer.build_vocab(tagged_train_docs)
print(str(doc_vectorizer))


# In[67]:


# 벡터 문서 학습
from time import time

start = time()


# In[ ]:


for epoch in tqdm(range(5)):
    doc_vectorizer.train(tagged_train_docs, total_examples=doc_vectorizer.corpus_count, epochs=doc_vectorizer.iter)
    doc_vectorizer.alpha -= 0.002 #decrease the learing rate
    doc_vectorizer.min_alpha = doc_vectorizer.alpha # fix the learning rate, no decay
    
    #doc_vectorizer.train(tagged_train_docs, total_examples=doc_vectorizer.corpus_count, epochs=doc_vectorizer.iter)
end = time()
print("During Time: {}".format(end-start))


# In[ ]:


doc_vectorizer.docvecs.most_similar('Toy Story', topn=20)


# In[ ]:


doc_vectorizer.docvecs.most_similar('Harry Potter and the Deathly Hallows: Part 1', topn=20)

