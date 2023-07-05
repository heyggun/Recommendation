#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as pd
import numpy as np
import plotnine
from plotnine import *
import os, sys, gc
from tqdm.notebook import tqdm
import warnings
warnings.filterwarnings('ignore')


# In[14]:


path = "../Desktop/DT/books/"
print(os.listdir(path))


# - books.csv : 책의 메타정보
# - book_tags.csv : 책-태그의 매핑정보
# - ratings.csv : 사용자가 책에 대해 점수를 준 평점정보
# - tags.csv : 태그의 정보
# - to_read.csv : 사용자가 읽으려고 기록해둔 책 (장바구니)

# In[15]:


books = pd.read_csv(path + 'books.csv')
book_tags = pd.read_csv(path + 'book_tags.csv')
train = pd.read_csv(path + 'train.csv')
test = pd.read_csv(path + 'test.csv')
tags = pd.read_csv(path + 'tags.csv')
to_read = pd.read_csv(path + 'to_read.csv')


# In[17]:


train['book_id'] = train['book_id'].astype(str)
test['book_id'] = test['book_id'].astype(str)
books['books_id'] = books['book_id'].astype(str)


# In[18]:


popular_rec_model = books.sort_values(by='books_count', ascending=False)['book_id'].values[0:500]


# In[32]:


sol = test.groupby(['user_id'])['book_id'].agg({'unique'}).reset_index()
gt = {}
for user in tqdm(sol['user_id'].unique()):
    gt[user] = list(sol[sol['user_id'] == user]['unique'].values[0])


# In[20]:


rec_df = pd.DataFrame()
rec_df['user_id'] = train['user_id'].unique()


# # TF-IDF를 이용한 Contents Based Model

# In[21]:


from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(books['title'])
print(tfidf_matrix.shape)


# In[23]:


from sklearn.metrics.pairwise import cosine_similarity


# In[24]:


cosine_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
cosine_matrix


# In[27]:


# book tilte과 id를 매핑할 dictionary 생성

book2id = {}
for i, c in enumerate(books['title']) : book2id[i] =c
    
# id 와 book title을 매핑할 dictionary 생성
id2book = {}
for i,c in book2id.items() : id2book[c] = i
    
# book_id 와 title을 매핑할 dictionary를 생성
bookid2book = {}
for i,j in zip(books['title'].values, books['book_id'].values):
    bookid2book[i] = j


# In[28]:


books['title'].head()


# In[29]:


idx = id2book['Twilight (Twilight, #1)']
sim_scores = [(book2id[i], c) for i,c in enumerate(cosine_matrix[idx]) if i != idx]
sim_scores = sorted(sim_scores, key= lambda x: x[1], reverse=True)
sim_scores[0:10]


# 1. 학습셋에서 제목이 있는 경우에 대해서만 진행
# 2. 각 유저별로 읽은 책의 목록을 수집
# 3. 읽은 책과 유사한 책 추출
# 4. 모든 책에 대해서 유사도를 더한 값을 계산
# 5. 3에서 유사도가 가장 높은 순서대로 추출
# 

# In[37]:


books['title'] = books['title'].astype(str)
books['book_id'] = books['book_id'].astype(str)


# In[39]:


train = pd.merge(train, books[['book_id', 'title']], how='left', on= 'book_id')
train.head()


# In[42]:


# 0. 학습셋에서 제목이 있는 경우에 대해서만 진행
tf_train = train[train['title_x'].notnull()].reset_index(drop=True)
tf_train['idx2title'] = tf_train['title_x'].apply(lambda x: id2book[x])
tf_train.head()


# In[45]:


idx2title2book = {}
for i,j in zip(tf_train['idx2title'].values, tf_train['book_id'].values):
    idx2title2book[i] = j


# In[47]:


# 1. 각 유저별로 읽은 책의 목록을 수집
user = 7
read_list = tf_train.groupby(['user_id'])['idx2title'].agg({'unique'}).reset_index()
seen = read_list[read_list['user_id'] == user]['unique'].values[0]
seen


# In[48]:


# 2. 읽은 책과 유사한 책 추천
### 343 번째 책과 다른 책들간의 유사도
cosine_matrix[343]


# In[49]:


# 2. 읽은 책과 유사한 책 추출
total_cosine_sim = np.zeros(len(book2id))
for book_ in seen:
    # 3. 모든 책에 대해서 유사도를 더한 값을 계산
    # 343번째 책과 248의 유사도가 모두 결합된 유사도
    total_cosine_sim += cosine_matrix[book_]


# In[51]:


# 4. 3에서 유사도가 가장 높은 순서대로 추출
sim_scores = [(i,c) for i,c in enumerate(total_cosine_sim) if i not in seen] 
# 자기 자신을 제외한 영화들의 유사도 및 인덱스를 추출
sim_scores = sorted(sim_scores, key = lambda x: x[1], reverse= True)
# 유사도가 높은 순서대로 정렬
sim_scores[0:5]


# In[52]:


book2id[4809]


# In[53]:


bookid2book[book2id[4809]]


# In[54]:


tf_train['user_id'].unique()


# In[56]:


tf_train.head()


# In[ ]:


## 전체 영화에 대해서 진행 
total_rec_list = {}

read_list1 = train.groupby(['user_id'])['book_id'].agg({'unique'}).reset_index()
read_list2 = tf_train.groupby(['user_id'])['idx2title'].agg({'unique'}).reset_index()

for user in tqdm(train['user_id'].unique()):
    rec_list = []
        
    # 만약 TF-IDF 소속의 추천대상이라면 Contents 기반의 추천 
    if user in tf_train['user_id'].unique():
        # 1. 각 유저별로 읽은 책의 목록을 수집 
        seen = read_list2[read_list2['user_id'] == user]['unique'].values[0]
        # 2. 읽은 책과 유사한 책 추출 
        total_cosine_sim = np.zeros(len(book2id))
        for book_ in seen: 
            # 3. 모든 책에 대해서 유사도를 더한 값을 계산 
            # 343번째 책과 248의 유사도가 모두 결합된 유사도
            total_cosine_sim += cosine_matrix[book_]
            
              # 4. 3에서 유사도가 가장 높은 순서대로 추출
        sim_scores = [(bookid2book[book2id[i]], c) for i, c in enumerate(total_cosine_sim) if i not in seen] # 자기 자신을 제외한 영화들의 유사도 및 인덱스를 추출 
        recs = sorted(sim_scores, key = lambda x: x[1], reverse=True)[0:300] # 유사도가 높은 순서대로 정렬 
        for rec in recs: 
            if rec not in seen:
                rec_list.append(rec)   
        
    # 그렇지 않으면 인기도 기반의 추천 
    else: 
        seen = read_list1[read_list1['user_id'] == user]['unique'].values[0]
        for rec in popular_rec_model[0:400]:
            if rec not in seen:
                rec_list.append(rec)
                
    total_rec_list[user] = rec_list[0:200]


# In[ ]:




