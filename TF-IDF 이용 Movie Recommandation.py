#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


# In[5]:


import os
print(os.listdir("../desktop/DT/movies"))


# In[8]:


path = "../desktop/DT/movies/"
# 대용량 데이터를 불러오는 경우 각 컬럼의 데이터 타입(dtype)을 추측하느라
# 메모리 할당이 많으므로 대용량 데이터를 불러 올때 메모리 에러가 난다면 
# low_memory를 False로 설정하는 것을 권장
data = pd.read_csv(path + 'movies_metadata.csv', low_memory=False)
data.head()


# In[10]:


# overview 항목 추출
data.columns


# In[11]:


# 데이터 전처리
# overview의 결측치가 있는 항목은 모두 제거

data = data[data['overview'].notnull()].reset_index(drop=True)
data.shape


# In[27]:


data = data.loc[0:5000].reset_index(drop=True)


# In[29]:


# 불용어 : 유의미하지 않은 단어 토큰 제거
tfidf = TfidfVectorizer(stop_words='english', max_features=10000)

# overview에 대해서 tf-idf 수행
tfidf_matrix = tfidf.fit_transform(data['overview'])
print(tfidf_matrix.shape)


# In[30]:


from sklearn.metrics.pairwise import cosine_similarity
cosine_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)


# In[32]:


cosine_matrix.shape


# In[33]:


np.round(cosine_matrix,4)


# In[35]:


for i,c in enumerate(data['title']):
    print(i)
    print(c)
    break


# In[36]:


# movie title과 id를 매핑할 dictionary 생성
movie2id = {}
for i,c in enumerate(data['title']) : 
    movie2id[i] =c
    
# id와 movie title을 매핑할 dictionary를 생성함
id2movies = {}
for i, c in movie2id.items() :
    id2movies[c] = i


# In[38]:


# Toy Stroy의 id 추출
idx = id2movies['Toy Story'] # Toy Story는 0번 인덱스
sim_scores= [(i,c) for i, c in enumerate(cosine_matrix[idx])
            if i != idx] # 자기 자신을 제외한 영화들의 유사도 및 인덱스 추출
sim_scores= sorted(sim_scores, key = lambda x: x[1], reverse=True)

#ㅇ사도가 높은 순서대로 정렬
sim_scores[0:10]


# In[39]:


sim_scores = [(movie2id[i], score) for i, score in sim_scores[0:10]]
sim_scores


# In[ ]:




