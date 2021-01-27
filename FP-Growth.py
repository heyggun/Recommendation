#!/usr/bin/env python
# coding: utf-8

# In[4]:


get_ipython().system('pip install mlxtend')


# In[5]:


import mlxtend
import numpy as np
import pandas as pd


# In[6]:


data = np.array([
                 ['우유','기저귀','주스'],
                 ['양상추','기저귀','맥주'],
                 ['우유','양상추','기저귀','맥주'],
                 ['양상추','맥주']
])


# In[7]:


from mlxtend.preprocessing import TransactionEncoder
te = TransactionEncoder()
te_ary = te.fit(data).transform(data)
df = pd.DataFrame(te_ary, columns=te.columns_)
df


# In[8]:


get_ipython().run_cell_magic('time', '', '\nfrom mlxtend.frequent_patterns import apriori\napriori(df, min_support=0.5, use_colnames=True)')


# In[9]:


get_ipython().run_cell_magic('time', '', 'from mlxtend.frequent_patterns import fpgrowth\n\nfpgrowth(df, min_support=0.5, use_colnames=True)')


# In[ ]:




