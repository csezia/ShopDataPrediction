#!/usr/bin/env python
# coding: utf-8

# In[37]:


import pandas as pd
import streamlit as st


# In[38]:


df=pd.read_csv('shop data.csv')


# In[39]:


df


# In[40]:


x=df.drop(['buys'], axis=1)


# In[41]:


x


# In[42]:


y=df['buys']


# In[43]:


y


# In[44]:


from sklearn.preprocessing import LabelEncoder


# In[45]:


le_x=LabelEncoder


# In[46]:


x=x.apply(LabelEncoder().fit_transform)
x


# In[47]:


from sklearn.model_selection import train_test_split


# In[48]:


xtrain, xtest, ytrain, ytest=train_test_split(x,y,test_size=0.25, random_state=0)


# In[49]:


xtest


# In[50]:


from sklearn.linear_model import LogisticRegression


# In[51]:


model=LogisticRegression()


# In[52]:


model.fit(xtrain,ytrain)


# In[53]:


model.predict(xtest)


# In[54]:


model.predict([[1,0,1,0]])


# In[55]:


import pickle


# In[56]:


with open('lecture.pkl','wb') as file:
    pickle.dump(model,file)


# In[57]:


with open('lecture.pkl','rb') as file:
    mp = pickle.load(file)


# In[ ]:




