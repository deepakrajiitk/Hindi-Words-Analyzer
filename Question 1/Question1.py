#!/usr/bin/env python
# coding: utf-8

# ### Installing required m`odules

# In[1]:


get_ipython().system('pip install gensim')


# ### importing required libraries

# In[2]:


import random
import numpy as np
from gensim.models import Word2Vec
import pandas as pd
import glob


# ### Loading dataset

# In[4]:


hindi_data = open('hindi.txt',encoding='utf-8').read()
dataset = []
for i in hindi_data.split('\n'):
    dataset.append(i.split(',')[:3])


# In[5]:


dataset = dataset[:65]
ground_truth_score = [i[2] for i in dataset]
# ground_truth_score


# ### Creating Ground Truth

# In[6]:


ground_truth = {'0.4':[],'0.5':[],'0.6':[],'0.7':[],'0.8':[]}
for j in dataset[:65]:
    if float(j[2])>4:
        ground_truth['0.4'].append(1)
    else:
        ground_truth['0.4'].append(0)
    if float(j[2])>5:
        ground_truth['0.5'].append(1)
    else:
        ground_truth['0.5'].append(0)
    if float(j[2])>6:
        ground_truth['0.6'].append(1)
    else:
        ground_truth['0.6'].append(0)
    if float(j[2])>7:
        ground_truth['0.7'].append(1)
    else:
        ground_truth['0.7'].append(0)
    if float(j[2])>8:
        ground_truth['0.8'].append(1)
    else:
        ground_truth['0.8'].append(0)


# In[7]:


ground_truth


# In[9]:


folders1 = ['50','100','200','300']
folders2 = ['cbow','fastext','sg']
thresholds = [0.4, 0.5, 0.6, 0.7,0.8]


# ### Loading required models

# In[10]:


from gensim.models import FastText
models = []
for folder1 in folders1:
    for folder2 in folders2:
        for file in glob.glob("hi/"+folder1+"/"+folder2+"/"+"*.model"):
            if folder2=="fasttext":
                models.append(FastText.load(file))
            else:
                models.append(Word2Vec.load(file))


# In[ ]:


models


# In[ ]:


# this function calculates cosine similarities between two vectors
def find_cosine_sim(vector1, vector2):
    dot = np.dot(vector1, vector2)
    vector1_norm = np.linalg.norm(vector1)
    vector2_norm = np.linalg.norm(vector1)
    return dot/(vector1_norm*vector2_norm)


# In[ ]:


# function to calcute truth
def find_vector(dataset, model, threshold):
    vector = []
    sim = []
    for i in dataset:
        vector1 = model.wv[i[0]]
        vector2 = model.wv[i[1]]
        cosine_sim = find_cosine_sim(vector1, vector2)
        sim.append(cosine_sim)
        if cosine_sim>=threshold:
            vector.append(1)
        else:
            vector.append(0)
    return vector, sim


# In[ ]:


# function to find accuracy
def find_accuracy(result_vector, ground_truth):
    total_matches = len([i for i, j in zip(result_vector, ground_truth) if i==j])
    l = len(result_vector)
    return ((total_matches)/l)*100


# ### Saving Files

# In[ ]:


j=0
for model in models:
    i+=1
    for threshold in thresholds:
        vector, sims = find_vector(dataset, model, threshold)
        print(find_accuracy(vector, ground_truth[str(threshold)]))
        df = pd.DataFrame()
        df['Word1'] = [i[0] for i in dataset]
        df['Word2'] = [i[1] for i in dataset]
        df['Similarity Score'] = sims
        df['Ground Truth'] = ground_truth_score
        df['Label'] = vector
        df.to_csv(folders2[models.index(model)]+"_"+folder1[j]+"_"+str(threshold)+".csv")
    if i==3:
        j += 1
        


# ### Different preprocessing for glove model

# In[ ]:


count = 0
glove_dict = {}
with open("hi/50/glove/hi-d50-glove.txt",encoding= 'utf-8',errors='replace') as f:
    for line in f:
        line = line.strip("\n")
        temp = line.split(" ")
        glove_dict[temp[0]] = temp[1:]


# In[ ]:


for i in glove_dict:
    glove_dict[i] = [float(j) for j in glove_dict[i]]


# In[ ]:


def find_vector_glove(dataset, threshold, glove_dict):
    vector = []
    sim = []
    for i in dataset:
        vector1 = np.array(glove_dict[i[0]])
        vector2 = np.array(glove_dict[i[1]])
        cosine_sim = find_cosine_sim(vector1, vector2)
        sim.append(cosine_sim)
        if cosine_sim>=threshold:
            vector.append(1)
        else:
            vector.append(0)
    return vector, sim


# In[ ]:


for threshold in thresholds:
        vector, sims = find_vector_glove(dataset, threshold, glove_dict)
        print(find_accuracy(vector, ground_truth[str(threshold)]))
        df = pd.DataFrame()
        df['Word1'] = [i[0] for i in dataset]
        df['Word2'] = [i[1] for i in dataset]
        df['Similarity Score'] = sims
        df['Ground Truth'] = ground_truth_score
        df['Label'] = vector
        df.to_csv("glove_"+str(threshold)+".csv")

