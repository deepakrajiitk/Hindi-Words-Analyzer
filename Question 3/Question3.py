#!/usr/bin/env python
# coding: utf-8

# ## Question 3

# ### Importing libraries

# In[103]:


import numpy as np
import re
from nltk import ngrams
from collections import Counter


# In[104]:


vowels = ["ा", "ि", "ी", "ु", "ू", "े", "ो", "ै", "ौ", "ृ", "ॄ", "ॉ", "ं", "्","अ", "आ", "इ", "ई", "उ", "ऊ", "ऋ", "ॠ", "ए", "ऐ", "ओ", "औ", "अं", "अः"]
punctuations=["।",";",",",":","!",'"',"?",":-","-","{","(","}",")","_","०","S","―","=","[","]","......",":-",".","॥",'”',"|"]


# In[105]:


# function to find ngrams
def find_ngrams(doc, n):
  return list(zip(*(doc[i:] for i in range(n))))


# In[106]:


# function to change words
def change_word(word):
    new_word = ""
    l = len(word)
    for i in range(l-1):
        new_word += word[i]
        if word[i] not in vowels and word[i+1] not in vowels and not word[i].isdigit():
            new_word += 'अ'
#     print(new_word)
#     print(l)
    new_word += word[l-1]
    if word[l-1] not in vowels and not word[l-1].isdigit():
        new_word += 'अ'
    return new_word


# In[107]:


# function to split words
def split_word(word):
    temp = ""
    split_words = []
    for i in range(len(word)):
        temp += word[i]
        if word[i] in vowels:
            if word[i]=='अ' and i!=0:
                temp = temp[:-1]
            split_words.append(temp)
            temp = ""
    if temp != "":
        split_words.append(temp)
    return split_words


# In[108]:


for i in "अपअने":
    print(i)


# In[109]:


split_word("क्षअत्रियअ")


# In[110]:


change_word('क्षत्रिय')


# In[111]:


unigrams_char = {}
bigrams_char = {}
trigrams_char = {}
quadgrams_char= {}
unigrams_words = {}
bigrams_words = {}
trigrams_words = {}
unigrams_syllables = {}
bigrams_syllables = {}
trigrams_syllables = {}

syllable_words = []

syllable_words_dict = {}
change_word_dict = {}

count = 0


# In[112]:


import time

start = time.time()
print("starting time", start)
with open('hi/hi.txt',encoding= 'utf-8') as f:
    for line in f:
        syllable_words = []
        if count>500000:
            break
        count+=1
        
# -------------------- Removing unnecessary part of words -------------------
        
        new_line = ""
        for i in line:
            if i not in punctuations:
                new_line += i
        line = new_line
        line = line.replace("\n","")
        line = str(" ".join(line.split()))
    
# ---------------------------------------------------------------------------

        words = line.split(" ")
        for word in words:
            if word == "":
                continue
            try:
                if word not in change_word_dict:
                    new_word = change_word(word)
                    change_word_dict[word] = new_word
                else:
                    new_word = change_word_dict[word]

# -------------------- Finding n-grams for characters --------------------------
                
        
                for i in new_word:
                    if i not in unigrams_char:
                        unigrams_char[i]=1
                    else:
                        unigrams_char[i]+=1
                
                for i in find_ngrams(new_word, 2):
                    if i not in bigrams_char:
                        bigrams_char[i]=1
                    else:
                        bigrams_char[i]+=1
                
                for i in find_ngrams(new_word, 3):
                    if i not in trigrams_char:
                        trigrams_char[i]=1
                    else:
                        trigrams_char[i]+=1
                
                for i in find_ngrams(new_word, 4):
                    if i not in quadgrams_char:
                        quadgrams_char[i]=1
                    else:
                        quadgrams_char[i]+=1

    # ------------------- Finding n-grams for syllables ---------------------------

                if new_word not in syllable_words_dict:
                    split_words = split_word(new_word)
                    syllable_words_dict[new_word] = split_words
                else:
                    split_words = syllable_words_dict[new_word]
                
                for i in split_words:
                    if i not in unigrams_syllables:
                        unigrams_syllables[i]=1
                    else:
                        unigrams_syllables[i]+=1

                for i in find_ngrams(split_words, 2):
                    if i not in bigrams_syllables:
                        bigrams_syllables[i]=1
                    else:
                        bigrams_syllables[i]+=1

                for i in find_ngrams(split_words, 3):
                    if i not in trigrams_syllables:
                        trigrams_syllables[i]=1
                    else:
                        trigrams_syllables[i]+=1
            except:
                print("Error in word",word)
            

# -------------------- Finding n-grams for words -----------------------------
        
        for i in words:
            if i not in unigrams_words:
                unigrams_words[i]=1
            else:
                unigrams_words[i]+=1
        
        for i in find_ngrams(words, 2):
            if i not in bigrams_words:
                bigrams_words[i]=1
            else:
                bigrams_words[i]+=1

        for i in find_ngrams(words, 3):
            if i not in trigrams_words:
                trigrams_words[i]=1
            else:
                trigrams_words[i]+=1
        
print("ending time",time.time()-start)


# In[113]:


count


# In[114]:


unigrams_char


# In[115]:


bigrams_char


# In[116]:


trigrams_char


# In[117]:


quadgrams_char


# In[118]:


f = open("top_uni_char.txt",'w',encoding='utf-8')
f.write(str(sorted(unigrams_char, key=unigrams_char.get, reverse=True)[:100]))
f.close()


# In[119]:


f = open("top_bi_char.txt",'w',encoding='utf-8')
f.write(str(sorted(bigrams_char, key=bigrams_char.get, reverse=True)[:100]))
f.close()


# In[120]:


f = open("top_tri_char.txt",'w',encoding='utf-8')
f.write(str(sorted(trigrams_char, key=trigrams_char.get, reverse=True)[:100]))
f.close()


# In[121]:


f = open("top_quad_char.txt",'w',encoding='utf-8')
f.write(str(sorted(quadgrams_char, key=quadgrams_char.get, reverse=True)[:100]))
f.close()


# In[122]:


unigrams_words


# In[123]:


bigrams_words


# In[124]:


trigrams_words


# In[125]:


f = open("top_uni_words.txt",'w',encoding='utf-8')
f.write(str(sorted(unigrams_words, key=unigrams_words.get, reverse=True)[:100]))
f.close()


# In[126]:


f = open("top_bi_words.txt",'w',encoding='utf-8')
f.write(str(sorted(bigrams_words, key=bigrams_words.get, reverse=True)[:100]))
f.close()


# In[127]:


f = open("top_tri_words.txt",'w',encoding='utf-8')
f.write(str(sorted(trigrams_words, key=trigrams_words.get, reverse=True)[:100]))
f.close()


# In[128]:


unigrams_syllables


# In[129]:


bigrams_syllables


# In[130]:


trigrams_syllables


# In[131]:


f = open("top_uni_char.txt",'w',encoding='utf-8')
f.write(str(sorted(unigrams_syllables, key=unigrams_syllables.get, reverse=True)[:100]))
f.close()


# In[132]:


f = open("top_uni_syllables.txt",'w',encoding='utf-8')
f.write(str(sorted(bigrams_syllables, key=bigrams_syllables.get, reverse=True)[:100]))
f.close()


# In[133]:


f = open("top_tri_syllables.txt",'w',encoding='utf-8')
f.write(str(sorted(trigrams_syllables, key=trigrams_syllables.get, reverse=True)[:100]))
f.close()


# ## Question 4

# In[134]:


import matplotlib.pyplot as plt

#convert value of frequency to numpy array
s = list(unigrams_char.values())
s = np.array(s)

#Calculate zipf and plot the data
count, bins, ignored = plt.hist(s[s<50], 50)
plt.show()


# In[135]:


#convert value of frequency to numpy array
s = list(bigrams_char.values())
s = np.array(s)

#Calculate zipf and plot the data
count, bins, ignored = plt.hist(s[s<50], 50)
plt.show()


# In[136]:


#convert value of frequency to numpy array
s = list(trigrams_char.values())
s = np.array(s)

#Calculate zipf and plot the data
count, bins, ignored = plt.hist(s[s<50], 50)
plt.show()


# In[137]:


#convert value of frequency to numpy array
s = list(unigrams_words.values())
s = np.array(s)

#Calculate zipf and plot the data
count, bins, ignored = plt.hist(s[s<50], 50)
plt.show()


# In[138]:


#convert value of frequency to numpy array
s = list(trigrams_words.values())
s = np.array(s)

#Calculate zipf and plot the data
count, bins, ignored = plt.hist(s[s<50], 50)
plt.show()


# In[139]:


#convert value of frequency to numpy array
s = list(trigrams_words.values())
s = np.array(s)

#Calculate zipf and plot the data
count, bins, ignored = plt.hist(s[s<50], 50)
plt.show()


# In[140]:


#convert value of frequency to numpy array
s = list(unigrams_syllables.values())
s = np.array(s)

#Calculate zipf and plot the data
count, bins, ignored = plt.hist(s[s<50], 50)
plt.show()


# In[141]:


#convert value of frequency to numpy array
s = list(bigrams_syllables.values())
s = np.array(s)

#Calculate zipf and plot the data
count, bins, ignored = plt.hist(s[s<50], 50)
plt.show()


# In[142]:


#convert value of frequency to numpy array
s = list(trigrams_syllables.values())
s = np.array(s)

#Calculate zipf and plot the data
count, bins, ignored = plt.hist(s[s<50], 50)
plt.show()


# ### All of the above follow Zipfian distribution
