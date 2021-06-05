# -*- coding: utf-8 -*-
"""
Created on Fri Jun  4 09:36:24 2021

@author: kjell
"""
from keras.preprocessing.text import Tokenizer
import pickle
import numpy as np

# Data hantering
data_file = "reddit_short_stories.txt"
cleaned_file= "reddit_cleaned_word.txt"

# tar bort innehållet från filen
open(cleaned_file, "w").close()

f = open(data_file,"r")

# Tecken som ska tas bort från texten
blacklisted_char = '#$%&()*+-/:;<=>@[\\]^_`{|}~\t\n1234567890'
new_line=set('!.?')

# Använder keras Tokenizer för att ta de vanligaste 4000 orden samt ta bort 
# onödiga tecken
tokenizer=Tokenizer(num_words=4000, filters=blacklisted_char, lower=True, oov_token=("OOV"))

text=[]
text_all=""


for lines in f:
    # Tar bort <sos> och <eos> i varje novell
    lines=lines[5:-4]
    # Separera .!,? med ordet framför för att kunna vektorisera dessa 
    # tecken enskillt
    lines = lines.replace(".", " .")
    lines = lines.replace("!", " !")
    lines = lines.replace("?", " ?")
    lines = lines.replace(",", " ,")
    
    # Adderar ihop alla noveller
    text_all+=lines
    
    # Delar upp texten i meningar för att kunna använda keras tokenizer
    sentence=""
    for i in lines:
        
        if i not in new_line:
            sentence+=i
        else:
            sentence+=i
            text.append(sentence)
            sentence=""
        
del f
tokenizer.fit_on_texts(text)
words = text_all.split()

del text_all
#--------------------------------------------
# Koden mellan de sträckade linjerna är till stor del hämtad från 
# https://keras.io/examples/generative/lstm_character_level_text_generation/
maxlen_of_sentence = 10
steps = 6
sentences = []
next_word=[]

lengh=int((len(words)-maxlen_of_sentence-1)/3)

for i in range(0, lengh, steps):
    sentences.append(words[i:i+maxlen_of_sentence])
    next_word.append(words[i+maxlen_of_sentence])
#-------------------------------------------

#Kodar texten till index i dict. 
data_tokenized=[]
for i in sentences:
    data=tokenizer.texts_to_sequences(i)
    data_tokenized.append(data)
next_tokenized=tokenizer.texts_to_sequences(next_word)

# Förvanlar next_tokenized till vektorer
data_next=tokenizer.sequences_to_matrix(next_tokenized)
data_next =data_next.astype(bool)

# Sparar data_next
f = open("yData_word", "wb")
np.save(f,data_next)
f.close()

del data_next

data_ind=[]

# Förvalar datan till vektorer av typen bool
for i in data_tokenized:
    d = tokenizer.sequences_to_matrix(i)
    d = d.astype(bool)
    data_ind.append(d)

del data_tokenized 
 
del next_tokenized
dict_tokenizer = tokenizer.word_index

# Sparar data i filer
f=open("tokenizer_word_index.pkl", "wb")
pickle.dump(dict_tokenizer, f)
f.close()

f = open("xData_word", "wb")
np.save(f,data_ind)
f.close()


    
    
    
    
    
    
    
    
    