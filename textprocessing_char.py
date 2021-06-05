# -*- coding: utf-8 -*-
"""
Created on Mon May 24 11:04:56 2021

@author: Viktor Kjellberg
"""
"""
Denna kod förbereder data för att kunna användas för att generera text baserat 
på tecken
"""
import pickle
import numpy as np

# Filer för att hämta och spara data
data_file = "reddit_short_stories.txt"
cleaned_file= "reddit_cleaned.txt"

# tar bort innehållet från filen
open(cleaned_file, "w").close()

#Öppnar filen med datamängden
f = open(data_file)

# Tecken som ska tas bort från texten
blacklisted_char = set('-^[]_;*%¤$@+-!#&()/')

# Går igenom novellerna en för en och tar bort onödiga tecken samt onödiga ord
# Slår sedan ihop alla noveller i en fil 
for lines in f:
        
   text = ""
   cleaned_text =""
        
   for i in lines:
       if i not in blacklisted_char:
           cleaned_text +=i
       else:
           cleaned_text +=" "
        
   words = cleaned_text.split()
        
   for w in words:
       if w != "<eos>" and w != "<sos>" and not w.startswith("http://") and not w.startswith("http:") and w != "\n":
           text = text +" " + w
        
        
   with open(cleaned_file, "a") as text_file:
       text_file.write(text)

f.close()

data = open(cleaned_file, "r")
data = data.read()

#--------------------------------------------
# Koden mellan de sträckade linjerna är till stor del hämtad från 
# https://keras.io/examples/generative/lstm_character_level_text_generation/

# Skapar ett dict som innehåller alla tecken och index för dessa
characters = sorted(list(set(data)))
char_indices = dict((c, i) for i, c in enumerate(characters))

maxlen_of_sentence = 50
steps = 6
sentences = []
next_label = []

# Hur stor del av datamängden som ska används
lengh=int ((len(data)-maxlen_of_sentence)/2)

# Går igenom datan och delar upp den i delar av 50 tecken samt det förväntade 
# predikterade resultatet. 
for i in range(0, lengh, steps):
    sentences.append(data[i:i+maxlen_of_sentence])
    next_label.append(data[i+maxlen_of_sentence])

# Omvandlar varje array i sentences till vektorer av storleken 50x78
# samt next_label till vektorer av storlek 1x78
x = np.zeros((len(sentences), maxlen_of_sentence, len(characters)), dtype=bool)
y = np.zeros((len(sentences), len(characters)),dtype=bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        x[i, t, char_indices[char]] = 1
    y[i, char_indices[next_label[i]]] = 1

#-------------------------------------------


# Sparar data i filer
f=open("char_indices.pkl", "wb")
pickle.dump(char_indices, f)
f.close()

f = open("xData", "wb")
np.save(f,x)
f.close()

f = open("yData", "wb")
np.save(f,y)
f.close()

