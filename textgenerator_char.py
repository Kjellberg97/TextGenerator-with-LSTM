# -*- coding: utf-8 -*-
"""
Created on Thu May 27 20:31:49 2021

@author: Viktor Kjellberg
"""
"""
Denna fil genererar text genom att generera en tecken i taget

"""
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.optimizers import RMSprop, Adam
from sklearn.model_selection import train_test_split
import numpy as np
import random
import pickle


f=open("xData", "rb")
x_data=np.load(f)
f.close()

f=open("yData", "rb")
y_data=np.load(f)
f.close()

data = open("reddit_cleaned.txt", "rb")
text = data.read()
text = text.decode()
data.close()

# laddar in dict av tecknen
f=open("char_indices.pkl", "rb")
char_indices=pickle.load(f)
f.close()
indices_char = dict(map(reversed, char_indices.items()))

x_train,x_test,y_train, y_test= train_test_split(x_data,y_data, test_size=0.2, random_state=4)

split = int(len(x_test)/2)
x_val = x_test[:split]
y_val = y_test[:split]
x_test = x_test[split:]
y_test = y_test[split:]

char_len= x_data.shape[2]
maxlen=x_data.shape[1]

# build the model: a single LSTM
print('Build model...')
model = Sequential()
model.add(LSTM(128, input_shape=(maxlen, char_len)))
model.add(Dense(char_len, activation='softmax'))

optimizer = Adam(lr=0.001)

model.compile(loss='categorical_crossentropy', optimizer=optimizer)

model.summary()

callbacks= keras.callbacks.EarlyStopping(monitor="val_loss", patience=3)

model.fit(x_train, y_train,
          batch_size=16,
          epochs=10,
          validation_data=(x_val, y_val),
          callbacks=callbacks)

model.save("./saved_model")

model=keras.models.load_model("./saved_model")

acc = model.evaluate(x_test, y_test)
print(acc)

# softmax och text_gen är delvis inspirerade av:
#https://keras.io/examples/generative/lstm_character_level_text_generation/

def softmax(pred):
    """
    Denna funktion är i princip en soft max funktion som används för att 
    bestämma vad nästa värdet från modellens prediction är. 
    """
    p = np.zeros(len(pred))
    pred = pred.astype("float64")
    pred = np.exp(np.log(pred) / 1)
    pred = pred / np.sum(pred)
    probas = np.random.multinomial(1, pred, 1)
    
    p[np.argmax(probas)]=1
    max_element=np.argmax(probas)

    return p,max_element

def text_gen(model, x_test,char_len,maxlen):
    
    print("-----------------------------\nGenerating text....\n")
    # Tar en slumpmässing datapunkt från test datan.
    start = random.randint(0, len(x_test)-1)
    
    start_text = x_test[start]
    s=""
    # Omvandlar den från en lista av vektorer till text
    for i in start_text:
        ind=np.where(i==np.amax(i))
        ind=ind[0][0]
        char = indices_char[ind]
        s=s+char
        
    start_text=start_text.reshape(1,maxlen,char_len)

    generated_text=""
    """
    Denna for loop använder modellen för att predicera nästa tecken baserat på 
    start datan. Detta tecken läggs sedan in i slutet av listan samtidigt som 
    det första tecknet tas bort. Denna nya lista används sedan för att 
    predicera näst tecken osv. 
    """    
    for i in range(300):
        start_text=start_text.reshape(1,maxlen,char_len)
        pred= model.predict(start_text, verbose=0)[0]
        pred, index =softmax(pred)
        start_text=start_text.reshape(maxlen,char_len)
        start_text= start_text[1:]
        start_text = np.append(start_text, pred)
        generated_text += indices_char[index]
    
    print("Inside (...) is used to start the generation of text.")
    print("Generated text: (", s, ")", generated_text)
        

#Genererar txt baserat på slumpmässig data punt ifrån x_test
text_gen(model, x_test,char_len,maxlen)
text_gen(model, x_test,char_len,maxlen)
text_gen(model, x_test,char_len,maxlen)
        
        
    
    
       
    
    
    
  
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    




