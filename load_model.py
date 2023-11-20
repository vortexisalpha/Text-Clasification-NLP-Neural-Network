import tensorflow as tf
from tensorflow import keras
import numpy as np

data = keras.datasets.imdb

#only take the most common 88000 words
(train_data, train_labels ), ( test_data, test_labels) = data.load_data(num_words=88000)


#this just translates the numbers into strings
word_index = data.get_word_index()
word_index = {k:(v+3) for k,v in word_index.items()}
word_index['<PAD>'] = 0
word_index['<START>'] = 1
word_index['<UNK>'] = 2
word_index['<UNUSED>'] = 3

reverse_word_index = dict([(value,key) for (key,value) in word_index.items()])
#edit our data so that its all the same length:
#this makes it all a length of 250 adding <PAD> tags if its shorter and cutting it if its too big
train_data = keras.preprocessing.sequence.pad_sequences(train_data, value=word_index['<PAD>'], padding='post' , maxlen=250)
test_data = keras.preprocessing.sequence.pad_sequences(test_data, value=word_index['<PAD>'], padding='post' , maxlen=250)


def decode_review(text):
    including_excess = ' '.join([reverse_word_index.get(i,'?') for i in text])
    excluding_excess = including_excess.replace('<PAD>', '').replace('<START>', '').replace('<UNK>', '')
    return excluding_excess

model = keras.models.load_model('model.h5')

def review_encode(s):
    encoded = [1]

    for word in s:
        if word.lower() in word_index:
            encoded.append(word_index[word.lower()]) #encode it from the word index dictionary 
        else:
            encoded.append(2) # if its not in our vocabulary make it <UNK>
    return encoded

with open('shawshank.txt', encoding='utf-8') as f:
    for line in f.readlines():
        nline = line.replace(',','').replace('.','').replace(')','').replace('(','').replace(':','').replace('\"','').strip().split(' ')
        encode = review_encode(nline)
        encode = keras.preprocessing.sequence.pad_sequences([encode], value=word_index['<PAD>'], padding='post' , maxlen=250)
        predict = model.predict(encode)
        print(line)
        print(encode)
        print(predict[0])
        assessment = ['negative', 'positive']
        prediction_accuracy = predict[0]
        prediction = 1
        if prediction_accuracy < 0.5: 
            prediction = 0
            prediction_accuracy = 1- prediction_accuracy
        prediction_accuracy = prediction_accuracy*100
        
        print('AI predicts: ' + str(assessment[int(prediction)]) +' with %' + str(int(prediction_accuracy))+ ' accuracy')