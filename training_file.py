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
    excluding_excess = including_excess.replace('<PAD>', '')
    excluding_excess = excluding_excess.replace('<START>', '')
    excluding_excess = excluding_excess.replace('<UNK>', '')
    return excluding_excess

###NEURAL MODEL###

model = keras.Sequential()
model.add(keras.layers.Embedding(88000,16)) #Embedding layer groups words on similarity
#It finds word vectors for each parsed word (^in this case we are using 16 dimensions^)

model.add(keras.layers.GlobalAveragePooling1D())
#global average pooling puts our data in a lower dimension for computational purposes

model.add(keras.layers.Dense(16,activation='relu'))
#regular hidden layer 16 neurons relu activation function

model.add(keras.layers.Dense(1,activation='sigmoid')) 
#output layer: sigmoid as we want a value between 0-1

model.summary()

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
#binary_crossentropy means our loss function is adapted adjust values based on 1 and 0

###TRAINING THE NEURAL MODEL###

#splitting our data into 2 sets so the nnetwork isnt working off memorisation

x_val = train_data[:10000]
x_train = train_data[10000:]

y_val = train_labels[:10000]
y_train = train_labels[10000:]

fitModel = model.fit(x_train, y_train, epochs=40, batch_size=512, validation_data=(x_val,y_val), verbose=1)
#batch size is the amount of reviews we feed the NN at once

#prints the accuracy
results = model.evaluate(test_data,test_labels)
print('model accuracy: '+ str(results[1]))

model.save('model.h5')

###TESTING THE NEURAL MODEL###
'''
assesment = ['negative', 'positive']

number_to_predict = 12
n = number_to_predict 

test_review = test_data[n] #review we are testing
predict = predict = model.predict(np.array([test_review])) #predict if it is good or bad
print('Review: ')
print(decode_review(test_review))

prediction_certainty = predict[0]
if prediction_certainty < 0.5:
    prediction_certainty = (1.-prediction_certainty)
prediction_certainty = prediction_certainty*100


print('Prediction: ' + str(assesment[int(predict[0])]) + ' with %' +str(int(prediction_certainty)) + ' certainty')
print('Actual: ' + str(assesment[test_labels[n]]) + ' : '+ str(test_labels[n]))
'''