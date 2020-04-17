# Wade Strain
# AI - RNN Homework Assignment
# April 7, 2020
# adapted from https://machinelearningmastery.com/how-to-develop-a-word-level-neural-language-model-in-keras/
# https://www.pythoncentral.io/how-to-pickle-unpickle-tutorial/

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Import Libraries Section
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
import pandas as pd
import numpy as np
from pickle import dump
from keras.preprocessing.text import text_to_word_sequence,Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Embedding, Dense, LSTM, Dropout, Flatten
from keras.callbacks import ModelCheckpoint
import datetime as dt
import matplotlib.pyplot as plt
import string

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Parameters Section
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
filename = 'abcnews-date-text.csv'
cleaned_filename = 'cleanedup-headlines-file.csv'
seq_length = 9
max_words = 10000        # consider only top 1,000 words in dataset, for memory sake
num_samples = 15000

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Load Data Section
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# dataframe = pd.read_csv(filename, delimiter=',')
# dataframe = dataframe.drop(columns=['publish_date'])
# dataframe = dataframe.sample(n=num_samples, replace=False, random_state=32)   # shrink number of headlines, for computer efficiency
# dataframe.to_csv(cleaned_filename, index=False, header=False)

text = open(cleaned_filename, encoding='utf8').read()
# words = text_to_word_sequence(text)
words = text.split()
table = str.maketrans('', '', string.punctuation)
words = [w.translate(table) for w in words]
words = [word for word in words if word.isalpha()]

print('Total Tokens: %d' % len(words))
print('Unique Tokens: %d' % len(set(words)))

# create sequences of tokens
length = seq_length+1
headlines = list()
for i in range(length, len(words)):
    seq = words[i-length:i]
    headline = ' '.join(seq)
    headlines.append(headline)

start_time = dt.datetime.now()
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Pretreat Data Section
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# vectorize the headlines
print('Vectorizing...')
tokenizer = Tokenizer()
tokenizer.fit_on_texts(headlines)
sequences = tokenizer.texts_to_sequences(headlines)
num_of_words = len(tokenizer.word_index)+1
# print('Found %s unique tokens' % num_of_words)

# to pad shorter sequences with 0 to make sure each sequence is the same size
sequences = np.array(sequences)
print('Splitting input & ouptut...')
X, y = sequences[:,:-1], sequences[:,-1]

del headlines  # to save memory
del words
del sequences
y = to_categorical(y)    # one hot encode output variable
seq_length = X.shape[1]

print(X.shape)
print(y.shape)
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Define Model Section
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
model = Sequential()
model.add(Embedding(num_of_words, 100, input_length=seq_length))
# model.add(LSTM(300, return_sequences=True))
# model.add(LSTM(300, return_sequences=True))
model.add(LSTM(600, return_sequences=True))
model.add(LSTM(600))
# model.add(Dense(300, activation='relu'))
model.add(Dense(num_of_words, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Train Model Section
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# filepath="Model-LSTM-weights-improvement-{epoch:02d}-{loss:.4f}-dropout.hdf5"
# checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=2, period = 10, mode='min')
# callbacks_list = [checkpoint]
#
# history = model.fit(X, y, batch_size=512, epochs=50, callbacks=callbacks_list)

history = model.fit(X, y, batch_size=512, epochs=50)

stop_time = dt.datetime.now()
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Show output Section
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
print(model.summary())
print("Time required for training:", stop_time-start_time)

# plot training loss vs. number of training epochs
plt.plot(history.history['loss'])
plt.title('training loss vs. num training epochs')
plt.ylabel('loss')
plt.xlabel('epochs')
plt.show()

# save the model to file
model.save('model.h5')
# save the tokenizer
dump(tokenizer, open('tokenizer.pkl', 'wb'))
