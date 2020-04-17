# Wade Strain
# AI - RNN Assignment - text generation
# April 9, 2020

from random import randint
from pickle import load
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences

# generate sequence from my language model
def generate_seq(model, tokenizer, seq_length, seed_text, n_words):
    result = list()
    in_text = seed_text
    # generate a fixed number of words
    for _ in range(n_words):
        # encode the text as integer
        encoded = tokenizer.texts_to_sequences([in_text])[0]
        # truncate sequences to a fixed length
        encoded = pad_sequences([encoded], maxlen=seq_length, truncating='pre')
        # predict probabilities for each word
        yhat = model.predict_classes(encoded, verbose=0)
        # map predicted word index to word
        out_word = ''
        for word, index in tokenizer.word_index.items():
            if index == yhat:
                out_word = word
                break
        # append to input
        in_text += ' ' + out_word
        result.append(out_word)
    return ' '.join(result)

# load and read csv file into memory as text
cleaned_filename = 'cleanedup-headlines-file.csv'
text = open(cleaned_filename, encoding='utf8').read()
words = text.split()

seq_length = 8
# create the text sequences
length = seq_length+1
headlines = list()
for i in range(length, len(words)):
    seq = words[i-length:i]
    headline = ' '.join(seq)
    headlines.append(headline)

model = load_model('model.h5')
tokenizer = load(open('tokenizer.pkl', 'rb'))

for i in range(15):
    seed_text = headlines[randint(0,len(headlines))]
    print('Seed text: ' + seed_text)
    generated = generate_seq(model, tokenizer, seq_length+1, seed_text, randint(4,8))
    print('Generated: ' + generated + '\n')
