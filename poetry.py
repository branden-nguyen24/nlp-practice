import tensorflow as tf
import numpy as np

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

tokenizer = Tokenizer()

data = "In the town of Athy one Jeremy Lanigan\n Battered away til he hadnt a pound.\n His father died and made him a man again\n Left him a farm and ten acres of ground.\n He gave a grand party for friends and relations\n Who didnt forget him when come to the wall,\n And if youll but listen Ill make your eyes glisten\n Of the rows and the ructions of Lanigan's Ball.\n Myself to be sure got free invitation,\n For all the nice girls and boys I might ask,\n And just in a minute both friends and relations\n Were dancing round merry as bees round a cask.\n Judy ODaly, that nice little milliner,\n She tipped me a wink for to give her a call,\n And I soon arrived with Peggy McGilligan\n Just in time for Lanigan's Ball.\n"

corpus = data.lower().split("\n")

tokenizer.fit_on_texts(corpus)
total_words = len(tokenizer.word_index) + 1  # <OOV> token

input_sequences = []
for line in corpus:
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)

max_sequence_len = max([len(x) for x in input_sequences])

input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))

xs = input_sequences[:, :-1]
labels = input_sequences[:, -1]
ys = tf.keras.utils.to_categorical(labels, num_classes=total_words)

model = tf.keras.Sequential()
model.add(tf.keras.layers.Embedding(total_words, 240, input_length=max_sequence_len-1))
model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(150)))
model.add(tf.keras.layers.Dense(total_words, activation='softmax'))
adam = tf.keras.optimizers.Adam(learning_rate=0.01)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
history = model.fit(xs, ys, epochs=100, verbose=1)

seed_text = "I made a poetry machine"
next_words = 20

for _ in range(next_words):
    token_list = tokenizer.texts_to_sequences([seed_text])[0]
    token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
    #  predicted = model.predict_classes(token_list, verbose=0)
    predict_x = model.predict(token_list)
    predicted = np.argmax(predict_x, axis=1)
    output_word = ""
    for word, index in tokenizer.word_index.items():
        if index == predicted:
            output_word = word
            break
    seed_text += " " + output_word
print(seed_text)





