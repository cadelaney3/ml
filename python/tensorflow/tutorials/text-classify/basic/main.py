from __future__ import absolute_import, division, print_function

import tensorflow as tf 
from tensorflow import keras

import numpy as np 
import matplotlib.pyplot as plt 

print(np.__version__)
print(tf.__version__)

# downloads dataset
imdb = keras.datasets.imdb

# num_words keeps top 10k most frequent words in train data
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

print("Training entries: {}, labels: {}".format(len(train_data), len(train_labels)))

# each int reps a specific word in a dict
print(train_data[0])

# number of words in first and second reviews of dataset
print(len(train_data[0]), len(train_data[1]))

# convert int back to words

# dict mapping words to an int index
word_index = imdb.get_word_index()

# first indices are reserved
word_index = {k:(v+3) for k,v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2 # unknown
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])

print(decode_review(train_data[0]))

# use pad_sequences function to pad arrays so they are same length
train_data = keras.preprocessing.sequence.pad_sequences(train_data,
                                                        value=word_index["<PAD>"],
                                                        padding='post',
                                                        maxlen=256)

test_data = keras.preprocessing.sequence.pad_sequences(test_data,
                                                       value=word_index["<PAD>"],
                                                       padding='post',
                                                       maxlen=256)

print(len(train_data[0]), len(train_data[1]))
print(train_data[0])

# neural net created by stacking layers. Choose how many layers to use and how many hidden units to use for each layer
# input data consists of array of word-indices. Labels to predict are 0 or 1.

# input shape is the vocab count used for movie reviews (10k words)
vocab_size = 10000

model = keras.Sequential()
model.add(keras.layers.Embedding(vocab_size, 16))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation=tf.nn.relu))
model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))

model.summary()

# The first layer is an Embedding layer. This layer takes the integer-encoded vocabulary and 
# looks up the embedding vector for each word-index. These vectors are learned as the model trains. 
# The vectors add a dimension to the output array. The resulting dimensions are: (batch, sequence, embedding)

# Next, a GlobalAveragePooling1D layer returns a fixed-length output vector for each example by averaging over the sequence dimension. 
# This allows the model to handle input of variable length, in the simplest way possible.

# This fixed-length output vector is piped through a fully-connected (Dense) layer with 16 hidden units.

# The last layer is densely connected with a single output node. Using the sigmoid activation function, 
# this value is a float between 0 and 1, representing a probability, or confidence level.

# configure model to use an optimizer and a loss function
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['acc'])

# create validation set so we can check accuracy of model on data not seen before
# create validation set by setting apart 10k examples from original training data.
x_val = train_data[:10000]
partial_x_train = train_data[10000:]

y_val = train_labels[:10000]
partial_y_train = train_labels[10000:]

# train model for 40 epochs in mini-batches of 512 samples.
# this is 40 iterations over all samples in x_train & y_train tensors.
history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=40,
                    batch_size=512,
                    validation_data=(x_val, y_val),
                    verbose=1)

# evaluate the model. Two values returns are loss (lower values better) and accuracy
results = model.evaluate(test_data, test_labels)
print(results)                    

# create graph of accuracy and loss over time
# model.fit() returns a History object that contains a dict w/ everything that happened during training
history_dict = history.history
print(history_dict.keys())

acc = history_dict['acc']
val_acc = history_dict['val_acc']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(acc) + 1)

# 'bo' is for "blue dot"
plt.plot(epochs, loss, 'bo', label='Training loss')
# b is for 'solid blue line'
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training & validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

plt.clf()   # clear figure

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()
#plt.savefig("graph.png")