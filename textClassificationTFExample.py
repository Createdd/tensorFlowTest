# The code base is from TensorFlows text classification tutorial
# https://www.tensorflow.org/tutorials/keras/basic_text_classification

import tensorflow as tf
from tensorflow import keras
import numpy as np

# ======= Import the IMDB dataset

imdb = keras.datasets.imdb
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

# ======= Explore the data

print("Training entries: {}, labels: {}".format(len(train_data), len(train_labels)))
print(train_data[0])

# ======= Convert integers back to words

# Retrieves the dictionary mapping word indices back to words.
word_index = imdb.get_word_index()

# Add 3 new indices
# Exchange key and values
# Create function to create the text according to the indices
word_index = {k: (v + 3) for k, v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2
word_index["<UNUSED>"] = 3
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])


def decode_review(integerList):
    return " ".join([reverse_word_index.get(i, "?") for i in integerList])


# ====== Prepare (test/training)data by converting into tensors
train_data = keras.preprocessing.sequence.pad_sequences(
    train_data, value=word_index["<PAD>"], padding="post", maxlen=256
)
test_data = keras.preprocessing.sequence.pad_sequences(
    test_data, value=word_index["<PAD>"], padding="post", maxlen=256
)

# ========== Build the model

# input shape is the vocabulary count used for the movie reviews (10,000 words)
vocab_size = 10000

# 1. layer to turn positive integers (indexes) into dense vectors of fixed size
# 2. layer returns a fixed-length output vector for each example by averaging over the sequence dimension
# 3. output vector is piped through a fully-connected (Dense) layer with 16 hidden units and a rectified linear unit activation
# 4. last layer is densely connected with a single output node. Using the sigmoid activation function, the result is between 0 and 1
model = keras.Sequential()
model.add(keras.layers.Embedding(vocab_size, 16))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation=tf.nn.relu))
model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))


# because of the binary classification problem we'll use the binary_crossentropy loss function
model.compile(
    optimizer=tf.train.AdamOptimizer(), loss="binary_crossentropy", metrics=["accuracy"]
)

# create a validation set

x_val = train_data[:10000]
partial_x_train = train_data[10000:]

y_val = train_labels[:10000]
partial_y_train = train_labels[10000:]

# ========== use the fit method, which trains the model for a given number of epochs (iterations on a dataset).

history = model.fit(
    partial_x_train,
    partial_y_train,
    epochs=40,
    batch_size=512,
    validation_data=(x_val, y_val),
    verbose=0,
)

# evaluate model

results = model.evaluate(test_data, test_labels)
print(results)

# ========== create a graph for accuracy and loss over time
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

# use the history object from the training of the model
history_dict = history.history
history_dict.keys()

# choose 4 metrics to monitor training and set the x-axis to the epochs
acc = history_dict["acc"]
val_acc = history_dict["val_acc"]
loss = history_dict["loss"]
val_loss = history_dict["val_loss"]
epochs = range(1, len(acc) + 1)

# plot the loss
plt.plot(epochs, loss, "go", label="Training loss")
plt.plot(epochs, val_loss, "b", label="Validation loss")
plt.title("Training and validation loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()

plt.show()

# plot the accuracy
plt.clf()
acc_values = acc
val_acc_values = val_acc

plt.plot(epochs, acc, "go", label="Training acc")
plt.plot(epochs, val_acc, "b", label="Validation acc")
plt.title("Training and validation accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()

plt.show()

# ============ Several copyright credits
# @title Licensed under the Apache License, Version 2.0 (the "License");#@title
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# @title MIT License
#
# Copyright (c) 2017 François Chollet
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
