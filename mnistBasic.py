#
#   simple_mnist.py
#   Simple NN to classify handwritten digits from MNIST dataset.
#

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# We use the TF helper function to pull down the data from the MNIST site
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# x is the placeholder for the 28 x 28 image data
# y_ is called "y bar" and is a 10 element vector, containing the predicted probability of each
# digit (0-9) class. Such as [0.14, 0.8, 0, 0, 0, 0, 0, 0, 0, 0.06]
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

# Define the weights and balances
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# Define the model
y = tf.nn.softmax(tf.matmul(x, W) + b)

# Loss is cross entropy
cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y)
)

# Each training step in gradient descent we want to minimize cross entropy
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# Initialize all variables
init = tf.global_variables_initializer()

# Create an interactive session that can span multiple code blocks. Don't
# forget to explicity close the session with sess.close()
sess = tf.Session()

# Perform the initialization which is only the initialization of all global variables
sess.run(init)

# Perform 1000 training steps
for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)  # get 100 random data points

    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

# Evaluate how well the model did. Do this by compating the digit with the highest probability in
# actual (y) and predicted (y_)
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
test_accuracy = sess.run(
    accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}
)
print("Test Accuracy: {0}%".format(test_accuracy * 100.0))

sess.close()
