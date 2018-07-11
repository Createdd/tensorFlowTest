import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# Define a path to TensorBoard log files
logPath = "./tfb_deep_logs/"

# Add summaries statistics to use in TensorBoard visualization.
def variable_summaries(var):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_mean(var))
        tf.summary.histogram('histogram', var)

# Create input objerct which reads data from MNIST datasets. Perform one-hot encoding to define the digit.
mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

# Using Interactive session makes it the default sessions so we do not need to pass sess
sess = tf.InteractiveSession()

# Define placeholders for MNIST input data
with tf.name_scope("MNIST_Input"):
    x = tf.placeholder(tf.float32, shape=[None, 784], name="x")
    y_ = tf.placeholder(tf.float32, shape=[None, 10], name="y_")

# Change the MNIST input data from a list of values to a 28 pixels x 28 pixels x 1 grayscale value cube
# which the Convolution network can use.
with tf.name_scope("Input_Reshape"):
    x_image = tf.reshape(x, shape=[-1, 28, 28, 1], name='x_image')
    tf.summary.image('input_img', x_image, 5)

# Define helper function to created weights and biases variables, and convolution, and pooling layers.
# We are using RELU as our activation function. These must be initialized to a small positive number
# and with some noise so you don't end up going to zero when comparing diff.
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1, name='variable')
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape, name='variable')
    return tf.Variable(initial)

# Convolution and Pooling - we do Convolution, and then pooling to control overfitting
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME', name='conv')

def max_pool_2x2(x, name=None):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], 
                             strides=[1, 2, 2, 1], 
                             padding='SAME',
                             name=name)


# Define layers in the NN

with tf.name_scope('Conv1'):
    # 1st Convolution layer
    # 32 features for each 5x5 patch of the image]
    with tf.name_scope('weights'):
        W_conv1 = weight_variable([5, 5, 1, 32])
        variable_summaries(W_conv1)
    
    with tf.name_scope('bias'):
        b_conv1 = bias_variable([32])
        variable_summaries(b_conv1)

    # Do convolution on images, add bias and push through RELU activation
    conv1_wx_b = conv2d(x_image, W_conv1) + b_conv1
    tf.summary.histogram('conv1_wx_b', conv1_wx_b)
    h_conv1 = tf.nn.relu(conv1_wx_b, name='relu')
    tf.summary.histogram('h_conv1', h_conv1)

    # Take result and run through max_pool
    h_pool1 = max_pool_2x2(h_conv1, name='pool')


with tf.name_scope('Conv2'):
    # 2nd Convolution layer
    # Process the 32 features from Convolution layer 1, in 5x5 patch. Return 64 features weights and biases.
    with tf.name_scope('weights'):
        W_conv2 = weight_variable([5, 5, 32, 64])
        variable_summaries(W_conv2)
    
    with tf.name_scope('bias'):
        b_conv2 = bias_variable([64])
        variable_summaries(b_conv2)

    # Do convolution of the output of the 1st convolutional layer. Pool result.
    conv2_wx_b = conv2d(h_pool1, W_conv2) + b_conv2
    tf.summary.histogram('conv2_wx_b', conv2_wx_b)
    h_conv2 = tf.nn.relu(conv2_wx_b, name='relu')
    tf.summary.histogram('h_conv2', h_conv2)
    h_pool2 = max_pool_2x2(h_conv2, name='pool')


with tf.name_scope('FC'):
    # Fully Connected Layer
    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])

    # Connect ouput of pooling layer 2 as input to the full connected layer
    h_pool2_flat = tf.reshape(h_pool2, shape=[-1, 7 * 7 * 64], name='reshape')
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1, name='relu')

with tf.name_scope('Dropout'):
    # Dropout some neuros to reduce overfitting
    keep_prob = tf.placeholder(tf.float32)  # get dropou probability as a training input.
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

with tf.name_scope('Output'):
    # Readout layer
    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])

    # Define the model
    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2


# Loss measurement
with tf.name_scope("cross_entropy"):
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_conv, labels=y_))

# Loss optimization
with tf.name_scope("loss_optimiser"):
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

with tf.name_scope("accuracy"):
    # What is correct
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))

    # How accurate is it?
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

tf.summary.scalar('cross_entropy', cross_entropy)
tf.summary.scalar('training_accuracy', accuracy)

# TensorBoard - Merge summaries
summarize_all = tf.summary.merge_all()


# Initialize all of the variables
sess.run(tf.global_variables_initializer())

# TensorBoard - Write the default graph out so we can view it's structure
tbWriter = tf.summary.FileWriter(logPath, sess.graph)

# Train the model
import time

# Define the number of steps and how often we display progress
num_steps = 3000
display_every = 100

# Start timer
start_time = time.time()
end_time = time.time()

for i in range(num_steps):
    batch = mnist.train.next_batch(50)
    _, summary = sess.run([train_step, summarize_all], feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

    # Periodic status display
    if i % display_every == 0:
        train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
        end_time = time.time()
        print('Step {0}, elapsed time {1:.2f} seconds, training accuracy {2:.3f}%'.format(i, end_time - start_time, train_accuracy * 100))

        # Write summaries to log
        tbWriter.add_summary(summary, i)

# Display summary
#       Time to train
end_time = time.time()
print('Total training time for {0} batches: {1:.2f} seconds'.format(i + 1, end_time - start_time))


# Accuracy on the test data
print('Test accuracy {0:.3f}%'.format(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}) * 100.0))

sess.close()