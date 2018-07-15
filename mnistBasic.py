
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python import debug as tf_debug

# Only log errors (to prevent unnecessary cluttering of the console)
tf.logging.set_verbosity(tf.logging.ERROR)

# We use the TF helper function to pull down the data from the MNIST site
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# x is the placeholder for the 28 x 28 image data (the input)
# y_ is a 10 element vector, containing the predicted probability of each digit (0-9) class
# Define the weights and balances (always keep the dimensions in mind)
with tf.name_scope("variables_scope"):
    x = tf.placeholder(tf.float32, shape=[None, 784], name="x_placeholder")
    y_ = tf.placeholder(tf.float32, shape=[None, 10], name="y_placeholder")

    with tf.name_scope("weights_scope"):
        W = tf.Variable(tf.zeros([784, 10]), name="weights_variable")

    with tf.name_scope("bias_scope"):
        b = tf.Variable(tf.zeros([10]), name="bias_variable")

    # Define the activation function = the real y. Do not use softmax here, as it will be applied in the next step
    assert x.get_shape().as_list() == [None, 784]
    assert y_.get_shape().as_list() == [None, 10]
    assert W.get_shape().as_list() == [784, 10]
    assert b.get_shape().as_list() == [10]

    with tf.name_scope("yReal_scope"):
        y = tf.add(tf.matmul(x, W), b, name="y_calculated")

    assert y.get_shape().as_list() == [None, 10]

# Loss is defined as cross entropy between the prediction and the real value
# Each training step in gradient descent we want to minimize the loss
with tf.name_scope("loss_scope"):
    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=y_, logits=y, name="lossFunction"
        ),
        name="loss",
    )

with tf.name_scope("training_scope"):
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(
        loss, name="gradDescent"
    )

# Evaluate the accuracy of the model
with tf.name_scope("accuracy_scope"):
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="accuracy")

# Initialize all variables
# Perform the initialization which is only the initialization of all global variables
init = tf.global_variables_initializer()

# ------ Set Session or InteractiveSession
sess = tf.Session()
sess = tf_debug.LocalCLIDebugWrapperSession(sess)
sess.run(init)

# Perform 1000 training steps
# Feed the next batch and run the training
for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

print("============================================")
print(
    f"Accuracy of the model is: {sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})*100}%"
)

sess.close()

# Use this in the terminal to start the tensorboard server
# tensorboard --logdir=./tfb_logs/ --port=8090 --debugger_port 8080 --host=127.0.0.1
