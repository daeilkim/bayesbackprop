import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

graph = tf.Graph()
num_features = 784 # 28x28 pixels
num_labels = 10 # training
num_steps = 1000
batch_size = 100
num_units = 400

if __name__ == "__main__":
    # Initialize the tensorflow graph
    graph = tf.Graph()
    train_subset = 200

    with graph.as_default():
        x = tf.placeholder(tf.float32, [None, num_features], name="x-input")
        y_ = tf.placeholder(tf.float32, [None, num_labels], name='y-input')

        # Calculate weights and biases at the first layer given mu and sigma
        bias1 = tf.Variable(tf.constant(0.1, shape=[num_units]), name="bias1")
        bias2= tf.Variable(tf.constant(0.1, shape=[num_units]), name="bias2")
        bias_out = tf.Variable(tf.constant(0.1, shape=[num_labels]), name="bias_out")
        weights1 = tf.Variable(tf.random_normal([num_features, num_units]), name="weights_layer1")
        weights2 = tf.Variable(tf.random_normal([num_units, num_units]), name="weights_layer2")
        weights_out = tf.Variable(tf.random_normal([num_units, num_labels]), name="weights_out")

        # Rectified Linear Unit Layers
        h_relu1 = tf.nn.relu(tf.add(tf.matmul(x, weights1), bias1))
        h_relu2 = tf.nn.relu(tf.add(tf.matmul(h_relu1, weights2), bias2))
        y = tf.matmul(h_relu2, weights_out) + bias_out

        # Specify the cost function. We'll want to replace this with the BayesBackprop cost
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))
        _ = tf.scalar_summary('cross entropy', cost)

        # Specify the optimizer
        optimizer = tf.train.AdamOptimizer(1e-2).minimize(cost)

        # Calculate accuracy
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    with tf.Session(graph=graph) as session:
        tf.initialize_all_variables().run()
        for step in range(num_steps):
            if (step % 20 == 0):
                feed = {x: mnist.test.images, y_: mnist.test.labels}
                result = session.run([accuracy], feed_dict=feed)
                summary_str = result[0]
                print("Step %d: %2.3f" % (step, accuracy.eval(feed_dict=feed)*100) )
            else:
                batch = mnist.train.next_batch(batch_size)
                session.run(optimizer, feed_dict={x: batch[0], y_: batch[1]})
