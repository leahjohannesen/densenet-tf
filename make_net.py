import tensorflow as tf 
from tensorflow.examples.tutorials.mnist import input_data

def weights_init(shape):
    return tf.get_variable('weight', shape, initializer=tf.truncated_normal_initializer())

def bias_init(shape):
    return tf.get_variable('bias', shape, initializer=tf.constant_initializer())

def conv2d(x, w):
    return tf.nn.conv2d(x, w, strides=[1,1,1,1], padding='SAME')

def bn_relu_conv(n_layer, inputs, w_shape, b_shape):
    with tf.variable_scope(n_layer) as scope:
        weights = tf.get_variable('weights', w_shape, tf.truncated_normal_initializer())
        bias = tf.get_variable('bias', b_shape, tf.constant_initializer())
        conv = tf.nn.conv2d(inputs, weights, strides=[1,3,3,1], padding='SAME')
        relu = tf.nn.relu(conv + bias)
        return relu, weights, bias

def train_model():

    x = tf.placeholder(tf.float32, shape=[None, 784])
    y = tf.placeholder(tf.float32, shape=[None, 10])
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    

    #First conv layer
    x0 = tf.reshape(x, [-1,28,28,1])
    w_conv1 = weights_init([5,5,1,16])
    b_conv1 = bias_init([16])

    conv1 = conv2d(x0, w_conv1)
    relu1 = tf.nn.relu(conv1 + b_conv1)

    relu_list = [relu1]

    for i in range(1,3):
        name = str(i)
        with tf.variable_scope(name) as scope:
            weight = weights_init([3,3,16,16])
            bias = bias_init([16])
            conv = conv2d(relu_list[i-1], weight)
            relu = tf.nn.relu(conv + bias)
            relu_list.append(relu)

    #Flatten
    classes = 10
    
    flat = tf.reshape(relu_list[-1], shape=[-1, 144])
    weight = weights_init([144, 10])
    bias = bias_init([10])

    dense_out = tf.matmul(flat, w_out)
    output = tf.nn.softmax(dense_out + b_out)

    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y * tf.log(output), reduction_indices=[1]))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    correct_prediction = tf.equal(tf.argmax(output,1), tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    with tf.session() as sess:
        sess.run(tf.initialize_all_variables())
        for i in range(1000):
            batch = mnist.train.next_batch(100)
            if i%100 == 0:
                train_accuracy = accuracy.eval(feed_dict={x:batch[0], y: batch[1]})
                print "Accuracy: {}".format(train_accuracy)
            results = sess.run(train_step, feed_dict={x: batch[0], y: batch[1]})


if __name__ == "__main__":
    train_model()
