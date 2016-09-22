import tensorflow as tf 
from tensorflow.examples.tutorials.mnist import input_data

def weights_init(shape):
    return tf.Variable(tf.truncated_normal(shape))

def bias_init(shape):
    return tf.Variable(tf.constant(0.1))

def conv2d(x, w):
    return tf.nn.conv2d(x, w, strides=[1,1,1,1], padding='SAME')

def maxpool(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding="VALID")

def bn_relu_conv(inputs, w_shape, b_shape):
    w = weights_init(w_shape) 
    b = bias_init(b_shape) 
    conv = conv2d(inputs, w)
    relu = tf.nn.relu(conv + b)
    return relu

def train_model():

    x = tf.placeholder(tf.float32, shape=[None, 784])
    y = tf.placeholder(tf.float32, shape=[None, 10])
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    

    #First conv layer
    with tf.variable_scope("input"):
        x_shape = tf.reshape(x, [-1,28,28,1])
        relu = bn_relu_conv(x_shape, [3,3,1,16], [16])
        block1 = [relu]

    n_old = 16
    n_new = 32 

    for i in range(1,3):
        name = "block1-{}".format(i)
        with tf.variable_scope(name): 
            relu = bn_relu_conv(block1[i-1], [3,3,n_old, n_new], [n_new])
            block1.append(relu)
            n_old = n_new
            n_new *= 2

    pool1 = maxpool(block1[-1])    

    n_old = 64 
    n_new = 128

    block2 = [pool1]

    for i in range(1,3):
        name = "block2-{}".format(i)
        with tf.variable_scope(name): 
            relu = bn_relu_conv(block2[i-1], [3,3,n_old, n_new], [n_new])
            block2.append(relu)
            n_old = n_new
            n_new *= 2
        
    pool2 = maxpool(block2[-1])

    pool_shape = pool2.get_shape()
    n_flat = int(pool_shape[1])*int(pool_shape[2])*int(pool_shape[3])

    with tf.variable_scope("output"):
        flat = tf.reshape(pool2, shape=[-1, n_flat])
        w = weights_init([n_flat, 10])
        b = bias_init([10])
        dense = tf.matmul(flat, w)
        output = tf.nn.softmax(dense + b)

    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y * tf.log(output), reduction_indices=[1]))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    correct_prediction = tf.equal(tf.argmax(output,1), tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        summary_writer = tf.train.SummaryWriter('../logs/', sess.graph)
        for i in range(1000):
            batch = mnist.train.next_batch(100)
            if i%100 == 0:
                train_accuracy = accuracy.eval(feed_dict={x:batch[0], y: batch[1]})
                print "Accuracy: {}".format(train_accuracy)
            results = sess.run(train_step, feed_dict={x: batch[0], y: batch[1]})

        

if __name__ == "__main__":
    train_model()
