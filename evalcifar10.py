import tensorflow as tf
import numpy as np
import sys
import densenet

sys.path.append('/home/ubuntu/data-classes/')
from cifar10 import cifar10

def train(drop=0.8):
    data = cifar10()

    x = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
    y = tf.placeholder(tf.float32, shape=[None, 10])

    lr = tf.placeholder(tf.float32)
    keep = tf.placeholder(tf.float32)
    is_train = tf.placeholder(tf.bool)

    y_pred = densenet.pred(x, keep, is_train)
    acc = densenet.acc(y, y_pred)

    loss = tf.nn.softmax_cross_entropy_with_logits(y_pred, y)
    #This calls the optimizer from the opts.py module. Helps with clutter.
    train_step = tf.train.MomentumOptimizer(lr, 0.9, use_nesterov=True).minimize(loss)

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())

        #Some constats/variables used in the training
        epochs = 5
        batch_size = 64 
        total_train = len(data.x_trn)
        deciles = int(total_train/(10*batch_size))
        lrate = 0.01

        #Computes tstidation accuracy iteratively to avoid blowing up memory
        print '-----Starting the Session-----'
        tst_list = []
        while True:
            batch_tst = data.next_tst(batch_size)
            if not batch_tst:
                break
            tst_acc = sess.run(acc, feed_dict={x: batch_tst[0], y: batch_tst[1], 
                                            keep: 1.0, is_train: False})
            tst_list.append(tst_acc)

        print '\n' + '- '*10
        print "Starting Test Accuray: {}".format(np.mean(tst_list))
        print '- '*10 + '\n'
        loss_list = []

        #The actual training regimen
        for epoch in range(epochs):
            print "-----Starting Epoch {}-----".format(epoch)
            n = 0
            if epoch % 75 == 0: lrate = 0.001

            while True:
                #Gets next batch of data, returns tuple of x/y if it hasn't gone through
                #the epoch, otherwise returns false and goes into the tstidation regime
                batch = data.next_trn(batch_size)
                if not batch:
                    tst_list = []
                    while True:
                        batch_tst = data.next_tst(batch_size)
                        if not batch_tst:
                            break
                        tst_acc = sess.run(acc, feed_dict={x: batch_tst[0], y: batch_tst[1],
                                            lr: lrate, keep: 1.0, is_train: False})
                        tst_list.append(tst_acc)
                    print "End of Epoch"
                    print "Test Accuracy: {}\n".format(np.mean(tst_list))
                    break

                #Prints the status of the run, every 10%
                if n%deciles == 0:
                    print "Percent of epoch complete: {}0%.".format(n/deciles)
                n += 1
                loss_trn, _ = sess.run([loss, train_step], feed_dict={x: batch[0],
                                                     y: batch[1], keep: drop, is_train: True})
                loss_list.append(loss_trn)

if __name__ == '__main__':
    train()
