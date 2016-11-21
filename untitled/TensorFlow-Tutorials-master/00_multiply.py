#!/usr/bin/env python

import tensorflow as tf

a = tf.placeholder("float", 2) # Create a symbolic variable 'a'
b = tf.placeholder("float", 2) # Create a symbolic variable 'b'

y = tf.mul(a, b) # multiply the symbolic variables

with tf.Session() as sess: # create a session to evaluate the symbolic expressions

    print(sess.run(y, feed_dict={a: [1, 2], b: [2, 5]})) # eval expressions with parameters for a and b

    # print(sess.run(y, feed_dict={a: 3, b: 3}))
