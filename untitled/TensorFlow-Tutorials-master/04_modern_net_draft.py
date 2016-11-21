import tensorflow as tf
import numpy as np
import input_data
sess = tf.InteractiveSession()



def nn(X, w_h, w_o):

    # h = tf.nn.relu(tf.matmul(X, w_h))

    h = tf.mul(X, w_h)

    # h2 = tf.nn.relu(tf.matmul(h, w_h2))

    o = tf.mul(h, w_o)

    return X, h, o

X = tf.Variable([1, 2, 3, 4])
w_h = tf.Variable([1,1,1,1],[1,1,1,1])
w_o = tf.Variable([5,5,5,5])


op = nn(X, w_h, w_o)
# print(tf.mul(X, w_h))

model = tf.initialize_all_variables()

with tf.Session() as session:
    session.run(model)
    result = session.run(op)
    print(result)
