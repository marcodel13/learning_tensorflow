# placeholders

import tensorflow as tf

# *********************************************


# so, you can either have a predifined variable x ...
# x = tf.Variable(3)
#
# op = x + 1
#
# model = tf.initialize_all_variables()
#
# with tf.Session() as session:
#
#     session.run(model)
#
#     print(session.run(op))


# *********************************************

# ... or create a placeholder

x = tf.placeholder('float', None) # none: we can pass to x any number of values

op = x + 1


# --> note: here you don't need to initialize the variable, because x is a placeholder

with tf.Session() as session:

    res = session.run(op, feed_dict={x: [3, 5, 6, 77]})
    # shape = res.get_shape()

    print(res)


# c = tf.Variable([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
# d = tf.placeholder('float', [10, 3])
# print(c.get_shape())
# print(d.get_shape())

# *********************************************

# you can use placeholders not only for variables, of course, but also matrices

x = tf.placeholder('float', [None, None]) # none: we can pass to x any number of values

a = tf.Variable([[1.,2.,3.], [4.,5.,6.]])

initialize = tf.initialize_all_variables()

op = x + 1

with tf.Session() as session:

    matrix = [[1,2,3],[4,5,6]] # again: this is a 2*3 matrix: 2 rows, 3 columns

    results = session.run(op, feed_dict={x: matrix})

    print(results)

    session.run(initialize)

    print(session.run(a))