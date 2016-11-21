import tensorflow as tf

x = tf.Variable([[1, 2], [3, 4]])
y = tf.Variable([5,6])

op = tf.mul(x, y)

model = tf.initialize_all_variables()

with tf.Session() as session:

    session.run(model)
    result = session.run(op)
    print(result)
