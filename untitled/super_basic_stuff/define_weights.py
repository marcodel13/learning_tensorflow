import tensorflow as tf

weights = tf.Variable(tf.random_normal([10, 5], stddev=0.35), name="weights")

bias = tf.Variable(tf.random_normal([10], stddev=0.35), name="bias")


model = tf.initialize_all_variables()


with tf.Session() as session:

    session.run(model)

    w = session.run(weights)
    b = session.run(bias)
    print(w)
    print(b)