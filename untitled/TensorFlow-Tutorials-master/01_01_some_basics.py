import tensorflow as tf

matrix1 = tf.constant([[3., 3.]])
matrix2 = tf.constant([[2.],[5.]])

product = tf.matmul(matrix1, matrix2)

# Launch the default graph.
with tf.Session() as sess:
  result = sess.run([product])
  print(result)