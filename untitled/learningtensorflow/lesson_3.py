import tensorflow as tf
import numpy as np

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
# First, load the image
filename = "MarshOrchid.jpg"
image = mpimg.imread(filename)

# Print out its shape
# print(image.shape)
#
# plt.imshow(image)
# plt.show()


# Create a TensorFlow Variable
x = tf.Variable(image, name='x')

model = tf.initialize_all_variables()

# transpose the image
with tf.Session() as session:

    x = tf.transpose(x, perm=[1, 0, 2])
    session.run(model)
    result = session.run(x)

plt.imshow(result)
plt.show()