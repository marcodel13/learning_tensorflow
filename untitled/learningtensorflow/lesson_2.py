# http://learningtensorflow.com/lesson2/

import tensorflow as tf
import numpy as np


x = tf.constant(35, name='x')  # x is a scalar
x_1 = tf.constant([35, 40, 45], name='x_1')
y = tf.Variable(x + 5, name='y')
y_1 = tf.Variable(x_1 + 5, name='y_1')

data = np.random.randint(1000, size=10000)
variable = tf.Variable(5*data^2 - 3*data + 15, name='variable')

print(data)
# --> <tensorflow.python.ops.variables.Variable object at 0x101b175c0>
# when you print y you don't get '40' because this is different fomr a normal python code
# y is effectively an equation that means “when this variable is computed, take the value of x (as it is then) and add 5 to
# it”. The computation of the value of y is never actually performed in the above program

# if you want to run it, you have to create a session

model = tf.initialize_all_variables()

with tf.Session() as session:
    session.run(model)
    print(session.run(variable))
