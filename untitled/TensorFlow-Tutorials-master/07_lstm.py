#Inspired by https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3%20-%20Neural%20Networks/recurrent_network.py
import tensorflow as tf

import numpy as np
import input_data

# configuration
#                        O * W + b -> 10 labels for each image, O[? 28], W[28 10], B[10]
#                       ^ (O: output 28 vec from 28 vec input)
#                       |
#      +-+  +-+       +--+
#      |1|->|2|-> ... |28| time_step_size = 28
#      +-+  +-+       +--+
#       ^    ^    ...  ^
#       |    |         |
# img1:[28] [28]  ... [28]
# img2:[28] [28]  ... [28]
# img3:[28] [28]  ... [28]
# ...
# img128 or img256 (batch_size or test_size 256)
#      each input size = input_vec_size=lstm_size=28

# M: open the dataset and create a one hot representation. anyway this is a sd hoc script, so in this moment it is not interesting for me
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels
trX = trX.reshape(-1, 28, 28)
teX = teX.reshape(-1, 28, 28)



# configuration variables
# M: define the number of units in the LSTM - note: the value we will be using is 'lstm_size', not 'input_vec_size'
input_vec_size = lstm_size = 28
# M: number of elements in the sequences. This is the second value in the three needed: Number of examples, number of input, dimension of each input
time_step_size = 28

# M: number x-y pairs for each batch
batch_size = 128

# M: not sure what this is
test_size = 256


# M: define (a function that initializes) weights. Weights are a variable that has to have a shape. Such a shape basically defines the size of the matrix, and it can be different:
# in case of weights you want to shape to be <size of the layer from which weights start> x <size of the layer to which weights arrive>. As a result you have a 2-d matrix of the kind:
# [[-0.23223054 -0.16686235 -0.44882786  0.28732216  0.20905587]
#  [ 0.44168109 -0.35944489 -0.3395052  -0.18816268 -0.09728211]
#  [ 0.0842329   0.38425198 -0.2567867   0.71504617 -0.55237728]
#  [ 0.19866824  0.79201376  0.31991485 -0.00461882 -0.30853665]
#  [-0.166702   -0.01597206 -0.24885109 -0.06365411 -0.38819805]
#  [-0.76675725  0.56328106 -0.04897104 -0.0717795  -0.25141913]
#  [ 0.19430462 -0.32525453 -0.69064105 -0.00812313 -0.24793796]
#  [-0.19759524 -0.12641869 -0.31907535  0.04258627  0.03036178]
#  [ 0.02176806  0.36872005 -0.2298775  -0.11932793  0.14894287]
#  [-0.05779798 -0.07805374 -0.12266409  0.34272981 -0.05321882]]
# in case of the bias, you just want them to be <size of the layer to which weights arrive>. As a result you have a 1-d vector of the kind:
# [-0.26305592  0.15021914 -0.2998521  -0.13653564 -0.30410272 -0.48367888
#  -0.17051893  0.35504109  0.38076493 -0.44539294]
#
def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))

# get lstm_size and output 10 labels
# M: you are defining W as a 2-d matrix and B as a 1-d matrix (see explanation few lines above)
W = init_weights([lstm_size, 10])
B = init_weights([10])

# M: define the placeholders: remember they are just empty variables you will fill later on.
# X is a single training example, which has three values (remember: X is a sequence of thing, can be words, images and so on):
# --> Number of examples: None --> you don't know how many examples you will give the classifier during training, i.e. you don't know the size of the batch
# --> Number of elements in the sequences: 28 --> how many elements do you have in each sequence? in this case it is a fixed lenght because we are working with images;
# with text we'll have different lenghts, that's why we will need padding
# --> Dimension of each element of the sequence: 28 --> in this case 28 because we're working with images (see line 16)
# Y is a the gold label associated to the example X. It has two dimensions>
# --> Number of examples: None --> as before: you don't know how many examples you will give the classifier during training, i.e. you don't know the size of the batch
# --> Number of possible classes --> in this task we have ten possible classes as output, therefore Y is a one hot vector that says hich is the correct target class for the example X
X = tf.placeholder("float", [None, 28, 28])
Y = tf.placeholder("float", [None, 10])

# M: define the model, that takes as arguments:
# --> X: the incoming example
# --> W: the weights
# --> B: the bias
# --> lstm_size: the size of the lstm layer

def model(X, W, B, lstm_size):


    # M: transpose the first and the second value of X, i.e. the Number of examples and Number of elements in the sequences. I don't know why this is done
    # X, input shape: (batch_size, time_step_size, input_vec_size)
    XT = tf.transpose(X, [1, 0, 2])  # permute time_step_size and batch_size

    # M: not sure why this is done
    # XT shape: (time_step_size, batch_size, input_vec_size)
    XR = tf.reshape(XT, [-1, lstm_size]) # each row has input for each lstm cell (lstm_size=input_vec_size)

    # M: not sure why this is done
    # XR shape: (time_step_size * batch_size, input_vec_size)
    X_split = tf.split(0, time_step_size, XR) # split them to time_step_size (28 arrays)
    # Each array shape: (batch_size, input_vec_size)

    # M: define the units in the lstm layer
    # Make lstm with lstm_size (each input vector size)
    lstm = tf.nn.rnn_cell.BasicLSTMCell(lstm_size, forget_bias=1.0, state_is_tuple=True)

    # M:
    # tf.nn.rnn: Creates a recurrent neural network specified by rnn_cell, i.e. it takes as inout the cell created in the previous step (in our case what is called 'lstm').
    # the second argument (in our case 'X_split') is the input data
    # note that in the other tutorial this step is easier to understand: val, _ = tf.nn.dynamic_rnn(cell, data, dtype=tf.float32)
    # in any case, still I cannot understand why it returns a tuple
    # Get lstm cell output, time_step_size (28) arrays with lstm_size output: (batch_size, lstm_size)
    outputs, _states = tf.nn.rnn(lstm, X_split, dtype=tf.float32)

    # Linear activation
    # Get the last output
    return tf.matmul(outputs[-1], W) + B, lstm.state_size # State size to initialize the stat







py_x, state_size = model(X, W, B, lstm_size)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(py_x, Y))
train_op = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cost)
predict_op = tf.argmax(py_x, 1)

# Launch the graph in a session
with tf.Session() as sess:
    # you need to initialize all variables
    tf.initialize_all_variables().run()

    for i in range(100):
        for start, end in zip(range(0, len(trX), batch_size), range(batch_size, len(trX)+1, batch_size)):
            sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end]})

        test_indices = np.arange(len(teX))  # Get A Test Batch
        np.random.shuffle(test_indices)
        test_indices = test_indices[0:test_size]

        print(i, np.mean(np.argmax(teY[test_indices], axis=1) ==
                         sess.run(predict_op, feed_dict={X: teX[test_indices]})))
