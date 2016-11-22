import tensorflow as tf
import numpy as np
from random import shuffle


'''
************ randomly create train and test ************
'''
NUM_EXAMPLES = 10000

train_input = ['{0:020b}'.format(i) for i in range(2**10)]
shuffle(train_input)
train_input = [map(int,i) for i in train_input]
ti  = []
for i in train_input:
    temp_list = []
    for j in i:
            temp_list.append([j])
    ti.append(np.array(temp_list))
train_input = ti

train_output = []

for i in train_input:
    count = 0
    for j in i:
        if j[0] == 1:
            count+=1
    temp_list = ([0]*21)
    temp_list[count]=1
    train_output.append(temp_list)

test_input = train_input[NUM_EXAMPLES:]
test_output = train_output[NUM_EXAMPLES:]
train_input = train_input[:NUM_EXAMPLES]
train_output = train_output[:NUM_EXAMPLES]

print ("test and training X loaded")

'''
************ model ************
'''

x = tf.placeholder(tf.float32, [None, 20, 1]) #Number of examples, number of input, dimension of each input
y_ = tf.placeholder(tf.float32, [None, 21])

# Create the model
W = tf.Variable(tf.zeros([20, 21]))
b = tf.Variable(tf.zeros([21]))
y = tf.matmul(x, W) + b

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))

# Train
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

batch_size = 10
no_of_batches = int(len(train_input)) / batch_size
epoch = 20

with tf.Session() as sess:
    # Train
    tf.initialize_all_variables().run()
    for i in range(epoch):
        ptr = 0
        batch_xs, batch_ys = train_input[ptr:ptr+batch_size], train_output[ptr:ptr+batch_size]
        ptr += batch_size
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
        # sess.run(minimize, {X: inp, Y: out})

    # Test trained model
    # correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    # accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    # print(sess.run(accuracy, feed_dict={x: mnist.test.images,
    #                                     y_: mnist.test.labels}))



# batch_size = 1000
# no_of_batches = int(len(train_input)) / batch_size
# epoch = 20
# for i in range(epoch):
#     ptr = 0
#     for j in range(int(no_of_batches)):
#         inp, out = train_input[ptr:ptr+batch_size], train_output[ptr:ptr+batch_size]
#         ptr+=batch_size
#         sess.run(minimize,{X: inp, Y: out})
#     print ("Epoch ",str(i))
# incorrect = sess.run(error,{X: test_input, Y: test_output})
# print (sess.run(prediction,{X: [[[1],[0],[0],[1],[1],[0],[1],[1],[1],[0],[1],[0],[0],[1],[1],[0],[1],[1],[1],[0]]]}))
# print('Epoch {:2d} error {:3.1f}%'.format(i + 1, 100 * incorrect))
# sess.close()