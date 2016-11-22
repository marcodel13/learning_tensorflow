import tensorflow as tf
import numpy as np

# ******************** create the representation of the sentences ********************

# first youu have to translate a sentence (e.g. 'I run the program') into a sentence of indexes like [1, 65, 2, 1789].
# you do that with a simple python code: take the glove file, and write a code that takes each word in the sentence, looks for the position in the glove file, and return the index

# then when you have that each sentence is list of words like the one below
words = [2, 1, 0, 5]

# and you have a glove matrix like this one (in this case is a random matrix)
glove = np.random.randn([10, 5])

# you use this line to look up the embedding of the indexes you have in the sentence (i.e. in the list 'words')
sent_wembs = tf.nn.embedding_lookup(glove, words)

# ******************** batch ********************

# since I don't have that many sentences, I can think about not using batch, and compute one sentence at a time

with tf.variable_scope("rnn"):
    model = tf.nn.basic

# ******************** output the result for each state ********************

# first, you can use place holder of this kind:
X = tf.placeholder("float", [None, None, 300])
# that is, my input X is a float, I don't know how many of them I will get (None), I don't know the time-step size, i.e. how many words I will have in the sentence, but I know that the each
# word embedding representing the words has dimensionality 300. Note that the second value could be set if I wanted to be more precise: I could put, for example, 50, and then use padding.
# note also that I could use None for everything: X = tf.placeholder("float", None)

# then create the lstm cell
size_of_the_lstm_layer = 100
cell = tf.nn.rnn_cell.LSTMCell(size_of_the_lstm_layer, state_is_tuple=True)

# you need to use the dynamic lstm cell, which takes as argument the cell just created and the input data X
output, state = tf.nn.bidirectional_dynamic_rnn(cell, X, dtype=tf.float32)




