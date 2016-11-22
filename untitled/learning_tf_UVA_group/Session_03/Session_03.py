from __future__ import print_function
import os
import numpy as np
import random
import string
import tensorflow as tf
import zipfile
# from six.moves import range
# from six.moves.urllib.request import urlretrieve

url = 'http://mattmahoney.net/dc/'

def maybe_download(filename, expected_bytes):
  """Download a file if not present, and make sure it's the right size."""
  if not os.path.exists(filename):
    # filename, _ = urlretrieve(url + filename, filename)
    print('no file')
  statinfo = os.stat(filename)
  if statinfo.st_size == expected_bytes:
    print('Found and verified %s' % filename)
  else:
    print(statinfo.st_size)
    raise Exception(
      'Failed to verify ' + filename + '. Can you get to it with a browser?')
  return filename

filename = maybe_download('text8.zip', 31344016)


def read_data(filename):
    f = zipfile.ZipFile(filename)
    for name in f.namelist():
        return tf.compat.as_str(f.read(name))
    f.close()


text = read_data(filename)
# print('Data size %d' % len(text))

# create the validation set
valid_size = 1000
valid_text = text[:valid_size]
train_text = text[valid_size:]
train_size = len(train_text)

vocabulary_size = len(string.ascii_lowercase) + 1  # [a-z] + ' ' = 27
first_letter = ord(string.ascii_lowercase[0])


# map a char to an index
def char2id(char):
    # print(string.ascii_lowercase)
    if char in string.ascii_lowercase: #note: string.ascii_lowercase is just a list of characters: abcdefghijklmnopqrstuvwxyz
        return ord(char) - first_letter + 1
    elif char == ' ':
        return 0
    else:
        print('Unexpected character: %s' % char)
        return 0


# map a index to a char
def id2char(dictid):
    if dictid > 0:
        return chr(dictid + first_letter - 1)
    else:
        return ' '

# print(id2char(1))
# print(id2char(26))
# print(id2char(0))


# Function to generate a training batch for the LSTM model.

batch_size = 64
num_unrollings = 10


class BatchGenerator(object):
    def __init__(self, text, batch_size, num_unrollings):
        self._text = text
        self._text_size = len(text)
        self._batch_size = batch_size
        self._num_unrollings = num_unrollings
        segment = self._text_size // batch_size
        self._cursor = [offset * segment for offset in range(batch_size)]
        self._last_batch = self._next_batch()


    def _next_batch(self): # I'm not able to go into the details, but this func first creates a matrix ('batch') of zeros whose dimension are the sieze of the bacth for the rows (of course,
        # this only means that I'm deciding how many examples I want), while columns are as much as the words in the vocabulary: this is done because the aim is to create a one hot vector
        # representation using the cha2id function that is called below.
        """Generate a single batch from the current cursor position in the data."""
        batch = np.zeros(shape=(self._batch_size, vocabulary_size), dtype=np.float) # this is just a matrix of zeros 64(batch size) x 27(vocabulary size)
        for b in range(self._batch_size):
            batch[b, char2id(self._text[self._cursor[b]])] = 1.0
            self._cursor[b] = (self._cursor[b] + 1) % self._text_size
        return batch

    def next(self):
        """Generate the next array of batches from the data. The array consists of
        the last batch of the previous array, followed by num_unrollings new ones.
        """
        batches = [self._last_batch]
        for step in range(self._num_unrollings):
            batches.append(self._next_batch())
        self._last_batch = batches[-1]
        return batches

# honestly I don't get weel what these two funcs below do
def characters(probabilities):
    """Turn a 1-hot encoding or a probability distribution over the possible
    characters back into its (most likely) character representation."""
    return [id2char(c) for c in np.argmax(probabilities, 1)]


def batches2string(batches):
    """Convert a sequence of batches back into their (most likely) string
    representation."""
    s = [''] * batches[0].shape[0]
    for b in batches:
        s = [''.join(x) for x in zip(s, characters(b))]
    return s


train_batches = BatchGenerator(train_text, batch_size, num_unrollings)
valid_batches = BatchGenerator(valid_text, 1, 1)

print(batches2string(train_batches.next()))
print(batches2string(train_batches.next()))
# print(batches2string(valid_batches.next()))
# print(batches2string(valid_batches.next()))