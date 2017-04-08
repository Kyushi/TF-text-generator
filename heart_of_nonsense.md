
# TV Script Generation
In this project, you'll generate your own [Simpsons](https://en.wikipedia.org/wiki/The_Simpsons) TV scripts using RNNs.  You'll be using part of the [Simpsons dataset](https://www.kaggle.com/wcukierski/the-simpsons-by-the-data) of scripts from 27 seasons.  The Neural Network you'll build will generate a new TV script for a scene at [Moe's Tavern](https://simpsonswiki.com/wiki/Moe's_Tavern).
## Get the Data
The data is already provided for you.  You'll be using a subset of the original dataset.  It consists of only the scenes in Moe's Tavern.  This doesn't include other versions of the tavern, like "Moe's Cavern", "Flaming Moe's", "Uncle Moe's Family Feed-Bag", etc..


```python
"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
import helper

# data_dir = './data/simpsons/moes_tavern_lines.txt'
data_dir = './data/josephconrad/hod.txt'
text = helper.load_data(data_dir)
# Ignore notice, since we don't use it for analysing the data
# text = text[81:]
```

## Explore the Data
Play around with `view_sentence_range` to view different parts of the data.


```python
view_sentence_range = (0, 10)

"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
import numpy as np

print('Dataset Stats')
print('Roughly the number of unique words: {}'.format(len({word: None for word in text.split()})))
scenes = text.split('\n\n')
print('Number of scenes: {}'.format(len(scenes)))
sentence_count_scene = [scene.count('\n') for scene in scenes]
print('Average number of sentences in each scene: {}'.format(np.average(sentence_count_scene)))

sentences = [sentence for scene in scenes for sentence in scene.split('\n')]
print('Number of lines: {}'.format(len(sentences)))
word_count_sentence = [len(sentence.split()) for sentence in sentences]
print('Average number of words in each line: {}'.format(np.average(word_count_sentence)))

print()
print('The sentences {} to {}:'.format(*view_sentence_range))
print('\n'.join(text.split('\n')[view_sentence_range[0]:view_sentence_range[1]]))
```

    Dataset Stats
    Roughly the number of unique words: 8899
    Number of scenes: 208
    Average number of sentences in each scene: 13.927884615384615
    Number of lines: 3105
    Average number of words in each line: 12.208695652173914
    
    The sentences 0 to 10:
    HEART OF DARKNESS
    
    By Joseph Conrad
    
    
    
    
    I
    
    


## Implement Preprocessing Functions
The first thing to do to any dataset is preprocessing.  Implement the following preprocessing functions below:
- Lookup Table
- Tokenize Punctuation

### Lookup Table
To create a word embedding, you first need to transform the words to ids.  In this function, create two dictionaries:
- Dictionary to go from the words to an id, we'll call `vocab_to_int`
- Dictionary to go from the id to word, we'll call `int_to_vocab`

Return these dictionaries in the following tuple `(vocab_to_int, int_to_vocab)`


```python
import numpy as np
import problem_unittests as tests
from collections import Counter

def create_lookup_tables(text):
    """
    Create lookup tables for vocabulary
    :param text: The text of tv scripts split into words
    :return: A tuple of dicts (vocab_to_int, int_to_vocab)
    """
    word_count = Counter(text)
    words_sorted = sorted(word_count, key=word_count.get, reverse=True)
    int_to_vocab = {i:word for i, word in enumerate(words_sorted)}
    vocab_to_int = {word:i for i, word in int_to_vocab.items()}
    return vocab_to_int, int_to_vocab


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_create_lookup_tables(create_lookup_tables)
```

    Tests Passed


### Tokenize Punctuation
We'll be splitting the script into a word array using spaces as delimiters.  However, punctuations like periods and exclamation marks make it hard for the neural network to distinguish between the word "bye" and "bye!".

Implement the function `token_lookup` to return a dict that will be used to tokenize symbols like "!" into "||Exclamation_Mark||".  Create a dictionary for the following symbols where the symbol is the key and value is the token:
- Period ( . )
- Comma ( , )
- Quotation Mark ( " )
- Semicolon ( ; )
- Exclamation mark ( ! )
- Question mark ( ? )
- Left Parentheses ( ( )
- Right Parentheses ( ) )
- Dash ( -- )
- Return ( \n )

This dictionary will be used to token the symbols and add the delimiter (space) around it.  This separates the symbols as it's own word, making it easier for the neural network to predict on the next word. Make sure you don't use a token that could be confused as a word. Instead of using the token "dash", try using something like "||dash||".


```python
def token_lookup():
    """
    Generate a dict to turn punctuation into a token.
    :return: Tokenize dictionary where the key is the punctuation and the value is the token
    """
    return {
        '.': "||Period||",
        ',': "||Comma||",
        '"': "||Quotation_Mark||",
        ';': "||Semicolon||",
        '!': "||Exclamation_Mark||",
        '?': "||Question_Mark||",
        '(': "||Left_Paretheses||",
        ')': "||Right+Parentheses||",
        '--': "||Dash||",
        '\n': "||Return||"
        }

"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_tokenize(token_lookup)
```

    Tests Passed


## Preprocess all the data and save it
Running the code cell below will preprocess all the data and save it to file.


```python
"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
# Preprocess Training, Validation, and Testing Data
helper.preprocess_and_save_data(data_dir, token_lookup, create_lookup_tables)
```

# Check Point
This is your first checkpoint. If you ever decide to come back to this notebook or have to restart the notebook, you can start from here. The preprocessed data has been saved to disk.


```python
"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
import helper
import numpy as np
import problem_unittests as tests

int_text, vocab_to_int, int_to_vocab, token_dict = helper.load_preprocess()
```

## Build the Neural Network
You'll build the components necessary to build a RNN by implementing the following functions below:
- get_inputs
- get_init_cell
- get_embed
- build_rnn
- build_nn
- get_batches

### Check the Version of TensorFlow and Access to GPU


```python
"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
from distutils.version import LooseVersion
import warnings
import tensorflow as tf

# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer'
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
```

    TensorFlow Version: 1.0.0
    Default GPU Device: /gpu:0


### Input
Implement the `get_inputs()` function to create TF Placeholders for the Neural Network.  It should create the following placeholders:
- Input text placeholder named "input" using the [TF Placeholder](https://www.tensorflow.org/api_docs/python/tf/placeholder) `name` parameter.
- Targets placeholder
- Learning Rate placeholder

Return the placeholders in the following the tuple `(Input, Targets, LearingRate)`


```python
def get_inputs():
    """
    Create TF Placeholders for input, targets, and learning rate.
    :return: Tuple (input, targets, learning rate)
    """
    # TODO: Implement Function
    input  = tf.placeholder(tf.int32, shape=(None, None), name="input")
    targets = tf.placeholder(tf.int32, shape=(None, None), name="targets")
    learning_rate = tf.placeholder(tf.float32, name="learning_rate")
    return input, targets, learning_rate


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_get_inputs(get_inputs)
```

    Tests Passed


### Build RNN Cell and Initialize
Stack one or more [`BasicLSTMCells`](https://www.tensorflow.org/api_docs/python/tf/contrib/rnn/BasicLSTMCell) in a [`MultiRNNCell`](https://www.tensorflow.org/api_docs/python/tf/contrib/rnn/MultiRNNCell).
- The Rnn size should be set using `rnn_size`
- Initalize Cell State using the MultiRNNCell's [`zero_state()`](https://www.tensorflow.org/api_docs/python/tf/contrib/rnn/MultiRNNCell#zero_state) function
    - Apply the name "initial_state" to the initial state using [`tf.identity()`](https://www.tensorflow.org/api_docs/python/tf/identity)

Return the cell and initial state in the following tuple `(Cell, InitialState)`


```python
def get_init_cell(batch_size, rnn_size):
    """
    Create an RNN Cell and initialize it.
    :param batch_size: Size of batches
    :param rnn_size: Size of RNNs
    :return: Tuple (cell, initialize state)
    """
    lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)
    
    cell = tf.contrib.rnn.MultiRNNCell([lstm]*2)
    initialize = cell.zero_state(batch_size, tf.float32)
    initial_state = tf.identity(initialize, name='initial_state')
    return cell, initial_state


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_get_init_cell(get_init_cell)
```

    Tests Passed


### Word Embedding
Apply embedding to `input_data` using TensorFlow.  Return the embedded sequence.


```python
def get_embed(input_data, vocab_size, embed_dim):
    """
    Create embedding for <input_data>.
    :param input_data: TF placeholder for text input.
    :param vocab_size: Number of words in vocabulary.
    :param embed_dim: Number of embedding dimensions
    :return: Embedded input.
    """
    embedding = tf.Variable(tf.random_uniform((vocab_size, embed_dim), -1, 1))
    embed = tf.nn.embedding_lookup(embedding, input_data)
    return embed


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_get_embed(get_embed)
```

    Tests Passed


### Build RNN
You created a RNN Cell in the `get_init_cell()` function.  Time to use the cell to create a RNN.
- Build the RNN using the [`tf.nn.dynamic_rnn()`](https://www.tensorflow.org/api_docs/python/tf/nn/dynamic_rnn)
 - Apply the name "final_state" to the final state using [`tf.identity()`](https://www.tensorflow.org/api_docs/python/tf/identity)

Return the outputs and final_state state in the following tuple `(Outputs, FinalState)` 


```python
def build_rnn(cell, inputs):
    """
    Create a RNN using a RNN Cell
    :param cell: RNN Cell
    :param inputs: Input text data
    :return: Tuple (Outputs, Final State)
    """
    outputs, finalstate = tf.nn.dynamic_rnn(cell, inputs, dtype=tf.float32)
    final_state = tf.identity(finalstate, name='final_state')
    return outputs, final_state


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_build_rnn(build_rnn)
```

    Tests Passed


### Build the Neural Network
Apply the functions you implemented above to:
- Apply embedding to `input_data` using your `get_embed(input_data, vocab_size, embed_dim)` function.
- Build RNN using `cell` and your `build_rnn(cell, inputs)` function.
- Apply a fully connected layer with a linear activation and `vocab_size` as the number of outputs.

Return the logits and final state in the following tuple (Logits, FinalState) 


```python
def build_nn(cell, rnn_size, input_data, vocab_size):
    """
    Build part of the neural network
    :param cell: RNN cell
    :param rnn_size: Size of rnns
    :param input_data: Input data
    :param vocab_size: Vocabulary size
    :return: Tuple (Logits, FinalState)
    """
    embed = get_embed(input_data, vocab_size, rnn_size*2)
    outputs, final_state = build_rnn(cell, embed)
    logits = tf.contrib.layers.fully_connected(outputs, vocab_size, activation_fn=None)
    return logits, final_state


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_build_nn(build_nn)
```

    Tests Passed


### Batches
Implement `get_batches` to create batches of input and targets using `int_text`.  The batches should be a Numpy array with the shape `(number of batches, 2, batch size, sequence length)`. Each batch contains two elements:
- The first element is a single batch of **input** with the shape `[batch size, sequence length]`
- The second element is a single batch of **targets** with the shape `[batch size, sequence length]`

If you can't fill the last batch with enough data, drop the last batch.

For exmple, `get_batches([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], 2, 3)` would return a Numpy array of the following:
```
[
  # First Batch
  [
    # Batch of Input
    [[ 1  2  3], [ 7  8  9]],
    # Batch of targets
    [[ 2  3  4], [ 8  9 10]]
  ],
 
  # Second Batch
  [
    # Batch of Input
    [[ 4  5  6], [10 11 12]],
    # Batch of targets
    [[ 5  6  7], [11 12 13]]
  ]
]
```


```python
def get_batches(int_text, batch_size, seq_length):
    """
    Return batches of input and target
    :param int_text: Text with the words replaced by their ids
    :param batch_size: The size of batch
    :param seq_length: The length of sequence
    :return: Batches as a Numpy array
    """
    n_batches = len(int_text) // (batch_size * seq_length)
    xdata = np.array(int_text[: n_batches * batch_size * seq_length])
    ydata = np.array(int_text[1: n_batches * batch_size * seq_length + 1])

    x_batches = np.split(xdata.reshape(batch_size, -1), n_batches, 1)
    y_batches = np.split(ydata.reshape(batch_size, -1), n_batches, 1)

    return np.array(list(zip(x_batches, y_batches)))


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_get_batches(get_batches)
```

    Tests Passed


## Neural Network Training
### Hyperparameters
Tune the following parameters:

- Set `num_epochs` to the number of epochs.
- Set `batch_size` to the batch size.
- Set `rnn_size` to the size of the RNNs.
- Set `seq_length` to the length of sequence.
- Set `learning_rate` to the learning rate.
- Set `show_every_n_batches` to the number of batches the neural network should print progress.


```python
# Number of Epochs
num_epochs = 64
# Batch Size
batch_size = 128
# RNN Size
rnn_size = 1024
# Sequence Length
seq_length = 8
# Learning Rate
learning_rate = 0.001
# Show stats for every n number of batches
show_every_n_batches = int(batch_size/10)

"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
save_dir = './save'
```

### Build the Graph
Build the graph using the neural network you implemented.


```python
"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
from tensorflow.contrib import seq2seq

train_graph = tf.Graph()
with train_graph.as_default():
    vocab_size = len(int_to_vocab)
    input_text, targets, lr = get_inputs()
    input_data_shape = tf.shape(input_text)
    cell, initial_state = get_init_cell(input_data_shape[0], rnn_size)
    logits, final_state = build_nn(cell, rnn_size, input_text, vocab_size)

    # Probabilities for generating words
    probs = tf.nn.softmax(logits, name='probs')

    # Loss function
    cost = seq2seq.sequence_loss(
        logits,
        targets,
        tf.ones([input_data_shape[0], input_data_shape[1]]))

    # Optimizer
    optimizer = tf.train.AdamOptimizer(lr)

    # Gradient Clipping
    gradients = optimizer.compute_gradients(cost)
    capped_gradients = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradients]
    train_op = optimizer.apply_gradients(capped_gradients)
```

## Train
Train the neural network on the preprocessed data.  If you have a hard time getting a good loss, check the [forms](https://discussions.udacity.com/) to see if anyone is having the same problem.


```python
"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
batches = get_batches(int_text, batch_size, seq_length)

with tf.Session(graph=train_graph) as sess:
    sess.run(tf.global_variables_initializer())

    for epoch_i in range(num_epochs):
        state = sess.run(initial_state, {input_text: batches[0][0]})

        for batch_i, (x, y) in enumerate(batches):
            feed = {
                input_text: x,
                targets: y,
                initial_state: state,
                lr: learning_rate}
            train_loss, state, _ = sess.run([cost, final_state, train_op], feed)

            # Show every <show_every_n_batches> batches
            if (epoch_i * len(batches) + batch_i) % show_every_n_batches == 0:
                print('Epoch {:>3} Batch {:>4}/{}   train_loss = {:.3f}'.format(
                    epoch_i,
                    batch_i,
                    len(batches),
                    train_loss))

    # Save Model
    saver = tf.train.Saver()
    saver.save(sess, save_dir)
    print('Model Trained and Saved')
```

    Epoch   0 Batch    0/23   train_loss = 8.654
    Epoch   0 Batch   12/23   train_loss = 6.603
    Epoch   1 Batch    1/23   train_loss = 6.269
    Epoch   1 Batch   13/23   train_loss = 6.065
    Epoch   2 Batch    2/23   train_loss = 6.002
    Epoch   2 Batch   14/23   train_loss = 5.995
    Epoch   3 Batch    3/23   train_loss = 5.842
    Epoch   3 Batch   15/23   train_loss = 5.800
    Epoch   4 Batch    4/23   train_loss = 5.595
    Epoch   4 Batch   16/23   train_loss = 5.523
    Epoch   5 Batch    5/23   train_loss = 5.345
    Epoch   5 Batch   17/23   train_loss = 5.221
    Epoch   6 Batch    6/23   train_loss = 5.155
    Epoch   6 Batch   18/23   train_loss = 4.932
    Epoch   7 Batch    7/23   train_loss = 4.803
    Epoch   7 Batch   19/23   train_loss = 4.660
    Epoch   8 Batch    8/23   train_loss = 4.652
    Epoch   8 Batch   20/23   train_loss = 4.533
    Epoch   9 Batch    9/23   train_loss = 4.509
    Epoch   9 Batch   21/23   train_loss = 4.247
    Epoch  10 Batch   10/23   train_loss = 4.190
    Epoch  10 Batch   22/23   train_loss = 4.054
    Epoch  11 Batch   11/23   train_loss = 4.051
    Epoch  12 Batch    0/23   train_loss = 3.923
    Epoch  12 Batch   12/23   train_loss = 3.888
    Epoch  13 Batch    1/23   train_loss = 3.708
    Epoch  13 Batch   13/23   train_loss = 3.607
    Epoch  14 Batch    2/23   train_loss = 3.546
    Epoch  14 Batch   14/23   train_loss = 3.546
    Epoch  15 Batch    3/23   train_loss = 3.364
    Epoch  15 Batch   15/23   train_loss = 3.327
    Epoch  16 Batch    4/23   train_loss = 3.293
    Epoch  16 Batch   16/23   train_loss = 3.178
    Epoch  17 Batch    5/23   train_loss = 3.156
    Epoch  17 Batch   17/23   train_loss = 2.989
    Epoch  18 Batch    6/23   train_loss = 2.931
    Epoch  18 Batch   18/23   train_loss = 2.697
    Epoch  19 Batch    7/23   train_loss = 2.614
    Epoch  19 Batch   19/23   train_loss = 2.423
    Epoch  20 Batch    8/23   train_loss = 2.476
    Epoch  20 Batch   20/23   train_loss = 2.318
    Epoch  21 Batch    9/23   train_loss = 2.326
    Epoch  21 Batch   21/23   train_loss = 2.249
    Epoch  22 Batch   10/23   train_loss = 2.160
    Epoch  22 Batch   22/23   train_loss = 2.038
    Epoch  23 Batch   11/23   train_loss = 2.037
    Epoch  24 Batch    0/23   train_loss = 1.901
    Epoch  24 Batch   12/23   train_loss = 1.815
    Epoch  25 Batch    1/23   train_loss = 1.674
    Epoch  25 Batch   13/23   train_loss = 1.549
    Epoch  26 Batch    2/23   train_loss = 1.595
    Epoch  26 Batch   14/23   train_loss = 1.535
    Epoch  27 Batch    3/23   train_loss = 1.348
    Epoch  27 Batch   15/23   train_loss = 1.273
    Epoch  28 Batch    4/23   train_loss = 1.327
    Epoch  28 Batch   16/23   train_loss = 1.042
    Epoch  29 Batch    5/23   train_loss = 1.124
    Epoch  29 Batch   17/23   train_loss = 1.138
    Epoch  30 Batch    6/23   train_loss = 1.111
    Epoch  30 Batch   18/23   train_loss = 0.991
    Epoch  31 Batch    7/23   train_loss = 0.932
    Epoch  31 Batch   19/23   train_loss = 0.807
    Epoch  32 Batch    8/23   train_loss = 0.843
    Epoch  32 Batch   20/23   train_loss = 0.702
    Epoch  33 Batch    9/23   train_loss = 0.674
    Epoch  33 Batch   21/23   train_loss = 0.702
    Epoch  34 Batch   10/23   train_loss = 0.587
    Epoch  34 Batch   22/23   train_loss = 0.530
    Epoch  35 Batch   11/23   train_loss = 0.514
    Epoch  36 Batch    0/23   train_loss = 0.481
    Epoch  36 Batch   12/23   train_loss = 0.495
    Epoch  37 Batch    1/23   train_loss = 0.473
    Epoch  37 Batch   13/23   train_loss = 0.396
    Epoch  38 Batch    2/23   train_loss = 0.433
    Epoch  38 Batch   14/23   train_loss = 0.401
    Epoch  39 Batch    3/23   train_loss = 0.394
    Epoch  39 Batch   15/23   train_loss = 0.367
    Epoch  40 Batch    4/23   train_loss = 0.358
    Epoch  40 Batch   16/23   train_loss = 0.274
    Epoch  41 Batch    5/23   train_loss = 0.284
    Epoch  41 Batch   17/23   train_loss = 0.278
    Epoch  42 Batch    6/23   train_loss = 0.330
    Epoch  42 Batch   18/23   train_loss = 0.232
    Epoch  43 Batch    7/23   train_loss = 0.254
    Epoch  43 Batch   19/23   train_loss = 0.218
    Epoch  44 Batch    8/23   train_loss = 0.257
    Epoch  44 Batch   20/23   train_loss = 0.235
    Epoch  45 Batch    9/23   train_loss = 0.225
    Epoch  45 Batch   21/23   train_loss = 0.245
    Epoch  46 Batch   10/23   train_loss = 0.225
    Epoch  46 Batch   22/23   train_loss = 0.213
    Epoch  47 Batch   11/23   train_loss = 0.213
    Epoch  48 Batch    0/23   train_loss = 0.196
    Epoch  48 Batch   12/23   train_loss = 0.187
    Epoch  49 Batch    1/23   train_loss = 0.202
    Epoch  49 Batch   13/23   train_loss = 0.192
    Epoch  50 Batch    2/23   train_loss = 0.210
    Epoch  50 Batch   14/23   train_loss = 0.213
    Epoch  51 Batch    3/23   train_loss = 0.209
    Epoch  51 Batch   15/23   train_loss = 0.206
    Epoch  52 Batch    4/23   train_loss = 0.209
    Epoch  52 Batch   16/23   train_loss = 0.180
    Epoch  53 Batch    5/23   train_loss = 0.194
    Epoch  53 Batch   17/23   train_loss = 0.201
    Epoch  54 Batch    6/23   train_loss = 0.209
    Epoch  54 Batch   18/23   train_loss = 0.170
    Epoch  55 Batch    7/23   train_loss = 0.193
    Epoch  55 Batch   19/23   train_loss = 0.163
    Epoch  56 Batch    8/23   train_loss = 0.202
    Epoch  56 Batch   20/23   train_loss = 0.190
    Epoch  57 Batch    9/23   train_loss = 0.182
    Epoch  57 Batch   21/23   train_loss = 0.203
    Epoch  58 Batch   10/23   train_loss = 0.191
    Epoch  58 Batch   22/23   train_loss = 0.179
    Epoch  59 Batch   11/23   train_loss = 0.182
    Epoch  60 Batch    0/23   train_loss = 0.169
    Epoch  60 Batch   12/23   train_loss = 0.162
    Epoch  61 Batch    1/23   train_loss = 0.182
    Epoch  61 Batch   13/23   train_loss = 0.173
    Epoch  62 Batch    2/23   train_loss = 0.193
    Epoch  62 Batch   14/23   train_loss = 0.197
    Epoch  63 Batch    3/23   train_loss = 0.194
    Epoch  63 Batch   15/23   train_loss = 0.189
    Model Trained and Saved


## Save Parameters
Save `seq_length` and `save_dir` for generating a new TV script.


```python
"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
# Save parameters for checkpoint
helper.save_params((seq_length, save_dir))
```

# Checkpoint


```python
"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
import tensorflow as tf
import numpy as np
import helper
import problem_unittests as tests

_, vocab_to_int, int_to_vocab, token_dict = helper.load_preprocess()
seq_length, load_dir = helper.load_params()
```

## Implement Generate Functions
### Get Tensors
Get tensors from `loaded_graph` using the function [`get_tensor_by_name()`](https://www.tensorflow.org/api_docs/python/tf/Graph#get_tensor_by_name).  Get the tensors using the following names:
- "input:0"
- "initial_state:0"
- "final_state:0"
- "probs:0"

Return the tensors in the following tuple `(InputTensor, InitialStateTensor, FinalStateTensor, ProbsTensor)` 


```python
def get_tensors(loaded_graph):
    """
    Get input, initial state, final state, and probabilities tensor from <loaded_graph>
    :param loaded_graph: TensorFlow graph loaded from file
    :return: Tuple (InputTensor, InitialStateTensor, FinalStateTensor, ProbsTensor)
    """
    input_tensor = loaded_graph.get_tensor_by_name('input:0')
    init_state_tensor = loaded_graph.get_tensor_by_name('initial_state:0')
    fin_state_tensor = loaded_graph.get_tensor_by_name('final_state:0')
    probs_tensor = loaded_graph.get_tensor_by_name('probs:0')
    return input_tensor, init_state_tensor, fin_state_tensor, probs_tensor


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_get_tensors(get_tensors)
```

    Tests Passed


### Choose Word
Implement the `pick_word()` function to select the next word using `probabilities`.


```python
def pick_word(probabilities, int_to_vocab):
    """
    Pick the next word in the generated text
    :param probabilities: Probabilites of the next word
    :param int_to_vocab: Dictionary of word ids as the keys and words as the values
    :return: String of the predicted word
    """
    wordlist = list(int_to_vocab.values())
    return np.random.choice(wordlist, p=probabilities)


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_pick_word(pick_word)
```

    Tests Passed


## Generate TV Script
This will generate the TV script for you.  Set `gen_length` to the length of TV script you want to generate.


```python
gen_length = 500
# homer_simpson, moe_szyslak, or Barney_Gumble
prime_word = 'evening'

"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
loaded_graph = tf.Graph()
with tf.Session(graph=loaded_graph) as sess:
    # Load saved model
    loader = tf.train.import_meta_graph(load_dir + '.meta')
    loader.restore(sess, load_dir)

    # Get Tensors from loaded model
    input_text, initial_state, final_state, probs = get_tensors(loaded_graph)

    # Sentences generation setup
    gen_sentences = [prime_word]
    prev_state = sess.run(initial_state, {input_text: np.array([[1]])})

    # Generate sentences
    for n in range(gen_length):
        # Dynamic Input
        dyn_input = [[vocab_to_int[word] for word in gen_sentences[-seq_length:]]]
        dyn_seq_length = len(dyn_input[0])

        # Get Prediction
        probabilities, prev_state = sess.run(
            [probs, final_state],
            {input_text: dyn_input, initial_state: prev_state})
        
        pred_word = pick_word(probabilities[dyn_seq_length-1], int_to_vocab)

        gen_sentences.append(pred_word)
    
    # Remove tokens
    tv_script = ' '.join(gen_sentences)
    for key, token in token_dict.items():
        ending = ' ' if key in ['\n', '(', '"'] else ''
        tv_script = tv_script.replace(' ' + token.lower(), key)
    tv_script = tv_script.replace('\n ', '\n')
    tv_script = tv_script.replace('( ', '(')
        
    print(tv_script)
```

    evening, a dim suspicion of there being a meaning in it
    which you-- you so to have a uncouth hole. it were within time to the
    faint uneasiness risk. a deep voice reached
    me head-dresses of stupid-- and. they beguiled the one of the land went a
    large shining crawling on the earth. we almost an
    large shining crawling on that the steamer was at an end of unequal
    hand, more than a year since the news came; 'we struck this told you beyond the
    land, where i did not go to join kurtz there and then. i have
    not him whether i am now. '
    
    " his aspect reminded me of an opinion-- something a little experience. one of kurtz
    done? they had no inherited experience to teach no more
    to us than a long time.
    
    " i pulled the name of seamen given get
    hold of mr. kurtz-- these government on it a pair of fuss. well, you see, i don't
    see that....... 'i have
    on the time, he stared at me with an intention. 'of course this only
    of the pilgrims. 'but when well, you see, i had also to be carried in
    a helmet; she had brass leggings, and educated still a
    laugh. then with their staves being-- an villages round that you feeling i knew
    something. but he meant to him-- straw else, burst into a dishonouring who
    was not crawling, but i had a cup of tea-- the last
    flickers of a middle-aged negro with a hopeless anywhere, and the
    lotus-flower--" mind, none of us would feel exactly profitable, so to his
    respective flipper of those and condemnation. i could the idea of the
    half-caste, who, as far as i could see, with a trifle. the thing was not in the
    faint uneasiness; and the anchor had failed them. the blinding
    uniform notions on this hillside, i had to crawl in three years impressive, and
    collapsed. it was lying on the earth to the last screech of all the snags, too-- because
    , 'now i didn't go to the devil of your adversary. mr. kurtz was
    , if he had good up and soothing? i had
    a way. for his pages to interrupt the contrary. it appears their intercourse had
    been teaching one, and the voice of these confidences i
    no vital anxieties now, i didn't make go ready to
    his uproar. the


# The TV Script is Nonsensical
It's ok if the TV script doesn't make any sense.  We trained on less than a megabyte of text.  In order to get good results, you'll have to use a smaller vocabulary or get more data.  Luckly there's more data!  As we mentioned in the begging of this project, this is a subset of [another dataset](https://www.kaggle.com/wcukierski/the-simpsons-by-the-data).  We didn't have you train on all the data, because that would take too long.  However, you are free to train your neural network on all the data.  After you complete the project, of course.
# Submitting This Project
When submitting this project, make sure to run all the cells before saving the notebook. Save the notebook file as "dlnd_tv_script_generation.ipynb" and save it as a HTML file under "File" -> "Download as". Include the "helper.py" and "problem_unittests.py" files in your submission.
