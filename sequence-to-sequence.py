import tensorflow as tf
import numpy as np
from tensorflow.python.ops.rnn_cell_impl import LSTMCell, LSTMStateTuple

padding = 0
end_of_sentence = 1

vocab_size = 10
input_embedding_size = 20

encoder_hidden_units = 20  # thousands
decoder_hidden_units = encoder_hidden_units * 2  # output is little different to input in oder to optimize the result?

# placeholders
encoder_inputs = tf.placeholder(shape=(None, None), dtype=tf.int32, name='encoder_inputs')
encoder_inputs_length = tf.placeholder(shape=(None,), dtype=tf.int32, name='encoder_inputs_length')
decoder_targets = tf.placeholder(shape=(None, None), dtype=tf.int32, name='decoder_targets')

# embeddings
embeddings = tf.Variable(tf.random_uniform([vocab_size, input_embedding_size], -1, 0, 1), dtype=tf.float32)
encoder_inputs_embedded = tf.nn.embedding_lookup(embeddings, encoder_inputs)

# define encoder
encoder_cell = LSTMCell(encoder_hidden_units)
((encoder_fw_outputs,
  encoder_bw_outputs),
 (encoder_fw_final_state,
  encoder_bw_final_state)) = (
    tf.nn.bidirectional_dynamic_rnn(cell_fw=encoder_cell,
                                    cell_bw=encoder_cell,
                                    inputs=encoder_inputs_embedded,
                                    sequence_length=encoder_inputs_length,
                                    initial_state_fw=None,
                                    initial_state_bw=None,
                                    parallel_iterations=None,
                                    dtype=tf.float32,
                                    swap_memory=False,
                                    time_major=True,
                                    scope=None)
)

# bidirectional step(forward and backward) : expensive but better prediction
encoder_outputs = tf.concat((encoder_fw_outputs, encoder_bw_outputs), 2)
encoder_final_state_c = tf.concat(
    (encoder_fw_final_state.c, encoder_bw_final_state.c), 1)
encoder_final_state_h = tf.concat(
    (encoder_fw_final_state.h, encoder_bw_final_state.h), 1)

# combine all together(backward and forward final state) for decoder feed
encoder_final_state = LSTMStateTuple(
    c=encoder_final_state_c,
    h=encoder_final_state_h
)

# defining decoder :batch size is the most important one !!
# LSTM (Long short term memory units)
decoder_cell = LSTMCell(decoder_hidden_units)
encoder_max_time, batch_size = tf.unstack(tf.shape(encoder_inputs))
decoder_lengths = encoder_inputs_length + 3
# ass 3 bcz 2 additional steps below
# 1 for the leading end of sentence token for the decoder input
# we want it to be a little bigger for the end of sentence token which indicates the end of sequence

# dividing into small batch size=> make prediction better (little more computationally expensive), not always
# GRU has less gates than LSTM (less expensive but tends to have better results specifically for dynamic network=>coooooool!!!)

# defining weights and biases
W = tf.Variable(tf.random_uniform([decoder_hidden_units, vocab_size], -1, 1), dtype=tf.float32)
b = tf.Variable(tf.zeros([vocab_size]), dtype=tf.float32)


assert end_of_sentence == 1 and padding == 0

eos_time_slice = tf.ones([batch_size], dtype=tf.int32, name='end_of_sentence')
pad_time_slice = tf.zeros([batch_size], dtype=tf.int32, name='padding')

eos_step_embedded = tf.nn.embedding_lookup(embeddings, eos_time_slice)
pad_step_embedded = tf.nn.embedding_lookup(embeddings, pad_time_slice)

# loop initialization for next step
def loop_fn_initial():
    initial_elements_finished = (0 >= decoder_lengths)
    initial_input = eos_step_embedded
    initial_cell_state = encoder_final_state
    initial_cell_output = None
    initial_loop_state = None
    return (initial_elements_finished,
            initial_input,
            initial_cell_state,
            initial_cell_output,
            initial_loop_state)

# transitioning for next loop(attention mechanism? is it kind of one hot?)
def loop_fn_transition(time, previous_output, previous_state, previous_loop_state):
    def get_next_input():
       output_logits = tf.add(tf.matmul(previous_output, W), b)
       # attention
       prediction = tf.argmax(output_logits, axis=1)
       next_input = tf.nn.embedding_lookup(embeddings, prediction)
       return next_input

    elements_finished = (time >= decoder_lengths)
    finished = tf.reduce_all(elements_finished)
    input = tf.cond(finished, lambda: pad_step_embedded, get_next_input)

    # set previous to current
    state = previous_state
    output = previous_output
    loop_state = None

    return (elements_finished,
            input,
            state,
            output,
            loop_state)

# fill the data and looping
def loop_fn(time, previous_output, previous_state, previous_loop_state):
    if previous_state is None:  # time == 0
        assert previous_output is None and previous_state is None
        return loop_fn_initial()
    else:
        return loop_fn_transition(time, previous_output, previous_state, previous_loop_state)

decoder_outputs_ta, decoder_final_state, _ = tf.nn.raw_rnn(decoder_cell, loop_fn)
decoder_outputs = decoder_outputs_ta.stack()

# formatting decoder output into valid prediction
decoder_max_steps, decoder_batch_size, decoder_dim = tf.unstack(tf.shape(decoder_outputs))
decoder_outputs_flat = tf.reshape(decoder_outputs, (-1, decoder_dim))
decoder_logits_flat = tf.add(tf.matmul(decoder_outputs_flat, W), b)
decoder_logits = tf.reshape(decoder_logits_flat, (decoder_max_steps, decoder_batch_size, vocab_size))

# final prediction value
decoder_prediction = tf.argmax(decoder_logits, 2)

# cross entropy loss
stepwise_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
    labels=tf.one_hot(decoder_targets, depth=vocab_size, dtype=tf.float32),
    logits=decoder_logits,
)

# loss
loss = tf.reduce_mean(stepwise_cross_entropy)

# train : Adamoptimizer
train_op = tf.train.AdamOptimizer().minimize(loss)

# helper functions : to generate data
def helper_batch(inputs, max_sequence_length=None):
    sequence_lengths = [len(seq) for seq in inputs]
    batch_size = len(inputs)

    if max_sequence_length is None:
        max_sequence_length = max(sequence_lengths)

    inputs_batch_major = np.zeros(shape=[batch_size, max_sequence_length], dtype=np.int32)  # == PAD

    for i, seq in enumerate(inputs):
        for j, element in enumerate(seq):
            inputs_batch_major[i, j] = element

    inputs_time_major = inputs_batch_major.swapaxes(0, 1)

    return inputs_time_major, sequence_lengths


def helper_random_sequences(length_from, length_to,
                     vocab_lower, vocab_upper,
                     batch_size):

    if length_from > length_to:
        raise ValueError('length_from > length_to')

    def random_length():
        if length_from == length_to:
            return length_from
        return np.random.randint(length_from, length_to + 1)

    while True:
        yield [
            np.random.randint(low=vocab_lower,
                              high=vocab_upper,
                              size=random_length()).tolist()
            for _ in range(batch_size)
        ]
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

batch_size = 100

batches = helper_random_sequences(length_from=3, length_to=8,
                                   vocab_lower=2, vocab_upper=10,
                                   batch_size=batch_size)

print('head of the batch:')
for seq in next(batches)[:10]:
    print(seq)

# continuously generate data using next_feed function
def next_feed():
    batch = next(batches)
    encoder_inputs_, encoder_input_lengths_ = helper_batch(batch)
    decoder_targets_, _ = helper_batch(
        [(sequence) + [end_of_sentence] + [padding] * 2 for sequence in batch]
    )
    return {
        encoder_inputs: encoder_inputs_,
        encoder_inputs_length: encoder_input_lengths_,
        decoder_targets: decoder_targets_,
    }

loss_track = []

# training step
max_batches = 3001
batches_in_epoch = 1000

try:
    for batch in range(max_batches):
        feed_dict = next_feed()
        _, l = sess.run([train_op, loss], feed_dict=feed_dict)
        loss_track.append(l)

        if batch == 0 or batch % batches_in_epoch == 0:
            print('batch {}'.format(batch))
            print('  minibatch loss: {}'.format(sess.run(loss, feed_dict=feed_dict)))
            predict_ = sess.run(decoder_prediction, feed_dict=feed_dict)
            for i, (inp, pred) in enumerate(zip(feed_dict[encoder_inputs].T, predict_.T)):
                print('  sample {}:'.format(i + 1))
                print('    input     > {}'.format(inp))
                print('    predicted > {}'.format(pred))
                if i >= 2:
                    break
            print()

except KeyboardInterrupt:
    print('training interrupted')


print('Learning Finished!')

