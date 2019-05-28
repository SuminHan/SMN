from keras.models import Model
from keras.layers import Input, GRU, Dense

import tensorflow as tf
tf.enable_eager_execution()


import os
import cPickle
import numpy as np
ted_prep_corpus_dir = './prep/ted_test_corpus.pkl'
ted_prep_nvmap_dir = './prep/ted_prep_nvmap.pkl'
if not os.path.isfile(ted_prep_corpus_dir) \
	or not os.path.isfile(ted_prep_nvmap_dir):
	print 'Error: {} or {} has not found.'.format(ted_prep_corpus_dir, ted_prep_nvmap_dir)
else:
	print '{} has found, load using cPickle.'.format(ted_prep_corpus_dir)
	print '{} has found, load using cPickle.'.format(ted_prep_nvmap_dir)
	prep_corpus = cPickle.load(open(ted_prep_corpus_dir, "rb"))
	prep_nvmap = cPickle.load(open(ted_prep_nvmap_dir, "rb"))

print(len(prep_corpus), len(prep_nvmap))


len_sentences = 0
for url in prep_corpus:
	len_sentences += prep_corpus[url]['length']
	
span_size = 40
num_voca = len(prep_nvmap)

# Constants
num_encoder_tokens = num_voca
num_decoder_tokens = num_voca
max_encoder_seq_length = span_size
max_decoder_seq_length = span_size

batch_size = 64  # Batch size for training.
epochs = 1  # Number of epochs to train for.
latent_dim = 256  # Latent dimensionality of the encoding space.
num_samples = 1000  # Number of samples to train on.

encoder_input_data = np.zeros(
		(num_samples, max_encoder_seq_length, num_encoder_tokens),
		dtype='float32')
decoder_input_data = np.zeros(
		(num_samples, max_decoder_seq_length, num_decoder_tokens),
		dtype='float32')
decoder_target_data = np.zeros(
		(num_samples, max_decoder_seq_length, num_decoder_tokens),
		dtype='float32')

count = 0
for url in prep_corpus:
	for i, sentence in enumerate(prep_corpus[url]['script']):
		for t, wi in enumerate(sentence):
			encoder_input_data[i, t, wi] = 1.
			decoder_input_data[i, t, wi] = 1.
			if t > 0:
				decoder_target_data[i, t - 1, wi] = 1.
		if count > num_samples:
			break
			


encoder_inputs = Input(shape=(None, num_encoder_tokens))
encoder = GRU(latent_dim, return_state=True)
encoder_outputs, encoder_state_h = encoder(encoder_inputs)

decoder_inputs = Input(shape=(None, num_decoder_tokens))
decoder_gru = GRU(latent_dim, return_sequences=True)
decoder_outputs = decoder_gru(decoder_inputs, initial_state=encoder_state_h)
decoder_dense = Dense(num_decoder_tokens, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
          batch_size=batch_size,
          epochs=epochs,
          validation_split=0.2)

# Save model
model.save('s2s.h5')



encoder_model = Model(encoder_inputs, encoder_state_h)

decoder_state_input_h = Input(shape=(latent_dim,))
decoder_states_inputs = [decoder_state_input_h]
decoder_outputs, state_h, state_c = decoder_gru(
    decoder_inputs, initial_state=decoder_states_inputs)
decoder_states = [state_h]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs] + decoder_states)



def decode_sequence(input_seq):
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1, num_decoder_tokens))
    # Populate the first character of target sequence with the start character.
    target_seq[0, 0, target_token_index['\t']] = 1.

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value)
        print(h)

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = reverse_target_char_index[sampled_token_index]
        decoded_sentence += sampled_char

        # Exit condition: either hit max length
        # or find stop character.
        if (sampled_char == '\n' or
           len(decoded_sentence) > max_decoder_seq_length):
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1.

        # Update states
        states_value = [h, c]

    return decoded_sentence


for seq_index in range(100):
	# Take one sequence (part of the training set)
	# for trying out decoding.
	input_seq = encoder_input_data[seq_index: seq_index + 1]
	decoded_sentence = decode_sequence(input_seq)
	print('-')
	print('Input sentence:', input_texts[seq_index])
	print('Decoded sentence:', decoded_sentence)
def decode_sequence(input_seq):
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1, num_decoder_tokens))
    # Populate the first character of target sequence with the start character.
    target_seq[0, 0, target_token_index['\t']] = 1.

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value)
        print(h)

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = reverse_target_char_index[sampled_token_index]
        decoded_sentence += sampled_char

        # Exit condition: either hit max length
        # or find stop character.
        if (sampled_char == '\n' or
           len(decoded_sentence) > max_decoder_seq_length):
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1.

        # Update states
        states_value = [h, c]

    return decoded_sentence


for seq_index in range(100):
    # Take one sequence (part of the training set)
    # for trying out decoding.
    input_seq = encoder_input_data[seq_index: seq_index + 1]
    decoded_sentence = decode_sequence(input_seq)
    print('-')
    print('Input sentence:', input_texts[seq_index])
    print('Decoded sentence:', decoded_sentence)
