import os
import cPickle

#from __future__ import print_function
from keras.callbacks import LambdaCallback
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
import numpy as np
import random
import sys
import io

#import theano
#import theano.tensor as T
#import lasagne

def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

if __name__ == '__main__':
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

	x = np.zeros((len_sentences, span_size, num_voca), dtype=np.bool)
	y = np.zeros((len_sentences, num_voca), dtype=np.bool)

	for url in prep_corpus:
		for i, sentence in enumerate(prep_corpus[url]['script']):
			for t, wi in enumerate(sentence):
				x[i, t, wi] = 1

	model = Sequential()
	model.add(LSTM(num_voca, input_shape=(span_size, num_voca), return_sequences=True))

	optimizer = RMSprop(lr=0.01)
	model.compile(loss='categorical_crossentropy', optimizer=optimizer)
	model.fit(x[:100], x[:100],
			  batch_size=1,
			  epochs=1)

	preds = model.predict(x_pred, verbose=0)[0]

	#prep_corpus[url]['length'] = len(tot_sents)
	#prep_corpus[url]['script'] = np.array(tot_sents)
	#prep_corpus[url]['laughter'] = np.array(tot_laughter)
	#prep_corpus[url]['applause'] = np.array(tot_applause)
