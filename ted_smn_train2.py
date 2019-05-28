import theano, cPickle, h5py, lasagne, random, csv, gzip, time                                                  
import numpy as np
import theano.tensor as T 
from layers import *
from util import *
from lasagne.layers import *

import ast, json
import os
import argparse

from nltk.corpus import stopwords  
stopWords = stopwords.words('english')



# assemble the network
def build_rmn(span_size, d_word, d_hidden, len_voc, num_descs, eps=1e-5, lr=0.01):
	print 'd_hidden', d_hidden
	print 'len_voc', len_voc
	print 'num_desc', num_descs

	# input theano vars
	in_spans = T.imatrix(name='spans')
	in_slen = T.imatrix('sent_length')
	in_empty = T.imatrix('empty')
	in_laughter = T.imatrix('num_laughter')
	in_applause = T.imatrix('num_applause')
	in_currmasks = T.matrix(name='curr_masks')

	# define network
	l_inspans = lasagne.layers.InputLayer(shape=(None, span_size), 
        input_var=in_spans)
	l_currmask = lasagne.layers.InputLayer(shape=(None, span_size), 
		input_var=in_currmasks)
	l_slen = lasagne.layers.InputLayer(shape=(None, 1), 
		input_var=in_slen)
	l_empty = lasagne.layers.InputLayer(shape=(None, 1), 
		input_var=in_empty)
	l_laughter = lasagne.layers.InputLayer(shape=(None, 1), 
		input_var=in_laughter)
	l_applause = lasagne.layers.InputLayer(shape=(None, 1), 
		input_var=in_applause)


	l_emb = MyEmbeddingLayer(l_inspans, len_voc, 
			d_word, name='word_emb')

	# freeze embeddings
	#if freeze_words:
	#	l_emb.params[l_emb.W].remove('trainable')

		

	# average each span's embeddings
	l_currsum = AverageLayer([l_emb, l_currmask], d_word)
	#l_curr = ElemwiseMergeLayer([l_spans, l_currmask], T.mul)

	# pass all embeddings thru feed-forward layer
	l_concat = ConcatLayer([l_currsum, l_slen, l_empty, l_laughter, l_applause])
	l_mix = DenseLayer(l_concat, d_word)

	# compute recurrent weights over dictionary
	l_rels = RecurrentRelationshipLayer(\
			l_mix, d_word, d_hidden, num_descs)

	# multiply weights with dictionary matrix
	l_recon = ReconLayer(l_rels, d_word, num_descs)

	# compute loss
	currsums = lasagne.layers.get_output(l_currsum)
	recon = lasagne.layers.get_output(l_recon)

	currsums /= currsums.norm(2, axis=1)[:, None]
	recon /= recon.norm(2, axis=1)[:, None]
	correct = T.sum(recon * currsums, axis=1)
	loss = T.sum(T.maximum(0., 
				T.sum(1. - correct[:, None], axis=1)))

	# enforce orthogonality constraint
	norm_R = l_recon.R / l_recon.R.norm(2, axis=1)[:, None]
	ortho_penalty = eps * T.sum((T.dot(norm_R, norm_R.T) - \
				T.eye(norm_R.shape[0])) ** 2)
	loss += ortho_penalty

	all_params = lasagne.layers.get_all_params(l_recon, trainable=True)
	updates = lasagne.updates.adam(loss, all_params, learning_rate=lr)

	traj_fn = theano.function([in_spans, in_currmasks, in_slen, 
			in_empty, in_laughter, in_applause],
			lasagne.layers.get_output(l_rels))
	
	#ex_cost, ex_ortho = train_fn(span, mask, slen, empty, laughter, applause, allow_input_downcast=True)
	train_fn = theano.function([in_spans, in_currmasks, in_slen, 
			in_empty, in_laughter, in_applause],
			[loss, ortho_penalty], updates=updates)
	return train_fn, traj_fn, l_recon



if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Process some integers.')
	parser.add_argument('--init', dest='init', action='store_true')
	parser.add_argument('--test', dest='test', action='store_true')
	parser.add_argument('--no-init', dest='init', action='store_false')
	parser.set_defaults(init=False, test=False)
	 
	args = parser.parse_args()
	ted_prep_corpus_dir = './prep/ted_prep_corpus.pkl'
	if args.test:
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

	print 'loading completed!'

	span_size = 40
	 
	rnvmap = {v:k for k, v in prep_nvmap.items()}
	nv_counter = {}
	for i in range(len(prep_nvmap)):
		nv_counter[i] = 0
	for url, data in prep_corpus.items():
		length = data['length']
		lines = data['script']
		for line in lines:
			for v in line:
				nv_counter[v] += 1

	d = nv_counter
	num_fb = 1000 # number of frequent bigram
	fbmap = {None: 0} # frequent bigram map
	for i in sorted(d, key=d.get, reverse=True):
		if rnvmap[i] == '_' or rnvmap[i] == '_ _': continue
		fbmap[i] = len(fbmap)
		if len(fbmap) >= num_fb: break
		if d[i] == 0: break
		print i, '\t', rnvmap[i], '\t', d[i]
	rfbmap = {v:k for k, v in fbmap.items()}

	
	ted_train_data = []
	for url, data in prep_corpus.items():
		length = data['length']
		data_script = data['script']
		data_mask = np.zeros(data_script.shape, dtype=np.int32)
		data_empty = np.zeros(length, dtype=np.int32)
		data_slen = np.zeros(length, dtype=np.int32)
		for i in range(length):
			is_empty = True
			scount = 0
			for j in range(span_size):
				wbg = data_script[i, j]
				if wbg != 0:
					scount += 1
				if wbg not in fbmap:
					data_script[i, j] = 0
					data_mask[i, j] = 0
				else:
					data_script[i, j] = fbmap[wbg]
					data_mask[i, j] = 1
					is_empty = False
		
			if is_empty:
				data_empty[i] = 1
			data_slen[i] = scount

		data['mask'] = data_mask
		data['empty'] = data_empty
		data['slen'] = data_slen
		ted_train_data.append(data)
		print(data)
		

	descriptor_log = 'models/descriptors.log'
	trajectory_log = 'models/trajectories.log'

	
	# d_word
	d_word = 300

	# embedding/hidden dimensionality
	d_hidden = 256

	# number of descriptors
	num_descs = 30

	# word dropout probability
	# p_drop = 0.75

	n_epochs = 300
	lr = 0.01
	eps = 1e-5
	num_traj = len(prep_corpus)
	len_voc = len(fbmap)
	
	print 'compiling...'
	train_fn, traj_fn, final_layer = build_rmn(
		span_size, d_word, d_hidden, len_voc, num_descs, eps=eps, lr=lr)
	print 'done compiling, now training...'

	# training loop
	min_cost = float('inf')
	for epoch in range(n_epochs):
		cost = 0.
		random.shuffle(ted_train_data)
		start_time = time.time()
		for data in ted_train_data:
			length = data['length']
			span = data['script']
			mask = data['mask']
			slen = data['slen'].reshape((length, 1))
			empty = data['empty'].reshape((length, 1))
			laughter = data['laughter'].reshape((length, 1))
			applause = data['applause'].reshape((length, 1))


			print(span.shape, mask.shape, slen.shape, empty.shape, laughter.shape, applause.shape)

			ex_cost, ex_ortho = train_fn(span, mask, slen, empty, laughter, applause)
			cost += ex_cost

		end_time = time.time()
		# save params if cost went down
		if cost < min_cost:
			min_cost = cost
			params = lasagne.layers.get_all_params(final_layer)
			p_values = [p.get_value() for p in params]
			p_dict = dict(zip([str(p) for p in params], p_values))
			cPickle.dump(p_dict, open('models/tmn_params.pkl', 'wb'),
				protocol=cPickle.HIGHEST_PROTOCOL)

			# compute nearest neighbors of descriptors
			R = p_dict['R']
			log = open(descriptor_log, 'w')
			for ind in range(len(R)):
				desc = R[ind] / np.linalg.norm(R[ind])
				sims = We.dot(desc.T)
				ordered_words = np.argsort(sims)[::-1]
				desc_list = [ revmap[w] for w in ordered_words[:10]]
				log.write(' '.join(desc_list) + '\n')
				print 'descriptor %d:' % ind
				print desc_list
			log.flush()
			log.close()

			# save relationship trajectories
			print 'writing trajectories...'
			tlog = open(trajectory_log, 'wb')
			traj_writer = csv.writer(tlog)
			traj_writer.writerow(['Title'] + \
				['Topic ' + str(i) for i in range(num_descs)])

			for idx, data in enumerate(ted_train_data):
				span = data['script']
				mask = data['mask']
				slen = data['slen']
				empty = data['empty']
				laughter = data['laughter']
				applause = data['applause']

				traj = traj_fn(span, mask, slen, empty, laughter, applause)

				for ind in range(len(traj)):
					step = traj[ind]
					traj_writer.writerow([str(idx)] + list(step) )   

			tlog.flush()
			tlog.close()

        print 'done with epoch: ', epoch, ' cost =',\
			cost / len(span_data), 'time: ', end_time-start_time
