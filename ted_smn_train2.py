import theano, cPickle, h5py, lasagne, random, csv, gzip, time                                                  
import numpy as np
import theano.tensor as T 
from layers import *
from util import *

import ast, json
import os
import argparse

from nltk.corpus import stopwords  
stopWords = stopwords.words('english')



# assemble the network
def build_rmn(d_word, d_rating, d_split, d_hidden, len_voc, num_descs, We, kidxes=None,
	freeze_words=True, eps=1e-5, lr=0.01, negs=10):
	if kidxes != None:
		num_descs = len(kidxes)
	print 'd_word', d_word
	print 'd_rating', d_rating
	print 'd_split', d_split
	print 'd_hidden', d_hidden
	print 'len_voc', len_voc
	print 'num_desc', num_descs

	# input theano vars
	in_spans = T.imatrix(name='spans')
	in_neg = T.imatrix(name='neg_spans')
	in_rating = T.dvector(name='rating')
	in_title = T.dvector(name='title')
	in_currmasks = T.matrix(name='curr_masks')
	in_dropmasks = T.matrix(name='drop_masks')
	in_negmasks = T.matrix(name='neg_masks')

	# define network
	l_inspans = lasagne.layers.InputLayer(shape=(None, span_size), 
        input_var=in_spans)
	l_inneg = lasagne.layers.InputLayer(shape=(negs, span_size), 
		input_var=in_neg)
	l_inrating = lasagne.layers.InputLayer(shape=(d_rating,),
		input_var=in_rating)
	l_intitle = lasagne.layers.InputLayer(shape=(d_word,),
		input_var=in_title)
	l_currmask = lasagne.layers.InputLayer(shape=(None, span_size), 
		input_var=in_currmasks)
	l_dropmask = lasagne.layers.InputLayer(shape=(None, span_size), 
		input_var=in_dropmasks)
	l_negmask = lasagne.layers.InputLayer(shape=(negs, span_size), 
		input_var=in_negmasks)

	# negative examples should use same embedding matrix
	l_emb = MyEmbeddingLayer(l_inspans, len_voc, 
			d_word, W=We, name='word_emb')
	l_negemb = MyEmbeddingLayer(l_inneg, len_voc, 
			d_word, W=l_emb.W, name='word_emb_copy1')

	# freeze embeddings
	if freeze_words:
		l_emb.params[l_emb.W].remove('trainable')
		l_negemb.params[l_negemb.W].remove('trainable')

		

	# average each span's embeddings
	l_currsum = AverageLayer([l_emb, l_currmask], d_word)
	l_dropsum = AverageLayer([l_emb, l_dropmask], d_word)
	l_negsum = AverageLayer([l_negemb, l_negmask], d_word)

	# pass all embeddings thru feed-forward layer
	l_mix = TedMixingLayer([l_dropsum, l_intitle, l_inrating],
			d_word, d_rating)

	# compute recurrent weights over dictionary
	l_rels = RecurrentRelationshipLayer(\
			l_mix, d_word, d_hidden, num_descs)

	# multiply weights with dictionary matrix
	l_recon = ReconLayer(l_rels, d_word, num_descs)
	if kidxes != None:
		#l_recon.params[l_recon.R].remove('trainable')
		myR = np.zeros((num_descs, d_word))
		for i, ki in enumerate(kidxes):
			myR[i] = We[ki]
		l_recon.R.set_value(myR)

	# compute loss
	currsums = lasagne.layers.get_output(l_currsum)
	negsums = lasagne.layers.get_output(l_negsum)
	recon = lasagne.layers.get_output(l_recon)

	currsums /= currsums.norm(2, axis=1)[:, None]
	recon /= recon.norm(2, axis=1)[:, None]
	negsums /= negsums.norm(2, axis=1)[:, None]
	correct = T.sum(recon * currsums, axis=1)
	negs = T.dot(recon, negsums.T)
	loss = T.sum(T.maximum(0., 
				T.sum(1. - correct[:, None] + negs, axis=1)))

	# enforce orthogonality constraint
	norm_R = l_recon.R / l_recon.R.norm(2, axis=1)[:, None]
	ortho_penalty = eps * T.sum((T.dot(norm_R, norm_R.T) - \
				T.eye(norm_R.shape[0])) ** 2)
	loss += ortho_penalty

	all_params = lasagne.layers.get_all_params(l_recon, trainable=True)
	updates = lasagne.updates.adam(loss, all_params, learning_rate=lr)
	#traj = traj_fn(rating, title, curr, cm)
	traj_fn = theano.function([in_title, in_rating, 
			in_spans, in_dropmasks], 
			lasagne.layers.get_output(l_rels))
	
	#ex_cost, ex_ortho = train_fn(title, rating, curr, cm, drop_mask, ns, nm)
	train_fn = theano.function([in_title, in_rating, 
			in_spans, in_currmasks, in_dropmasks,
			in_neg, in_negmasks], 
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
	fbmap = {} # frequent bigram map
	for i in sorted(d, key=d.get, reverse=True):
		if rnvmap[i] == '_' or rnvmap[i] == '_ _': continue
		fbmap[i] = len(fbmap)+1 # starts from 0; which represents  not listed on fbmap
		if len(fbmap) >= num_fb: break
		if d[i] == 0: break
		print i, '\t', rnvmap[i], '\t', d[i]
	rfbmap = {v:k for k, v in fbmap.items()}


	for url, data in prep_corpus.items():
		length = data['length']
		data_script = data['script']
		data_mask = np.zeros(data_script.shape, dtype=np.int32)
		data_empty = np.zeros(length, dtype=np.int32)
		for i in range(length):
			is_empty = True
			for j in range(span_size):
				wbg = data_script[i, j]
				if wbg not in fbmap:
					data_script[i, j] = 0
					data_mask[i, j] = 0
				else:
					data_script[i, j] = fbmap[wbg]
					data_mask[i, j] = 1
					is_empty = False
		
			if is_empty:
				data_empty[i] = 1

		data['mask'] = data_mask
		data['empty'] = data_empty
		print(data)

	

	descriptor_log = 'models/descriptors.log'
	trajectory_log = 'models/trajectories.log'


	# embedding/hidden dimensionality
	d_hidden = 256

	# number of descriptors
	num_descs = 30

	# word dropout probability
	# p_drop = 0.75

	n_epochs = 300
	lr = 0.001
	eps = 1e-6
	num_traj = len(prep_corpus)
	len_voc = len(fbmap)
	
	print 'compiling...'

	keys = ['_', 'people', 'time', 'world', 'laughter', 'life', 
			'applause', 'human', 'love', 'joy', 'sadness', 'lonely',
			'technology', 'entertainment', 'design', 'science', 'idea',
			'business', 'culture', 'motivation', 'goal', 'potential', 'psychology', 'emotion']
	#keytags['_'] = 10000
	#kidxes = [Vi[w] for w in keytags if w in Vi and w in w2v_model and keytags[w] > 100]
	#kidxes = [Vi[w] for w in keys]
	kidxes=None
	#for i in kidxes:
	#	print revmap[i],
	#print ''
	train_fn, traj_fn, final_layer = build_rmn(
			d_word, d_rating, d_split, d_hidden, len_voc, num_descs, We, kidxes=kidxes,
			freeze_words=True, eps=1e-5, lr=0.01, negs=10)
	print 'done compiling, now training...'

	# training loop
	min_cost = float('inf')
	for epoch in range(n_epochs):
		cost = 0.
		random.shuffle(ted_train_data)
		start_time = time.time()
		for title, rating, curr, cm in span_data:
			ns, nm = generate_negative_samples(\
				num_traj, span_size, num_negs, span_data)

			# word dropout
			drop_mask = (np.random.rand(*(cm.shape)) < (1 - p_drop)).astype('float32')
			drop_mask *= cm

			#print title.shape, rating.shape, curr.shape, cm.shape
			ex_cost, ex_ortho = train_fn(title, rating, curr, cm, drop_mask, ns, nm)
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
			for idx, (title, rating, curr, cm) in enumerate(span_data):
				#c1, c2 = [cmap[c] for c in chars]
				#bname = bmap[book[0]]

				# feed unmasked inputs to get trajectories
				traj = traj_fn(title, rating, curr, cm)
				for ind in range(len(traj)):
					step = traj[ind]
					traj_writer.writerow([ted_titles[idx]] + \
					list(step) )   

			tlog.flush()
			tlog.close()

        print 'done with epoch: ', epoch, ' cost =',\
			cost / len(span_data), 'time: ', end_time-start_time
