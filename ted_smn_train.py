import theano, cPickle, h5py, lasagne, random, csv, gzip, time                                                  
import numpy as np
import theano.tensor as T 
from layers import *
from util import *

import ast, json
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

from gensim.models import Word2Vec
from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec

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
	descriptor_log = 'models/descriptors.log'
	trajectory_log = 'models/trajectories.log'
	initial_run = True

	print 'loading data...'
	ted_data = cPickle.load(open('ted_data.pkl', 'rb'))

	#rating_tags = {}
	rating_tags = {'Ingenious': 1, 'Funny': 0, 'Inspiring': 4, 'OK': 13, 'Fascinating': 5, 'Persuasive': 7, 'Longwinded': 10, 'Informative': 2, 'Unconvincing': 12, 'Beautiful': 6, 'Jaw-dropping': 11, 'Obnoxious': 8, 'Courageous': 3, 'Confusing': 9}
	d_rating = len(rating_tags)

	
	Vi = {'_': 0}
	textdata = []
	ted_titles = []
	keytags = {}
	for key, data in ted_data.items():
		rating_array = np.zeros(len(rating_tags), dtype=float)
		if int(data['duration']) > 300 and 'transcript' in data:
			ted_titles.append(''.join(c for c in data['title'] if c.isalpha() or c == ' '))
			data['title'] = ''.join(c for c in data['title'].lower() if c.isalpha() or c == ' ')
			data['description'] = ''.join(c for c in data['description'].lower() if c.isalpha() or c == ' ')
			data['transcript'] = data['transcript'].lower()
			textdata.append(data['title'])
			textdata.append(data['description'])
			textdata.append(data['transcript'])
			json_data = ast.literal_eval(data['tags'])
			for tag in json_data:
				for word in tag.split(' '):
					keytags.setdefault(word, 0)
					keytags[word] += 1

			json_data = ast.literal_eval(data['ratings'])
			for tag_data in json_data:
				rating_array[rating_tags[tag_data['name']]] = int(tag_data['count'])
			data['rating_array'] = rating_array

	d_split = 25
	d_word = 50 #either 50, 100, 200, or 300
	glove_file = "data/glove.6B." + str(d_word) + "d.txt"
	gensim_file = "data/gensim.glove.6B." + str(d_word) + "d.txt"

	if not os.path.isfile(gensim_file):
		print("convert", glove_file, "to", gensim_file)
		glove2word2vec(glove_input_file=glove_file, word2vec_output_file=gensim_file)

	We_file = "data/We."+str(d_word)+".glove.p"
	Vi_file = "data/Vi."+str(d_word)+".p"
	w2v_model = None
	if not os.path.isfile(We_file) or not os.path.isfile(Vi_file) or initial_run:
		print("loading", gensim_file)
		model = KeyedVectors.load_word2vec_format(gensim_file,binary=False)
		w2v_model = model

		for text in textdata:
			for word in text.split():
				if word not in Vi:
					Vi[word] = len(Vi)
					if len(Vi) == 10:
						print(Vi)

		We = np.zeros((len(Vi), d_word), dtype=float)
		for word, idx in Vi.items():
			if word in model:
				We[idx] = model[word]
			We[idx] /= sum(We[idx])
			
		norm_We = We / np.linalg.norm(We, axis=1)[:, None]
		We = np.nan_to_num(norm_We)
		cPickle.dump(We, open(We_file, 'wb'))
		cPickle.dump(Vi, open(Vi_file, 'wb'))
	else:
		We = cPickle.load(open(We_file, 'rb'))
		Vi = cPickle.load(open(Vi_file, 'rb'))
	
	span_size = 120
	def data_vectorize(title, rating, transcript):
		vec_title = np.zeros(d_word, dtype=float)
		vec_rating = rating / sum(rating)
		vec_script = []
		mask_script = []

		for word in title.split():
			vec_title += We[Vi[word]]
		if sum(vec_title) != 0:
			vec_title /= sum(vec_title) 

		script_tokens = transcript.split()
		i_from = 0
		while i_from < len(script_tokens):
			vec = np.zeros(span_size, dtype=np.int32)
			mask = np.zeros(span_size, dtype=float)
			i_to = min(i_from + span_size, len(script_tokens))
			ind = 0
			for i in range(i_from, i_to):
				if script_tokens[i] in stopWords:
					vec[ind] = 0
				else:
					vec[ind] = Vi[script_tokens[i]]
					
				mask[ind] = 1.0
				ind += 1
			vec_script.append(vec)
			mask_script.append(mask)
			i_from = i_to
		
		vec_script = np.array(vec_script)
		mask_script = np.array(mask_script)

		# print(vec_title.shape, vec_rating.shape, vec_script.shape, mask_script.shape)
		# ((50,), (14,), (*23, 100), (*23, 100))
		return vec_title, vec_rating, vec_script, mask_script
		



	ted_train_data = []
	ted_neg_samples = []
	for key, data in ted_data.items():
		rating_array = np.zeros(len(rating_tags), dtype=float)
		if int(data['duration']) > 300 and 'transcript' in data:
			vec_title, vec_rating, vec_spans, vec_mask = data_vectorize(\
				data['description'], data['rating_array'], data['transcript'])
			ted_train_data.append((vec_title, vec_rating, vec_spans, vec_mask))


	# embedding/hidden dimensionality
	d_hidden = 50

	# number of descriptors
	num_descs = 30

	# number of negative samples per relationship
	num_negs = 50

	# word dropout probability
	p_drop = 0.75

	n_epochs = 300
	lr = 0.001
	eps = 1e-6
	span_data = ted_train_data
	num_traj = len(span_data)
	
	len_voc = len(Vi)
	wmap = Vi
	revmap = {}
	for w in wmap:
		revmap[wmap[w]] = w
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


	
