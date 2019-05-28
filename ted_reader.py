# Renewed 2019-05-24: pkls
# Renewed 2019-05-27: Prep

import string
import csv
import cPickle
import ast, json
import numpy as np
import os.path
import argparse
from sklearn.feature_extraction.text import TfidfVectorizer

import nltk
from nltk import tokenize
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.tag import pos_tag
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
stopwords = stopwords.words('english')

ted_main_dir = "./data/ted_main.csv"
ted_transcript_dir = "./data/transcripts.csv"


def ct(char):
	if char == '(':
		return 1
	elif char == ')':
		return 2
	elif char == '\'':
		return 3
	elif char == '[':
		return 5
	elif char == ']':
		return 6
	elif char in string.punctuation + '\"':
		return 4
	elif char in '0123456789':
		return 7
	elif char == ' ':
		return -1
	else:
		return 0


def text_cleaner(text):
	ntext = ''
	text = text.lower()
	i = 0
	prep_gr = 20
	while i < len(text):
		c = text[i]
		ctc = ct(c)
		ctp = -1 if i == 0 else ct(text[i-1])
		ctn = -1 if i == len(text)-1 else ct(text[i+1])

		if ctc == 5:
			i += 1
			pgr = 0
			while ct(text[i]) != 6:
				pgr += 1
				if pgr > prep_gr: # [ sentence.... ] is too long (error: not found)
					break
			i += pgr
			continue
		elif ctc == 6:
			i += 1
			continue
		elif ctc == 1:
			tmptext = ''
			i += 1
			while ct(text[i]) != 2:
				if text[i] != ' ':
					tmptext += text[i]
				i += 1
			if tmptext.isalpha() and len(tmptext) > 5 and len(tmptext) <= 15:
				ntext += ' (' + tmptext + ') '
			i += 1
			continue
		#elif ctc == 2 and ctn != -1:
		#	ntext += c + ' '
		elif ctc == 4:
			if ctp != -1:
				ntext += ' '
			ntext += c
			if ctn != -1:
				ntext += ' '
		elif ctc == 7:
			if ctp != 7:
				ntext += ' '
			ntext += c
			if ctn != 7:
				ntext += ' '
		elif ctc == 3:
			ntext += '\''
		else:
			ntext += c
		i += 1
	
	#return ntext
	#return ''.join(c for c in ntext if c.isalpha() or c == ' ')
	new_tokens = []
	word_tokens = ntext.split()
	for idx, word in enumerate(word_tokens):
		if word == ':':
			if (idx >= 2 and ct(word_tokens[idx-2]) == 4) or idx == 1:
				if new_tokens:
					new_tokens.pop()
		else:
			new_tokens.append(unicode(word, errors='ignore'))
	return new_tokens


def load_data(ted_main_dir, ted_transcript_dir):
	ted_data = {}
	ted_voca = {}
	ted_actn = {}
	act_count = {}
	ted_freq = {}
	with open(ted_main_dir, 'r') as ted_main_file:
		ted_main_csv = csv.reader(ted_main_file, delimiter=',')
		ted_main_headers = next(ted_main_csv)
		for row in ted_main_csv:
			video = {}
			for i, k in enumerate(ted_main_headers):
				video[k] = row[i].strip()
			video['num_speaker'] = int(video['num_speaker'])
			if video['num_speaker'] > 1:
				continue
			video['transcript'] = []
			ted_data[video['url']] = video

	with open(ted_transcript_dir, 'r') as ted_transcripts:
		ted_transcript_csv = csv.reader(ted_transcripts, delimiter=',')
		ted_transcript_headers = next(ted_transcript_csv)
		for i, row in enumerate(ted_transcript_csv):
			url  = row[1].strip()
			if url not in ted_data:
				continue
			text = row[0].strip().lower()
			text_tokens = text_cleaner(text)
			ted_data[url]['transcript'] = text_tokens
			for word in text_tokens:
				if word == '': continue
				if word not in ted_voca:
					ted_voca[word] = len(ted_voca)
				ted_freq.setdefault(word, 0)
				ted_freq[word] += 1
				if word[0] == '(' and word[-1] == ')':
					if word not in ted_actn:
						ted_actn[word] = len(ted_actn)
						act_count[word] = 0
					act_count[word] += 1
	
	urls = ted_data.keys()
	for url in urls:
		if len(ted_data[url]['transcript']) == 0:
			del ted_data[url]

	with open("./prep/ted-voca-freq.txt", 'w') as wf:
		d = ted_freq
		for w in sorted(d, key=d.get, reverse=True):
			wf.write("%s\t%d\n"%(w, d[w]))
			
	with open("./prep/ted-actn-freq.txt", 'w') as wf:
		d = act_count
		for w in sorted(d, key=d.get, reverse=True):
			wf.write("%s\t%d\n"%(w, d[w]))
			
	return ted_data, ted_voca, ted_actn, ted_freq


def prep_data(ted_data, ted_voca, ted_actn, ted_freq):
	corpus = {url:ted_data[url]['transcript'] for url in ted_data if 'transcript' in ted_data[url]}

	voca_list = ted_freq.keys()
	for voca in voca_list:
		if ted_freq[voca] < 5000:
			del ted_freq[voca]
	
	span_size = 40
	nvmap = {'_': 0}
	stopfreq = set(stopwords + ted_freq.keys())
	prep_corpus = {}
	count = 0
	totcount = len(corpus)
	pt = 10
	print "Preparation process of data"
	for url, text in corpus.items():
		count += 1
		if count >= len(corpus)//100*pt:
			print pt, '% processed'
			pt += 10
		tot_sents = []
		tot_laughter = []
		tot_applause = []
		for sent in ' '.join(text).split('.'):
			word_tokens = sent.split(' ')
			word_filtered = []
			pos_laughter = []
			pos_applause = []
			idx = 0
			for word in word_tokens:
				if word == '': continue
				if word in ted_actn:
					actword = word[1:-1]
					if actword == 'laughter' or actword == 'laughs' or actword == 'laughting':
						pos_laughter.append(idx)
					elif actword == 'applause':
						pos_applause.append(idx)
				else:
					word_filtered.append(word)
				idx += 1
				
			sentlen = len(word_filtered)
			if sentlen == 0:
				continue

			tagged_list = pos_tag(word_tokenize(' '.join(word_filtered)))
			new_tokens = []
			for w, t in tagged_list:
				if w in stopfreq:
					new_tokens.append(w)
				else:
					new_tokens.append(t)

			#print '#', ' '.join(new_tokens)

			nsents = []
			tmp = []
			if len(new_tokens) == 1:
				w = new_tokens[0]
				if w not in nvmap:
					nvmap[w] = len(nvmap)
				if len(tmp) < span_size:
					tmp.append(nvmap[w])
				else:
					nsents.append(tmp)
					tmp = []
					tmp.append(nvmap[w])
			else:
				for idx in range(len(new_tokens)-1):
					w = new_tokens[idx] + ' ' + new_tokens[idx+1]
					if w not in nvmap:
						nvmap[w] = len(nvmap)

					if len(tmp) < span_size:
						tmp.append(nvmap[w])
					else:
						nsents.append(tmp)
						tmp = []
						tmp.append(nvmap[w])

			if len(tmp) > 0:
				tmp.extend([0]*(span_size - len(tmp)))
				nsents.append(tmp)

			#if len(nsents) == 0:
			#	nsents.append([0]*span_size)

			if (len(pos_laughter) > 0 and (pos_laughter[-1] - len(pos_laughter) + 1) // span_size > len(nsents)) or \
				(len(pos_applause) > 0 and (pos_applause[-1] - len(pos_applause) + 1) // span_size > len(nsents)):
				nsents.append([0]*span_size)
				
			tot_sents.extend(nsents)
			
			nlaughter = [0] * len(nsents)
			napplause = [0] * len(nsents)
			for it, l_idx in enumerate(pos_laughter):
				nlaughter[(l_idx-it)//span_size] += 1
			for it, a_idx in enumerate(pos_applause):
				napplause[(a_idx-it)//span_size] += 1
			tot_laughter.extend(nlaughter)
			tot_applause.extend(napplause)

		if len(tot_sents) != 0:
			prep_corpus[url] = {}
			prep_corpus[url]['length'] = len(tot_sents)
			prep_corpus[url]['script'] = np.array(tot_sents)
			prep_corpus[url]['laughter'] = np.array(tot_laughter)
			prep_corpus[url]['applause'] = np.array(tot_applause)

			'''print prep_corpus[url]
			for tag in prep_corpus[url]:
				if tag != 'length':
					print tag, prep_corpus[url][tag].shape
				else:
					print tag, prep_corpus[url][tag]'''
	print 100, '% processed'

	ted_test_corpus_dir = './prep/ted_test_corpus.pkl'
	test_corpus = {}
	count = 0
	for url in prep_corpus:
		test_corpus[url] = prep_corpus[url]
		count += 1
		if count >= 100:
			break
	cPickle.dump(test_corpus, open(ted_test_corpus_dir, "wb"))
	
	return prep_corpus, nvmap

				
	# (2491, 635, 18)



if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Process some integers.')
	parser.add_argument('--init', dest='init', action='store_true')
	parser.add_argument('--init-prep', dest='init_prep', action='store_true')
	parser.add_argument('--no-init', dest='init', action='store_false')
	parser.set_defaults(init=False, init_prep=False)
	args = parser.parse_args()

	ted_data_pkl_dir = './prep/ted_data.pkl'
	ted_voca_pkl_dir = './prep/ted_voca.pkl'
	ted_actn_pkl_dir = './prep/ted_actn.pkl'
	ted_freq_pkl_dir = './prep/ted_freq.pkl'
	if not os.path.isfile(ted_data_pkl_dir) \
		or not os.path.isfile(ted_voca_pkl_dir) \
		or not os.path.isfile(ted_actn_pkl_dir) \
		or not os.path.isfile(ted_freq_pkl_dir) \
		or args.init:
		print 'loading and preprocess transcript data...'
		ted_data, ted_voca, ted_actn, ted_freq = load_data(ted_main_dir, ted_transcript_dir)
		cPickle.dump(ted_data, open(ted_data_pkl_dir, "wb"))
		cPickle.dump(ted_voca, open(ted_voca_pkl_dir, "wb"))
		cPickle.dump(ted_actn, open(ted_actn_pkl_dir, "wb"))
		cPickle.dump(ted_freq, open(ted_freq_pkl_dir, "wb"))
	else:
		print '{} has found, load using cPickle.'.format(ted_data_pkl_dir)
		print '{} has found, load using cPickle.'.format(ted_voca_pkl_dir)
		print '{} has found, load using cPickle.'.format(ted_actn_pkl_dir)
		print '{} has found, load using cPickle.'.format(ted_freq_pkl_dir)
		ted_data = cPickle.load(open(ted_data_pkl_dir, "rb"))
		ted_voca = cPickle.load(open(ted_voca_pkl_dir, "rb"))
		ted_actn = cPickle.load(open(ted_actn_pkl_dir, "rb"))
		ted_freq = cPickle.load(open(ted_freq_pkl_dir, "rb"))
	print(len(ted_data), len(ted_voca), len(ted_actn), len(ted_freq))

	ted_prep_corpus_dir = './prep/ted_prep_corpus.pkl'
	ted_prep_nvmap_dir = './prep/ted_prep_nvmap.pkl'
	if not os.path.isfile(ted_prep_corpus_dir) \
		or not os.path.isfile(ted_prep_nvmap_dir) \
		or args.init \
		or args.init_prep:
		prep_corpus, prep_nvmap = prep_data(ted_data, ted_voca, ted_actn, ted_freq)
		cPickle.dump(prep_corpus, open(ted_prep_corpus_dir, "wb"))
		cPickle.dump(prep_nvmap, open(ted_prep_nvmap_dir, "wb"))
	else:
		print '{} has found, load using cPickle.'.format(ted_prep_corpus_dir)
		print '{} has found, load using cPickle.'.format(ted_prep_nvmap_dir)
		prep_corpus = cPickle.load(open(ted_prep_corpus_dir, "rb"))
		prep_nvmap = cPickle.load(open(ted_prep_nvmap_dir, "rb"))

