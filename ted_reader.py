import string
import csv
#import pickle
import cPickle
import numpy as np
import os.path

import nltk
from nltk import tokenize
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
nltk.download('punkt')
nltk.download('stopwords')
stopwords = set(stopwords.words('english'))

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
			ntext += ' '
		else:
			ntext += c
		i += 1
	
	return ntext

def load_data(ted_main_dir, ted_transcript_dir):
	ted_data = {}
	with open(ted_main_dir, 'r') as ted_main_file:
		ted_main_csv = csv.reader(ted_main_file, delimiter=',')
		ted_main_headers = next(ted_main_csv)

		for row in ted_main_csv:
			video = {}
			for i, k in enumerate(ted_main_headers):
				video[k] = row[i].strip()

			ted_data[video['url']] = video


	with open(ted_transcript_dir, 'r') as ted_transcripts:
		ted_transcript_csv = csv.reader(ted_transcripts, delimiter=',')
		ted_transcript_headers = next(ted_transcript_csv)

		for i, row in enumerate(ted_transcript_csv):
			url  = row[1].strip()
			text = row[0].strip().lower()
			text = text_cleaner(text)

			ted_data[url]['transcript'] = text

	cPickle.dump(ted_data, open("ted_data.pkl", "wb"))


if __name__ == '__main__':
	print 'loading data...'
	load_data(ted_main_dir, ted_transcript_dir)
